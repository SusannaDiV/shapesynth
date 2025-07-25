import dataclasses
import os
import pickle
import torch
import time
from torch import nn
from tqdm.auto import tqdm

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.chem.stack import Stack
from synformer.data.common import ProjectionBatch, TokenType
from synformer.models.encoder import get_encoder
from synformer.models.decoder import Decoder
from synformer.models.adapter import UniMolAdapter, ContinuousCodeProjector
from .classifier_head import ClassifierHead
from .decoder import Decoder
from .encoder import get_encoder, ShapeEncoder
from .fingerprint_head import ReactantRetrievalResult, get_fingerprint_head

# Import for DESERT functionality
from synformer.models.desert.inference import run_desert_inference
from synformer.models.desert.encoder import create_fragment_encoder


@dataclasses.dataclass
class _ReactantItem:
    reactant: Molecule
    index: int
    score: float

    def __iter__(self):
        return iter([self.reactant, self.index, self.score])


@dataclasses.dataclass
class _ReactionItem:
    reaction: Reaction
    index: int
    score: float

    def __iter__(self):
        return iter([self.reaction, self.index, self.score])


@dataclasses.dataclass
class PredictResult:
    token_logits: torch.Tensor
    token_sampled: torch.Tensor
    reaction_logits: torch.Tensor
    retrieved_reactants: ReactantRetrievalResult

    def to(self, device: torch.device):
        self.__class__(
            self.token_logits.to(device),
            self.token_sampled.to(device),
            self.reaction_logits.to(device),
            self.retrieved_reactants,
        )
        return self

    def best_token(self) -> list[TokenType]:
        return [TokenType(t) for t in self.token_logits.argmax(dim=-1).detach().cpu().tolist()]

    def top_reactions(self, topk: int, rxn_matrix: ReactantReactionMatrix) -> list[list[_ReactionItem]]:
        topk = min(topk, self.reaction_logits.size(-1))
        logit, index = self.reaction_logits.topk(topk, dim=-1, largest=True)
        bsz = logit.size(0)
        out: list[list[_ReactionItem]] = []
        for i in range(bsz):
            out_i: list[_ReactionItem] = []
            for j in range(topk):
                idx = int(index[i, j].item())
                out_i.append(
                    _ReactionItem(
                        reaction=rxn_matrix.reactions[idx],
                        index=idx,
                        score=float(logit[i, j].item()),
                    )
                )
            out.append(out_i)
        return out

    def top_reactants(self, topk: int) -> list[list[_ReactantItem]]:
        bsz = self.retrieved_reactants.reactants.shape[0]
        score_all = 1.0 / (self.retrieved_reactants.distance.reshape(bsz, -1) + 0.1)
        index_all = self.retrieved_reactants.indices.reshape(bsz, -1)
        mols = self.retrieved_reactants.reactants.reshape(bsz, -1)

        topk = min(topk, mols.shape[-1])
        best_index = (-score_all).argsort(axis=-1)

        out: list[list[_ReactantItem]] = []
        for i in range(bsz):
            out_i: list[_ReactantItem] = []
            for j in range(topk):
                idx = int(best_index[i, j])
                out_i.append(
                    _ReactantItem(
                        reactant=mols[i, idx],
                        index=index_all[i, idx],
                        score=score_all[i, idx],
                    )
                )
            out.append(out_i)
        return out


@dataclasses.dataclass
class GenerateResult:
    code: torch.Tensor
    code_padding_mask: torch.Tensor

    token_types: torch.Tensor
    token_padding_mask: torch.Tensor

    rxn_indices: torch.Tensor

    reactant_fps: torch.Tensor
    predicted_fps: torch.Tensor
    reactant_indices: torch.Tensor

    reactants: list[list[Molecule | None]]
    reactions: list[list[Reaction | None]]

    @property
    def batch_size(self):
        return self.token_types.size(0)

    @property
    def seq_len(self):
        return self.token_types.size(1)

    def to_(self, device: str | torch.device):
        self.code = self.code.to(device)
        self.code_padding_mask = self.code_padding_mask.to(device)
        self.token_types = self.token_types.to(device)
        self.token_padding_mask = self.token_padding_mask.to(device)
        self.rxn_indices = self.rxn_indices.to(device)
        self.reactant_fps = self.reactant_fps.to(device)
        self.predicted_fps = self.predicted_fps.to(device)
        self.reactant_indices = self.reactant_indices.to(device)

    def build(self):
        stacks = [Stack() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                if self.token_types[i, j] == TokenType.START:
                    continue
                if self.token_types[i, j] == TokenType.END:
                    break
                if self.token_types[i, j] == TokenType.REACTION:
                    rxn = self.reactions[i][j]
                    if rxn is None:
                        break
                    success = stacks[i].push_rxn(rxn, int(self.rxn_indices[i, j].item()))
                    if not success:
                        break
                elif self.token_types[i, j] == TokenType.REACTANT:
                    mol = self.reactants[i][j]
                    if mol is None:
                        break
                    stacks[i].push_mol(mol, int(self.reactant_indices[i, j].item()))
        return stacks


class Synformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Start timing
        self.start_time = time.time()
        
        # Store encoder type for later use
        self.encoder_type = cfg.encoder_type
        
        if cfg.encoder_type == "shape_pretrained":
            print("\nLoading pretrained encoder...")
            # Initialize encoder on CPU
            self.pretrained_encoder = ShapeEncoder.from_pretrained(
                cfg.encoder.pretrained,
                device='cpu'  # Force CPU initialization
            )
            self.pretrained_encoder.eval()  # Set to eval mode
            self.encoder = None
        elif cfg.encoder_type == "desert":
            print("\nUsing DESERT encoder...")
            # Store paths for DESERT model and vocabulary
            self.desert_model_path = cfg.encoder.desert_model_path
            self.vocab_path = cfg.encoder.vocab_path
            self.shape_patches_path = cfg.encoder.shape_patches_path if hasattr(cfg.encoder, 'shape_patches_path') else None
            print(f"Initialized DESERT with shape_patches_path: {self.shape_patches_path}")
            self.encoder = None
        else:
            self.encoder = get_encoder(cfg.encoder_type, cfg.encoder)
        
        print(f"Initialized Synformer with encoder_type: {cfg.encoder_type}")
        
        cfg.decoder.d_model = 768#1024  # Override the 768 from config
        decoder_kwargs = {}
        if "decoder_only" not in cfg.decoder and cfg.encoder_type == "none":
            decoder_kwargs["decoder_only"] = True
        self.decoder = Decoder(**cfg.decoder, **decoder_kwargs)
        self.d_model: int = 768#int = 1024
        
        # Remove projection since we're matching dimensions
        self.token_head = ClassifierHead(self.d_model, max(TokenType) + 1)
        self.reaction_head = ClassifierHead(self.d_model, cfg.decoder.num_reaction_classes)
        self.fingerprint_head = get_fingerprint_head(cfg.fingerprint_head_type, cfg.fingerprint_head)

        # Initialize both projector and adapter if adapter config exists
        if hasattr(cfg, 'adapter'):
            self.projector = ContinuousCodeProjector(
                in_dim=1024,  # Encoder output dimension
                out_dim=768   # Decoder input dimension
            )
            self.adapter = UniMolAdapter(**cfg.adapter)
        else:
            self.projector = None
            self.adapter = None

    def encode(self, batch, mixture_weight: float | None = None):
        # Get device from model parameters
        device = next(self.decoder.parameters()).device
        
        # Check if we're using DESERT encoder
        if hasattr(self, 'encoder_type') and self.encoder_type == "desert":
            # DESERT molecule generation
            if not os.path.exists(self.desert_model_path):
                raise FileNotFoundError(f"DESERT model file not found: {self.desert_model_path}")
            
            # Load vocabulary to understand fragment IDs
            vocab_path = self.vocab_path
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            
            # Create reverse mapping from ID to token
            id_to_token = {}
            for token, (_, _, idx) in vocab.items():
                id_to_token[idx] = token
            
            # Get SMILES string - use smiles_str if available (for inference)
            if 'smiles_str' in batch:
                smiles_str = batch['smiles_str']
                print(f"Using SMILES string from batch: {smiles_str}")
            else:
                # For training, we need to handle the tokenized SMILES
                # This is a placeholder - in practice, you might need to convert tokens back to SMILES
                # or store the original SMILES in the batch
                raise ValueError("DESERT encoder requires 'smiles_str' in the batch during inference")
            
            # Use shape_patches_path from instance variable if not provided in batch
            shape_patches_path = batch.get('shape_patches_path', self.shape_patches_path)
            
            # Run DESERT inference
            print(f"Running DESERT inference with shape_patches_path: {shape_patches_path}")
            desert_sequences = run_desert_inference( # WORKS! desert_sequences = run_desert_inference(smiles_str, self.desert_model_path, shape_patches_path)
                model_path=self.desert_model_path, 
                shape_patches_path=shape_patches_path, 
                device=device
            )
            desert_sequence = desert_sequences[0]  # Take the first sequence
            
            # Each element in desert_sequence is a tuple (fragment_id, translation, rotation)
            print(f"Received {len(desert_sequence)} fragments with spatial information")
            for i, (frag_id, trans, rot) in enumerate(desert_sequence[:5]):
                frag_name = id_to_token.get(frag_id, "Unknown")
                print(f"Fragment {i+1}: ID={frag_id} ({frag_name}), Translation={trans}, Rotation={rot}")
            if len(desert_sequence) > 5:
                print(f"... and {len(desert_sequence)-5} more fragments")
            
            # Create and initialize the fragment encoder
            print("\n=== Step 2: Running Fragment Encoder ===\n")
            
            # mixture_weight=1 means we are using 100% spatial information from DESERT
            # mixture_weight=0 would mean using all zeros (no spatial information)
            # Use the passed mixture_weight if available, otherwise default (e.g., 0.8 or a config value)
            current_mixture_weight = mixture_weight if mixture_weight is not None else 0.8 # Default to 0.8 if not provided
            print(f"Using DESERT mixture_weight = {current_mixture_weight} for fragment encoding.")

            # Pass grid_resolution and max_dist parameters to ensure proper spatial encoding
            encoder = create_fragment_encoder(
                vocab_path=vocab_path, 
                device=device,
                grid_resolution=0.5,  # Default value from excellent.py
                max_dist=6.75,        # Default value from excellent.py
                mixture_weight=current_mixture_weight # Use the determined mixture_weight
            )
            
            # Encode the DESERT sequence
            print(f"Encoding DESERT sequence with {len(desert_sequence)} fragments including spatial information")
            print(f"Using mixture_weight={current_mixture_weight} (0=all zeros, 1=full spatial encoding)")
            encoder_output = encoder.encode_desert_sequence(desert_sequence, device=device)
            
            # Handle different return types based on encoder implementation
            if hasattr(encoder_output, 'code'):
                code = encoder_output.code
                code_padding_mask = encoder_output.code_padding_mask
                print(f"Generated embeddings tensor with shape: {code.shape}")
                print(f"Generated padding mask with shape: {code_padding_mask.shape}")
            else:
                # Backward compatibility with older encoder implementations
                code = encoder_output
                print(f"Generated embeddings tensor with shape: {code.shape}")
                # Create a dummy padding mask (all False)
                batch_size = code.shape[0]
                seq_len = code.shape[1]
                code_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
                print(f"Created dummy code_padding_mask with shape: {code_padding_mask.shape}")
                
            # Apply additional normalization to ensure the embeddings are in the expected distribution
            # This is critical for the decoder to work properly
            code_mean = code.mean()
            code_std = code.std()
            print(f"Encoder output before normalization - mean: {code_mean.item():.4f}, std: {code_std.item():.4f}")
            
            # Always normalize - this is mandatory to get proper decoder behavior
            print("Applying additional normalization to encoder output")
            # Center and scale the embeddings to have a fixed mean and std that works well with the decoder
            # Target a distribution with mean around 0 and std around 0.05-0.1
            code = (code - code_mean) / (code_std + 1e-5) * 0.05
            
            # Check values after normalization
            code_min = code.min().item()
            code_max = code.max().item()
            print(f"Encoder output after normalization - mean: {code.mean().item():.4f}, std: {code.std().item():.4f}, min: {code_min:.4f}, max: {code_max:.4f}")
            
            # Empty loss dict for encoder
            encoder_loss_dict = {}
            
            return code, code_padding_mask, encoder_loss_dict
        
        # Handle case where encoder is None
        if self.encoder is None:
            # If encoder is None and not using DESERT, we can't proceed
            raise ValueError("Encoder is not initialized and not using DESERT encoder")
        
        # Determine encoder type from the encoder instance
        encoder_type = "unknown"
        if isinstance(self.encoder, ShapeEncoder):
            encoder_type = "shape"
        else:
            # Assume SMILES encoder for any other type
            encoder_type = "smiles"
        
        # Get d_model from encoder - handle different encoder types
        if hasattr(self.encoder, '_d_model'):
            d_model = self.encoder._d_model
        elif hasattr(self.encoder, 'd_model'):
            d_model = self.encoder.d_model
        else:
            # Default value for SMILES encoder
            d_model = 768  # Standard transformer dimension
            
        # Create embedding for shape encoder
        embed = nn.Embedding(2, d_model).to(device)
        
        # For inference, batch might be a Molecule or a simple dict
        # Check if we're in inference mode
        is_inference = False
        if isinstance(batch, dict):
            # Check if this is a simple inference batch
            if 'smiles' in batch or 'molecule' in batch:
                is_inference = True
        
        # Handle inference mode
        if is_inference:
            # In inference mode, we just need to encode the SMILES
            if encoder_type == "smiles":
                # For inference, we'll use the batch directly
                # The encoder should handle the molecule object
                code, code_padding_mask, encoder_loss_dict = self.encoder(batch)
                return code, code_padding_mask, encoder_loss_dict
            else:
                raise ValueError(f"Inference not implemented for encoder type: {encoder_type}")
        
        # Regular training/validation mode
        if encoder_type == "shape":
            # Shape encoder
            # Encode
            code, code_padding_mask, encoder_loss_dict = self.encoder(
                batch['shape_grids'],
                batch['shape_grid_info'],
                batch['shape_grid_mask'],
                batch['shape_grid_labels'],
            )
            
            # Add shape type embedding
            code = code + embed(torch.zeros_like(code_padding_mask, dtype=torch.long))
        elif encoder_type == "smiles":
            # SMILES encoder
            code, code_padding_mask, encoder_loss_dict = self.encoder(
                batch['smiles_tokens'],
                batch['smiles_padding_mask'],
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        return code, code_padding_mask, encoder_loss_dict

    def get_loss(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        **options,
    ):
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=token_padding_mask,
        )[:, :-1]

        token_types_gt = token_types[:, 1:].contiguous()
        rxn_indices_gt = rxn_indices[:, 1:].contiguous()
        reactant_fps_gt = reactant_fps[:, 1:].contiguous()

        loss_dict: dict[str, torch.Tensor] = {}
        aux_dict: dict[str, torch.Tensor] = {}

        # NOTE: token_padding_mask is True for padding tokens: ~token_padding_mask[:, :-1].contiguous()
        # We set the mask to None so the model perfers producing the `END` token when the embedding makes no sense
        loss_dict["token"] = self.token_head.get_loss(h, token_types_gt, None)
        loss_dict["reaction"] = self.reaction_head.get_loss(h, rxn_indices_gt, token_types_gt == TokenType.REACTION)

        fp_loss, fp_aux = self.fingerprint_head.get_loss(
            h,
            reactant_fps_gt,
            token_types_gt == TokenType.REACTANT,
            **options,
        )
        loss_dict.update(fp_loss)
        aux_dict.update(fp_aux)

        return loss_dict, aux_dict

    def get_loss_shortcut(self, batch: ProjectionBatch, **options):
        # Note: mixture_weight is not typically varied during training loss calculation
        # It's primarily an inference/sampling time parameter. If needed here, it would require passing through.
        code, code_padding_mask, encoder_loss_dict = self.encode(batch)
        loss_dict, aux_dict = self.get_loss(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=batch["token_types"],
            rxn_indices=batch["rxn_indices"],
            reactant_fps=batch["reactant_fps"],
            token_padding_mask=batch["token_padding_mask"],
            **options,
        )
        loss_dict.update(encoder_loss_dict)
        return loss_dict, aux_dict

    def get_log_likelihood(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        **options,
    ) -> dict[str, torch.Tensor]:
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=token_padding_mask,
        )[:, :-1]

        token_types_gt = token_types[:, 1:].contiguous()
        rxn_indices_gt = rxn_indices[:, 1:].contiguous()
        reactant_fps_gt = reactant_fps[:, 1:].contiguous()

        ll_token_types = self.token_head.get_log_likelihood(h, token_types_gt, ~token_padding_mask[:, 1:])
        ll_rxn = self.reaction_head.get_log_likelihood(h, rxn_indices_gt, token_types_gt == TokenType.REACTION)
        ll_bb = self.fingerprint_head.get_log_likelihood(h, reactant_fps_gt, token_types_gt == TokenType.REACTANT)
        ll = ll_token_types + ll_rxn + ll_bb
        return {
            "token": ll_token_types,
            "reaction": ll_rxn,
            "reactant": ll_bb,
            "total": ll,
        }

    def get_log_likelihood_shortcut(self, batch: ProjectionBatch, **options):
        # Similar to get_loss_shortcut, mixture_weight not typically varied here.
        code, code_padding_mask, _ = self.encode(batch)
        return self.get_log_likelihood(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=batch["token_types"],
            rxn_indices=batch["rxn_indices"],
            reactant_fps=batch["reactant_fps"],
            token_padding_mask=batch["token_padding_mask"],
            **options,
        )

    @torch.no_grad()
    def predict(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        topk: int = 4,
        temperature_token: float = 0.1,
        **options,
    ):
        # Print total execution time before prediction
        end_time = time.time()
        print(f"\nTotal Synformer execution time: {end_time - self.start_time:.2f} seconds")
        
        # Debug prints for input tensors
        print(f"DEBUG - predict input shapes:")
        print(f"  code: {code.shape if code is not None else None}")
        print(f"  code_padding_mask: {code_padding_mask.shape if code_padding_mask is not None else None}")
        print(f"  token_types: {token_types.shape}")
        print(f"  rxn_indices: {rxn_indices.shape}")
        print(f"  reactant_fps: {reactant_fps.shape}")
        
        # Check for NaNs in encoder output
        if code is not None:
            has_nan = torch.isnan(code).any()
            print(f"  code contains NaN: {has_nan}")
            if has_nan:
                print(f"  NaN percentage: {torch.isnan(code).sum() / code.numel() * 100:.2f}%")
        
        # Generate embeddings
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=None,
        )
        h_next = h[:, -1]  # (bsz, h_dim)
        print(f"  decoder output shape: {h.shape}")
        print(f"  h_next shape: {h_next.shape}")
        print(f"  h_next distribution - mean: {h_next.mean().item():.4f}, std: {h_next.std().item():.4f}")
        
        # Get token logits and sample token
        token_logits = self.token_head.predict(h_next)
        print(f"  token_logits shape: {token_logits.shape}")
        
        # Print logits stats - handle multi-batch case properly
        logits_min = token_logits.min().item()
        logits_max = token_logits.max().item()
        logits_mean = token_logits.mean().item()
        print(f"  token logits min/max/mean: {logits_min:.4f}/{logits_max:.4f}/{logits_mean:.4f}")
        
        # ALWAYS normalize logits to prevent numerical issues
        print(f"  Normalizing token logits to prevent numerical issues")
        # Center around the max value to maintain relative relationships
        token_logits = token_logits - token_logits.max(dim=-1, keepdim=True)[0]
        # Scale down to prevent underflow/overflow
        token_logits = token_logits * 1.0  # Use a small scale factor
        # Apply a temperature directly to the logits
        token_logits = token_logits / temperature_token
        
        # Check the normalized logits
        norm_logits_min = token_logits.min().item()
        norm_logits_max = token_logits.max().item()
        norm_logits_mean = token_logits.mean().item()
        print(f"  Normalized token logits min/max/mean: {norm_logits_min:.4f}/{norm_logits_max:.4f}/{norm_logits_mean:.4f}")
        
        # Apply softmax to normalized logits (no temperature here since already applied)
        softmax_token = torch.nn.functional.softmax(token_logits, dim=-1)
        
        # Print softmax distribution for each token type (for single examples)
        if token_types.shape[0] == 1:
            for token_type in range(4):  # Assuming 4 token types
                prob = softmax_token[0, token_type].item()
                print(f"  Token type {token_type} ({TokenType(token_type).name}) probability: {prob:.8f}")
        
        # Calculate entropy - handle multi-batch case properly
        entropy = (-softmax_token * torch.log(softmax_token + 1e-10)).sum(dim=-1)
        # For multi-batch, just report mean entropy
        mean_entropy = entropy.mean().item()
        print(f"  token distribution entropy: {mean_entropy:.4f}")
        
        # Force temperature annealing - try progressively higher temperatures until we get some entropy
        if mean_entropy < 0.1 and token_types.shape[0] == 1:
            print(f"  Entropy too low, trying higher temperatures")
            for temp in [0.5, 1.0, 2.0, 5.0, 10.0]:
                print(f"  Trying temperature {temp}")
                # Apply new temperature
                temp_logits = token_logits * (temperature_token / temp)
                temp_softmax = torch.nn.functional.softmax(temp_logits, dim=-1)
                # Calculate new entropy
                temp_entropy = (-temp_softmax * torch.log(temp_softmax + 1e-10)).sum(dim=-1)
                mean_temp_entropy = temp_entropy.mean().item()
                print(f"  New entropy with temp={temp}: {mean_temp_entropy:.4f}")
                
                # Print distribution with this temperature
                for token_type in range(4):
                    prob = temp_softmax[0, token_type].item()
                    print(f"    Token type {token_type} ({TokenType(token_type).name}) probability: {prob:.8f}")
                
                # If entropy is good enough, use this temperature
                if mean_temp_entropy > 0.1:
                    print(f"  Using temperature {temp} with entropy {mean_temp_entropy:.4f}")
                    softmax_token = temp_softmax
                    break
        
        token_sampled = torch.multinomial(
            softmax_token,
            num_samples=1,
        )
        
        # For batches, we report the first sampled token or indicate this is a batch
        if token_types.shape[0] == 1:
            print(f"  sampled token: {token_sampled.item()} ({TokenType(token_sampled.item()).name})")
        else:
            print(f"  sampled {token_types.shape[0]} tokens (batch mode)")
        
        # Get reaction logits
        reaction_logits = self.reaction_head.predict(h_next)[..., : len(rxn_matrix.reactions)]
        print(f"  reaction_logits shape: {reaction_logits.shape}")
        
        # Get retrieved reactants
        retrieved_reactants = self.fingerprint_head.retrieve_reactants(
            h_next,
            fpindex,
            topk,
            mask=token_sampled == TokenType.REACTANT,
            **options,
        )
        
        # Only show detailed reactant info for single samples
        if token_types.shape[0] == 1 and token_sampled.item() == TokenType.REACTANT.value:
            print(f"  retrieved_reactants: {len(retrieved_reactants.reactants.flatten())} reactants")
            if hasattr(retrieved_reactants, 'distance'):
                print(f"  closest reactant distance: {retrieved_reactants.distance.min():.4f}")
        elif token_sampled.eq(TokenType.REACTANT).any():
            print(f"  retrieved reactants for batch with {token_sampled.eq(TokenType.REACTANT).sum().item()} REACTANT tokens")
        
        return PredictResult(token_logits, token_sampled, reaction_logits, retrieved_reactants)

    @torch.no_grad()
    def generate_without_stack(
        self,
        batch: ProjectionBatch,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_len: int = 24,
        temperature_token: float = 1.0,
        temperature_reaction: float = 1.0,
        temperature_reactant: float = 1.0,
        **options,
    ):
        print("\nDEBUG - Starting generation...")
        code, code_padding_mask, _ = self.encode(batch)
        bsz = code.size(0)
        fp_dim = self.fingerprint_head.fingerprint_dim

        token_padding_mask = torch.full([bsz, 1], fill_value=False, dtype=torch.bool, device=code.device)
        token_types = torch.full([bsz, 1], fill_value=TokenType.START, dtype=torch.long, device=code.device)
        rxn_indices = torch.full([bsz, 1], fill_value=0, dtype=torch.long, device=code.device)
        reactant_fps = torch.zeros([bsz, 1, fp_dim], dtype=torch.float, device=code.device)
        predicted_fps = torch.zeros([bsz, 1, fp_dim], dtype=torch.float, device=code.device)
        reactant_indices = torch.full([bsz, 1], fill_value=-1, dtype=torch.long, device=code.device)
        reactants: list[list[Molecule | None]] = [[None] for _ in range(bsz)]
        reactions: list[list[Reaction | None]] = [[None] for _ in range(bsz)]

        print(f"DEBUG - Initial shapes:")
        print(f"  code: {code.shape}")
        print(f"  code_padding_mask: {code_padding_mask.shape}")
        print(f"  token_types: {token_types.shape}")
        print(f"  rxn_indices: {rxn_indices.shape}")
        print(f"  reactant_fps: {reactant_fps.shape}")
        
        generation_steps = []

        for step in tqdm(range(max_len - 1)):
            print(f"\nDEBUG - Generation step {step+1}:")
            pred = self.predict(
                code=code,
                code_padding_mask=code_padding_mask,
                token_types=token_types,
                rxn_indices=rxn_indices,
                reactant_fps=reactant_fps,
                rxn_matrix=rxn_matrix,
                fpindex=fpindex,
                temperature_token=temperature_token,
                **options,
            )

            token_padding_mask_next = torch.logical_or(
                token_types[:, -1:] == TokenType.END, token_padding_mask[:, -1:]
            )
            token_padding_mask = torch.cat([token_padding_mask, token_padding_mask_next], dim=-1)

            token_next = pred.token_sampled
            token_types = torch.cat([token_types, token_next], dim=-1)
            
            generation_steps.append({
                'step': step + 1,
                'token': token_next.item(),
                'token_name': TokenType(token_next.item()).name
            })

            # Break if we sampled an END token
            if token_next.item() == TokenType.END.value:
                print(f"DEBUG - Found END token at step {step+1}, stopping generation")
                break

            # Reaction
            rxn_idx_next = torch.multinomial(
                torch.nn.functional.softmax(pred.reaction_logits / temperature_reaction, dim=-1),
                num_samples=1,
            )[..., 0]
            rxn_indices = torch.cat([rxn_indices, rxn_idx_next[..., None]], dim=-1)
            for b, idx in enumerate(rxn_idx_next):
                reactions[b].append(rxn_matrix.reactions[int(idx.item())])
                if token_next.item() == TokenType.REACTION.value:
                    print(f"DEBUG - Sampled reaction: {rxn_matrix.reactions[int(idx.item())].name}")

            # Reactant (building block)
            fp_scores = (
                torch.from_numpy(1.0 / (pred.retrieved_reactants.distance + 1e-4)).to(reactant_fps).reshape(bsz, -1)
            )
            fp_idx_next = torch.multinomial(
                torch.nn.functional.softmax(fp_scores / temperature_reactant, dim=-1),
                num_samples=1,
            )[..., 0]

            fp_next = (
                torch.from_numpy(pred.retrieved_reactants.fingerprint_retrieved)
                .to(reactant_fps)
                .reshape(bsz, -1, fp_dim)  # (bsz, n_fps*topk, fp_dim)
            )[range(bsz), fp_idx_next]
            reactant_fps = torch.cat([reactant_fps, fp_next[..., None, :]], dim=-2)

            pfp_next = (
                torch.from_numpy(pred.retrieved_reactants.fingerprint_predicted)
                .to(predicted_fps)
                .reshape(bsz, -1, fp_dim)  # (bsz, n_fps*topk, fp_dim)
            )[range(bsz), fp_idx_next]
            predicted_fps = torch.cat([predicted_fps, pfp_next[..., None, :]], dim=-2)

            ridx_next = (
                torch.from_numpy(pred.retrieved_reactants.indices)
                .to(reactant_indices)
                .reshape(bsz, -1)[range(bsz), fp_idx_next]
            )
            reactant_indices = torch.cat([reactant_indices, ridx_next[..., None]], dim=-1)

            reactant_next = pred.retrieved_reactants.reactants.reshape(bsz, -1)[range(bsz), fp_idx_next.cpu().numpy()]
            for b, m in enumerate(reactant_next):
                reactants[b].append(m)
                if token_next.item() == TokenType.REACTANT.value:
                    print(f"DEBUG - Sampled reactant: {m.smiles if m else 'None'}")
        
        print("\nDEBUG - Generation summary:")
        for step_info in generation_steps:
            print(f"  Step {step_info['step']}: {step_info['token_name']} ({step_info['token']})")
        
        result = GenerateResult(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            token_padding_mask=token_padding_mask,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            predicted_fps=predicted_fps,
            reactant_indices=reactant_indices,
            reactants=reactants,
            reactions=reactions,
        )
        
        # Debug final stack
        stacks = result.build()
        print("\nDEBUG - Final stacks:")
        for i, stack in enumerate(stacks):
            top_mols = stack.get_top()
            print(f"  Stack {i+1}: {len(top_mols)} molecules")
            for j, mol in enumerate(top_mols):
                print(f"    Molecule {j+1}: {mol.smiles}")
        
        return result

    def load_pretrained_decoder(self, smiles_checkpoint_path, device='cuda'):
        """
        Load a pretrained decoder from a checkpoint file.
        This replaces the current decoder with the pretrained one.
        
        Args:
            smiles_checkpoint_path: Path to the checkpoint file
            device: Device to load the decoder on
        """
        from omegaconf import OmegaConf
        
        print(f"\nLoading pretrained decoder from {smiles_checkpoint_path}...")
        
        # Load the full model checkpoint to get the decoder
        full_model_checkpoint = torch.load(smiles_checkpoint_path, map_location=device)
        config = OmegaConf.create(full_model_checkpoint['hyper_parameters']['config'])
        
        # Create and load the decoder
        # First, check the actual parameters in the checkpoint
        decoder_params = {}
        for k in full_model_checkpoint['state_dict'].keys():
            if k.startswith('model.decoder.'):
                parts = k.split('.')
                if len(parts) > 2:
                    # Extract layer information
                    if parts[2] == 'dec' and parts[3] == 'layers' and len(parts) > 4:
                        layer_num = int(parts[4])
                        if 'num_layers' not in decoder_params or layer_num + 1 > decoder_params['num_layers']:
                            decoder_params['num_layers'] = layer_num + 1
                    # Extract pe_max_len from pe_dec.pe shape
                    if parts[2] == 'pe_dec' and parts[3] == 'pe':
                        pe_shape = full_model_checkpoint['state_dict'][k].shape
                        decoder_params['pe_max_len'] = pe_shape[1]
        
        print(f"Detected decoder parameters: {decoder_params}")
        
        # Create decoder with the correct parameters
        pretrained_decoder = Decoder(
            d_model=768,
            nhead=16,
            dim_feedforward=4096,
            num_layers=decoder_params.get('num_layers', 10),  # Use detected or default to 10
            pe_max_len=decoder_params.get('pe_max_len', 32),  # Use detected or default to 32
            output_norm=False,  # Set to False to match checkpoint architecture
            fingerprint_dim=config.model.decoder.fingerprint_dim,
            num_reaction_classes=config.model.decoder.num_reaction_classes
        )
        
        # Extract decoder weights from the checkpoint
        decoder_state_dict = {}
        for k, v in full_model_checkpoint['state_dict'].items():
            if k.startswith('model.decoder.'):
                # Remove 'model.decoder.' prefix
                new_key = k.replace('model.decoder.', '')
                decoder_state_dict[new_key] = v
        
        # Load weights into decoder
        pretrained_decoder.load_state_dict(decoder_state_dict)
        pretrained_decoder.to(device)
        pretrained_decoder.eval()
        
        # Replace the current decoder with the pretrained one
        self.decoder = pretrained_decoder
        
        # Load token head, reaction head, and fingerprint head
        token_head = ClassifierHead(768, max(TokenType) + 1)
        token_head_state_dict = {}
        for k, v in full_model_checkpoint['state_dict'].items():
            if k.startswith('model.token_head.'):
                new_key = k.replace('model.token_head.', '')
                token_head_state_dict[new_key] = v
        token_head.load_state_dict(token_head_state_dict)
        token_head.to(device)
        token_head.eval()
        self.token_head = token_head
        
        reaction_head = ClassifierHead(768, config.model.decoder.num_reaction_classes)
        reaction_head_state_dict = {}
        for k, v in full_model_checkpoint['state_dict'].items():
            if k.startswith('model.reaction_head.'):
                new_key = k.replace('model.reaction_head.', '')
                reaction_head_state_dict[new_key] = v
        reaction_head.load_state_dict(reaction_head_state_dict)
        reaction_head.to(device)
        reaction_head.eval()
        self.reaction_head = reaction_head
        
        # Load fingerprint head
        fingerprint_head = get_fingerprint_head(
            config.model.fingerprint_head_type, 
            config.model.fingerprint_head
        )
        fingerprint_head_state_dict = {}
        for k, v in full_model_checkpoint['state_dict'].items():
            if k.startswith('model.fingerprint_head.'):
                new_key = k.replace('model.fingerprint_head.', '')
                fingerprint_head_state_dict[new_key] = v
        fingerprint_head.load_state_dict(fingerprint_head_state_dict)
        fingerprint_head.to(device)
        fingerprint_head.eval()
        self.fingerprint_head = fingerprint_head
        
        print("Successfully loaded pretrained decoder and heads")

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that handles the case when using DESERT encoder.
        When using DESERT, we filter out encoder weights from the state_dict.
        """
        if hasattr(self, 'encoder_type') and self.encoder_type == "desert":
            # Filter out encoder weights
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('encoder.'):
                    filtered_state_dict[k] = v
            return super().load_state_dict(filtered_state_dict, strict=False)
        else:
            return super().load_state_dict(state_dict, strict=strict)


def draw_generation_results(result: GenerateResult):
    from PIL import Image

    from synformer.utils.image import draw_text, make_grid

    bsz, len = result.token_types.size()
    im_list: list[Image.Image] = []
    for b in range(bsz):
        im: list[Image.Image] = []
        for l in range(len):
            if result.token_types[b, l] == TokenType.START:
                im.append(draw_text("START"))
            elif result.token_types[b, l] == TokenType.END:
                im.append(draw_text("END"))
                break
            elif result.token_types[b, l] == TokenType.REACTION:
                rxn = result.reactions[b][l]
                if rxn is not None:
                    im.append(rxn.draw())
            elif result.token_types[b, l] == TokenType.REACTANT:
                reactant = result.reactants[b][l]
                if reactant is not None:
                    im.append(reactant.draw())

        im_list.append(make_grid(im))
    return im_list
