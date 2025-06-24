from torch import nn
import torch
import math
import pickle
import os
import re
from synformer.data.common import ProjectionBatch, TokenType
from synformer.models.transformer.positional_encoding import PositionalEncoding
from synformer.models.encoder.base import BaseEncoder, EncoderOutput
from synformer.models.encoder.smiles import SMILESEncoder
from synformer.chem.featurize import tokenize_smiles
from sascorer import calculateScore

class Spatial3DPositionalEncoding(nn.Module):
    """
    Custom positional encoding that captures 3D spatial relationships between fragments.
    This encoding combines traditional sequence position encoding with 
    spatial information derived from fragment translations and rotations.
    """
    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1, 
                 trans_bins: int = 27000, rot_bins: int = 8000, grid_resolution: float = 0.5, 
                 max_dist: float = 6.75):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.grid_resolution = grid_resolution
        self.max_dist = max_dist
        self.box_size = int(2 * max_dist / grid_resolution + 1)
        
        # Traditional position encoding (sequence-based)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 3, 2) * (-math.log(10000.0) / (d_model // 3)))
        
        # Allocate 1/3 of dimensions for sequence position encoding
        pe_seq = torch.zeros(1, max_len, d_model // 3)
        pe_seq[0, :, 0::2] = torch.sin(position * div_term)
        pe_seq[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_seq", pe_seq)
        
        # Create learnable embeddings for translation and rotation spatial information
        self.trans_embedding = nn.Embedding(trans_bins, d_model // 3)
        self.rot_embedding = nn.Embedding(rot_bins, d_model // 3)
        
        # Layer norm for combining the different types of embeddings
        self.combine_norm = nn.LayerNorm(d_model)
        
        # Additional MLP to further process the combined embeddings
        self.spatial_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Initialize with small weights to avoid dominating the input embeddings
        with torch.no_grad():
            for layer in self.spatial_mlp:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, 0.01)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    def get_spatial_encodings(self, translations, rotations):
        """
        Generate spatial encodings based on translations and rotations.
        """
        # Get embeddings for translations and rotations
        trans_emb = self.trans_embedding(translations)  # [batch_size, seq_len, d_model//3]
        rot_emb = self.rot_embedding(rotations)         # [batch_size, seq_len, d_model//3]
        
        return trans_emb, rot_emb

    def forward(self, x: torch.Tensor, translations=None, rotations=None) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            translations: Tensor, shape [batch_size, seq_len] with translation bin indices
            rotations: Tensor, shape [batch_size, seq_len] with rotation bin indices
        """
        batch_size, seq_len, _ = x.size()
        
        # Add sequence position encoding to first third of dimensions
        seq_encoding = self.pe_seq[:, :seq_len, :]
        seq_encoding = seq_encoding.expand(batch_size, -1, -1)
        
        # If no translations/rotations provided, use only sequence encoding
        if translations is None or rotations is None:
            # Expand sequence encoding to fill all dimensions
            seq_encoding_expanded = torch.zeros_like(x)
            seq_encoding_expanded[:, :, :self.d_model // 3] = seq_encoding
            return self.dropout(x + seq_encoding_expanded)
        
        # Get spatial encodings for translations and rotations
        trans_emb, rot_emb = self.get_spatial_encodings(translations, rotations)
        
        # Combine all encodings
        combined_encoding = torch.cat([seq_encoding, trans_emb, rot_emb], dim=2)
        
        # Process combined encoding with MLP
        spatial_encoding = self.spatial_mlp(combined_encoding)
        
        # Add to input embeddings and normalize
        x = x + spatial_encoding
        x = self.combine_norm(x)
        
        return self.dropout(x)

class FragmentEncoder(BaseEncoder):
    def __init__(
        self,
        d_model: int = 768,  # Same as SMILES encoder
        nhead: int = 16,     # Same as SMILES encoder
        dim_feedforward: int = 3072,  # 4x d_model as in paper
        num_layers: int = 6,  # Same as SMILES encoder
        pe_max_len: int = 32,
        vocab_path: str = None,  # Path to vocab.pkl
        num_trans_bins: int = 27000,  # Maximum translation bin value
        num_rot_bins: int = 8000,    # Maximum rotation bin value
        grid_resolution: float = 0.5,
        max_dist: float = 6.75,
        mixture_weight: float = 0.2,  # How much of original spatial info to keep
    ):
        super().__init__()
        self._dim = d_model
        self.mixture_weight = mixture_weight
        
        # Load vocabulary
        if vocab_path is None:
            vocab_path = "/workspace/data/desert/vocab.pkl"
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Get vocab size from the actual vocabulary
        vocab_size = len(self.vocab)
        print(f"Loaded vocabulary with {vocab_size} tokens")
        
        # Fragment embeddings - use same size as vocab to maintain compatibility
        self.fragment_emb = nn.Embedding(vocab_size, d_model, padding_idx=self.vocab['PAD'][2])
        
        # Use the custom 3D-aware positional encoding
        self.pe_enc = Spatial3DPositionalEncoding(
            d_model=d_model,
            max_len=pe_max_len,
            trans_bins=num_trans_bins,
            rot_bins=num_rot_bins,
            grid_resolution=grid_resolution,
            max_dist=max_dist
        )
        
        # Transformer encoder - same architecture as SMILES encoder
        self.enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        
        # Initial normalization
        self.initial_norm = nn.LayerNorm(d_model)
        
        # 2-layer MLP with smaller hidden dimension to compress the information
        self.adapter_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # More precise distribution matching
        self.distribution_matcher = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # Constrain values to (-1, 1) range
            nn.Linear(d_model, d_model)
        )
        
        # Initialize the final layer to produce small values like SMILES encoder
        with torch.no_grad():
            # Last layer should output values mostly in the ±0.1 range
            self.distribution_matcher[-1].weight.data.normal_(0, 0.01)
            self.distribution_matcher[-1].bias.data.zero_()

    @property
    def dim(self) -> int:
        return self._dim
        
    def set_mixture_weight(self, weight: float):
        """Set the mixture weight between encoder output and zeros."""
        self.mixture_weight = max(0.0, min(1.0, weight))
        print(f"Set encoder mixture weight to {self.mixture_weight}")

    def forward(self, fragment_ids, translations, rotations, padding_mask=None):
        """
        Args:
            fragment_ids: Tensor of shape [batch_size, seq_len] containing fragment IDs
            translations: Tensor of shape [batch_size, seq_len] containing translation bin indices
            rotations: Tensor of shape [batch_size, seq_len] containing rotation bin indices
            padding_mask: Optional boolean mask of shape [batch_size, seq_len] where True indicates padding
        """
        # Get fragment embeddings
        frag_emb = self.fragment_emb(fragment_ids)
        
        # Apply spatial positional encoding that incorporates translations and rotations
        h = self.pe_enc(frag_emb, translations, rotations)
        
        # Apply transformer encoder
        if padding_mask is None:
            padding_mask = torch.zeros_like(fragment_ids, dtype=torch.bool)
            
        out = self.enc(h, src_key_padding_mask=padding_mask)
        
        # Apply improved adapter stack
        out = self.initial_norm(out)
        out = out + 0.1 * self.adapter_mlp(out)  # Residual connection with small contribution
        
        # Match distribution to what decoder expects (SMILES-like)
        out = self.distribution_matcher(out)
        
        # Further constrain the range to match SMILES encoder
        out_mean = out.mean()
        out_std = out.std()
        out = (out - out_mean) / (out_std + 1e-5) * 0.05  # Target std of 0.05
        
        # Apply mixture weight to blend spatial information with zeros
        zeros = torch.zeros_like(out)
        blended_out = self.mixture_weight * out + (1 - self.mixture_weight) * zeros
        
        # Print statistics for debugging
        print(f"Encoder output stats - mean: {blended_out.mean().item():.4f}, "
              f"std: {blended_out.std().item():.4f}, "
              f"min: {blended_out.min().item():.4f}, "
              f"max: {blended_out.max().item():.4f}")
        
        return EncoderOutput(blended_out, padding_mask)

    def encode_desert_sequence(self, desert_sequence, device='cpu', max_seq_len=32):
        """
        Encode a DESERT sequence into the format expected by the Synformer decoder.
        
        Args:
            desert_sequence: List of tuples (fragment_id, translation, rotation)
            device: Device to put tensors on
            max_seq_len: Maximum sequence length for padding
            
        Returns:
            EncoderOutput containing the encoded sequence and padding mask
        """
        # Find EOS token index if present
        eos_idx = None
        for i, (frag_id, _, _) in enumerate(desert_sequence):
            if frag_id == 3:  # EOS token
                eos_idx = i
                break
        
        # If no EOS token found, use all tokens
        if eos_idx is None:
            eos_idx = len(desert_sequence)
        
        # Include EOS token in the sequence
        seq_len = eos_idx + 1
        
        # Prepare tensors with padding
        fragment_ids = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        translations = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        rotations = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        padding_mask = torch.ones(max_seq_len, dtype=torch.bool, device=device)  # True means padding
        
        # Fill in actual values and mark as non-padding
        for i in range(min(seq_len, max_seq_len)):
            if i < len(desert_sequence):
                frag_id, trans, rot = desert_sequence[i]
                fragment_ids[i] = frag_id
                translations[i] = trans
                rotations[i] = rot
                padding_mask[i] = False  # Not padding
        
        # Add batch dimension
        fragment_ids = fragment_ids.unsqueeze(0)
        translations = translations.unsqueeze(0)
        rotations = rotations.unsqueeze(0)
        padding_mask = padding_mask.unsqueeze(0)
        
        return self.forward(fragment_ids, translations, rotations, padding_mask)

def create_fragment_encoder(vocab_path=None, device='cpu', grid_resolution=0.5, max_dist=6.75, mixture_weight=0.2):
    """Create and initialize a FragmentEncoder"""
    encoder = FragmentEncoder(
        vocab_path=vocab_path,
        grid_resolution=grid_resolution,
        max_dist=max_dist,
        mixture_weight=mixture_weight
    )
    encoder.to(device)
    encoder.eval()
    return encoder

if __name__ == "__main__":
    # Example usage
    '''
    desert_sequence = [
        (6, 20575, 3577),    # Fragment 1: *C
        (53, 11369, 7673),   # Fragment 2: *O*
        (9, 16933, 7650),    # Fragment 3: *C(*)=O
        (9, 13692, 7674),    # Fragment 4: *C(*)=O
        (10, 12592, 7674),   # Fragment 5: *N*
        (9, 2870, 4401),     # Fragment 6: *C(*)=O
        (53, 20205, 4176),   # Fragment 7: *O*
        (11, 14789, 1399),   # Fragment 8: *C*
        (53, 12196, 6430),   # Fragment 9: *O*
        (11, 12196, 3928),   # Fragment 10: *C*
        (53, 10601, 0),      # Fragment 11: *O*
        (11, 11387, 3928),   # Fragment 12: *C*
        (53, 2808, 5513),    # Fragment 13: *O*
        (11, 9715, 4911),    # Fragment 14: *C*
        (70, 8691, 5513),    # Fragment 15: *C1CC1
        (3, 12492, 3849),    # Fragment 16: EOS
    ]
    '''
    
    desert_sequence = [
        (6, 1, 3577),        # Fragment 1: ID=6 (*C), Translation=1, Rotation=3577
        (53, 1, 7673),       # Fragment 2: ID=53 (*O*), Translation=1, Rotation=7673
        (9, 8261, 7627),     # Fragment 3: ID=9 (*C(*)=O), Translation=8261, Rotation=7627
        (9, 12980, 7926),    # Fragment 4: ID=9 (*C(*)=O), Translation=12980, Rotation=7926
        (10, 12173, 384),    # Fragment 5: ID=10 (*N*), Translation=12173, Rotation=384
        (9, 19762, 4724),    # Fragment 6: ID=9 (*C(*)=O), Translation=19762, Rotation=4724
        (53, 20564, 7672),   # Fragment 7: ID=53 (*O*), Translation=20564, Rotation=7672
        (11, 3625, 7604),    # Fragment 8: ID=11 (*C*), Translation=3625, Rotation=7604
        (53, 18743, 4176),   # Fragment 9: ID=53 (*O*), Translation=18743, Rotation=4176
        (11, 17594, 7397),   # Fragment 10: ID=11 (*C*), Translation=17594, Rotation=7397
        (53, 13108, 4176),   # Fragment 11: ID=53 (*O*), Translation=13108, Rotation=4176
        (11, 9348, 1158),    # Fragment 12: ID=11 (*C*), Translation=9348, Rotation=1158
        (70, 10626, 4176),   # Fragment 13: ID=70 (*C1CC1), Translation=10626, Rotation=4176
        (3, 11744, 3849),    # Fragment 14: ID=3 (EOS), Translation=11744, Rotation=3849
    ]

    # Get vocab path from the checkpoint path
    smiles_checkpoint_path = "/workspace/data/processed/sf_ed_default.ckpt"
    vocab_path = "/workspace/data/desert/vocab.pkl"
    smiles = "CC1=NN(CC=Cc2cc(F)ccc2[C@H]2CCC[C@@H]2N)C2=NN=C[C@@H]12"#"CC1=CC=C(C=C1)C2=CC=C(C=C2)N"
    
    encoder = create_fragment_encoder(vocab_path=vocab_path)
    output = encoder.encode_desert_sequence(desert_sequence)
    print("Encoder output shape:", output.code.shape)
    print("Padding mask shape:", output.code_padding_mask.shape) 
    print("Output:", output)
    
    # Test with SMILES encoder
    smiles_encoder = SMILESEncoder(
        d_model=768,
        num_token_types=200,  # map somehow all those 21000 to 200
        nhead=16,
        dim_feedforward=4096,
        num_layers=6,
        pe_max_len=256
    )
    
    # Load pretrained weights
    state_dict = torch.load(smiles_checkpoint_path, map_location='cpu')['state_dict']
    # Filter encoder weights and remove the model.encoder prefix
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.encoder.'):
            # Remove 'model.encoder.' prefix
            new_key = k.replace('model.encoder.', '')
            encoder_state_dict[new_key] = v
    
    smiles_encoder.load_state_dict(encoder_state_dict)
    smiles_encoder.to('cpu')
    smiles_encoder.eval()
    
    # Create input batch for SMILES encoder using the official tokenizer
    smiles_tokens = tokenize_smiles(smiles)
    smiles_tensor = torch.tensor(smiles_tokens, device='cpu').unsqueeze(0)
    smiles_batch = ProjectionBatch({"smiles": smiles_tensor})
    
    # Run SMILES encoder
    with torch.no_grad():
        smiles_output = smiles_encoder(smiles_batch)
    
    print("SMILES Encoder output shape:", smiles_output.code.shape)
    print("SMILES Encoder padding mask shape:", smiles_output.code_padding_mask.shape)
    print("SMILES Encoder output:", smiles_output)
    
    # Compare outputs
    print("\nComparison:")
    print(f"Fragment encoder d_model: {output.code.shape[-1]}")
    print(f"SMILES encoder d_model: {smiles_output.code.shape[-1]}")
    print(f"Fragment sequence length: {output.code.shape[1]}")
    print(f"SMILES sequence length: {smiles_output.code.shape[1]}")
    
    # Print padding information
    print("\nPadding information:")
    print(f"Fragment encoder padded positions: {output.code_padding_mask.sum().item()} out of {output.code_padding_mask.numel()}")
    print(f"SMILES encoder padded positions: {smiles_output.code_padding_mask.sum().item()} out of {smiles_output.code_padding_mask.numel()}")
    
    # Print a sample of the padding masks
    print("\nFragment encoder padding mask")
    print(output.code_padding_mask)
    print("\nSMILES encoder padding mask")
    print(smiles_output.code_padding_mask)
    
    # Now run the decoder with the fragment encoder output
    print("\n\n=== Running Decoder with Fragment Encoder Output ===\n")
    
    # Import necessary modules for decoder
    from synformer.models.decoder import Decoder
    from synformer.data.common import TokenType
    from synformer.chem.fpindex import FingerprintIndex
    from synformer.chem.matrix import ReactantReactionMatrix
    from synformer.models.synformer import Synformer
    from synformer.chem.mol import Molecule
    from synformer.sampler.analog.state_pool import StatePool
    from omegaconf import OmegaConf
    import pandas as pd
    from tqdm.auto import tqdm
    
    # Load the full model checkpoint to get the decoder
    full_model_checkpoint = torch.load(smiles_checkpoint_path, map_location='cpu')
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
    decoder = Decoder(
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
    decoder.load_state_dict(decoder_state_dict)
    decoder.to('cpu')
    decoder.eval()
    
    # Load token head, reaction head, and fingerprint head
    from synformer.models.classifier_head import ClassifierHead
    
    token_head = ClassifierHead(768, max(TokenType) + 1)
    token_head_state_dict = {}
    for k, v in full_model_checkpoint['state_dict'].items():
        if k.startswith('model.token_head.'):
            new_key = k.replace('model.token_head.', '')
            token_head_state_dict[new_key] = v
    token_head.load_state_dict(token_head_state_dict)
    token_head.to('cpu')
    token_head.eval()
    
    reaction_head = ClassifierHead(768, config.model.decoder.num_reaction_classes)
    reaction_head_state_dict = {}
    for k, v in full_model_checkpoint['state_dict'].items():
        if k.startswith('model.reaction_head.'):
            new_key = k.replace('model.reaction_head.', '')
            reaction_head_state_dict[new_key] = v
    reaction_head.load_state_dict(reaction_head_state_dict)
    reaction_head.to('cpu')
    reaction_head.eval()
    
    # Load fingerprint head
    from synformer.models.fingerprint_head import get_fingerprint_head
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
    fingerprint_head.to('cpu')
    fingerprint_head.eval()
    
    # Load reaction matrix and fingerprint index
    rxn_matrix_path = "/workspace/data/processed/comp_2048/matrix.pkl"
    fpindex_path = "/workspace/data/processed/comp_2048/fpindex.pkl"
    
    print(f"Loading reaction matrix from: {rxn_matrix_path}")
    print(f"Loading fingerprint index from: {fpindex_path}")
    
    import pickle
    rxn_matrix = pickle.load(open(rxn_matrix_path, 'rb'))
    fpindex = pickle.load(open(fpindex_path, 'rb'))
    
    # Create a simple state pool for inference
    class SimpleStatePool:
        def __init__(self, fpindex, rxn_matrix, encoder_output, decoder, token_head, reaction_head, fingerprint_head):
            self.fpindex = fpindex
            self.rxn_matrix = rxn_matrix
            self.encoder_output = encoder_output
            self.decoder = decoder
            self.token_head = token_head
            self.reaction_head = reaction_head
            self.fingerprint_head = fingerprint_head
            
            # Initialize an empty stack
            from synformer.chem.stack import Stack
            self.stack = Stack()
            
            # Initialize token types, reaction indices, and reactant fingerprints
            self.token_types = torch.tensor([[TokenType.START.value]], dtype=torch.long)
            self.rxn_indices = torch.zeros_like(self.token_types)
            # Get fingerprint dimension from the fp_option attribute
            fp_dim = fpindex.fp_option.dim
            self.reactant_fps = torch.zeros((1, 1, fp_dim), dtype=torch.float32)
            
        def step(self):
            # Run decoder
            with torch.no_grad():
                # Ensure reactant_fps has the right shape for the current token_types
                if self.reactant_fps.shape[1] != self.token_types.shape[1]:
                    # Pad or truncate reactant_fps to match token_types length
                    fp_dim = self.fpindex.fp_option.dim
                    new_fps = torch.zeros((1, self.token_types.shape[1], fp_dim), dtype=torch.float32)
                    # Copy existing fingerprints (up to min length)
                    min_len = min(self.reactant_fps.shape[1], new_fps.shape[1])
                    new_fps[:, :min_len, :] = self.reactant_fps[:, :min_len, :]
                    self.reactant_fps = new_fps
                
                h = self.decoder(
                    code=self.encoder_output.code,
                    code_padding_mask=self.encoder_output.code_padding_mask,
                    token_types=self.token_types,
                    rxn_indices=self.rxn_indices,
                    reactant_fps=self.reactant_fps,
                    token_padding_mask=None
                )
                h_next = h[:, -1]  # (bsz, h_dim)
                
                # Get token logits and sample
                token_logits = self.token_head.predict(h_next)
                token_probs = torch.nn.functional.softmax(token_logits / 0.1, dim=-1)
                token_sampled = torch.multinomial(token_probs, num_samples=1)
                
                # Get reaction logits
                reaction_logits = self.reaction_head.predict(h_next)[..., :len(self.rxn_matrix.reactions)]
                reaction_probs = torch.nn.functional.softmax(reaction_logits / 0.1, dim=-1)
                rxn_idx_next = torch.multinomial(reaction_probs, num_samples=1)[..., 0]
                
                # Get reactant fingerprints if needed
                if token_sampled.item() == TokenType.REACTANT.value:
                    retrieved_reactants = self.fingerprint_head.retrieve_reactants(
                        h_next,
                        self.fpindex,
                        topk=4,
                        mask=token_sampled == TokenType.REACTANT,
                    )
                    
                    # Sample a reactant
                    fp_scores = torch.from_numpy(1.0 / (retrieved_reactants.distance + 1e-4)).reshape(1, -1)
                    fp_probs = torch.nn.functional.softmax(fp_scores / 0.1, dim=-1)
                    fp_idx_next = torch.multinomial(fp_probs, num_samples=1)[..., 0]
                    
                    # Get the reactant molecule
                    reactant_mol = retrieved_reactants.reactants[0, fp_idx_next.item()]
                    reactant_idx = retrieved_reactants.indices[0, fp_idx_next.item()]
                    
                    # Update stack with reactant
                    self.stack.push_mol(reactant_mol, reactant_idx)
                    
                    # Update reactant fingerprints
                    fp_retrieved = torch.from_numpy(retrieved_reactants.fingerprint_retrieved[0, fp_idx_next]).unsqueeze(0).unsqueeze(1)
                    self.reactant_fps = torch.cat([self.reactant_fps, fp_retrieved], dim=1)
                
                elif token_sampled.item() == TokenType.REACTION.value:
                    # Get the reaction
                    reaction = self.rxn_matrix.reactions[rxn_idx_next.item()]
                    
                    # Update stack with reaction
                    success = self.stack.push_rxn(reaction, rxn_idx_next.item())
                    
                    # If reaction failed, change token to END
                    if not success:
                        token_sampled[0, 0] = TokenType.END.value
                
                # Update token types and reaction indices
                self.token_types = torch.cat([self.token_types, token_sampled], dim=1)
                self.rxn_indices = torch.cat([self.rxn_indices, rxn_idx_next.unsqueeze(1)], dim=1)
                
                return token_sampled.item()
    
    # Run inference with fragment encoder output
    print("\nRunning inference with fragment encoder output...")
    state_pool = SimpleStatePool(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        encoder_output=output,
        decoder=decoder,
        token_head=token_head,
        reaction_head=reaction_head,
        fingerprint_head=fingerprint_head
    )
    
    # Generate a molecule
    max_steps = 24
    for step in range(max_steps):
        token = state_pool.step()
        print(f"Step {step+1}: Token = {TokenType(token).name}")
        
        # Stop if END token is generated
        if token == TokenType.END.value:
            break
    
    # Get the generated molecule
    generated_mols = state_pool.stack.get_top()
    if generated_mols:
        for i, mol in enumerate(generated_mols):
            print(f"\nGenerated molecule {i+1}: {mol.smiles}")
    else:
        print("\nNo molecules generated")
    
    # Run inference with SMILES encoder output for comparison
    print("\n\nRunning inference with SMILES encoder output for comparison...")
    state_pool_smiles = SimpleStatePool(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        encoder_output=smiles_output,
        decoder=decoder,
        token_head=token_head,
        reaction_head=reaction_head,
        fingerprint_head=fingerprint_head
    )
    
    # Generate a molecule
    for step in range(max_steps):
        token = state_pool_smiles.step()
        print(f"Step {step+1}: Token = {TokenType(token).name}")
        
        # Stop if END token is generated
        if token == TokenType.END.value:
            break
    
    # Get the generated molecule
    generated_mols_smiles = state_pool_smiles.stack.get_top()
    if generated_mols_smiles:
        for i, mol in enumerate(generated_mols_smiles):
            print(f"\nGenerated molecule {i+1}: {mol.smiles}")
    else:
        print("\nNo molecules generated")
    
    # Compare the results
    print("\n\n=== Comparison of Results ===")
    if generated_mols and generated_mols_smiles:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from synformer.chem.fpindex import FingerprintOption
        import numpy as np
        from tabulate import tabulate
        
        # Convert sets to lists for indexing
        generated_mols_list = list(generated_mols)
        generated_mols_smiles_list = list(generated_mols_smiles)
        
        if generated_mols_list and generated_mols_smiles_list:
            mol1 = generated_mols_list[0]
            mol2 = generated_mols_smiles_list[0]
            
            print(f"Fragment encoder generated: {mol1.smiles}")
            print(f"SMILES encoder generated: {mol2.smiles}")
            
            # Calculate similarity
            fp1 = mol1.get_fingerprint(FingerprintOption.morgan_for_tanimoto_similarity())
            fp2 = mol2.get_fingerprint(FingerprintOption.morgan_for_tanimoto_similarity())
            similarity = mol1.tanimoto_similarity(mol2, FingerprintOption.morgan_for_tanimoto_similarity())
            
            print(f"Tanimoto similarity between generated molecules: {similarity:.4f}")
            
            # Now run multiple generations and collect results
            print("\n\n=== Running Multiple Generations (10 times) ===\n")
            
            # Try to import sascorer
            try:
                import sascorer
                has_sascorer = True
            except ImportError:
                print("Warning: Could not import sascorer module. SAS scores will not be calculated.")
                has_sascorer = False
            
            # Try to import docking function
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
                from experiments.graoh_sequential import dock_best_molecule, prepare_ligand_pdbqt, QVinaOption
                has_docking = True
            except ImportError as e:
                print(f"Warning: Could not import docking function: {e}")
                print("Docking scores will not be calculated.")
                has_docking = False
            
            # Create result lists for each encoder
            fragment_results = []
            smiles_results = []
            
            # Run 10 generations
            for i in range(2):
                print(f"\n--- Run {i+1}/10 ---")
                
                # Fragment encoder generation
                print("\nRunning fragment encoder generation...")
                state_pool_fragment = SimpleStatePool(
                    fpindex=fpindex,
                    rxn_matrix=rxn_matrix,
                    encoder_output=output,
                    decoder=decoder,
                    token_head=token_head,
                    reaction_head=reaction_head,
                    fingerprint_head=fingerprint_head
                )
                
                # Run generation until END token or max steps
                for step in range(max_steps):
                    token = state_pool_fragment.step()
                    if token == TokenType.END.value:
                        break
                
                # Get generated molecule
                generated_mols_fragment = state_pool_fragment.stack.get_top()
                if generated_mols_fragment:
                    fragment_mol = list(generated_mols_fragment)[0]
                    fragment_smiles = fragment_mol.smiles
                    print(f"Fragment encoder generated: {fragment_smiles}")
                    
                    # Calculate SAS score
                    sas_score_fragment = None
                    if has_sascorer:
                        rdkit_mol_fragment = Chem.MolFromSmiles(fragment_smiles)
                        if rdkit_mol_fragment:
                            try:
                                sas_score_fragment = sascorer.calculateScore(rdkit_mol_fragment)
                                print(f"SAS score: {sas_score_fragment:.2f}")
                            except Exception as e:
                                print(f"Error calculating SAS score: {e}")
                    
                    # Calculate docking score
                    docking_score_fragment = None
                    if has_docking:
                        rdkit_mol_fragment = Chem.MolFromSmiles(fragment_smiles)
                        if rdkit_mol_fragment:
                            docking_score_fragment = dock_best_molecule(rdkit_mol_fragment)
                            print(f"Docking score: {docking_score_fragment}")
                    
                    # Store results
                    fragment_results.append({
                        'run': i+1,
                        'smiles': fragment_smiles,
                        'sas_score': sas_score_fragment,
                        'docking_score': docking_score_fragment
                    })
                else:
                    print("No molecules generated")
                
                # SMILES encoder generation
                print("\nRunning SMILES encoder generation...")
                state_pool_smiles = SimpleStatePool(
                    fpindex=fpindex,
                    rxn_matrix=rxn_matrix,
                    encoder_output=smiles_output,
                    decoder=decoder,
                    token_head=token_head,
                    reaction_head=reaction_head,
                    fingerprint_head=fingerprint_head
                )
                
                # Run generation until END token or max steps
                for step in range(max_steps):
                    token = state_pool_smiles.step()
                    if token == TokenType.END.value:
                        break
                
                # Get generated molecule
                generated_mols_smiles = state_pool_smiles.stack.get_top()
                if generated_mols_smiles:
                    smiles_mol = list(generated_mols_smiles)[0]
                    smiles_smiles = smiles_mol.smiles
                    print(f"SMILES encoder generated: {smiles_smiles}")
                    
                    # Calculate SAS score
                    sas_score_smiles = None
                    if has_sascorer:
                        rdkit_mol_smiles = Chem.MolFromSmiles(smiles_smiles)
                        if rdkit_mol_smiles:
                            try:
                                sas_score_smiles = sascorer.calculateScore(rdkit_mol_smiles)
                                print(f"SAS score: {sas_score_smiles:.2f}")
                            except Exception as e:
                                print(f"Error calculating SAS score: {e}")
                    
                    # Calculate docking score
                    docking_score_smiles = None
                    if has_docking:
                        rdkit_mol_smiles = Chem.MolFromSmiles(smiles_smiles)
                        if rdkit_mol_smiles:
                            docking_score_smiles = dock_best_molecule(rdkit_mol_smiles)
                            print(f"Docking score: {docking_score_smiles}")
                    
                    # Store results
                    smiles_results.append({
                        'run': i+1,
                        'smiles': smiles_smiles,
                        'sas_score': sas_score_smiles,
                        'docking_score': docking_score_smiles
                    })
                else:
                    print("No molecules generated")
            
            # Print summary table
            print("\n\n=== Summary of Results ===\n")
            
            # Fragment encoder summary
            print("Fragment Encoder Results:")
            fragment_table = []
            for result in fragment_results:
                fragment_table.append([
                    result['run'],
                    result['smiles'],
                    f"{result['sas_score']:.2f}" if result['sas_score'] is not None else "N/A",
                    f"{result['docking_score']:.2f}" if result['docking_score'] is not None else "N/A"
                ])
            
            print(tabulate(fragment_table, headers=["Run", "SMILES", "SAS Score", "Docking Score"], tablefmt="grid"))
            
            # SMILES encoder summary
            print("\nSMILES Encoder Results:")
            smiles_table = []
            for result in smiles_results:
                smiles_table.append([
                    result['run'],
                    result['smiles'],
                    f"{result['sas_score']:.2f}" if result['sas_score'] is not None else "N/A",
                    f"{result['docking_score']:.2f}" if result['docking_score'] is not None else "N/A"
                ])
            
            print(tabulate(smiles_table, headers=["Run", "SMILES", "SAS Score", "Docking Score"], tablefmt="grid"))
            
            # Calculate average statistics
            frag_sas_scores = [r['sas_score'] for r in fragment_results if r['sas_score'] is not None]
            frag_docking_scores = [r['docking_score'] for r in fragment_results if r['docking_score'] is not None]
            smiles_sas_scores = [r['sas_score'] for r in smiles_results if r['sas_score'] is not None]
            smiles_docking_scores = [r['docking_score'] for r in smiles_results if r['docking_score'] is not None]
            
            print("\nAverage Statistics:")
            stats_table = [
                ["Fragment Encoder", 
                 f"{np.mean(frag_sas_scores):.2f} ± {np.std(frag_sas_scores):.2f}" if frag_sas_scores else "N/A",
                 f"{np.mean(frag_docking_scores):.2f} ± {np.std(frag_docking_scores):.2f}" if frag_docking_scores else "N/A"],
                ["SMILES Encoder", 
                 f"{np.mean(smiles_sas_scores):.2f} ± {np.std(smiles_sas_scores):.2f}" if smiles_sas_scores else "N/A",
                 f"{np.mean(smiles_docking_scores):.2f} ± {np.std(smiles_docking_scores):.2f}" if smiles_docking_scores else "N/A"]
            ]
            
            print(tabulate(stats_table, headers=["Encoder", "Avg SAS Score", "Avg Docking Score"], tablefmt="grid"))
            
            # Calculate win counts: how many times did fragment encoder perform better?
            better_docking_count = 0
            better_sas_count = 0
            valid_comparisons = 0
            
            for i in range(min(len(fragment_results), len(smiles_results))):
                frag_result = fragment_results[i]
                smiles_result = smiles_results[i]
                
                # For docking scores, compare when both are available (lower is better)
                if frag_result['docking_score'] is not None and smiles_result['docking_score'] is not None:
                    valid_comparisons += 1
                    if frag_result['docking_score'] < smiles_result['docking_score']:
                        better_docking_count += 1
                
                # For SAS scores, compare when both are available (lower is better)
                if frag_result['sas_score'] is not None and smiles_result['sas_score'] is not None:
                    if frag_result['sas_score'] < smiles_result['sas_score']:
                        better_sas_count += 1
            
            print("\nWin Rate Comparison:")
            if valid_comparisons > 0:
                print(f"Fragment encoder had better docking scores in {better_docking_count}/{valid_comparisons} runs ({better_docking_count/valid_comparisons*100:.1f}%)")
                print(f"Fragment encoder had better SAS scores in {better_sas_count}/{valid_comparisons} runs ({better_sas_count/valid_comparisons*100:.1f}%)")
            else:
                print("Not enough valid comparisons to calculate win rates")
                
            # Create a head-to-head comparison table
            print("\nHead-to-Head Comparison:")
            head_to_head = []
            for i in range(min(len(fragment_results), len(smiles_results))):
                frag_result = fragment_results[i]
                smiles_result = smiles_results[i]
                
                # Format docking comparison
                if frag_result['docking_score'] is not None and smiles_result['docking_score'] is not None:
                    docking_diff = frag_result['docking_score'] - smiles_result['docking_score']
                    docking_winner = "Fragment" if docking_diff < 0 else "SMILES" if docking_diff > 0 else "Tie"
                    docking_comp = f"{frag_result['docking_score']:.2f} vs {smiles_result['docking_score']:.2f} ({docking_winner})"
                else:
                    docking_comp = "N/A"
                
                # Format SAS comparison
                if frag_result['sas_score'] is not None and smiles_result['sas_score'] is not None:
                    sas_diff = frag_result['sas_score'] - smiles_result['sas_score']
                    sas_winner = "Fragment" if sas_diff < 0 else "SMILES" if sas_diff > 0 else "Tie"
                    sas_comp = f"{frag_result['sas_score']:.2f} vs {smiles_result['sas_score']:.2f} ({sas_winner})"
                else:
                    sas_comp = "N/A"
                
                head_to_head.append([i+1, docking_comp, sas_comp])
            
            print(tabulate(head_to_head, headers=["Run", "Docking Score (Fragment vs SMILES)", "SAS Score (Fragment vs SMILES)"], tablefmt="grid"))
            
            # Comparison of best molecules
            print("\nBest Molecules by Docking Score:")
            best_frag_idx = np.argmin(frag_docking_scores) if frag_docking_scores else None
            best_smiles_idx = np.argmin(smiles_docking_scores) if smiles_docking_scores else None
            
            if best_frag_idx is not None:
                best_frag_result = fragment_results[best_frag_idx]
                print(f"Fragment Encoder Best: {best_frag_result['smiles']}")
                print(f"  SAS Score: {best_frag_result['sas_score']:.2f}")
                print(f"  Docking Score: {best_frag_result['docking_score']:.2f}")
            
            if best_smiles_idx is not None:
                best_smiles_result = smiles_results[best_smiles_idx]
                print(f"SMILES Encoder Best: {best_smiles_result['smiles']}")
                print(f"  SAS Score: {best_smiles_result['sas_score']:.2f}")
                print(f"  Docking Score: {best_smiles_result['docking_score']:.2f}")
            
        else:
            print("Cannot compare results as at least one generation has no molecules")
    else:
        print("Cannot compare results as at least one generation failed")
    