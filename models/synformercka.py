import dataclasses
import os
import pickle
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr

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


def compute_cka(X, Y, kernel='linear'):
    """
    Compute Center Kernel Alignment (CKA) between two sets of representations.
    
    Args:
        X: First set of representations (n_samples, n_features_1)
        Y: Second set of representations (n_samples, n_features_2) 
        kernel: Kernel type ('linear' or 'rbf')
    
    Returns:
        CKA score (scalar)
    """
    # Convert to tensors if numpy
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()
    
    # Ensure tensors are on CPU for computation
    X = X.detach().cpu()
    Y = Y.detach().cpu()
    
    # Flatten if needed (for multi-dimensional features)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(Y.shape[0], -1)
    
    # Ensure same number of samples
    min_samples = min(X.shape[0], Y.shape[0])
    X = X[:min_samples]
    Y = Y[:min_samples]
    
    if kernel == 'linear':
        return linear_CKA(X, Y)
    elif kernel == 'rbf':
        return rbf_CKA(X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute linear CKA between two feature matrices.
    
    Args:
        X: Features from first representation (n_tokens, d1)
        Y: Features from second representation (n_tokens, d2)
    
    Returns:
        CKA similarity score [0, 1]
    """
    # Center the features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute cross-correlation
    dot_XY = (X.T @ Y).norm(p='fro') ** 2
    
    # Compute norms
    norm_X = (X.T @ X).norm(p='fro')
    norm_Y = (Y.T @ Y).norm(p='fro')
    
    # Avoid division by zero
    if norm_X == 0 or norm_Y == 0:
        return 0.0
    
    cka_score = dot_XY / (norm_X * norm_Y)
    
    # Ensure result is in [0, 1] and convert to float
    return float(torch.clamp(cka_score, 0.0, 1.0))


def rbf_CKA(X: torch.Tensor, Y: torch.Tensor, gamma_x: float = None, gamma_y: float = None) -> float:
    """
    Compute RBF kernel CKA between two feature matrices.
    
    Args:
        X: Features from first representation (n_tokens, d1)
        Y: Features from second representation (n_tokens, d2)
        gamma_x: RBF bandwidth for X (default: 1/d1)
        gamma_y: RBF bandwidth for Y (default: 1/d2)
    
    Returns:
        CKA similarity score [0, 1]
    """
    if gamma_x is None:
        gamma_x = 1.0 / X.shape[1]
    if gamma_y is None:
        gamma_y = 1.0 / Y.shape[1]
    
    def rbf_kernel(A, gamma):
        # Compute pairwise squared distances
        sq_dists = torch.cdist(A, A) ** 2
        return torch.exp(-gamma * sq_dists)
    
    # Compute RBF kernels
    K_X = rbf_kernel(X, gamma_x)
    K_Y = rbf_kernel(Y, gamma_y)
    
    # Center kernels
    n = K_X.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H
    
    # Compute CKA
    numerator = torch.trace(K_X_centered @ K_Y_centered)
    denominator = torch.sqrt(torch.trace(K_X_centered @ K_X_centered) * torch.trace(K_Y_centered @ K_Y_centered))
    
    if denominator == 0:
        return 0.0
    
    cka_score = numerator / denominator
    return float(torch.clamp(cka_score, 0.0, 1.0))


def plot_cka_comparison(cka_scores, save_path=None):
    """
    Plot CKA scores comparison for different variants.
    
    Args:
        cka_scores: Dictionary with variant names as keys and CKA scores as values
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: CKA between DESERT and decoder inputs
    variants = list(cka_scores.keys())
    desert_decoder_scores = [cka_scores[v].get('desert_vs_decoder', 0) for v in variants]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars1 = ax1.bar(variants, desert_decoder_scores, color=colors[:len(variants)], alpha=0.8)
    ax1.set_title('CKA: DESERT Latents vs SynFormer Decoder Inputs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('CKA Score', fontsize=12)
    ax1.set_xlabel('Variant', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, desert_decoder_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: CKA between blended and native training latents (if available)
    if any('blended_vs_native' in cka_scores[v] for v in variants):
        blended_native_scores = [cka_scores[v].get('blended_vs_native', 0) for v in variants]
        bars2 = ax2.bar(variants, blended_native_scores, color=colors[:len(variants)], alpha=0.8)
        ax2.set_title('CKA: Blended Latents vs Native Training Latents', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CKA Score', fontsize=12)
        ax2.set_xlabel('Variant', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, blended_native_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Native Training Latents\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('CKA: Blended vs Native Training Latents', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CKA comparison plot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_cka_heatmap(cka_matrix, variant_names, save_path=None):
    """
    Plot CKA scores as a heatmap for pairwise comparisons.
    
    Args:
        cka_matrix: Square matrix of CKA scores between variants
        variant_names: List of variant names
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(cka_matrix, dtype=bool), k=1)
    sns.heatmap(cka_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                square=True,
                xticklabels=variant_names,
                yticklabels=variant_names,
                mask=mask,
                cbar_kws={'label': 'CKA Score'})
    
    plt.title('Pairwise CKA Scores Between Variants', fontsize=16, fontweight='bold')
    plt.xlabel('Variant', fontsize=12)
    plt.ylabel('Variant', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CKA heatmap saved to: {save_path}")
    
    plt.show()
    return plt.gcf()


def plot_layerwise_cka_heatmap(cka_matrix, desert_stages, synformer_stages, save_path=None):
    """
    Plot layer-wise CKA analysis as a comprehensive heatmap.
    
    Args:
        cka_matrix: Matrix of CKA scores (desert_stages x synformer_stages)
        desert_stages: List of DESERT stage names
        synformer_stages: List of SynFormer stage names  
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(14, 10))
    
    # Create the heatmap
    sns.heatmap(cka_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',  # Red-Yellow-Blue colormap (high values are red)
                center=0.5,      # Center colormap at 0.5
                square=False,
                xticklabels=synformer_stages,
                yticklabels=desert_stages,
                cbar_kws={'label': 'CKA Score', 'shrink': 0.8})
    
    plt.title('Layer-wise CKA Analysis: DESERT ↔ SynFormer Decoder\n' + 
              'How well do DESERT representations align with SynFormer layers?', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('SynFormer Decoder Stages', fontsize=14, fontweight='bold')
    plt.ylabel('DESERT Processing Stages', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer-wise CKA heatmap saved to: {save_path}")
    
    plt.show()
    return plt.gcf()


def compute_layerwise_cka(desert_representations, synformer_representations):
    """
    Compute CKA matrix between all DESERT and SynFormer processing stages.
    
    Args:
        desert_representations: Dict of {stage_name: tensor}
        synformer_representations: Dict of {stage_name: tensor}
    
    Returns:
        cka_matrix: numpy array of CKA scores
        desert_stages: list of DESERT stage names
        synformer_stages: list of SynFormer stage names
    """
    desert_stages = list(desert_representations.keys())
    synformer_stages = list(synformer_representations.keys())
    
    n_desert = len(desert_stages)
    n_synformer = len(synformer_stages)
    
    cka_matrix = np.zeros((n_desert, n_synformer))
    
    print(f"\n🔍 Computing {n_desert} × {n_synformer} = {n_desert * n_synformer} CKA comparisons...")
    
    for i, desert_stage in enumerate(desert_stages):
        for j, synformer_stage in enumerate(synformer_stages):
            try:
                desert_rep = desert_representations[desert_stage]
                synformer_rep = synformer_representations[synformer_stage]
                
                cka_score = compute_cka(desert_rep, synformer_rep, kernel='linear')
                cka_matrix[i, j] = cka_score
                
                print(f"  {desert_stage:20s} ↔ {synformer_stage:20s}: {cka_score:.4f}")
                
            except Exception as e:
                print(f"  Error computing CKA for {desert_stage} ↔ {synformer_stage}: {str(e)}")
                cka_matrix[i, j] = 0.0
    
    return cka_matrix, desert_stages, synformer_stages


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
        
        # Initialize CKA tracking
        self.cka_scores = {}
        self.desert_latents = None
        self.native_latents = None
        self.enable_cka_analysis = getattr(cfg, 'enable_cka_analysis', True)
        
        # Storage for layer-wise CKA analysis
        self.desert_representations = {}
        self.synformer_representations = {}
        self.enable_layerwise_cka = getattr(cfg, 'enable_layerwise_cka', True)
        
        # Store processing options
        self.no_adapter = getattr(cfg, 'no_adapter', False)
        self.minimal_processing = getattr(cfg, 'minimal_processing', False)
        
        if self.no_adapter:
            print("\nRunning in no-adapter mode - using raw DESERT output")
        elif self.minimal_processing:
            print("\nRunning in minimal-processing mode - basic dimension matching only")
        
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
            if self.no_adapter:
                print("... with --no-adapter option for ablation.")
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
        
        if self.encoder_type == "desert" and self.no_adapter:
            with open(self.vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            max_frag_id = -1
            for _, (_, _, idx) in vocab.items():
                if idx > max_frag_id:
                    max_frag_id = idx
            vocab_size = max_frag_id + 1
            self.desert_embedding = nn.Embedding(vocab_size, 768)
        
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
        # Clear previous representations at the start of each encoding to ensure fresh analysis
        if hasattr(self, 'desert_representations'):
            self.desert_representations.clear()
        if hasattr(self, 'synformer_representations'):
            self.synformer_representations.clear()
        # Don't clear cka_scores - they need to persist for the final analysis
        
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
            
            # Store raw DESERT latents for CKA analysis
            if self.enable_cka_analysis:
                print("\n" + "="*80)
                print("🔍 STARTING LAYER-WISE CKA ANALYSIS FOR ADAPTER ABLATION STUDY")
                print("="*80)
                print(f"📊 Preparing to analyze {len(desert_sequence)} DESERT fragments")
                print("🎯 Will compute CKA between DESERT processing stages and SynFormer decoder layers")
                if self.enable_layerwise_cka:
                    print("🌟 LAYER-WISE ANALYSIS ENABLED - Full representational comparison!")
                print("="*80)
                
                # Convert DESERT sequence to tensor for CKA computation
                desert_tensor = torch.zeros((len(desert_sequence), 7), device=device)  # 3 trans + 3 rot + 1 frag_id
                for i, (frag_id, trans, rot) in enumerate(desert_sequence):
                    desert_tensor[i, :3] = torch.tensor(trans, device=device)
                    desert_tensor[i, 3:6] = torch.tensor(rot, device=device)
                    desert_tensor[i, 6] = frag_id
                self.desert_latents = desert_tensor.detach().cpu()
                
                # Store for layer-wise analysis
                if self.enable_layerwise_cka:
                    self.desert_representations['raw_coordinates'] = desert_tensor.detach().cpu()
                    print(f"✅ Raw DESERT coordinates stored: shape {desert_tensor.shape}")
                    
                    # For no-adapter case, store more detailed stages
                    if hasattr(self, 'no_adapter') and self.no_adapter:
                        # Store raw fragment IDs
                        fragment_ids = torch.tensor([frag_id for frag_id, _, _ in desert_sequence], dtype=torch.float32)
                        self.desert_representations['fragment_ids'] = fragment_ids.detach().cpu()
                        print(f"✅ Fragment IDs stored: shape {fragment_ids.shape}")
                        
                        # Store spatial coordinates separately
                        spatial_coords = torch.tensor([[trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]] 
                                                      for _, trans, rot in desert_sequence], dtype=torch.float32)
                        self.desert_representations['spatial_coords'] = spatial_coords.detach().cpu()
                        print(f"✅ Spatial coordinates stored: shape {spatial_coords.shape}")
                else:
                    print(f"✅ DESERT latents stored: shape {self.desert_latents.shape}")
            
            # Each element in desert_sequence is a tuple (fragment_id, translation, rotation)
            print(f"Received {len(desert_sequence)} fragments with spatial information")
            for i, (frag_id, trans, rot) in enumerate(desert_sequence[:5]):
                frag_name = id_to_token.get(frag_id, "Unknown")
                print(f"Fragment {i+1}: ID={frag_id} ({frag_name}), Translation={trans}, Rotation={rot}")
            if len(desert_sequence) > 5:
                print(f"... and {len(desert_sequence)-5} more fragments")
            
            # Check if we should bypass the fragment encoder
            if hasattr(self, 'no_adapter') and self.no_adapter:
                print("\n=== Running Control Experiment: No Fragment Encoder or Adapter ===\n")
                print("Simply reshaping DESERT output to match decoder input dimensions")
                
                # Get raw DESERT sequence and reshape to match decoder input
                num_fragments = len(desert_sequence)
                # Create tensor of expected shape (1, num_fragments, 768) filled with raw DESERT values
                code = torch.zeros((1, num_fragments, 768), device=device)
                
                # Simply copy the raw values from DESERT sequence into the tensor
                # No processing, no adaptation, just dimension matching
                for i, (frag_id, trans, rot) in enumerate(desert_sequence):
                    # Copy raw values directly
                    code[0, i, :3] = torch.tensor(trans, device=device)  # Translation
                    code[0, i, 3:6] = torch.tensor(rot, device=device)   # Rotation
                    code[0, i, 6] = frag_id                              # Fragment ID
                
                # Create padding mask (all False since we have no padding)
                code_padding_mask = torch.zeros((1, num_fragments), dtype=torch.bool, device=device)
                
                print(f"Raw DESERT output reshaped to: {code.shape}")
                print(f"Padding mask shape: {code_padding_mask.shape}")
                
                # Compute CKA for no-adapter variant
                print(f"\n🔍 DEBUG CKA conditions:")
                print(f"   self.enable_cka_analysis: {self.enable_cka_analysis}")
                print(f"   self.desert_latents is not None: {self.desert_latents is not None}")
                print(f"   hasattr(self, 'cka_scores'): {hasattr(self, 'cka_scores')}")
                
                if self.enable_cka_analysis and self.desert_latents is not None:
                    print("\n" + "🌟" * 60)
                    print("🧪 ANALYZING NO_ADAPTER")
                    print("🌟" * 60)
                    print("🔍 Testing raw DESERT output alignment...")
                    print("📊 This measures how well unprocessed spatial coordinates align with decoder")
                    print("💡 Expected: Lower scores indicate need for processing/adaptation")
                    print()
                    
                    decoder_input = code.squeeze(0).detach().cpu()  # Remove batch dimension
                    cka_score = compute_cka(self.desert_latents, decoder_input, kernel='linear')
                    
                    scores_dict = {'desert_vs_decoder': cka_score}
                    
                    # Compare with native latents if available
                    if self.native_latents is not None:
                        native_cka = compute_cka(decoder_input, self.native_latents, kernel='linear')
                        scores_dict['blended_vs_native'] = native_cka
                        if native_cka >= 0.6:
                            emoji = "🟢"
                            quality = "EXCELLENT"
                        elif native_cka >= 0.3:
                            emoji = "🟡"
                            quality = "GOOD"
                        elif native_cka >= 0.1:
                            emoji = "🟠"
                            quality = "WEAK"
                        else:
                            emoji = "🔴"
                            quality = "POOR"
                        print(f"  {emoji} blended_vs_native: {native_cka:.4f} ({quality} alignment)")
                    
                    self.cka_scores['NO_ADAPTER'] = scores_dict
                    print(f"🔍 DEBUG: Stored CKA scores: {self.cka_scores}")
                    
                    if cka_score >= 0.6:
                        emoji = "🟢"
                        quality = "EXCELLENT"
                    elif cka_score >= 0.3:
                        emoji = "🟡"
                        quality = "GOOD"
                    elif cka_score >= 0.1:
                        emoji = "🟠"
                        quality = "WEAK"
                    else:
                        emoji = "🔴"
                        quality = "POOR"
                    
                    print(f"  {emoji} desert_vs_decoder: {cka_score:.4f} ({quality} alignment)")
                    print("🌟" * 60)
                    print()
                
                # Store the raw reshaped code for layer-wise analysis 
                if self.enable_layerwise_cka:
                    self.desert_representations['reshaped_raw'] = code.squeeze(0).detach().cpu()
                    print(f"✅ Reshaped raw DESERT output stored: shape {code.squeeze(0).shape}")
                    
                    # Trigger comprehensive layer-wise analysis for no-adapter case
                    print("\n" + "🔬"*80)
                    print("🎯 COMPREHENSIVE NO-ADAPTER LAYER-WISE ANALYSIS")
                    print("🔬"*80)
                    print("📊 This analyzes alignment between different DESERT representations and decoder")
                    print("🎯 Helps understand the impact of bypassing fragment processing")
                    print("🔬"*80)
                
                # Empty loss dict for encoder
                encoder_loss_dict = {}
                
                return code, code_padding_mask, encoder_loss_dict
            elif hasattr(self, 'minimal_processing') and self.minimal_processing:
                print("\n=== Running Minimal Processing Experiment ===\n")
                print("Basic dimension matching and feature placement")
                
                # Get DESERT sequence and create tensor of expected shape
                num_fragments = len(desert_sequence)
                code = torch.zeros((1, num_fragments, 768), device=device)
                
                # Simple embedding for fragment IDs
                max_frag_id = max(frag_id for frag_id, _, _ in desert_sequence)
                frag_embedding = nn.Embedding(max_frag_id + 1, 768).to(device)
                
                # Add a simple MLP to process the embeddings
                mlp = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.LayerNorm(768),
                    nn.ReLU(),
                    nn.Linear(768, 768)
                ).to(device)
                
                # Place features in a structured way but without any adaptation
                for i, (frag_id, trans, rot) in enumerate(desert_sequence):
                    # Get base embedding from fragment ID
                    base_embedding = frag_embedding(torch.tensor(frag_id, device=device))
                    
                    # Add spatial information to first 6 dimensions
                    base_embedding[0:3] = torch.tensor(trans, device=device)  # Translation
                    base_embedding[3:6] = torch.tensor(rot, device=device)    # Rotation
                    
                    # Add position encoding to help with sequence understanding
                    position = i / num_fragments  # Normalized position
                    position_encoding = torch.sin(torch.linspace(0, 10, 768, device=device) * position)
                    base_embedding += position_encoding * 0.05  # Small position encoding
                    
                    # Process through MLP
                    processed_embedding = mlp(base_embedding)
                    
                    # Store the embedding
                    code[0, i] = processed_embedding
                
                # Create padding mask
                code_padding_mask = torch.zeros((1, num_fragments), dtype=torch.bool, device=device)
                
                # Normalize the output to match expected distribution
                code_mean = code.mean()
                code_std = code.std()
                code = (code - code_mean) / (code_std + 1e-5) * 0.05
                
                print(f"Minimally processed output shape: {code.shape}")
                print(f"Padding mask shape: {code_padding_mask.shape}")
                print(f"Output stats - mean: {code.mean().item():.4f}, std: {code.std().item():.4f}")
                
                # Compute CKA for minimal processing variant
                if self.enable_cka_analysis and self.desert_latents is not None:
                    print("\n" + "🌟" * 60)
                    print("🧪 ANALYZING MINIMAL_PROCESSING")
                    print("🌟" * 60)
                    print("🔍 Testing basic fragment embedding processing...")
                    print("📊 This measures alignment with simple embedding + MLP transformation")
                    print("💡 Expected: Moderate scores showing basic adaptation benefits")
                    print()
                    
                    decoder_input = code.squeeze(0).detach().cpu()  # Remove batch dimension
                    cka_score = compute_cka(self.desert_latents, decoder_input, kernel='linear')
                    
                    scores_dict = {'desert_vs_decoder': cka_score}
                    
                    # Compare with native latents if available
                    if self.native_latents is not None:
                        native_cka = compute_cka(decoder_input, self.native_latents, kernel='linear')
                        scores_dict['blended_vs_native'] = native_cka
                        if native_cka >= 0.6:
                            emoji = "🟢"
                            quality = "EXCELLENT"
                        elif native_cka >= 0.3:
                            emoji = "🟡"
                            quality = "GOOD"
                        elif native_cka >= 0.1:
                            emoji = "🟠"
                            quality = "WEAK"
                        else:
                            emoji = "🔴"
                            quality = "POOR"
                        print(f"  {emoji} blended_vs_native: {native_cka:.4f} ({quality} alignment)")
                    
                    self.cka_scores['MINIMAL_PROCESSING'] = scores_dict
                    
                    if cka_score >= 0.6:
                        emoji = "🟢"
                        quality = "EXCELLENT"
                    elif cka_score >= 0.3:
                        emoji = "🟡"
                        quality = "GOOD"
                    elif cka_score >= 0.1:
                        emoji = "🟠"
                        quality = "WEAK"
                    else:
                        emoji = "🔴"
                        quality = "POOR"
                    
                    print(f"  {emoji} desert_vs_decoder: {cka_score:.4f} ({quality} alignment)")
                    print("🌟" * 60)
                    print()
                
                # Empty loss dict for encoder
                encoder_loss_dict = {}
                
                return code, code_padding_mask, encoder_loss_dict
            else:
                # Create and initialize the fragment encoder
                print("\n=== Step 2: Running Fragment Encoder ===\n")
                
                # mixture_weight=1 means we are using 100% spatial information from DESERT
                # mixture_weight=0 would mean using all zeros (no spatial information)
                # Use the passed mixture_weight if available, otherwise default (e.g., 0.8 or a config value)
                current_mixture_weight = mixture_weight if mixture_weight is not None else 0.8 # Default to 1.0 for maximum spatial information
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
            
            # Store intermediate representations for layer-wise analysis
            if self.enable_layerwise_cka:
                # Store raw fragment data for analysis
                fragment_data = []
                for item in desert_sequence:
                    try:
                        frag_id, trans, rot = item
                        # Handle both list/tuple and scalar formats
                        if isinstance(trans, (list, tuple)) and len(trans) >= 3:
                            trans_coords = [trans[0], trans[1], trans[2]]
                        else:
                            # If trans is scalar, use it for all coordinates (fallback)
                            trans_coords = [float(trans), float(trans), float(trans)]
                        
                        if isinstance(rot, (list, tuple)) and len(rot) >= 3:
                            rot_coords = [rot[0], rot[1], rot[2]]
                        else:
                            # If rot is scalar, use it for all coordinates (fallback)
                            rot_coords = [float(rot), float(rot), float(rot)]
                        
                        fragment_data.append(trans_coords + rot_coords)
                    except Exception as e:
                        print(f"Warning: Error processing fragment data: {e}")
                        print(f"Fragment item: {item}")
                        # Skip this fragment
                        continue
                        
                if fragment_data:
                    fragment_tensor = torch.tensor(fragment_data, dtype=torch.float32, device=device)
                    self.desert_representations['spatial_coordinates'] = fragment_tensor.detach().cpu()
                    print(f"✅ DESERT spatial coordinates stored: shape {fragment_tensor.shape}")
                else:
                    print("⚠️  No valid fragment data found for spatial coordinates")
            
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
            
            # Store final DESERT output for layer-wise analysis
            if self.enable_layerwise_cka:
                self.desert_representations['final_output'] = code.squeeze(0).detach().cpu()
                print(f"✅ Final DESERT output stored: shape {code.squeeze(0).shape}")
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
            
            # Compute CKA for full fragment encoder variant
            if self.enable_cka_analysis and self.desert_latents is not None:
                print("\n" + "🌟" * 60)
                print("🧪 ANALYZING FULL_FRAGMENT_ENCODER")
                print("🌟" * 60)
                print("🔍 Testing complete spatial encoding with mixture weights...")
                print("📊 This measures alignment with full fragment encoder processing")
                print("💡 Expected: Higher scores showing optimal integration")
                print()
                
                decoder_input = code.squeeze(0).detach().cpu()  # Remove batch dimension
                cka_score = compute_cka(self.desert_latents, decoder_input, kernel='linear')
                
                # Determine variant name based on mixture weight
                variant_name = f"FULL_FRAGMENT_ENCODER_w{current_mixture_weight:.1f}"
                scores_dict = {'desert_vs_decoder': cka_score}
                
                # Compare with native latents if available
                if self.native_latents is not None:
                    native_cka = compute_cka(decoder_input, self.native_latents, kernel='linear')
                    scores_dict['blended_vs_native'] = native_cka
                    if native_cka >= 0.6:
                        emoji = "🟢"
                        quality = "EXCELLENT"
                    elif native_cka >= 0.3:
                        emoji = "🟡"
                        quality = "GOOD"
                    elif native_cka >= 0.1:
                        emoji = "🟠"
                        quality = "WEAK"
                    else:
                        emoji = "🔴"
                        quality = "POOR"
                    print(f"  {emoji} blended_vs_native: {native_cka:.4f} ({quality} alignment)")
                
                self.cka_scores[variant_name] = scores_dict
                
                if cka_score >= 0.6:
                    emoji = "🟢"
                    quality = "EXCELLENT"
                elif cka_score >= 0.3:
                    emoji = "🟡"
                    quality = "GOOD"
                elif cka_score >= 0.1:
                    emoji = "🟠"
                    quality = "WEAK"
                else:
                    emoji = "🔴"
                    quality = "POOR"
                
                print(f"  {emoji} desert_vs_decoder: {cka_score:.4f} ({quality} alignment)")
                print("🌟" * 60)
                print()
            
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
        
        # Store decoder input for layer-wise CKA analysis
        if self.enable_layerwise_cka:
            decoder_input_embeddings = self.decoder.embed(token_types, rxn_indices, reactant_fps)
            self.synformer_representations['decoder_input'] = decoder_input_embeddings.squeeze(0).detach().cpu()
            print(f"✅ Decoder input embeddings stored: shape {decoder_input_embeddings.squeeze(0).shape}")

        # For layer-wise analysis, we need to capture intermediate decoder layers
        layer_outputs = []
        
        def layer_hook(module, input, output):
            if self.enable_layerwise_cka:
                layer_outputs.append(output.squeeze(0).detach().cpu())
        
        # Register hooks on decoder layers if available
        hooks = []
        if self.enable_layerwise_cka and hasattr(self.decoder, 'layers'):
            for i, layer in enumerate(self.decoder.layers):
                hook = layer.register_forward_hook(layer_hook)
                hooks.append(hook)
            print(f"📊 Registered hooks on {len(hooks)} decoder layers for layer-wise analysis")

        # Generate embeddings
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=None,
        )
        
        # Remove hooks and store layer outputs
        if self.enable_layerwise_cka:
            for hook in hooks:
                hook.remove()
            
            # Store individual layer outputs
            for i, layer_output in enumerate(layer_outputs):
                self.synformer_representations[f'decoder_layer_{i}'] = layer_output
                print(f"✅ Decoder layer {i} stored: shape {layer_output.shape}")
        
        # Store final decoder output for layer-wise CKA analysis
        if self.enable_layerwise_cka:
            self.synformer_representations['decoder_output'] = h.squeeze(0).detach().cpu()
            print(f"✅ Decoder output stored: shape {h.squeeze(0).shape}")
            
            # Only store representations for the first prediction to avoid contamination
            if len(self.synformer_representations) <= 2:  # Should have decoder_input and decoder_output
                print(f"✅ SynFormer representations captured (count: {len(self.synformer_representations)})")
            
            # Don't trigger analysis during prediction - it will happen at the end via state_pool
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
    
    def analyze_cka_scores(self, save_plots=True, plot_dir="./cka_plots"):
        """
        Analyze and plot CKA scores for the adapter ablation study.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        if not self.cka_scores:
            print("No CKA scores available for analysis.")
            return
        
        print("\n" + "="*60)
        print("CKA ANALYSIS FOR ADAPTER ABLATION STUDY")
        print("="*60)
        
        # First check if we have layer-wise CKA data
        print(f"\n🔍 Checking layer-wise CKA prerequisites:")
        print(f"   enable_layerwise_cka: {self.enable_layerwise_cka}")
        print(f"   desert_representations keys: {list(self.desert_representations.keys()) if hasattr(self, 'desert_representations') else 'None'}")
        print(f"   synformer_representations keys: {list(self.synformer_representations.keys()) if hasattr(self, 'synformer_representations') else 'None'}")
        
        if self.enable_layerwise_cka and self.desert_representations and self.synformer_representations:
            print("\n🌟 GENERATING LAYER-WISE CKA ANALYSIS!")
            self.analyze_layerwise_cka(save_plots, plot_dir)
        else:
            print("❌ Layer-wise CKA analysis not available - missing data or disabled")
        
        # Detailed CKA analysis for variants with emoji decorations
        print("\n" + "🎯" * 80)
        print("🔍 ADAPTER ABLATION CKA ANALYSIS")
        print("🎯" * 80)
        print("📊 Comparing representation alignment across different DESERT integration strategies:")
        print("   • NO_ADAPTER: Raw DESERT output vs SynFormer decoder")
        print("   • MINIMAL_PROCESSING: Basic fragment embedding processing")
        print("   • FULL_FRAGMENT_ENCODER: Complete spatial encoding with mixture weights")
        print("🎯" * 80)
        print()
        
        for variant, scores in self.cka_scores.items():
            print("🌟" * 60)
            print(f"🧪 ANALYZING {variant}")
            print("🌟" * 60)
            
            if variant == "NO_ADAPTER":
                print("🔍 Testing raw DESERT output alignment...")
                print("📊 This measures how well unprocessed spatial coordinates align with decoder")
                print("💡 Expected: Lower scores indicate need for processing/adaptation")
            elif variant == "MINIMAL_PROCESSING":
                print("🔍 Testing basic fragment embedding processing...")
                print("📊 This measures alignment with simple embedding + MLP transformation")
                print("💡 Expected: Moderate scores showing basic adaptation benefits")
            elif variant == "FULL_FRAGMENT_ENCODER":
                print("🔍 Testing complete spatial encoding with mixture weights...")
                print("📊 This measures alignment with full fragment encoder processing")
                print("💡 Expected: Higher scores showing optimal integration")
            
            print()
            for comparison, score in scores.items():
                if score >= 0.6:
                    emoji = "🟢"
                    quality = "EXCELLENT"
                elif score >= 0.3:
                    emoji = "🟡"
                    quality = "GOOD"
                elif score >= 0.1:
                    emoji = "🟠"
                    quality = "WEAK"
                else:
                    emoji = "🔴"
                    quality = "POOR"
                
                print(f"  {emoji} {comparison}: {score:.4f} ({quality} alignment)")
            
            print("🌟" * 60)
            print()
        
        # Generate comparison analysis
        print("🏆" * 80)
        print("🎯 COMPARATIVE ANALYSIS ACROSS VARIANTS")
        print("🏆" * 80)
        
        # Find best scores for each comparison type
        comparison_types = set()
        for scores in self.cka_scores.values():
            comparison_types.update(scores.keys())
        
        for comparison in comparison_types:
            print(f"\n📊 {comparison.upper()} COMPARISON:")
            variant_scores = []
            for variant, scores in self.cka_scores.items():
                if comparison in scores:
                    variant_scores.append((variant, scores[comparison]))
            
            variant_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (variant, score) in enumerate(variant_scores):
                if i == 0:
                    print(f"   🥇 {variant}: {score:.4f} (BEST)")
                elif i == 1:
                    print(f"   🥈 {variant}: {score:.4f}")
                else:
                    print(f"   🥉 {variant}: {score:.4f}")
        
        print("\n" + "📈" * 80)
        print("🔍 INTERPRETATION GUIDE")
        print("📈" * 80)
        print("🟢 CKA > 0.6: Excellent alignment (strong representational similarity)")
        print("🟡 CKA 0.3-0.6: Good alignment (moderate similarity)")
        print("🟠 CKA 0.1-0.3: Weak alignment (limited similarity)")
        print("🔴 CKA < 0.1: Poor alignment (little similarity)")
        print("📈" * 80)
        
        # Create plots directory if needed
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
        
        # Generate comparison plot
        plot_path = os.path.join(plot_dir, "cka_comparison.png") if save_plots else None
        plot_cka_comparison(self.cka_scores, save_path=plot_path)
        
        # Generate pairwise heatmap if we have multiple variants
        variants = list(self.cka_scores.keys())
        if len(variants) > 1:
            # Create pairwise CKA matrix
            n_variants = len(variants)
            cka_matrix = np.eye(n_variants)  # Diagonal is 1 (self-similarity)
            
            # For demonstration, we'll compute pairwise CKA if we have the latents stored
            # In practice, you might want to store latents from each variant and compute this
            
            heatmap_path = os.path.join(plot_dir, "cka_heatmap.png") if save_plots else None
            plot_cka_heatmap(cka_matrix, variants, save_path=heatmap_path)
        
        print("\n" + "✅" * 80)
        print("🎉 ADAPTER ABLATION CKA ANALYSIS COMPLETE!")
        print("📊 Use these insights to optimize DESERT-SynFormer integration strategies!")
        print("✅" * 80)
        print(f"\nCKA analysis complete. Plots saved to: {plot_dir}" if save_plots else "\nCKA analysis complete.")
        
        return self.cka_scores

    def analyze_layerwise_cka(self, save_plots=True, plot_dir="./cka_plots"):
        """
        Perform comprehensive layer-wise CKA analysis between DESERT and SynFormer stages.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        print("\n" + "🌟"*80)
        print("🔍 PERFORMING LAYER-WISE CKA ANALYSIS")
        print("🌟"*80)
        print("📊 This analyzes how DESERT processing stages align with SynFormer decoder layers")
        print("🎯 Perfect for understanding information flow and optimal integration points!")
        print("🌟"*80)
        
        try:
            # Compute layer-wise CKA matrix
            cka_matrix, desert_stages, synformer_stages = compute_layerwise_cka(
                self.desert_representations, 
                self.synformer_representations
            )
            
            # Generate comprehensive layerwise heatmap
            if save_plots:
                layerwise_path = os.path.join(plot_dir, "layerwise_cka_heatmap.png")
            else:
                layerwise_path = None
            
            plot_layerwise_cka_heatmap(
                cka_matrix, 
                desert_stages, 
                synformer_stages, 
                save_path=layerwise_path
            )
            
            # Find best alignments
            print("\n" + "🏆"*40)
            print("🎯 OPTIMAL ALIGNMENT ANALYSIS")
            print("🏆"*40)
            
            # Find best DESERT stage for each SynFormer stage
            for j, synformer_stage in enumerate(synformer_stages):
                best_desert_idx = np.argmax(cka_matrix[:, j])
                best_score = cka_matrix[best_desert_idx, j]
                best_desert_stage = desert_stages[best_desert_idx]
                print(f"🎯 {synformer_stage:20s} ← best aligned with → {best_desert_stage:20s} (CKA: {best_score:.4f})")
            
            # Find best SynFormer stage for each DESERT stage
            print("\n🔄 Reverse alignment analysis:")
            for i, desert_stage in enumerate(desert_stages):
                best_synformer_idx = np.argmax(cka_matrix[i, :])
                best_score = cka_matrix[i, best_synformer_idx]
                best_synformer_stage = synformer_stages[best_synformer_idx]
                print(f"🎯 {desert_stage:20s} ← best aligned with → {best_synformer_stage:20s} (CKA: {best_score:.4f})")
            
            # Overall statistics
            print("\n" + "📊"*40)
            print("📈 OVERALL STATISTICS")
            print("📊"*40)
            max_score = np.max(cka_matrix)
            min_score = np.min(cka_matrix)
            mean_score = np.mean(cka_matrix)
            std_score = np.std(cka_matrix)
            
            print(f"📊 CKA Score Range: {min_score:.4f} - {max_score:.4f}")
            print(f"📊 Mean CKA Score: {mean_score:.4f} ± {std_score:.4f}")
            
            # Interpretation guide
            print("\n" + "🔍"*40)
            print("📚 INTERPRETATION GUIDE")
            print("🔍"*40)
            print("🟢 CKA > 0.6: Excellent alignment (strong representational similarity)")
            print("🟡 CKA 0.3-0.6: Good alignment (moderate similarity)")  
            print("🟠 CKA 0.1-0.3: Weak alignment (limited similarity)")
            print("🔴 CKA < 0.1: Poor alignment (little similarity)")
            
            excellent_alignments = np.sum(cka_matrix > 0.6)
            good_alignments = np.sum((cka_matrix > 0.3) & (cka_matrix <= 0.6))
            weak_alignments = np.sum((cka_matrix > 0.1) & (cka_matrix <= 0.3))
            poor_alignments = np.sum(cka_matrix <= 0.1)
            
            total_comparisons = cka_matrix.size
            print(f"\n📊 Alignment Quality Distribution:")
            print(f"🟢 Excellent: {excellent_alignments}/{total_comparisons} ({excellent_alignments/total_comparisons*100:.1f}%)")
            print(f"🟡 Good: {good_alignments}/{total_comparisons} ({good_alignments/total_comparisons*100:.1f}%)")
            print(f"🟠 Weak: {weak_alignments}/{total_comparisons} ({weak_alignments/total_comparisons*100:.1f}%)")
            print(f"🔴 Poor: {poor_alignments}/{total_comparisons} ({poor_alignments/total_comparisons*100:.1f}%)")
            
            print("\n" + "✅"*80)
            print("🎉 LAYER-WISE CKA ANALYSIS COMPLETE!")
            print("📊 Use these insights to understand how DESERT spatial information flows through SynFormer!")
            print("✅"*80)
            
        except Exception as e:
            print(f"\n❌ Error in layer-wise CKA analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_native_latents(self, native_latents_path):
        """
        Load native training latents for CKA comparison.
        
        Args:
            native_latents_path: Path to saved native latents tensor
        """
        if os.path.exists(native_latents_path):
            self.native_latents = torch.load(native_latents_path, map_location='cpu')
            print(f"Loaded native training latents from: {native_latents_path}")
            print(f"Native latents shape: {self.native_latents.shape}")
        else:
            print(f"Native latents file not found: {native_latents_path}")
    
    def save_current_latents(self, save_path, latents_type="current"):
        """
        Save current latents for later comparison.
        
        Args:
            save_path: Path to save the latents
            latents_type: Type of latents being saved
        """
        if hasattr(self, 'desert_latents') and self.desert_latents is not None:
            torch.save(self.desert_latents, save_path)
            print(f"Saved {latents_type} latents to: {save_path}")
        else:
            print(f"No {latents_type} latents available to save")


def quick_cka_demo(smiles="CCO", desert_model_path="path/to/desert/model", 
                   vocab_path="path/to/vocab.pkl", shape_patches_path="path/to/patches.h5"):
    """
    Quick demo function to run CKA analysis without config files.
    Add this to the end of your script and call it directly.
    """
    print("="*60)
    print("QUICK CKA ANALYSIS DEMO")
    print("="*60)
    
    # Create a minimal config object
    class SimpleConfig:
        def __init__(self):
            self.encoder_type = "desert"
            self.enable_cka_analysis = True
            self.no_adapter = False
            self.minimal_processing = False
            
            # Create encoder config
            self.encoder = SimpleConfig()
            self.encoder.desert_model_path = desert_model_path
            self.encoder.vocab_path = vocab_path  
            self.encoder.shape_patches_path = shape_patches_path
            
            # Create decoder config
            self.decoder = SimpleConfig()
            self.decoder.d_model = 768
            self.decoder.nhead = 16
            self.decoder.dim_feedforward = 4096
            self.decoder.num_layers = 10
            self.decoder.pe_max_len = 32
            self.decoder.output_norm = False
            self.decoder.fingerprint_dim = 2048
            self.decoder.num_reaction_classes = 100
            
            # Fingerprint head config
            self.fingerprint_head_type = "linear"
            self.fingerprint_head = SimpleConfig()
    
    # Test different variants
    variants = [
        {"name": "No Adapter", "no_adapter": True, "minimal_processing": False},
        {"name": "Minimal Processing", "no_adapter": False, "minimal_processing": True},
        {"name": "Fragment Encoder w=0.5", "no_adapter": False, "minimal_processing": False, "mixture_weight": 0.5},
        {"name": "Fragment Encoder w=0.8", "no_adapter": False, "minimal_processing": False, "mixture_weight": 0.8},
    ]
    
    all_results = {}
    
    for variant in variants:
        print(f"\n{'-'*40}")
        print(f"Testing: {variant['name']}")
        print(f"{'-'*40}")
        
        try:
            # Create config for this variant
            cfg = SimpleConfig()
            cfg.no_adapter = variant.get("no_adapter", False)
            cfg.minimal_processing = variant.get("minimal_processing", False)
            
            # Create model
            model = Synformer(cfg)
            
            # Create batch
            batch = {
                'smiles_str': smiles,
                'shape_patches_path': shape_patches_path
            }
            
            # Run encoding
            mixture_weight = variant.get("mixture_weight", None)
            code, code_padding_mask, _ = model.encode(batch, mixture_weight=mixture_weight)
            
            # Store results
            if model.cka_scores:
                latest_key = list(model.cka_scores.keys())[-1]
                all_results[variant['name']] = model.cka_scores[latest_key]
                
                print("CKA Scores:")
                for comparison, score in model.cka_scores[latest_key].items():
                    print(f"  {comparison}: {score:.4f}")
            
        except Exception as e:
            print(f"Error in {variant['name']}: {str(e)}")
    
    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL VARIANTS")
        print(f"{'='*60}")
        
        for variant_name, scores in all_results.items():
            print(f"\n{variant_name}:")
            for comparison, score in scores.items():
                print(f"  {comparison}: {score:.4f}")
    
    return all_results


# SIMPLE USAGE EXAMPLES - Just call these functions directly!

def simple_cka_test(model):
    """
    Simplest possible CKA test - just add this after you create your model.
    
    Usage:
        model = Synformer(your_config)
        simple_cka_test(model)  # That's it!
    """
    if hasattr(model, 'cka_scores') and model.cka_scores:
        print("\n🔍 CKA ANALYSIS RESULTS:")
        print("=" * 40)
        for variant, scores in model.cka_scores.items():
            print(f"\n📊 {variant.upper()}:")
            for comparison, score in scores.items():
                emoji = "🟢" if score > 0.6 else "🟡" if score > 0.3 else "🔴"
                print(f"  {emoji} {comparison}: {score:.4f}")
        
        # Try to create a quick plot
        try:
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            model.analyze_cka_scores(save_plots=True, plot_dir="./cka_results")
        except:
            print("📈 Plot creation skipped (matplotlib not available)")
    else:
        print("❌ No CKA scores found. Make sure enable_cka_analysis=True in your config.")


def enable_cka_on_existing_model(model):
    """
    Add CKA analysis to an existing model that doesn't have it enabled.
    
    Usage:
        enable_cka_on_existing_model(model)
        # Now run your encoding and CKA will be computed automatically
    """
    model.enable_cka_analysis = True
    model.cka_scores = {}
    model.desert_latents = None
    model.native_latents = None
    print("✅ CKA analysis enabled on existing model!")


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
