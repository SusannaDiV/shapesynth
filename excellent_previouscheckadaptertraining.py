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
import dataclasses
import shutil
import subprocess
import tempfile
import uuid
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
from tabulate import tabulate

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
        self.trans_bins = trans_bins
        self.rot_bins = rot_bins
        
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
        # Clamp indices to valid range to prevent out-of-bounds errors
        translations_clamped = torch.clamp(translations, 0, self.trans_bins - 1)
        rotations_clamped = torch.clamp(rotations, 0, self.rot_bins - 1)
        
        # Get embeddings for translations and rotations
        trans_emb = self.trans_embedding(translations_clamped)  # [batch_size, seq_len, d_model//3]
        rot_emb = self.rot_embedding(rotations_clamped)         # [batch_size, seq_len, d_model//3]
        
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
        device: str = 'cuda',
        mixture_weight: float = 1,  # Weight to blend spatial info with zeros (exactly like excellent.py)
    ):
        super().__init__()
        self._dim = d_model
        self.mixture_weight = mixture_weight
        self.device = device
        
        # Load vocabulary
        if vocab_path is None:
            raise ValueError("vocab_path must be provided")
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
        
        # Improved output scaling to match SMILES encoder distribution
        self.output_scaling = nn.Sequential(
            nn.LayerNorm(d_model),  # Normalize to zero mean, unit variance
            nn.Linear(d_model, d_model),  # Linear projection to adjust scale
        )
        
        # Initialize the linear layer with identity + small noise to preserve most information
        with torch.no_grad():
            # Initialize close to identity matrix with small random noise
            self.output_scaling[1].weight.data.copy_(torch.eye(d_model) + torch.randn(d_model, d_model) * 0.01)
            # Initialize bias with small values to adjust mean slightly
            self.output_scaling[1].bias.data.uniform_(-0.01, 0.01)
        
        # NEW: Add dimension-specific feature scaling for better docking
        # These will learn which features are most important for docking
        self.feature_importance = nn.Parameter(torch.ones(d_model) * 0.8)
        
        # NEW: Add a secondary adapter for distribution matching
        self.distribution_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # NEW: Targeted dropout for spatial regularization
        self.spatial_dropout = nn.Dropout(0.2)
        
        # Move to device
        self.to(device)
    
    @property
    def dim(self) -> int:
        return self._dim
        
    def set_mixture_weight(self, weight: float):
        """Set the mixture weight between encoder output and zeros."""
        self.mixture_weight = max(0.0, min(1.0, weight))  # Clamp between 0 and 1
        print(f"Set encoder mixture weight to {self.mixture_weight}")

    def forward(self, fragment_ids, translations, rotations, padding_mask=None):
        """
        Args:
            fragment_ids: Tensor of shape [batch_size, seq_len] containing fragment IDs
            translations: Tensor of shape [batch_size, seq_len] containing translation bin indices
            rotations: Tensor of shape [batch_size, seq_len] containing rotation bin indices
            padding_mask: Optional boolean mask of shape [batch_size, seq_len] where True indicates padding
        """
        # Print translation/rotation ranges for debugging
        print(f"Translation range: min={translations.min().item()}, max={translations.max().item()}")
        print(f"Rotation range: min={rotations.min().item()}, max={rotations.max().item()}")
        
        # Get fragment embeddings
        frag_emb = self.fragment_emb(fragment_ids)
        
        # DEBUG: Print fragment embedding stats
        print(f"Fragment embeddings - mean: {frag_emb.mean().item():.4f}, std: {frag_emb.std().item():.4f}")
        
        # Apply spatial positional encoding that incorporates translations and rotations
        h = self.pe_enc(frag_emb, translations, rotations)
        
        # DEBUG: Print positional encoding stats
        print(f"After positional encoding - mean: {h.mean().item():.4f}, std: {h.std().item():.4f}")
        
        # Apply transformer encoder
        if padding_mask is None:
            padding_mask = torch.zeros_like(fragment_ids, dtype=torch.bool)
            
        out = self.enc(h, src_key_padding_mask=padding_mask)
        
        # DEBUG: Print transformer encoder stats
        print(f"After transformer encoder - mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
        
        # Apply output scaling
        out = self.output_scaling(out)
        
        # NEW: Apply selective dimension scaling to mimic mixture_weight=0.8 behavior
        # This applies learned feature importance factors to each dimension
        out = out * self.feature_importance.unsqueeze(0).unsqueeze(0)
        
        # NEW: Apply targeted dropout to regularize spatial information (similar to mixture_weight effect)
        out = self.spatial_dropout(out)
        
        # NEW: Apply the distribution adapter with residual connection
        # This helps match the distribution expected by the decoder while preserving spatial information
        out = out + 0.2 * self.distribution_adapter(out)
        
        # NEW: Apply selective L2 normalization to prevent extreme values while preserving direction
        out_norm = torch.norm(out, p=2, dim=-1, keepdim=True)
        scaling_factor = torch.clamp(out_norm, min=1.0) 
        out = out / scaling_factor * 0.1  # Scale to reasonable magnitude
        
        # DEBUG: Print output scaling stats
        print(f"After enhanced scaling - mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
        
        # Further normalize to achieve consistent standard deviation (similar to SMILES encoder)
        out_mean = out.mean(dim=-1, keepdim=True)
        out_std = out.std(dim=-1, keepdim=True) + 1e-5
        out = (out - out_mean) / out_std * 0.05
        
        # DEBUG: Print final stats
        print(f"Encoder output final stats - mean: {out.mean().item():.4f}, std: {out.std().item():.4f}, min: {out.min().item():.4f}, max: {out.max().item():.4f}")
        
        return EncoderOutput(out, padding_mask)

    def encode_desert_sequence(self, desert_sequence, device=None, max_seq_len=32):
        """
        Encode a DESERT sequence into the format expected by the Synformer decoder.
        
        Args:
            desert_sequence: List of tuples (fragment_id, translation, rotation)
            device: Device to put tensors on
            max_seq_len: Maximum sequence length for padding
            
        Returns:
            EncoderOutput containing the encoded sequence and padding mask
        """
        if device is None:
            device = self.device
        
        print(f"Encoding DESERT sequence with {len(desert_sequence)} fragments including spatial information")
        print(f"Using mixture_weight={self.mixture_weight} (0=all zeros, 1=full spatial encoding)")
        
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
        
        # DEBUG: Print sequence details
        print(f"Processing sequence with {seq_len} fragments, max_seq_len={max_seq_len}")
        
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
        
        # DEBUG: Print tensor shapes
        print(f"Tensor shapes: fragments={fragment_ids.shape}, translations={translations.shape}, rotations={rotations.shape}, padding_mask={padding_mask.shape}")
        
        # Process through the encoder
        encoder_output = self.forward(fragment_ids, translations, rotations, padding_mask)
        
        print(f"Generated embeddings tensor with shape: {encoder_output.code.shape}")
        print(f"Generated padding mask with shape: {encoder_output.code_padding_mask.shape}")
        
        # Return the encoder output
        return encoder_output

def create_fragment_encoder(vocab_path: str, embedding_dim: int = 768, device: str = 'cuda', 
                           grid_resolution: float = 0.5, max_dist: float = 6.75, 
                           mixture_weight: float = 1) -> FragmentEncoder:
    """
    Create a fragment encoder from a vocabulary file.
    
    Args:
        vocab_path: Path to the vocabulary pickle file
        embedding_dim: Dimension of the embeddings
        device: Device to place the encoder on
        grid_resolution: Resolution of the spatial grid
        max_dist: Maximum distance for spatial encoding
        mixture_weight: Weight to balance encoder output vs. zeros (0.0 = all zeros, 1.0 = full encoder output)
        
    Returns:
        FragmentEncoder instance
    """
    # Create encoder
    encoder = FragmentEncoder(
        d_model=embedding_dim,
        vocab_path=vocab_path,
        grid_resolution=grid_resolution,
        max_dist=max_dist,
        device=device,
        mixture_weight=mixture_weight
    )
    # encoder.to(device) # already done in FragmentEncoder.__init__
    encoder.eval()
    return encoder

# Copied from synformer/synformer/sampler/analog/state_pool.py
# Moved these definitions BEFORE the if __name__ == "__main__" block
@dataclasses.dataclass
class QVinaOption:
    """Options for QVina2 docking"""
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8
    num_modes: int = 1

def prepare_ligand_pdbqt(mol, obabel_path="obabel"):
    """
    Prepare a ligand for docking by converting it to PDBQT format using OpenBabel
    """
    try:
        # import tempfile # Already imported
        # import uuid # Already imported
        
        # Create unique filenames with absolute paths in the system temp directory
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        temp_mol_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.mol")
        temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.pdbqt")
        
        # Write the molecule to a temporary file
        Chem.MolToMolFile(mol, temp_mol_file)
        
        # Convert to PDBQT using OpenBabel
        cmd = [
            obabel_path,
            "-imol", temp_mol_file,
            "-opdbqt", "-O", temp_pdbqt_file,
            "--partialcharge", "gasteiger",
            "--gen3d", "best"
        ]
        
        # Run OpenBabel
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Check if the conversion was successful
        if process.returncode != 0:
            print(f"Error converting molecule to PDBQT: {process.stderr}")
            return None
        
        # Read the PDBQT file
        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()
            
        # Clean up temporary files
        try:
            os.remove(temp_mol_file)
            os.remove(temp_pdbqt_file)
        except:
            pass
        
        # Check if the PDBQT file is valid (contains ATOM or HETATM lines)
        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            print("Error: Generated PDBQT file does not contain valid atom entries")
            return None
            
        return pdbqt_content
    except Exception as e:
        print(f"Error preparing ligand: {str(e)}")
        # Clean up temporary files
        try:
            if os.path.exists(temp_mol_file):
                os.remove(temp_mol_file)
            if os.path.exists(temp_pdbqt_file):
                os.remove(temp_pdbqt_file)
        except:
            pass
        return None

def dock_best_molecule(mol, receptor_path, receptor_center):
    """Dock the molecule against receptor target"""
    try:
        # import tempfile # Already imported
        # import uuid # Already imported
        
        # Get SMILES for logging
        smiles = Chem.MolToSmiles(mol)
        print(f"Docking molecule: {smiles}")
        
        # Create unique ID for temporary files
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        
        # Use receptor center from args
        center = receptor_center
        box_size = [22.5, 22.5, 22.5] # Default box size, can be made configurable if needed

        # Find qvina path - check if it's in the current directory or bin subdirectory
        # Adjusted qvina path to be more general, user should ensure it's in PATH or provide full path
        qvina_path = shutil.which("qvina2.1")
        if qvina_path is None:
             # Fallback for common local path if not in PATH
            local_qvina_path = "bin/qvina2.1" # Check relative ./bin/ first
            if os.path.exists(local_qvina_path):
                qvina_path = local_qvina_path
            else:
                # Fallback to user-specified absolute path
                user_specific_qvina_path = "/workspace/synformer/bin/qvina2.1"
                if os.path.exists(user_specific_qvina_path):
                    qvina_path = user_specific_qvina_path
                else:
                    print("Error: QVina2 executable (qvina2.1) not found in PATH, ./bin/, or /workspace/synformer/bin/.")
                    return None
        
        obabel_path = shutil.which("obabel")
        
        if obabel_path is None:
            print("Error: OpenBabel (obabel) not found in PATH")
            return None
            
        # Check if receptor file exists
        if not os.path.exists(receptor_path):
            print(f"Error: Receptor file not found at {receptor_path}")
            return None
            
        # Prepare ligand
        ligand_pdbqt = prepare_ligand_pdbqt(mol, obabel_path)
        if ligand_pdbqt is None:
            print("Failed to prepare ligand for docking")
            return None
            
        # Write ligand to temporary file
        temp_ligand_file = os.path.join(temp_dir, f"temp_ligand_dock_{unique_id}.pdbqt")
        with open(temp_ligand_file, "w") as f:
            f.write(ligand_pdbqt)
            
        # Set up QVina options
        options = QVinaOption(
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            size_x=box_size[0],
            size_y=box_size[1],
            size_z=box_size[2]
        )
        
        # Create QVina command
        output_file = os.path.join(temp_dir, f"temp_ligand_dock_out_{unique_id}.pdbqt")
        cmd = [
            qvina_path,
            "--receptor", receptor_path,
            "--ligand", temp_ligand_file,
            "--center_x", str(options.center_x),
            "--center_y", str(options.center_y),
            "--center_z", str(options.center_z),
            "--size_x", str(options.size_x),
            "--size_y", str(options.size_y),
            "--size_z", str(options.size_z),
            "--exhaustiveness", str(options.exhaustiveness),
            "--num_modes", str(options.num_modes),
            "--out", output_file
        ]
        
        # Run QVina with timeout
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False
            )
        except subprocess.TimeoutExpired:
            print("Docking timed out after 5 minutes")
            return None
            
        # Parse output to get docking score
        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        try:
                            score = float(line.split()[3])
                            print(f"Docking score for {smiles}: {score}")
                            break
                        except (IndexError, ValueError):
                            pass
                            
        # Clean up temporary files
        try:
            if os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
            
        return score
        
    except Exception as e:
        print(f"Error during docking: {str(e)}")
        # Clean up temporary files
        try:
            if 'temp_ligand_file' in locals() and os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if 'output_file' in locals() and os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
        return None

# End of copied functions from state_pool.py

if __name__ == "__main__":
    from sascorer import calculateScore # Re-import sascorer here
    # Example usage
    # desert_sequence = [
    #     (6, 1, 3577),        # Fragment 1: ID=6 (*C), Translation=1, Rotation=3577
    #     (53, 1, 7673),       # Fragment 2: ID=53 (*O*), Translation=1, Rotation=7673
    #     (9, 8261, 7627),     # Fragment 3: ID=9 (*C(*)=O), Translation=8261, Rotation=7627
    #     (9, 12980, 7926),    # Fragment 4: ID=9 (*C(*)=O), Translation=12980, Rotation=7926
    #     (10, 12173, 384),    # Fragment 5: ID=10 (*N*), Translation=12173, Rotation=384
    #     (9, 19762, 4724),    # Fragment 6: ID=9 (*C(*)=O), Translation=19762, Rotation=4724
    #     (53, 20564, 7672),   # Fragment 7: ID=53 (*O*), Translation=20564, Rotation=7672
    #     (11, 3625, 7604),    # Fragment 8: ID=11 (*C*), Translation=3625, Rotation=7604
    #     (53, 18743, 4176),   # Fragment 9: ID=53 (*O*), Translation=18743, Rotation=4176
    #     (11, 17594, 7397),   # Fragment 10: ID=11 (*C*), Translation=17594, Rotation=7397
    #     (53, 13108, 4176),   # Fragment 11: ID=53 (*O*), Translation=13108, Rotation=4176
    #     (11, 9348, 1158),    # Fragment 12: ID=11 (*C*), Translation=9348, Rotation=1158
    #     (70, 10626, 4176),   # Fragment 13: ID=70 (*C1CC1), Translation=10626, Rotation=4176
    #     (3, 11744, 3849),    # Fragment 14: ID=3 (EOS), Translation=11744, Rotation=3849
    #]
    '''
    desert_sequence = [
        (6, 408, 3577),        # Fragment 1: ID=6 (*C), Translation=408, Rotation=3577
        (53, 21775, 1716),     # Fragment 2: ID=53 (*O*), Translation=21775, Rotation=1716
        (9, 16089, 6909),      # Fragment 3: ID=9 (*C(*)=O), Translation=16089, Rotation=6909
        (9, 12194, 6933),      # Fragment 4: ID=9 (*C(*)=O), Translation=12194, Rotation=6933
        (9, 12951, 6933),      # Fragment 5: ID=9 (*C(*)=O), Translation=12951, Rotation=6933
        (9, 11385, 6933),      # Fragment 6: ID=9 (*C(*)=O), Translation=11385, Rotation=6933
        (9, 10632, 6933),      # Fragment 7: ID=9 (*C(*)=O), Translation=10632, Rotation=6933
        (9, 12170, 6933),      # Fragment 8: ID=9 (*C(*)=O), Translation=12170, Rotation=6933
        (9, 15585, 6933),      # Fragment 9: ID=9 (*C(*)=O), Translation=15585, Rotation=6933
        (9, 17210, 6933),      # Fragment 10: ID=9 (*C(*)=O), Translation=17210, Rotation=6933
        (9, 10627, 6933),      # Fragment 11: ID=9 (*C(*)=O), Translation=10627, Rotation=6933
        (9, 10564, 6933),      # Fragment 12: ID=9 (*C(*)=O), Translation=10564, Rotation=6933
        (9, 12169, 6933),      # Fragment 13: ID=9 (*C(*)=O), Translation=12169, Rotation=6933
        (9, 18553, 7187),      # Fragment 14: ID=9 (*C(*)=O), Translation=18553, Rotation=7187
        (5367, 8277, 1158),    # Fragment 15: ID=5367 (*CC=C), Translation=8277, Rotation=1158
        (3, 15692, 1186),      # Fragment 16: ID=3 (EOS), Translation=15692, Rotation=1186
    ]
    '''
    desert_sequence = [
        (6, 21583, 3577),      # Fragment 1: ID=6 (*C), Translation=21583, Rotation=3577
        (53, 11211, 4182),     # Fragment 2: ID=53 (*O*), Translation=11211, Rotation=4182
        (9, 12740, 4400),      # Fragment 3: ID=9 (*C(*)=O), Translation=12740, Rotation=4400
        (10, 11383, 1393),     # Fragment 4: ID=10 (*N*), Translation=11383, Rotation=1393
        (9, 11412, 1393),      # Fragment 5: ID=9 (*C(*)=O), Translation=11412, Rotation=1393
        (9, 11412, 1393),      # Fragment 6: ID=9 (*C(*)=O), Translation=11412, Rotation=1393
        (9, 9814, 1393),       # Fragment 7: ID=9 (*C(*)=O), Translation=9814, Rotation=1393
        (9, 14593, 1416),      # Fragment 8: ID=9 (*C(*)=O), Translation=14593, Rotation=1416
        (10, 16946, 6956),     # Fragment 9: ID=10 (*N*), Translation=16946, Rotation=6956
        (27, 15432, 2609),     # Fragment 10: ID=27 (*CC*), Translation=15432, Rotation=2609
        (53, 1, 4959),         # Fragment 11: ID=53 (*O*), Translation=1, Rotation=4959
        (27, 19623, 3303),     # Fragment 12: ID=27 (*CC*), Translation=19623, Rotation=3303
        (53, 21175, 3577),     # Fragment 13: ID=53 (*O*), Translation=21175, Rotation=3577
        (9, 11716, 7604),      # Fragment 14: ID=9 (*C(*)=O), Translation=11716, Rotation=7604
        (3, 11511, 3577),      # Fragment 15: ID=3 (EOS), Translation=11511, Rotation=3577
    ]

    # Get vocab path from the checkpoint path
    smiles_checkpoint_path = "/workspace/data/processed/sf_ed_default.ckpt"
    vocab_path = "/workspace/data/desert/vocab.pkl"
    smiles = "CCCC(=O)ONCCC(=O)OC(=O)C(=O)NC(=O)C(C)=O"#"CC(=O)C(=O)C(=O)C(=O)CC(=O)C(=O)OC(=O)COC1CC1"#"CC1=NN(CC=Cc2cc(F)ccc2[C@H]2CCC[C@@H]2N)C2=NN=C[C@@H]12"#"CC1=NN(CC=Cc2cc(F)ccc2[C@H]2CCC[C@@H]2N)C2=NN=C[C@@H]12"#"CC1=CC=C(C=C1)C2=CC=C(C=C2)N"

    encoder = create_fragment_encoder(vocab_path=vocab_path, device='cpu', mixture_weight=1) # Corrected comment: mixture_weight is 0.6
    output = encoder.encode_desert_sequence(desert_sequence, device='cpu')
    print("Encoder output shape:", output.code.shape)
    print("Padding mask shape:", output.code_padding_mask.shape) 
    print("Output:", output)
    print("Mixture weight:", encoder.mixture_weight)
    
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
            
            # Define receptor path and center for docking (USER: PLEASE CONFIGURE THESE)
            receptor_path = "/workspace/data/3tym_A.pdbqt" 
            receptor_center = [8.122, 2.905, 24.389] # [x, y, z] coordinates
            print(f"INFO: Using receptor_path: {receptor_path}, receptor_center: {receptor_center}")
            print("INFO: Please ensure obabel and qvina2.1 are in your PATH or qvina2.1 is in ./bin/")

            # Try to import sascorer
            try:
                import sascorer
                has_sascorer = True
            except ImportError:
                print("Warning: Could not import sascorer module. SAS scores will not be calculated.")
                has_sascorer = False
            
            # Try to import docking function - REMOVE OLD LOGIC
            # No longer importing from experiments.graoh_sequential, functions are copied above.
            # The has_docking logic will implicitly be handled by dock_best_molecule itself.
            # If obabel or qvina2.1 are not found, dock_best_molecule will print an error and return None.
            # The existing result handling already checks if docking_score is None.
            has_docking = True # Assume we attempt docking; function will handle tool presence.
            
            # Create result lists for each encoder
            fragment_results = []
            smiles_results = []
            
            # Run 10 generations
            for i in range(10):
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
                    
                    sas_score_fragment = None
                    qed_score_fragment = None # Initialize QED score
                    docking_score_fragment = None

                    if has_sascorer:
                        rdkit_mol_fragment = Chem.MolFromSmiles(fragment_smiles)
                        if rdkit_mol_fragment:
                            try:
                                sas_score_fragment = sascorer.calculateScore(rdkit_mol_fragment)
                                print(f"SAS score: {sas_score_fragment:.2f}")
                            except Exception as e:
                                print(f"Error calculating SAS score: {e}")
                    
                    # Calculate QED score for fragment encoder
                    rdkit_mol_fragment_for_qed = Chem.MolFromSmiles(fragment_smiles) # Re-get or use existing if available
                    if rdkit_mol_fragment_for_qed:
                        try:
                            qed_score_fragment = QED.qed(rdkit_mol_fragment_for_qed)
                            print(f"QED score: {qed_score_fragment:.2f}")
                        except Exception as e:
                            print(f"Error calculating QED score: {e}")

                    if has_docking:
                        # rdkit_mol_fragment is already available if sascorer ran
                        if rdkit_mol_fragment is None: # If sascorer didn't run or failed
                            rdkit_mol_fragment = Chem.MolFromSmiles(fragment_smiles)

                        if rdkit_mol_fragment:
                            docking_score_fragment = dock_best_molecule(rdkit_mol_fragment, receptor_path, receptor_center)
                            if docking_score_fragment is not None:
                                print(f"Docking score: {docking_score_fragment:.2f}")
                            else:
                                print(f"Docking failed for {fragment_smiles}")
                    
                    # Store results
                    fragment_results.append({
                        'run': i+1,
                        'smiles': fragment_smiles,
                        'sas_score': sas_score_fragment,
                        'docking_score': docking_score_fragment,
                        'qed_score': qed_score_fragment # Store QED score
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
                    
                    sas_score_smiles = None
                    qed_score_smiles = None # Initialize QED score
                    docking_score_smiles = None

                    if has_sascorer:
                        rdkit_mol_smiles = Chem.MolFromSmiles(smiles_smiles)
                        if rdkit_mol_smiles:
                            try:
                                sas_score_smiles = sascorer.calculateScore(rdkit_mol_smiles)
                                print(f"SAS score: {sas_score_smiles:.2f}")
                            except Exception as e:
                                print(f"Error calculating SAS score: {e}")

                    # Calculate QED score for SMILES encoder
                    rdkit_mol_smiles_for_qed = Chem.MolFromSmiles(smiles_smiles) # Re-get or use existing
                    if rdkit_mol_smiles_for_qed:
                        try:
                            qed_score_smiles = QED.qed(rdkit_mol_smiles_for_qed)
                            print(f"QED score: {qed_score_smiles:.2f}")
                        except Exception as e:
                            print(f"Error calculating QED score: {e}")
                    
                    if has_docking:
                        # rdkit_mol_smiles is already available if sascorer ran
                        if rdkit_mol_smiles is None: # If sascorer didn't run or failed
                            rdkit_mol_smiles = Chem.MolFromSmiles(smiles_smiles)
                        
                        if rdkit_mol_smiles:
                            docking_score_smiles = dock_best_molecule(rdkit_mol_smiles, receptor_path, receptor_center)
                            if docking_score_smiles is not None:
                                print(f"Docking score: {docking_score_smiles:.2f}")
                            else:
                                print(f"Docking failed for {smiles_smiles}")
                    
                    # Store results
                    smiles_results.append({
                        'run': i+1,
                        'smiles': smiles_smiles,
                        'sas_score': sas_score_smiles,
                        'docking_score': docking_score_smiles,
                        'qed_score': qed_score_smiles # Store QED score
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
                    f"{result['docking_score']:.2f}" if result['docking_score'] is not None else "N/A",
                    f"{result['qed_score']:.2f}" if result['qed_score'] is not None else "N/A"
                ])
            
            print(tabulate(fragment_table, headers=["Run", "SMILES", "SAS Score", "Docking Score", "QED Score"], tablefmt="grid"))
            
            # SMILES encoder summary
            print("\nSMILES Encoder Results:")
            smiles_table = []
            for result in smiles_results:
                smiles_table.append([
                    result['run'],
                    result['smiles'],
                    f"{result['sas_score']:.2f}" if result['sas_score'] is not None else "N/A",
                    f"{result['docking_score']:.2f}" if result['docking_score'] is not None else "N/A",
                    f"{result['qed_score']:.2f}" if result['qed_score'] is not None else "N/A"
                ])
            
            print(tabulate(smiles_table, headers=["Run", "SMILES", "SAS Score", "Docking Score", "QED Score"], tablefmt="grid"))
            
            # Calculate average statistics
            frag_sas_scores = [r['sas_score'] for r in fragment_results if r['sas_score'] is not None]
            frag_docking_scores = [r['docking_score'] for r in fragment_results if r['docking_score'] is not None]
            frag_qed_scores = [r['qed_score'] for r in fragment_results if r['qed_score'] is not None]
            
            smiles_sas_scores = [r['sas_score'] for r in smiles_results if r['sas_score'] is not None]
            smiles_docking_scores = [r['docking_score'] for r in smiles_results if r['docking_score'] is not None]
            smiles_qed_scores = [r['qed_score'] for r in smiles_results if r['qed_score'] is not None]
            
            print("\nAverage Statistics:")
            stats_table = [
                ["Fragment Encoder", 
                 f"{np.mean(frag_sas_scores):.2f} ± {np.std(frag_sas_scores):.2f}" if frag_sas_scores else "N/A",
                 f"{np.mean(frag_docking_scores):.2f} ± {np.std(frag_docking_scores):.2f}" if frag_docking_scores else "N/A",
                 f"{np.mean(frag_qed_scores):.2f} ± {np.std(frag_qed_scores):.2f}" if frag_qed_scores else "N/A"],
                ["SMILES Encoder", 
                 f"{np.mean(smiles_sas_scores):.2f} ± {np.std(smiles_sas_scores):.2f}" if smiles_sas_scores else "N/A",
                 f"{np.mean(smiles_docking_scores):.2f} ± {np.std(smiles_docking_scores):.2f}" if smiles_docking_scores else "N/A",
                 f"{np.mean(smiles_qed_scores):.2f} ± {np.std(smiles_qed_scores):.2f}" if smiles_qed_scores else "N/A"]
            ]
            
            print(tabulate(stats_table, headers=["Encoder", "Avg SAS Score", "Avg Docking Score", "Avg QED Score"], tablefmt="grid"))
            
            # Calculate win counts: how many times did fragment encoder perform better?
            better_docking_count = 0
            better_sas_count = 0
            better_qed_count = 0
            valid_comparisons = 0
            valid_qed_comparisons = 0
            
            for i in range(min(len(fragment_results), len(smiles_results))):
                frag_result = fragment_results[i]
                smiles_result = smiles_results[i]
                
                # For docking scores, compare when both are available (lower is better)
                if frag_result['docking_score'] is not None and smiles_result['docking_score'] is not None:
                    if i == 0 or (fragment_results[i-1]['docking_score'] is None or smiles_results[i-1]['docking_score'] is None):
                        valid_comparisons = 0
                    valid_comparisons +=1 
                    if frag_result['docking_score'] < smiles_result['docking_score']:
                        better_docking_count += 1
                
                # For SAS scores, compare when both are available (lower is better)
                if frag_result['sas_score'] is not None and smiles_result['sas_score'] is not None:
                    if frag_result['sas_score'] < smiles_result['sas_score']:
                        better_sas_count += 1

                # For QED scores, compare when both are available (higher is better)
                if frag_result['qed_score'] is not None and smiles_result['qed_score'] is not None:
                    valid_qed_comparisons +=1
                    if frag_result['qed_score'] > smiles_result['qed_score']:
                        better_qed_count += 1
            
            print("\nWin Rate Comparison:")
            if valid_comparisons > 0:
                print(f"Fragment encoder had better docking scores in {better_docking_count}/{valid_comparisons} runs ({better_docking_count/valid_comparisons*100:.1f}%)")
                print(f"Fragment encoder had better SAS scores in {better_sas_count}/{valid_comparisons} runs ({better_sas_count/valid_comparisons*100:.1f}%)")
            else:
                print("Not enough valid comparisons for docking/SAS win rates")
            if valid_qed_comparisons > 0:
                 print(f"Fragment encoder had better QED scores in {better_qed_count}/{valid_qed_comparisons} runs ({better_qed_count/valid_qed_comparisons*100:.1f}%)")
            else:
                print("Not enough valid comparisons for QED win rates")
                
            # Create a head-to-head comparison table
            print("\nHead-to-Head Comparison:")
            head_to_head = []
            for i in range(min(len(fragment_results), len(smiles_results))):
                frag_result = fragment_results[i]
                smiles_result = smiles_results[i]
                
                # Format docking comparison
                docking_comp = "N/A"
                if frag_result['docking_score'] is not None and smiles_result['docking_score'] is not None:
                    docking_diff = frag_result['docking_score'] - smiles_result['docking_score']
                    docking_winner = "Fragment" if docking_diff < 0 else "SMILES" if docking_diff > 0 else "Tie"
                    docking_comp = f"{frag_result['docking_score']:.2f} vs {smiles_result['docking_score']:.2f} ({docking_winner})"
                
                # Format SAS comparison
                sas_comp = "N/A"
                if frag_result['sas_score'] is not None and smiles_result['sas_score'] is not None:
                    sas_diff = frag_result['sas_score'] - smiles_result['sas_score']
                    sas_winner = "Fragment" if sas_diff < 0 else "SMILES" if sas_diff > 0 else "Tie"
                    sas_comp = f"{frag_result['sas_score']:.2f} vs {smiles_result['sas_score']:.2f} ({sas_winner})"

                # Format QED comparison
                qed_comp = "N/A"
                if frag_result['qed_score'] is not None and smiles_result['qed_score'] is not None:
                    qed_diff = frag_result['qed_score'] - smiles_result['qed_score']
                    qed_winner = "Fragment" if qed_diff > 0 else "SMILES" if qed_diff < 0 else "Tie"
                    qed_comp = f"{frag_result['qed_score']:.2f} vs {smiles_result['qed_score']:.2f} ({qed_winner})"
                
                head_to_head.append([i+1, docking_comp, sas_comp, qed_comp])
            
            print(tabulate(head_to_head, headers=["Run", "Docking Score (Fragment vs SMILES)", "SAS Score (Fragment vs SMILES)", "QED Score (Fragment vs SMILES)"], tablefmt="grid"))
            
            # Comparison of best molecules
            print("\nBest Molecules by Docking Score:")
            best_frag_idx = np.argmin(frag_docking_scores) if frag_docking_scores else None
            best_smiles_idx = np.argmin(smiles_docking_scores) if smiles_docking_scores else None
            
            if best_frag_idx is not None:
                best_frag_result = fragment_results[best_frag_idx]
                print(f"Fragment Encoder Best: {best_frag_result['smiles']}")
                print(f"  SAS Score: {best_frag_result['sas_score']:.2f}" if best_frag_result['sas_score'] is not None else "  SAS Score: N/A")
                print(f"  Docking Score: {best_frag_result['docking_score']:.2f}" if best_frag_result['docking_score'] is not None else "  Docking Score: N/A")
                print(f"  QED Score: {best_frag_result['qed_score']:.2f}" if best_frag_result['qed_score'] is not None else "  QED Score: N/A")
            
            if best_smiles_idx is not None:
                best_smiles_result = smiles_results[best_smiles_idx]
                print(f"SMILES Encoder Best: {best_smiles_result['smiles']}")
                print(f"  SAS Score: {best_smiles_result['sas_score']:.2f}" if best_smiles_result['sas_score'] is not None else "  SAS Score: N/A")
                print(f"  Docking Score: {best_smiles_result['docking_score']:.2f}" if best_smiles_result['docking_score'] is not None else "  Docking Score: N/A")
                print(f"  QED Score: {best_smiles_result['qed_score']:.2f}" if best_smiles_result['qed_score'] is not None else "  QED Score: N/A")
            
        else:
            print("Cannot compare results as at least one generation has no molecules")
    else:
        print("Cannot compare results as at least one generation failed")

    # Helper function for new experiments
    def generate_and_evaluate_latent_batch(
        latent_code, padding_mask, num_generations, 
        decoder_model, token_head_model, reaction_head_model, fingerprint_head_model,
        fpindex_obj, rxn_matrix_obj, rec_path, rec_center, max_steps_decode
    ):
        all_smiles = []
        all_qed_scores = []
        all_docking_scores = []

        encoder_output_current = EncoderOutput(latent_code.to('cpu'), padding_mask.to('cpu'))

        for _ in range(num_generations):
            # Each generation needs a fresh state pool, as it consumes tokens from the start
            # The SimpleStatePool is stateful regarding the decoding process (self.token_types etc.)
            state_pool_gen = SimpleStatePool(
                fpindex=fpindex_obj, rxn_matrix=rxn_matrix_obj, 
                encoder_output=encoder_output_current, # Same latent input for all generations
                decoder=decoder_model, token_head=token_head_model, 
                reaction_head=reaction_head_model, fingerprint_head=fingerprint_head_model
            )
            for step in range(max_steps_decode):
                token = state_pool_gen.step()
                if token == TokenType.END.value: break
            
            mols_gen = state_pool_gen.stack.get_top()
            if mols_gen:
                mol_obj = list(mols_gen)[0]
                smi = mol_obj.smiles
                all_smiles.append(smi)
                rdkit_mol = Chem.MolFromSmiles(smi)
                if rdkit_mol:
                    try: 
                        qed = QED.qed(rdkit_mol)
                        all_qed_scores.append(qed)
                    except Exception: pass # QED calculation can fail
                    try: 
                        dock = dock_best_molecule(rdkit_mol, rec_path, rec_center)
                        if dock is not None: all_docking_scores.append(dock)
                    except Exception: pass # Docking can fail
        
        results = {
            'unique_smiles_count': len(set(all_smiles)),
            'total_valid_smiles': len(all_smiles),
            'best_qed': np.max(all_qed_scores) if all_qed_scores else None,
            'median_qed': np.median(all_qed_scores) if all_qed_scores else None,
            'best_docking': np.min(all_docking_scores) if all_docking_scores else None,
            'median_docking': np.median(all_docking_scores) if all_docking_scores else None,
            'all_smiles': list(set(all_smiles))[:5] # Show a few unique SMILES
        }
        return results

    # === Experiment 1: Random-direction probes ===
    print("\n\n=== Experiment 1: Random-direction probes ===")
    num_base_vectors_exp1 = 2 # Reduced for brevity, user can increase
    num_perturbations_exp1 = 2
    noise_scale_factor_exp1 = 0.05 
    base_z_scale_exp1 = 0.05 
    max_steps_exp1 = 10 
    num_generations_per_latent = 1 # Number of molecules to generate per latent vector

    original_padding_mask_exp = output.code_padding_mask.to('cpu')
    original_code_template_exp = output.code.to('cpu') 

    print(f"Generating {num_generations_per_latent} molecules per latent vector.")

    for i in range(num_base_vectors_exp1):
        print(f"\n--- Base Vector {i+1}/{num_base_vectors_exp1} ---")
        z_base_exp1 = torch.randn_like(original_code_template_exp) * base_z_scale_exp1
        
        base_results = generate_and_evaluate_latent_batch(
            z_base_exp1, original_padding_mask_exp, num_generations_per_latent,
            decoder, token_head, reaction_head, fingerprint_head,
            fpindex, rxn_matrix, receptor_path, receptor_center, max_steps_exp1
        )
        
        avg_qed_str = f"{base_results['best_qed']:.2f}" if base_results['best_qed'] is not None else 'N/A'
        median_qed_str = f"{base_results['median_qed']:.2f}" if base_results['median_qed'] is not None else 'N/A'
        avg_dock_str = f"{base_results['best_docking']:.2f}" if base_results['best_docking'] is not None else 'N/A'
        median_dock_str = f"{base_results['median_docking']:.2f}" if base_results['median_docking'] is not None else 'N/A'

        print(f"Base: Unique SMILES={base_results['unique_smiles_count']}/{base_results['total_valid_smiles']}, "
              f"BestQED={avg_qed_str}, MedQED={median_qed_str}, "
              f"BestDock={avg_dock_str}, MedDock={median_dock_str}")
        print(f"  Sample SMILES: {base_results['all_smiles']}")

        for j in range(num_perturbations_exp1):
            epsilon_exp1 = torch.randn_like(z_base_exp1) * noise_scale_factor_exp1 
            z_noisy_exp1 = z_base_exp1 + epsilon_exp1
            
            noisy_results = generate_and_evaluate_latent_batch(
                z_noisy_exp1, original_padding_mask_exp, num_generations_per_latent,
                decoder, token_head, reaction_head, fingerprint_head,
                fpindex, rxn_matrix, receptor_path, receptor_center, max_steps_exp1
            )
            
            avg_qed_noisy_str = f"{noisy_results['best_qed']:.2f}" if noisy_results['best_qed'] is not None else 'N/A'
            median_qed_noisy_str = f"{noisy_results['median_qed']:.2f}" if noisy_results['median_qed'] is not None else 'N/A'
            avg_dock_noisy_str = f"{noisy_results['best_docking']:.2f}" if noisy_results['best_docking'] is not None else 'N/A'
            median_dock_noisy_str = f"{noisy_results['median_docking']:.2f}" if noisy_results['median_docking'] is not None else 'N/A'

            print(f"  Perturbed {j+1}: Unique SMILES={noisy_results['unique_smiles_count']}/{noisy_results['total_valid_smiles']}, "
                  f"BestQED={avg_qed_noisy_str}, MedQED={median_qed_noisy_str}, "
                  f"BestDock={avg_dock_noisy_str}, MedDock={median_dock_noisy_str}")
            print(f"    Sample SMILES: {noisy_results['all_smiles']}")

    # === Experiment 2: Directional probes ===
    print("\n\n=== Experiment 2: Directional probes ===")
    num_interpolation_steps_exp2 = 10
    # num_generations_per_latent is reused from Exp1 settings
    max_steps_exp2 = 10

    z_high_exp2 = output.code.clone().detach().to('cpu') 
    z_low_exp2 = torch.randn_like(z_high_exp2) * base_z_scale_exp1 

    print(f"Generating {num_generations_per_latent} molecules per latent vector for directional probes.")

    print("\nDecoding z_low (random scaled vector):")
    low_results_exp2 = generate_and_evaluate_latent_batch(
        z_low_exp2, original_padding_mask_exp, num_generations_per_latent,
        decoder, token_head, reaction_head, fingerprint_head,
        fpindex, rxn_matrix, receptor_path, receptor_center, max_steps_exp2
    )
    
    avg_qed_low_str = f"{low_results_exp2['best_qed']:.2f}" if low_results_exp2['best_qed'] is not None else 'N/A'
    median_qed_low_str = f"{low_results_exp2['median_qed']:.2f}" if low_results_exp2['median_qed'] is not None else 'N/A'
    avg_dock_low_str = f"{low_results_exp2['best_docking']:.2f}" if low_results_exp2['best_docking'] is not None else 'N/A'
    median_dock_low_str = f"{low_results_exp2['median_docking']:.2f}" if low_results_exp2['median_docking'] is not None else 'N/A'

    print(f"z_low: Unique SMILES={low_results_exp2['unique_smiles_count']}/{low_results_exp2['total_valid_smiles']}, "
          f"BestQED={avg_qed_low_str}, MedQED={median_qed_low_str}, "
          f"BestDock={avg_dock_low_str}, MedDock={median_dock_low_str}")
    print(f"  Sample SMILES: {low_results_exp2['all_smiles']}")

    print("\nDecoding z_high (fragment encoder output for desert_sequence):")
    high_results_exp2 = generate_and_evaluate_latent_batch(
        z_high_exp2, original_padding_mask_exp, num_generations_per_latent,
        decoder, token_head, reaction_head, fingerprint_head,
        fpindex, rxn_matrix, receptor_path, receptor_center, max_steps_exp2
    )
    
    avg_qed_high_str = f"{high_results_exp2['best_qed']:.2f}" if high_results_exp2['best_qed'] is not None else 'N/A'
    median_qed_high_str = f"{high_results_exp2['median_qed']:.2f}" if high_results_exp2['median_qed'] is not None else 'N/A'
    avg_dock_high_str = f"{high_results_exp2['best_docking']:.2f}" if high_results_exp2['best_docking'] is not None else 'N/A'
    median_dock_high_str = f"{high_results_exp2['median_docking']:.2f}" if high_results_exp2['median_docking'] is not None else 'N/A'

    print(f"z_high: Unique SMILES={high_results_exp2['unique_smiles_count']}/{high_results_exp2['total_valid_smiles']}, "
          f"BestQED={avg_qed_high_str}, MedQED={median_qed_high_str}, "
          f"BestDock={avg_dock_high_str}, MedDock={median_dock_high_str}")
    print(f"  Sample SMILES: {high_results_exp2['all_smiles']}")

    print("\nInterpolating z_low to z_high:")
    for step_idx_exp2 in range(num_interpolation_steps_exp2 + 1):
        alpha = step_idx_exp2 / num_interpolation_steps_exp2
        z_interp_exp2 = z_low_exp2 + alpha * (z_high_exp2 - z_low_exp2)
        
        interp_results = generate_and_evaluate_latent_batch(
            z_interp_exp2, original_padding_mask_exp, num_generations_per_latent,
            decoder, token_head, reaction_head, fingerprint_head,
            fpindex, rxn_matrix, receptor_path, receptor_center, max_steps_exp2
        )
        
        avg_qed_interp_str = f"{interp_results['best_qed']:.2f}" if interp_results['best_qed'] is not None else 'N/A'
        median_qed_interp_str = f"{interp_results['median_qed']:.2f}" if interp_results['median_qed'] is not None else 'N/A'
        avg_dock_interp_str = f"{interp_results['best_docking']:.2f}" if interp_results['best_docking'] is not None else 'N/A'
        median_dock_interp_str = f"{interp_results['median_docking']:.2f}" if interp_results['median_docking'] is not None else 'N/A'

        print(f"Alpha={alpha:.2f}: Unique SMILES={interp_results['unique_smiles_count']}/{interp_results['total_valid_smiles']}, "
              f"BestQED={avg_qed_interp_str}, MedQED={median_qed_interp_str}, "
              f"BestDock={avg_dock_interp_str}, MedDock={median_dock_interp_str}")
        print(f"  Sample SMILES: {interp_results['all_smiles']}")
    