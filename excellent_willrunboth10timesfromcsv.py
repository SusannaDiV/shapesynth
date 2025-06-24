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
    import pandas as pd  # Add pandas import for CSV handling
    
    # Load the CSV file
    csv_path = "fragments_qed_docking.csv"
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")

    # Get vocab path from the checkpoint path
    smiles_checkpoint_path = "/workspace/data/processed/sf_ed_default.ckpt"
    vocab_path = "/workspace/data/desert/vocab.pkl"

    def parse_desert_sequence_from_row(row):
        """Parse desert sequence from a CSV row"""
        desert_sequence = []
        
        # Look for fragment columns (fragment_1_id, fragment_1_translation, etc.)
        for i in range(1, 18):  # fragments 1-17 based on CSV structure
            id_col = f'fragment_{i}_id'
            trans_col = f'fragment_{i}_translation'
            rot_col = f'fragment_{i}_rotation'
            
            if id_col in row and not pd.isna(row[id_col]):
                frag_id = int(row[id_col])
                translation = int(row[trans_col]) if not pd.isna(row[trans_col]) else 0
                rotation = int(row[rot_col]) if not pd.isna(row[rot_col]) else 0
                
                desert_sequence.append((frag_id, translation, rotation))
                
                # Stop if we hit EOS token (ID 3)
                if frag_id == 3:
                    break
            else:
                break
        
        return desert_sequence

    # Initialize models and data once before processing all rows
    print("Initializing models and loading data...")
    encoder = create_fragment_encoder(vocab_path=vocab_path, device='cpu', mixture_weight=1)
    
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
    
    # Import necessary modules for decoder
    from synformer.models.decoder import Decoder
    from synformer.data.common import TokenType
    from synformer.chem.fpindex import FingerprintIndex
    from synformer.chem.matrix import ReactantReactionMatrix
    from synformer.models.synformer import Synformer
    from synformer.chem.mol import Molecule
    from synformer.sampler.analog.state_pool import StatePool
    from omegaconf import OmegaConf
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
    
    # Define receptor path and center for docking
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
    
    has_docking = True # Assume we attempt docking; function will handle tool presence.
    
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
    
    # Load decoder and other necessary components
    print("Loading decoder and prediction heads...")
    
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

    # Collections for aggregated results across all rows
    all_fragment_results = []
    all_smiles_results = []
    
    print(f"\nProcessing {len(df)} rows from CSV...")
    
    # Main loop to process each row
    for idx, row in df.iterrows():
        print(f"\n{'='*80}")
        print(f"Processing row {idx+1}/{len(df)}: {row['shape_file_path']}")
        print(f"{'='*80}")
        
        # Parse current row data
        current_smiles = row['smiles_used']
        current_desert_sequence = parse_desert_sequence_from_row(row)
        
        print(f"Current SMILES: {current_smiles}")
        print(f"Current desert sequence: {len(current_desert_sequence)} fragments")
        
        if not current_desert_sequence:
            print("Warning: No valid desert sequence found in this row, skipping...")
            continue
        
        # Encode current inputs
        print("Encoding fragment sequence...")
        output = encoder.encode_desert_sequence(current_desert_sequence, device='cpu')
        
        print("Encoding SMILES sequence...")
        smiles_tokens = tokenize_smiles(current_smiles)
        smiles_tensor = torch.tensor(smiles_tokens, device='cpu').unsqueeze(0)
        smiles_batch = ProjectionBatch({"smiles": smiles_tensor})
        
        with torch.no_grad():
            smiles_output = smiles_encoder(smiles_batch)
        
        print(f"Fragment encoder output shape: {output.code.shape}")
        print(f"SMILES encoder output shape: {smiles_output.code.shape}")
        
        # Run multiple generations (10 times) for current row
        print(f"\n=== Running Multiple Generations (10 times) for Row {idx+1} ===\n")
        
        # Create result lists for each encoder for this row
        fragment_results = []
        smiles_results = []
        
        # Run 10 generations
        max_steps = 24
        for i in range(10):
            print(f"\n--- Row {idx+1}, Run {i+1}/10 ---")
            
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
                qed_score_fragment = None
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
                rdkit_mol_fragment_for_qed = Chem.MolFromSmiles(fragment_smiles)
                if rdkit_mol_fragment_for_qed:
                    try:
                        qed_score_fragment = QED.qed(rdkit_mol_fragment_for_qed)
                        print(f"QED score: {qed_score_fragment:.2f}")
                    except Exception as e:
                        print(f"Error calculating QED score: {e}")

                if has_docking:
                    if 'rdkit_mol_fragment' not in locals() or rdkit_mol_fragment is None:
                        rdkit_mol_fragment = Chem.MolFromSmiles(fragment_smiles)

                    if rdkit_mol_fragment:
                        docking_score_fragment = dock_best_molecule(rdkit_mol_fragment, receptor_path, receptor_center)
                        if docking_score_fragment is not None:
                            print(f"Docking score: {docking_score_fragment:.2f}")
                        else:
                            print(f"Docking failed for {fragment_smiles}")
                
                # Store results
                fragment_results.append({
                    'row': idx+1,
                    'run': i+1,
                    'smiles': fragment_smiles,
                    'sas_score': sas_score_fragment,
                    'docking_score': docking_score_fragment,
                    'qed_score': qed_score_fragment
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
                qed_score_smiles = None
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
                rdkit_mol_smiles_for_qed = Chem.MolFromSmiles(smiles_smiles)
                if rdkit_mol_smiles_for_qed:
                    try:
                        qed_score_smiles = QED.qed(rdkit_mol_smiles_for_qed)
                        print(f"QED score: {qed_score_smiles:.2f}")
                    except Exception as e:
                        print(f"Error calculating QED score: {e}")
                
                if has_docking:
                    if rdkit_mol_smiles is None:
                        rdkit_mol_smiles = Chem.MolFromSmiles(smiles_smiles)
                    
                    if rdkit_mol_smiles:
                        docking_score_smiles = dock_best_molecule(rdkit_mol_smiles, receptor_path, receptor_center)
                        if docking_score_smiles is not None:
                            print(f"Docking score: {docking_score_smiles:.2f}")
                        else:
                            print(f"Docking failed for {smiles_smiles}")
                
                # Store results
                smiles_results.append({
                    'row': idx+1,
                    'run': i+1,
                    'smiles': smiles_smiles,
                    'sas_score': sas_score_smiles,
                    'docking_score': docking_score_smiles,
                    'qed_score': qed_score_smiles
                })
            else:
                print("No molecules generated")
        
        # Add results from this row to global collections
        all_fragment_results.extend(fragment_results)
        all_smiles_results.extend(smiles_results)
        
        # Print summary for this row
        print(f"\n=== Summary for Row {idx+1} ===")
        if fragment_results and smiles_results:
            # Calculate averages for this row
            frag_scores = {
                'sas': [r['sas_score'] for r in fragment_results if r['sas_score'] is not None],
                'docking': [r['docking_score'] for r in fragment_results if r['docking_score'] is not None],
                'qed': [r['qed_score'] for r in fragment_results if r['qed_score'] is not None]
            }
            smiles_scores = {
                'sas': [r['sas_score'] for r in smiles_results if r['sas_score'] is not None],
                'docking': [r['docking_score'] for r in smiles_results if r['docking_score'] is not None],
                'qed': [r['qed_score'] for r in smiles_results if r['qed_score'] is not None]
            }
            
            print(f"Fragment encoder averages - SAS: {np.mean(frag_scores['sas']):.2f}, Docking: {np.mean(frag_scores['docking']):.2f}, QED: {np.mean(frag_scores['qed']):.2f}")
            print(f"SMILES encoder averages - SAS: {np.mean(smiles_scores['sas']):.2f}, Docking: {np.mean(smiles_scores['docking']):.2f}, QED: {np.mean(smiles_scores['qed']):.2f}")
    
    # Print final aggregated summary
    print(f"\n{'='*80}")
    print("FINAL AGGREGATED SUMMARY ACROSS ALL ROWS")
    print(f"{'='*80}")
    
    if all_fragment_results and all_smiles_results:
        # Calculate overall averages across all rows
        all_frag_scores = {
            'sas': [r['sas_score'] for r in all_fragment_results if r['sas_score'] is not None],
            'docking': [r['docking_score'] for r in all_fragment_results if r['docking_score'] is not None],
            'qed': [r['qed_score'] for r in all_fragment_results if r['qed_score'] is not None]
        }
        all_smiles_scores = {
            'sas': [r['sas_score'] for r in all_smiles_results if r['sas_score'] is not None],
            'docking': [r['docking_score'] for r in all_smiles_results if r['docking_score'] is not None],
            'qed': [r['qed_score'] for r in all_smiles_results if r['qed_score'] is not None]
        }
        
        print("\nOverall Average Statistics Across All Rows:")
        overall_stats_table = [
            ["Fragment Encoder", 
             f"{np.mean(all_frag_scores['sas']):.2f} ± {np.std(all_frag_scores['sas']):.2f}" if all_frag_scores['sas'] else "N/A",
             f"{np.mean(all_frag_scores['docking']):.2f} ± {np.std(all_frag_scores['docking']):.2f}" if all_frag_scores['docking'] else "N/A",
             f"{np.mean(all_frag_scores['qed']):.2f} ± {np.std(all_frag_scores['qed']):.2f}" if all_frag_scores['qed'] else "N/A"],
            ["SMILES Encoder", 
             f"{np.mean(all_smiles_scores['sas']):.2f} ± {np.std(all_smiles_scores['sas']):.2f}" if all_smiles_scores['sas'] else "N/A",
             f"{np.mean(all_smiles_scores['docking']):.2f} ± {np.std(all_smiles_scores['docking']):.2f}" if all_smiles_scores['docking'] else "N/A",
             f"{np.mean(all_smiles_scores['qed']):.2f} ± {np.std(all_smiles_scores['qed']):.2f}" if all_smiles_scores['qed'] else "N/A"]
        ]
        
        print(tabulate(overall_stats_table, headers=["Encoder", "Avg SAS Score", "Avg Docking Score", "Avg QED Score"], tablefmt="grid"))
        
        print(f"\nTotal results processed: {len(all_fragment_results)} fragment runs, {len(all_smiles_results)} SMILES runs")
        print("Processing complete!")
    else:
        print("No results collected!")
        
    print("\nScript execution finished.")
    