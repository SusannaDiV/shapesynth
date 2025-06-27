import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union

from synformer.models.encoder.base import BaseEncoder, EncoderOutput

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
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 3, 2) * (-math.log(10000.0) / (d_model // 3)))
        
        pe_seq = torch.zeros(1, max_len, d_model // 3)
        pe_seq[0, :, 0::2] = torch.sin(position * div_term)
        pe_seq[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_seq", pe_seq)
        
        self.trans_embedding = nn.Embedding(trans_bins, d_model // 3)
        self.rot_embedding = nn.Embedding(rot_bins, d_model // 3)
        
        self.combine_norm = nn.LayerNorm(d_model)
        
        self.spatial_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
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
        translations_clamped = torch.clamp(translations, 0, self.trans_bins - 1)
        rotations_clamped = torch.clamp(rotations, 0, self.rot_bins - 1)
        
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
        
        seq_encoding = self.pe_seq[:, :seq_len, :]
        seq_encoding = seq_encoding.expand(batch_size, -1, -1)
        
        if translations is None or rotations is None:
            seq_encoding_expanded = torch.zeros_like(x)
            seq_encoding_expanded[:, :, :self.d_model // 3] = seq_encoding
            return self.dropout(x + seq_encoding_expanded)
        
        trans_emb, rot_emb = self.get_spatial_encodings(translations, rotations)
        
        combined_encoding = torch.cat([seq_encoding, trans_emb, rot_emb], dim=2)
        
        spatial_encoding = self.spatial_mlp(combined_encoding)
        
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
        
        if vocab_path is None:
            raise ValueError("vocab_path must be provided")
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        vocab_size = len(self.vocab)
        print(f"Loaded vocabulary with {vocab_size} tokens")
        
        self.fragment_emb = nn.Embedding(vocab_size, d_model, padding_idx=self.vocab['PAD'][2])
        
        self.pe_enc = Spatial3DPositionalEncoding(
            d_model=d_model,
            max_len=pe_max_len,
            trans_bins=num_trans_bins,
            rot_bins=num_rot_bins,
            grid_resolution=grid_resolution,
            max_dist=max_dist
        )
        
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
        
        self.output_scaling = nn.Sequential(
            nn.LayerNorm(d_model),  
            nn.Linear(d_model, d_model), 
        )
        
        with torch.no_grad():
            self.output_scaling[1].weight.data.copy_(torch.eye(d_model) + torch.randn(d_model, d_model) * 0.01)
            self.output_scaling[1].bias.data.uniform_(-0.01, 0.01)
        
        self.feature_importance = nn.Parameter(torch.ones(d_model) * 0.8)
        
        self.distribution_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.spatial_dropout = nn.Dropout(0.2)
        self.to(device)
    
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
        print(f"Translation range: min={translations.min().item()}, max={translations.max().item()}")
        print(f"Rotation range: min={rotations.min().item()}, max={rotations.max().item()}")
        
        frag_emb = self.fragment_emb(fragment_ids)
        
        print(f"Fragment embeddings - mean: {frag_emb.mean().item():.4f}, std: {frag_emb.std().item():.4f}")
        
        h = self.pe_enc(frag_emb, translations, rotations)
        
        print(f"After positional encoding - mean: {h.mean().item():.4f}, std: {h.std().item():.4f}")
        
        if padding_mask is None:
            padding_mask = torch.zeros_like(fragment_ids, dtype=torch.bool)
            
        out = self.enc(h, src_key_padding_mask=padding_mask)
        
        print(f"After transformer encoder - mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
        
        out = out * self.mixture_weight
        
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
        
        print(f"Tensor shapes: fragments={fragment_ids.shape}, translations={translations.shape}, rotations={rotations.shape}, padding_mask={padding_mask.shape}")
        
        encoder_output = self.forward(fragment_ids, translations, rotations, padding_mask)
        
        print(f"Generated embeddings tensor with shape: {encoder_output.code.shape}")
        print(f"Generated padding mask with shape: {encoder_output.code_padding_mask.shape}")
        
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
    encoder = FragmentEncoder(
        d_model=embedding_dim,
        vocab_path=vocab_path,
        grid_resolution=grid_resolution,
        max_dist=max_dist,
        device=device,
        mixture_weight=mixture_weight
    )
    
    return encoder

