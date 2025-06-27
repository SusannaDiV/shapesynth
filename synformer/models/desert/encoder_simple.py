import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class FragmentEncoder(nn.Module):
    """
    Encoder for DESERT fragment sequences.
    This encoder converts fragment sequences into embeddings that can be used by the Synformer decoder.
    """
    
    def __init__(self, vocab: Dict, embedding_dim: int = 768, device: str = 'cuda'):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Create a mapping from fragment ID to index
        self.fragment_to_idx = {}
        for token, (_, _, idx) in vocab.items():
            self.fragment_to_idx[idx] = len(self.fragment_to_idx)
        
        # Create an embedding layer for the fragments
        self.embedding = nn.Embedding(len(self.fragment_to_idx) + 1, embedding_dim)  # +1 for padding
        
        # Initialize the embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Move to device
        self.to(device)
    
    def encode_desert_sequence(self, sequence: List[int], device: Optional[str] = None) -> torch.Tensor:
        """
        Encode a DESERT sequence into embeddings.
        
        Args:
            sequence: List of fragment IDs from DESERT
            device: Device to place the tensor on (defaults to self.device)
            
        Returns:
            Tensor of shape (1, seq_len, embedding_dim)
        """
        if device is None:
            device = self.device
        
        # Convert fragment IDs to indices
        indices = []
        for frag_id in sequence:
            if frag_id in self.fragment_to_idx:
                indices.append(self.fragment_to_idx[frag_id])
            else:
                # Use a default index for unknown fragments
                indices.append(len(self.fragment_to_idx))
        
        # Convert to tensor
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension
        
        # Get embeddings
        embeddings = self.embedding(indices_tensor)
        
        return embeddings


def create_fragment_encoder(vocab_path: str, embedding_dim: int = 768, device: str = 'cuda') -> FragmentEncoder:
    """
    Create a fragment encoder from a vocabulary file.
    
    Args:
        vocab_path: Path to the vocabulary pickle file
        embedding_dim: Dimension of the embeddings
        device: Device to place the encoder on
        
    Returns:
        FragmentEncoder instance
    """
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Create encoder
    encoder = FragmentEncoder(vocab, embedding_dim, device)
    
    return encoder

