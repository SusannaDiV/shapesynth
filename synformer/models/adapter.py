import torch
from torch import nn
from synformer.models.transformer.positional_encoding import PositionalEncoding

class ContinuousCodeProjector(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, continuous_code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous_code: can have shape
                - [B, L, in_dim]
                - [B, 1, L, in_dim]  (common case from Uni-Mol)
        Returns:
            Tensor of shape [B, L, out_dim]
        """
        # Squeeze extra dimension if present
        if continuous_code.dim() == 4 and continuous_code.size(1) == 1:
            continuous_code = continuous_code.squeeze(1)
        return self.proj(continuous_code)

class UniMolAdapter(nn.Module):
    def __init__(
        self,
        unimol_dim: int = 1024,      # UniMol output dimension
        d_model: int = 768,          # Target dimension for decoder
        num_tokens: int = 32,        # Number of output tokens
        num_layers: int = 2,         # Number of transformer layers
        nhead: int = 12,            # Number of attention heads
        dim_feedforward: int = 3072, # FFN dimension (4x d_model)
        dropout: float = 0.1,        # Dropout rate
        activation: str = "gelu",    # Activation function
        norm_first: bool = True,     # Pre-norm architecture
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.num_tokens = num_tokens
        self.d_model = d_model 
        self.dim_reduction = nn.Linear(unimol_dim, d_model) if unimol_dim != d_model else None
        
        # Learnable token embeddings that will be expanded
        self.token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=num_tokens)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # for stability
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, unimol_latent: torch.Tensor) -> torch.Tensor:
        """
        Convert UniMol latent vectors to a sequence of token embeddings.
        
        Args:
            unimol_latent: Tensor of shape [B, L, unimol_dim]
        Returns:
            tokens: Tensor of shape [B, num_tokens, d_model]
        """
        batch_size = unimol_latent.size(0)
        
        # Project to target dimension if needed BEFORE mean pooling
        if self.dim_reduction is not None:
            # Reshape to 2D for linear projection
            orig_shape = unimol_latent.shape
            unimol_latent = unimol_latent.reshape(-1, orig_shape[-1])  # [B*L, unimol_dim]
            unimol_latent = self.dim_reduction(unimol_latent)  # [B*L, d_model]
            unimol_latent = unimol_latent.reshape(orig_shape[0], -1, self.d_model)  # [B, L, d_model]
        
        # Average pool over the sequence dimension to get [B, d_model]
        unimol_latent = unimol_latent.mean(dim=1)
        
        token_embeddings = self.token_embedding.expand(batch_size, 1, -1)
        
        projected = token_embeddings + unimol_latent.unsqueeze(1)
        
        tokens = projected.expand(-1, self.num_tokens, -1)
        
        tokens = self.pos_encoding(tokens)
        
        tokens = self.transformer(tokens)
        
        tokens = self.final_norm(tokens)
        
        return tokens
