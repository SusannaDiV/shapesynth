import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import torch.onnx.operators
from torch.nn import Module

from synformer.data.common import ProjectionBatch
from synformer.models.transformer.positional_encoding import PositionalEncoding
from .base import BaseEncoder, EncoderOutput


def get_activation_fn(activation):
    """
    Get activation function by name

    Args:
        activation: activation function name

    Returns:
        - activation function
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise KeyError

class FFN(nn.Module):
    """
    Feed-forward neural network

    Args:
        d_model: input feature dimension
        dim_feedforward: dimensionality of inner vector space
        dim_out: output feature dimensionality
        activation: activation function
        bias: requires bias in output linear function
    """

    def __init__(self,
                 d_model,
                 dim_feedforward=None,
                 dim_out=None,
                 activation="relu",
                 bias=True):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model
        dim_out = dim_out or d_model

        self._fc1 = nn.Linear(d_model, dim_feedforward)
        self._fc2 = nn.Linear(dim_feedforward, dim_out, bias=bias)
        self._activation = get_activation_fn(activation)

    def forward(self, x):
        """
        Args:
            x: feature to perform feed-forward net
                :math:`(*, D)`, where D is feature dimension

        Returns:
            - feed forward output
                :math:`(*, D)`, where D is feature dimension
        """
        x = self._fc1(x)
        x = self._activation(x)
        x = self._fc2(x)
        return x
    
class AbstractEncoder(Module):
    """
    AbstractEncoder is the abstract for encoders, and defines general interface for encoders.

    Args:
        name: encoder name
    """

    def __init__(self, name=None):
        super().__init__()
        self._name = name
        self._cache = {}
        self._mode = 'train'

    def build(self, *args, **kwargs):
        """
        Build encoder with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of encoder. Outputs are cached until the encoder is reset.
        """
        if self._mode == 'train':
            if 'out' not in self._cache:
                out = self._forward(*args, **kwargs)
                self._cache['out'] = out
            return self._cache['out']
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        """
        Forward function to override. Its results can be auto cached in forward.
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def d_model(self):
        raise NotImplementedError

    @property
    def out_dim(self):
        raise NotImplementedError

    def _cache_states(self, name, state):
        """
        Cache a state into encoder cache

        Args:
            name: state key
            state: state value
        """
        self._cache[name] = state

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache.clear()
        self._mode = mode

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        self._cache = cache

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return self._cache

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    class __SinusoidalPositionalEmbedding(nn.Module):

        def __init__(self, embedding_dim, num_embeddings=1024):
            super().__init__()
            self._embedding_dim = embedding_dim
            self._num_embeddings = num_embeddings

            num_timescales = self._embedding_dim // 2
            log_timescale_increment = torch.FloatTensor([math.log(10000.) / (num_timescales - 1)])
            inv_timescales = nn.Parameter((torch.arange(num_timescales) * -log_timescale_increment).exp(), requires_grad=False)
            self.register_buffer('_inv_timescales', inv_timescales)

        def forward(
            self,
            input,
        ):
            """Input is expected to be of size [bsz x seqlen]."""
            mask = torch.ones_like(input).type_as(self._inv_timescales)
            positions = torch.cumsum(mask, dim=1) - 1

            scaled_time = positions[:, :, None] * self._inv_timescales[None, None, :]
            signal = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=-1)
            return signal.detach()

    __embed__ = None

    def __init__(self, embedding_dim, num_embeddings=1024):
        super().__init__()
        if not SinusoidalPositionalEmbedding.__embed__:
            SinusoidalPositionalEmbedding.__embed__ = SinusoidalPositionalEmbedding.__SinusoidalPositionalEmbedding(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings
            )
        self.embedding = SinusoidalPositionalEmbedding.__embed__

    def forward(self, input):
        return self.embedding(input)

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.

    Args:
        num_embeddings: number of embeddings
        embedding_dim: embedding dimension
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, post_mask=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
        # if post_mask = True, then padding_idx = id of pad token in token embedding, we first mark padding
        # tokens using padding_idx, then generate embedding matrix using positional embedding, finally set
        # marked positions with zero
        self._post_mask = post_mask

    def forward(
        self,
        input: Tensor,
        positions: Optional[Tensor] = None
    ):
        """
        Args:
             input: an input LongTensor
                :math:`(*, L)`, where L is sequence length
            positions: pre-defined positions
                :math:`(*, L)`, where L is sequence length

        Returns:
            - positional embedding indexed from input
                :math:`(*, L, D)`, where L is sequence length and D is dimensionality
        """
        if self._post_mask:
            mask = input.ne(self.padding_idx).long()
            if positions is None:
                positions = (torch.cumsum(mask, dim=1) - 1).long()
            emb = F.embedding(
                positions,
                self.weight,
                None,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )#[B,L,H]
            emb = emb * mask.unsqueeze(-1)
            return emb
        else:
            if positions is None:
                mask = torch.ones_like(input)
                positions = (torch.cumsum(mask, dim=1) - 1).long()
            return F.embedding(
                positions,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

class AbstractEncoderLayer(nn.Module):
    """
    AbstractEncoderLayer is an abstract class for encoder layers.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}
        self._mode = 'train'

    def reset(self, mode):
        """
        Reset encoder layer and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        self._cache.clear()

    def _update_cache(self, *args, **kwargs):
        """
        Update internal cache from outside states
        """
        pass

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return self._cache

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        self._cache = cache

class TransformerEncoderLayer(AbstractEncoderLayer):
    """
    TransformerEncoderLayer performs one layer of transformer operation, namely self-attention and feed-forward network.

    Args:
        d_model: feature dimension
        nhead: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0,
                 activation="relu",
                 normalize_before=False,):
        super(TransformerEncoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward=dim_feedforward, activation=activation)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                :math:`(S, B, D)`, where S is sequence length, B is batch size and D is feature dimension
            src_mask: the attention mask for the src sequence (optional).
                :math:`(S, S)`, where S is sequence length.
            src_key_padding_mask: the mask for the src keys per batch (optional).
                :math: `(B, S)`, where B is batch size and S is sequence length
        """
        residual = src
        if self.normalize_before:
            src = self.self_attn_norm(src)
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout1(src)
        src = residual + src
        if not self.normalize_before:
            src = self.self_attn_norm(src)

        residual = src
        if self.normalize_before:
            src = self.ffn_norm(src)
        src = self.ffn(src)
        src = self.dropout2(src)
        src = residual + src
        if not self.normalize_before:
            src = self.ffn_norm(src)
        return src

class TransformerEncoder(AbstractEncoder):
    """
    TransformerEncoder is a transformer encoder.

    Args:
        num_layers: number of encoder layers
        d_model: feature dimension
        n_head: head numbers of multihead attention
        dim_feedforward: dimensionality of inner vector space
        dropout: dropout rate
        activation: activation function used in feed-forward network
        learn_pos: learning postional embedding instead of sinusoidal one
        return_seed: return with sequence representation
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 n_head=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation='relu',
                 return_seed=False,
                 learn_pos=False,
                 normalize_before=False,
                 embed_scale=True,
                 embed_layer_norm=False,
                 max_pos=1024,
                 share_layers=False,
                 position_emb_post_mask=False,
                 name=None):
        super().__init__()
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._attention_dropout = attention_dropout
        self._activation = activation
        self._return_seed = return_seed
        self._learn_pos = learn_pos
        self._normalize_before = normalize_before
        self._name = name
        self._embed_scale = d_model ** .5 if embed_scale else None
        self._embed_layer_norm = embed_layer_norm
        self._max_pos = max_pos
        self._share_layers = share_layers

        self._special_tokens = None
        self._embed, self._pos_embed, self._embed_norm, self._embed_dropout, self._norm = None, None, None, None, None
        self._layer, self._layers = None, None
        self._pool_seed = None
        self._position_emb_post_mask = position_emb_post_mask

    def build(self, embed, special_tokens):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._embed = embed
        self._special_tokens = special_tokens
        if self._learn_pos:
            self._pos_embed = LearnedPositionalEmbedding(num_embeddings=self._max_pos,
                                                         embedding_dim=self._d_model,
                                                         padding_idx=special_tokens['pad'],
                                                         post_mask=self._position_emb_post_mask)
        else:
            self._pos_embed = SinusoidalPositionalEmbedding(self._d_model)
        self._embed_norm = nn.LayerNorm(self._d_model) if self._embed_layer_norm else None
        self._embed_dropout = nn.Dropout(self._dropout)
        if self._share_layers:
            self._layer = TransformerEncoderLayer(d_model=self._d_model,
                                                  nhead=self._n_head,
                                                  dim_feedforward=self._dim_feedforward,
                                                  dropout=self._dropout,
                                                  attention_dropout=self._attention_dropout,
                                                  activation=self._activation,
                                                  normalize_before=self._normalize_before)
            self._layers = [self._layer for _ in range(self._num_layers)]
        else:
            self._layers = nn.ModuleList([TransformerEncoderLayer(d_model=self._d_model,
                                                                  nhead=self._n_head,
                                                                  dim_feedforward=self._dim_feedforward,
                                                                  dropout=self._dropout,
                                                                  attention_dropout=self._attention_dropout,
                                                                  activation=self._activation,
                                                                  normalize_before=self._normalize_before)
                                          for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None

    def _forward(self, batch: dict):
        # Check required fields are present
        if "desert_input_frag_idx" not in batch:
            raise ValueError("desert_input_frag_idx must be in batch")
        
        # Get input tensors from batch
        src = batch["desert_input_frag_idx"]  # This is the tensor we want to process
        src_key_padding_mask = batch["desert_input_frag_idx_mask"]
        
        # Now we can get the batch size and sequence length
        bz, sl = src.size(0), src.size(1)
        
        # Rest of the forward pass
        x = self._embed(src)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            x = x + self._pos_embed(src)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = src.eq(self._special_tokens['pad'])
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)

        if self._norm is not None:
            x = self._norm(x)

        if self._return_seed:
            encoder_out = x[1:], src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x, src_padding_mask

        return encoder_out

    @property
    def d_model(self):
        return self._d_model

    @property
    def out_dim(self):
        return self._d_model

class ShapePretrainingEncoder(TransformerEncoder):
    def __init__(self, patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        self._patch_ffn = FFN(self._patch_size**3, self._d_model, self._d_model)
    
    def build(self,
              embed, 
              special_tokens):
        super().build(embed, special_tokens)
    
    def _forward(self, batch: dict):
        # Check required fields are present
        if "input_frag_idx" not in batch:
            raise ValueError("desert_input_frag_idx must be in batch")
        
        # Get input tensors from batch
        src = batch["input_frag_idx"]  # This is the tensor we want to process
        src_key_padding_mask = batch["input_frag_idx_mask"]
        
        # Now we can get the batch size and sequence length
        bz, sl = src.size(0), src.size(1)
        
        # Rest of the forward pass
        x = self._patch_ffn(src)
        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
            x = x + self._pos_embed(pos)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = torch.zeros((bz, sl), dtype=torch.bool).to(x.device)
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        if self._norm is not None:
            x = self._norm(x)
        
        if self._return_seed:
            encoder_out = x[1:], src_padding_mask[:, 1:], x[0]
        else:
            encoder_out = x, src_padding_mask
        
        print("Passed encoder")

        return encoder_out

    @property
    def dim(self):
        """Alias for d_model to maintain compatibility"""
        return self._d_model

class ShapeEncoder(BaseEncoder):
    _pretrained_instance = None  # Class variable to store singleton instance

    @classmethod
    def from_pretrained(cls, model_path: str, device: torch.device = None):
        """Create or return cached pretrained encoder instance"""
        if cls._pretrained_instance is None:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Initialize encoder with correct parameters
            instance = ShapePretrainingEncoder(
                patch_size=4,
                num_layers=12,
                d_model=1024,
                n_head=8,
                dim_feedforward=4096,
                dropout=0.1,
                attention_dropout=0.1,
                activation='relu',
                learn_pos=True
            )
            
            # Build the encoder
            instance.build(
                embed=nn.Embedding(2, 1024),
                special_tokens={'pad': 0}
            )
            
            # Load state dict
            encoder_state_dict = {k[9:]: v for k, v in checkpoint['model'].items() 
                                if k.startswith('_encoder.')}
            instance.load_state_dict(encoder_state_dict)
            
            # Move to device and set eval mode
            instance.to(device)
            instance.eval()
            
            cls._pretrained_instance = instance
        
        return cls._pretrained_instance

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int):
        super().__init__()
        self._dim = d_model
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

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: dict):
        # Check required fields are present
        if "desert_input_frag_idx" not in batch:
            raise ValueError("desert_input_frag_idx must be in batch")
            
        # Get input tensors from batch
        src = batch["desert_input_frag_idx"]  
        src_key_padding_mask = batch["desert_input_frag_idx_mask"]
        
        # Process through transformer
        out = self.enc(src, src_key_padding_mask=src_key_padding_mask)
        return EncoderOutput(out, src_key_padding_mask)