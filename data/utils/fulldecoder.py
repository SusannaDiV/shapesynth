import torch
import torch.nn as nn
from math import ceil
from typing import Dict
from typing import Optional
from torch import Tensor
from torch.nn import Module
import torch.onnx.operators
import torch.nn.functional as F
from collections import OrderedDict

class Embedding(nn.Embedding):
    """
    Embedding is a wrapped class of torch.nn.Embedding with normal initialization on weight
    and zero initialization on pad.

    Args:
        vocab_size: vocabulary size
        d_model: feature dimensionality
        padding_idx: index of pad, which is a special token to ignore
    """

    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.weight, mean=0, std=d_model ** -0.5)
        if padding_idx:
            nn.init.constant_(self.weight[padding_idx], 0)

import torch
import torch.nn.functional as F
import torch.distributions as D
from .utils import get_dock_fast, get_dock_fast_with_smiles
import pickle
from contextlib import contextmanager
from typing import Dict, List, Tuple
import subprocess
import time
import numpy as np
import logging


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



def recursive(fn):
    """
    Make a function to work recursively, regardless dict, list and tuple

    Args:
        fn: processing function

    Returns:
        - a recursive version of given function
    """

    def rfn(x, *args, **kwargs):
        if isinstance(x, dict):
            return {key: rfn(val, *args, **kwargs) for key, val in x.items()}
        elif isinstance(x, list):
            return [rfn(val, *args, **kwargs) for val in x]
        elif isinstance(x, tuple):
            return tuple([rfn(val, *args, **kwargs) for val in x])
        else:
            return fn(x, *args, **kwargs)

    return rfn


def get_ordered_values_from_table_by_key(table, reverse=False):
    """
    Get value list where the value orders are determined by their keys.

    Args:
        table: a table of data
        reverse: value list in a reversed order

    Returns:
        - an ordered list of values
    """
    keys = [_ for _ in table]
    keys.sort(reverse=reverse)
    values = [table[k] for k in keys]
    return values


def auto_map_args(d: Dict, slots: OrderedDict):
    """
    Auto map a dict of data to a pre-defined slots

    Args:
        d: a dict of data
        slots: pre-defined slots

    Returns:
        - a tuple of data, where the order of data corresponds to the key orders in slots
    """
    kwargs = OrderedDict()
    for key, val in slots.items():
        kwargs[key] = val
    for key, val in d.items():
        kwargs[key] = val
    args = tuple([v for _, v in kwargs.items()])
    while len(args) > 0:
        if args[-1] is None:
            args = args[:-1]
        else:
            break
    return args


def inspect_fn(fn):
    """
    Inspect arguments of a function

    Args:
        fn: a function to inspect

    Returns:
        - an ordered dict with arguments and defaulted values
    """
    args = OrderedDict()
    signature = inspect.signature(fn)
    for key, val in signature.parameters.items():
        if key not in ['args', 'kwargs']:
            args[key] = val.default
    return args


def auto_map(kwargs, fn):
    """
    Auto map function input to function arguments

    Args:
        kwargs: function input
        fn: a function

    Returns:
        - a tuple of function inputs
    """
    return auto_map_args(kwargs, inspect_fn(fn))


class AbstractCriterion(Module):
    """
    """

    def __init__(self):
        super().__init__()
        self._model = None

    def build(self, *args, **kwargs):
        """
        Construct a criterion for model training.
        Typically, `model` should be provided.
        """
        self._build(*args, **kwargs)

        e = Environment()
        if e.device.startswith('cuda'):
            logger.info('move criterion to {}'.format(e.device))
            self.cuda(e.device)

    def _build(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """
        Compute the loss from neural model input, and produce a loss.
        """
        raise NotImplementedError

    def step_update(self, *args, **kwargs):
        """
        Perform step-level update
        """
        pass
class BaseCriterion(AbstractCriterion):
    """
    """

    def __init__(self):
        super().__init__()

    def forward(self, net_input, net_output):
        """
        Compute loss from a batch of samples

        Args:
            net_input: neural network input and is used for compute the logits
            net_output (dict): oracle target for a network input
        Returns:
            tuple:
                - **loss**: loss for network backward and optimization
                - **logging_states**: logging information
        """
        if isinstance(net_input, Dict):
            lprobs = self._model(**net_input)
        elif isinstance(net_input, List) or isinstance(net_input, Tuple):
            lprobs = self._model(*net_input)
        else:
            lprobs = self._model(net_input)
        # fetch target with default index 0
        loss, logging_states = self.compute_loss(lprobs, **net_output)
        return loss, logging_states

    def compute_loss(self, *args, **kwargs):
        """
        Compute loss from model results
        """
        raise NotImplementedError



import importlib
import os

class AbstractModel(nn.Module):
    """
    AbstractModel is abstract class for models defining inferfaces.

    Args:
        path: path to restore checkpoints
    """

    def __init__(self, path=None):
        super().__init__()
        self._path = path

        self._mode = 'train'
        self._states = {}

    def build(self, *args, **kwargs):
        """
        Build neural model with task instances.
        It wraps `_build` function with restoring and moving to cuda.
        """
        self._build(*args, **kwargs)
        logger.info('neural network architecture\n{}'.format([_ for _ in self.children()]))
        logger.info('parameter size: {}'.format(sum(p.numel() for p in self.parameters())))

        e = E()
        if self._path is not None:
            logger.info(f'load model from {self._path}')
            self.load(self._path, e.device)

        if e.device.startswith('cuda'):
            logger.info('move model to {}'.format(e.device))
            self.cuda(e.device)

    def _build(self, *args, **kwargs):
        """
        Build neural model with task instances.
        """
        raise NotImplementedError

    def forward(self, *input):
        """
        Compute output with neural input

        Args:
            *input: neural inputs
        """
        raise NotImplementedError

    def load(self, path, device, strict=False):
        """
        Load model from path and move model to device.

        Args:
            path: path to restore model
            device: running device
            strict: load model strictly
        """
        with UniIO(path, 'rb') as fin:
            state_dict = torch.load(fin, map_location=device)
            mismatched = self.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict, strict=strict)

        if not strict:
            logger.info("keys IN this model but NOT IN loaded model >>> ")
            if len(mismatched.missing_keys) > 0:
                for ele in mismatched.missing_keys:
                    logger.info(f"    - {ele}")
            else:
                logger.info("    - None")
            logger.info("keys NOT IN this model but IN loaded model >>> ")
            if len(mismatched.unexpected_keys) > 0:
                for ele in mismatched.unexpected_keys:
                    logger.info(f"    - {ele}")
            else:
                logger.info("    - None")

    def save(self, path):
        """
        Save model to path.

        Args:
            path: path to save model
        """
        save_ckpt({'model': self.state_dict()}, path)

    def update_states(self, *args, **kwargs):
        """
        Update internal networks states.
        """
        raise NotImplementedError

    @property
    def states(self):
        return self._states

    def reset(self, *args, **kwargs):
        """
        Reset neural model states.
        """
        pass

    def is_pretrained(self):
        return self._path is not None
from typing import Optional

from torch import Tensor

import json
import logging
logger = logging.getLogger(__name__)

MODULE_REGISTRY = {}

from typing import Optional

from torch import Tensor


from torch.nn import Module

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


class AbstractSearch(Module):
    """
    AbstractSearch is search algorithm on original neural model to perform special inference.
    """

    def __init__(self):
        super().__init__()
        self._mode = 'infer'

    def build(self, *args, **kwargs):
        """
        Build search algorithm with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of search algorithm.
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
class SequenceSearch(AbstractSearch):
    """
    SequenceSearch algorithms are used to generate a complete sequence with strategies.
    It usually built from a one-step neural model and fledges the model to a full-step generation.
    """

    def __init__(self):
        super().__init__()

        self._decoder = None
        self._bos, self._eos, self._pad = None, None, None

    def build(self, decoder, bos, eos, pad, *args, **kwargs):
        """
        Build the search algorithm with task instances.

        Args:
            decoder: decoder of neural model.
            bos: begin-of-sentence index
            eos: end-of-sentence index
            pad: pad index
        """
        self._decoder = decoder
        self._bos, self._eos, self._pad = bos, eos, pad

    def forward(self,
                prev_tokens: Tensor,
                memory: Tensor,
                memory_padding_mask: Tensor,
                target_mask: Optional[Tensor] = None,
                prev_scores: Optional[Tensor] = None):
        """
        Decoding full-step sequence

        Args:
            prev_tokens: previous tokens or prefix of sequence
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(B, V)` where B is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(B)` where B is batch size

        Returns:
            - log probability of generated sequence
            - generated sequence
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        self._decoder.reset(mode)

from torch.nn import MSELoss
import torch
import torch.nn.functional as F
import torch.nn as nn




modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module

from torch import Tensor
import torch.nn as nn

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
from typing import Optional

from torch import Tensor
from torch import nn

import torch.nn as nn
import math

import torch
import torch.onnx.operators
from torch import nn


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
from torch.nn import Module



modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
class AbstractDecoder(Module):
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
        Build decoder with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of decoder.
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache.clear()
        self._mode = mode

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

modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file

from tqdm import tqdm
import torch
from functools import wraps


def singleton(cls):
    """
    Singleton decorator

    Args:
        cls: singleton class

    Returns:
        - an instance of a singleton class
    """
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance




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

    def _forward(self, src: Tensor):
        r"""
        Args:
            src: tokens in src side.
              :math:`(N, S)` where N is the batch size, S is the source sequence length.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
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


def fuse_key_value(key, value, key_padding_mask, value_padding_mask, fusing):
    """
    Fuse key representation and value representation

    Args:
        key:
            :math:`(K, N, E)` where N is the batch size, K is the key number, E is the embedding size.
        value:
            :math:`(L, K, N, E)` where L is the value length, N is the batch size, K is the key number,
            E is the embedding size.
        key_padding_mask:
            :math:`(N, K)` where N is the batch size, K is the key number, E is the embedding size.`
        value_padding_mask:
            :math:`(N, K, L)` where N is the batch size, K is the key number, L is the value length size.`
        fusing: fusing type

    Returns:
        - output: fused representation for key-value pair
    """
    if fusing == 'max-pool-value':
        value, _ = value.max(dim=0)
        return key + value, key_padding_mask
    elif fusing == 'expand-key':
        key = key.unsqueeze(0)
        return key + value, value_padding_mask
    else:
        raise NotImplementedError


def create_init_scores(prev_tokens, tensor):
    """
    Create init scores in search algorithms

    Args:
        prev_tokens: previous token
        tensor: a type tensor

    Returns:
        - initial scores as zeros
    """
    batch_size = prev_tokens.size(0)
    prev_scores = torch.zeros(batch_size).type_as(tensor)
    return prev_scores


def create_upper_triangular_mask(tensor: Tensor):
    """
    Create upper triangular mask. It is usually used in auto-regressive model in training

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).type_as(tensor).bool()
    return mask.detach()


def create_max_segment_mask(tensor: Tensor, max_segment_length):
    """
    Create max-segment mask.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - max-segment mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = [[i <= j < i + max_segment_length for j in range(sz)] for i in range(sz)]
    mask = torch.BoolTensor(mask).type_as(tensor).bool()
    return mask


def create_time_mask(tensor: Tensor):
    """
    Create time mask. It is usually used in auto-regressive model in training.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).type_as(tensor).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.detach()


def sample_from_gaussian(mean, logvar):
    """
    Sample a vector from gaussian distribution

    Args:
        mean: mean of gaussian distribution
        logvar: log-variance of gaussian distribution

    Returns:
        - sampled vector from gassian distribution
    """
    std = torch.exp(0.5 * logvar)
    z = torch.randn_like(mean)
    z = z * std + mean
    return z


def mean_pooling(x, x_padding_mask):
    """
    Mean pooling on representation

    Args:
        x: feature matrix
            :math:`(T, N, E)', where T is sequence length, N is batch size and E is feature dimension.
        x_padding_mask:
            :math:`(N, T)`, where T is sequence length and N is batch size.

    Returns:
    """
    sql = torch.sum((~x_padding_mask).long(), -1).unsqueeze(-1) # [bsz, 1]
    return torch.sum(x * (~x_padding_mask).transpose(0, 1).unsqueeze(-1).float(), dim=0) / sql


def create_source_target_modality(d_model,
                                  src_vocab_size,
                                  tgt_vocab_size,
                                  src_padding_idx,
                                  tgt_padding_idx,
                                  share_embedding=None):
    """
    Create source and target modality (embedding)

    Args:
        d_model: model dimension
        src_vocab_size: vocabulary size at source side
        tgt_vocab_size: vocabulary size at target side
        src_padding_idx: padding_idx in source vocabulary
        tgt_padding_idx: padding_idx in target vocabulary
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.

    Returns:
        - source embedding
            :math:`(V_s, E)` where V_s is source vocabulary size and E is feature dimension.
        - target embedding
            :math:`(V_t, E)` where V_s is target vocabulary size and E is feature dimension.
        - target output projection
            :math:`(V_t, E)` where V_s is target vocabulary size and E is feature dimension.
    """


    src_embed = Embedding(vocab_size=src_vocab_size,
                          d_model=d_model,
                          padding_idx=src_padding_idx)
    if share_embedding == 'all':
        assert src_vocab_size == tgt_vocab_size, \
            'The sizes of source and target vocabulary must be equal when sharing all the embedding'
        assert src_padding_idx == tgt_padding_idx, \
            'The padding idx must be the same by sharing all the embedding'
        tgt_embed = src_embed
    else:
        tgt_embed = Embedding(vocab_size=tgt_vocab_size,
                              d_model=d_model,
                              padding_idx=tgt_padding_idx)
    if share_embedding in ['all', 'decoder-input-output']:
        tgt_out_proj = nn.Linear(tgt_embed.weight.shape[1],
                                 tgt_embed.weight.shape[0],
                                 bias=False)
        tgt_out_proj.weight = tgt_embed.weight
    else:
        tgt_out_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        nn.init.normal_(tgt_out_proj.weight, mean=0, std=d_model ** -0.5)

    return src_embed, tgt_embed, tgt_out_proj


def create_padding_mask_from_length(length, maxlen=None):
    """
    Transform a sequence length matrix to padding mask

    Args:
        length: sequence length matrix
            :math:`(N)` where N is batch size

    Returns:
        - padding mask indicating length
            :math:`(N, L)` where N is batch size and L is maximum length in `length`
    """
    bsz = length.size(0)
    if maxlen is None:
        maxlen = length.max()
    index = torch.arange(maxlen).long().unsqueeze(0).repeat(bsz, 1).to(length)
    padding_mask = index.ge(length.unsqueeze(1))
    return padding_mask


def uniform_assignment(src_padding_mask, tgt_padding_mask):
    """
    Compute uniform assignment matrix between source sequence and target sequence

    Args:
        src_padding_mask: padding mask at source side
            :math:`(N, S)` where N is batch size and S is source sequence length
        tgt_padding_mask: padding mask at target side
            :math:`(N, T)` where N is batch size and T is source sequence length

    Returns:
        - uniform assignment matrix:
            :math:`(N, T, S)` where N is batch size, T is source sequence length and S is source sequence length
    """
    src_length = (~src_padding_mask.bool()).sum(dim=-1)
    tgt_length = (~tgt_padding_mask.bool()).sum(dim=-1)
    max_trg_len = tgt_padding_mask.size(-1)
    steps = (src_length.float() - 1) / (tgt_length.float() - 1 + 1e-4)
    # max_trg_len
    index_t = new_arange(tgt_length, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long()
    index_t = index_t.masked_fill(tgt_padding_mask, 0)
    return index_t.detach()


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def create_sequence(padding_mask, idx, pad_id=None):
    """
    Create a sequence filled with an index

    Args:
        padding_mask: padding mask of target sequence
        idx: filled value
        pad_id: index of pad

    Returns:
        - a long tensor that is of the same shape as padding_mask and filled with idx
    """
    seq = padding_mask.long()
    seq = seq.masked_fill(~padding_mask, idx)
    if pad_id is not None:
        seq = seq.masked_fill(padding_mask, pad_id)
    return seq


def param_summary(model):
    """
    Compute the number of trainable/total parameters

    Args:
        model: a torch module

    Returns:
        - a tuple of number of (trainable, total) parameters
    """
    numel_train = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000
    numel_total = sum(p.numel() for p in model.parameters()) // 1000000
    return numel_train, numel_total


class AbstractEncoderDecoderModel(AbstractModel):
    """
    AbstractEncoderDecoderModel defines interface for encoder-decoder model.
    It must contains two attributes: encoder and decoder.
    """

    def __init__(self, path, *args, **kwargs):
        super().__init__(path)
        self._args = args
        self._kwargs = kwargs

        self._encoder, self._decoder = None, None

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
class EncoderDecoderModel(AbstractEncoderDecoderModel):
    """
    EncoderDecoderModel defines overall encoder-decoder architecture.

    Args:
        encoder: encoder configurations to build an encoder
        decoder: decoder configurations to build an decoder
        d_model: feature embedding
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None):
        super().__init__(path=path)
        self._encoder_config, self._decoder_config = encoder, decoder
        self._d_model = d_model
        self._share_embedding = share_embedding
        self._path = path

    def _build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        """
        Build encoder-decoder model

        Args:
            src_vocab_size: vocabulary size at source sitde
            tgt_vocab_size: vocabulary size at target sitde
            src_special_tokens: special tokens in source vocabulary
            tgt_special_tokens: special tokens in target vocabulary
        """
        src_embed, tgt_embed, tgt_out_proj = create_source_target_modality(
            d_model=self._d_model,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_padding_idx=src_special_tokens['pad'],
            tgt_padding_idx=tgt_special_tokens['pad'],
            share_embedding=self._share_embedding
        )
        self._encoder = self._encoder_config
        self._decoder = self._decoder_config
        self._encoder.build(embed=src_embed, special_tokens=src_special_tokens)
        self._decoder.build(embed=tgt_embed,
                            special_tokens=tgt_special_tokens,
                            out_proj=tgt_out_proj)

    def reset(self, mode, *args, **kwargs):
        """
        Switch mode and reset internal states

        Args:
            mode: running mode
        """
        self._mode = mode
        self._encoder.reset(mode, *args, **kwargs)
        self._decoder.reset(mode, *args, **kwargs)

    def set_cache(self, cache):
        """
        Set internal cache with outside one

        Args:
            cache: neural model cache states
        """
        if 'encoder' in cache:
            self._encoder.set_cache(cache['encoder'])
        elif 'decoder' in cache:
            self._decoder.set_cache(cache['decoder'])
        else:
            raise LookupError

    def get_cache(self):
        """
        Retrieve internal cache

        Returns:
            - internal cache
        """
        return {
            'encoder': self._encoder.get_cache(),
            'decoder': self._decoder.get_cache()
        }


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple





modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file

class CrossEntropy(BaseCriterion):
    """
    Cross Entropy Loss.

    """

    def __init__(self, weight=None, logging_metric='acc'):
        super().__init__()
        self._weight = torch.FloatTensor(weight) if weight is not None else weight
        self._logging_metric = logging_metric
        self._padding_idx = None
        self._nll_loss = None

    def _build(self, model, padding_idx=-1):
        """
        Build a cross entropy loss over model.

        Args:
            model: a neural model for compute cross entropy.
            padding_idx: labels of padding_idx are all ignored to computed nll_loss
        """
        self._model = model
        self._padding_idx = padding_idx
        self._nll_loss = nn.NLLLoss(weight=self._weight, ignore_index=padding_idx)

    def compute_loss(self, lprobs, target):
        """
        Compute loss from a batch of samples

        Args:
            lprobs: neural network output logits
            target: oracle target for a network input

        Returns:
            - loss for network backward and optimization
            - logging information
        """
        lprobs = F.log_softmax(lprobs, dim=-1)

        # compute nll loss
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        nll_loss = self._nll_loss(lprobs, target)

        # record logging
        logging_states = {
            'loss': nll_loss.data.item(),
        }
        if self._logging_metric == 'acc':
            correct = (lprobs.max(dim=-1)[1] == target).sum().data.item()
            tot = target.size(0)
            logging_states['acc'] = correct / tot
        elif self._logging_metric == 'ppl':
            logging_states['ppl'] = 2 ** (nll_loss.data.item())
        return nll_loss, logging_states

import logging
logger = logging.getLogger(__name__)
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.multiprocessing import Process
import torch
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)






def list2tensor(x):
    if isinstance(x, Dict):
        return {k: list2tensor(v) for k, v in x.items()}
    elif isinstance(x, List):
        _x = get_example_obj(x)
        return create_tensor(x, type(_x))
    else:
        return x


def convert_idx_to_tensor(idx, pad, ndim=None):
    """
    Convert a nd list of indices to a torch tensor

    Args:
        idx: a nd list of indices
        pad: padding index
        ndim: dimension for idx

    Returns:
        - indices in torch tensor
    """
    max_lengths = maxlen(idx, ndim=ndim)
    tensor_type = type(pad)
    ndim = len(max_lengths)
    idx = pad_idx(idx, max_lengths, pad, ndim=ndim)
    idx = create_tensor(idx, tensor_type)
    return idx


def maxlen(idx, ndim=None):
    """
    Compute maxlen tuple from index

    Args:
        idx: a nd list of indices
        ndim: ndim for idx

    Returns:
        - tensor shape (tuple) of index list
    """
    def _max_tuple(tuples: List[Tuple]):
        return tuple(max(sizes) for sizes in zip(*tuples))

    if ndim is None:
        if isinstance(idx, list):
            tuples = [maxlen(i) for i in idx]
            return (len(idx),) + _max_tuple(tuples)
        else:
            return tuple()
    else:
        if ndim > 1:
            tuples = [maxlen(i, ndim-1) for i in idx]
            return (len(idx),) + _max_tuple(tuples)
        else:
            return len(idx),


def pad_idx(idx, max_lengths, pad_id, ndim):
    """
    Complete index list to a certain shape with padding

    Args:
        idx: a nd list of indices
        max_lengths: n-size tuple defining shape
        pad_id: padding index
        ndim: dimension for idx

    Returns:
        - a nd list of indices with padding
    """
    if ndim > 1:
        l, suff = max_lengths[0], max_lengths[1:]
        content = [pad_idx(i, suff, pad_id, ndim-1) for i in idx]
        if len(idx) < l:
            pad = create_pad((l - len(idx),) + suff, pad_id)
            content += pad
        return content
    else:
        return idx + [pad_id for _ in range(max_lengths[0] - len(idx))]


def create_pad(size, pad_id):
    """
    Create a padding list of a given size

    Args:
        size: nd list shape
        pad_id: padding index

    Returns:
        - padding list of the given size
    """
    if len(size) == 1:
        return [pad_id for _ in range(size[0])]
    else:
        return [create_pad(size[1:], pad_id) for _ in range(size[0])]


def create_tensor(idx: List, tensor_type) -> Tensor:
    """
    Create torch tensor from index

    Args:
        idx: index list
        tensor_type: type of tensor

    Returns:
        - a torch tensor created from index
    """
    if tensor_type is int:
        T = torch.LongTensor(idx)
    elif tensor_type is float:
        T = torch.FloatTensor(idx)
    elif tensor_type is bool:
        T = torch.BoolTensor(idx)
    else:
        raise TypeError
    return T


def convert_tensor_to_idx(tensor: Tensor, bos: int = None, eos: int = None, pad: int = None):
    """
    Convert a tensor to index.

    Args:
        tensor: original tensor
        bos: begin-of-sequence index
        eos: end-of-sequence index
        pad: padding index

    Returns:
        - a nd list of indices
    """
    idx = tensor.tolist()
    if bos and eos and pad:
        idx = remove_special_tokens(idx, bos, eos, pad)
    return idx


def remove_special_tokens(idx, bos: int, eos: int, pad: int):
    """
    Remove special tokens from nd index list

    Args:
        idx: a nd index list
        bos: begin-of-sequence index
        eos: end-of-sequence index
        pad: padding index

    Returns:
        - index list without special tokens
    """
    if isinstance(idx, list) and isinstance(idx[0], int):
        if idx[0] == bos:
            idx = idx[1:]
        eos_pos = find_eos(idx, eos)
        if eos_pos is not None:
            idx = idx[:eos_pos]
        idx = [i for i in idx if i != pad]
        return idx
    else:
        return [remove_special_tokens(i, bos, eos, pad) for i in idx]


def find_eos(idx: list, eos: int):
    """
    Find eos position

    Args:
        idx: index list
        eos: end-of-sequence index

    Returns:
        - position of eos
    """
    for pos, i in enumerate(idx):
        if i == eos:
            return pos
    return None


def _to_device(tensor, device, fp16=False):
    """
    Move a tensor to device

    Args:
        tensor: original tensor
        device: device name
        fp16: whether to perform fp16

    Returns:
        - tensor on the given device
    """
    if isinstance(tensor, torch.Tensor):
        if device.startswith('cuda'):
            tensor = tensor.cuda()
            if isinstance(tensor, torch.FloatTensor) and fp16:
                tensor = tensor.half()
        elif device == 'cpu':
            tensor = tensor.cpu()
    return tensor


def half_samples(samples):
    """
    Half tensor of the given samples

    Args:
        samples: samples to half

    Returns:
        - halved samples
    """
    if isinstance(samples, List):
        halved = []
        is_dummy = False
        for s in samples:
            hs, dummy = half_samples(s)
            is_dummy = dummy or is_dummy
            halved.append(hs)
        return halved, is_dummy
    elif isinstance(samples, Dict):
        t = get_example_obj(samples)
        size = t.size(0)
        idx = np.random.choice(list(range(size)), size // 2, replace=False)
        if len(idx) > 0:
            index = recursive(index_tensor)
            return index(samples, idx), False
        else:
            dummy = recursive(dummy_tensor)
            return dummy(samples), True
    else:
        raise NotImplementedError


def index_tensor(tensor, idx):
    """
    select tensor with the row of given indices

    Args:
        tensor: original
        idx: index to keep

    Returns:
        - tensor with selected row
    """
    return tensor[idx]


def dummy_tensor(tensor):
    size = tensor.size()
    new_size = tuple([1 for _ in size[1:]])
    tot = 1
    for s in size:
        tot *= s
    tensor = tensor.view((tot, ) + new_size)
    tensor = tensor[:1]
    return tensor


def get_example_obj(x):
    """
    Get a example object from List, Tuple or Dict

    Args:
        x: given object

    Returns:
        - an example object
    """
    if isinstance(x, List) or isinstance(x, Tuple):
        return get_example_obj(x[0])
    elif isinstance(x, Dict):
        for v in x.values():
            return get_example_obj(v)
    else:
        return x


@contextmanager
def possible_autocast():
    """
    Possibly perform autocast
    """
    env = Environment()
    if env.fp16:
        with autocast():
            yield
    else:
        yield


@singleton
class GradScalerSingleton:
    """
    GradScaler for fp16 training
    """

    def __init__(self) -> None:
        self._grad_scaler = GradScaler()

    def scale_loss(self, loss):
        return self._grad_scaler.scale(loss)

    def step(self, optimizer):
        self._grad_scaler.step(optimizer)

    def update(self):
        self._grad_scaler.update()


def possible_scale_loss(loss):
    """
    Possibly scale loss in fp training
    """
    env = Environment()
    if env.fp16:
        grad_scaler = GradScalerSingleton()
        return grad_scaler.scale_loss(loss)
    else:
        return loss


def save_avg_ckpt(last_ckpts, save_path, timeout=10000, wait=False):

    def _save(ckpts, path, timeout=10000):
        for ckpt in ckpts:
            if not wait_until_exist(ckpt, timeout=timeout):
                logger.info(f'timeout: {ckpt} not found')
                return
        time.sleep(10)
        avg_state_dict = get_avg_ckpt(ckpts)
        save_ckpt(avg_state_dict, path, wait=True)

    if wait:
        _save(last_ckpts, save_path, timeout)
    else:
        Process(target=_save, args=(last_ckpts, save_path, timeout)).start()


def save_ckpt(state_dict, path, retry=5, wait=False):

    def _save(state_dict, path):
        for _ in range(retry):
            try:
                tmp_path = f"tmp.put.{path.split('/')[-1]}"
                with open(tmp_path, 'wb') as fout:
                    torch.save(state_dict, fout)
                if path.startswith('hdfs:'):
                    subprocess.run(["hadoop", "fs", "-put", "-f", tmp_path, path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    subprocess.run(['rm', tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["mv", tmp_path, path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                logger.info(f'successfully save state_dict to {path}')
                break
            except Exception as e:
                logger.warning(f'saving checkpoint {path} fails: {e}')

    state_dict = to_device(state_dict, 'cpu')
    if wait:
        _save(state_dict, path)
    else:
        Process(target=_save, args=(state_dict, path)).start()


def get_avg_ckpt(ckpt_paths, device='cpu'):
    state_dict_list = []
    for path in ckpt_paths:
        if path.startswith('hdfs:'):
            local_path = f'tmp.get.{path.split("/")[-1]}'
            subprocess.run(['hadoop', 'fs', '-get', path, local_path])
            with open(local_path, 'rb') as fin:
                state_dict_list.append(torch.load(fin, map_location=device)['model'])
            subprocess.run(['rm', local_path])
        else:
            with open(path, 'rb') as fin:
                state_dict_list.append(torch.load(fin, map_location=device)['model'])
    state_dict = average_checkpoints(state_dict_list)
    return {"model": state_dict}


def average_checkpoints(state_dict_list: List):
    state_dict = OrderedDict()
    for i, sd in enumerate(state_dict_list):
        for key in sd:
            p = sd[key]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if i == 0:
                state_dict[key] = p.numpy()
            else:
                state_dict[key] = state_dict[key] + p.numpy()
    ckpt_num = len(state_dict_list)
    for key in state_dict:
        state_dict[key] = state_dict[key] / ckpt_num
        state_dict[key] = torch.from_numpy(state_dict[key])
    return state_dict


to_device = recursive(_to_device)




models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
class AbstractSearch(Module):
    """
    AbstractSearch is search algorithm on original neural model to perform special inference.
    """

    def __init__(self):
        super().__init__()
        self._mode = 'infer'

    def build(self, *args, **kwargs):
        """
        Build search algorithm with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of search algorithm.
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode

class SequenceSearch(AbstractSearch):
    """
    SequenceSearch algorithms are used to generate a complete sequence with strategies.
    It usually built from a one-step neural model and fledges the model to a full-step generation.
    """

    def __init__(self):
        super().__init__()

        self._decoder = None
        self._bos, self._eos, self._pad = None, None, None

    def build(self, decoder, bos, eos, pad, *args, **kwargs):
        """
        Build the search algorithm with task instances.

        Args:
            decoder: decoder of neural model.
            bos: begin-of-sentence index
            eos: end-of-sentence index
            pad: pad index
        """
        self._decoder = decoder
        self._bos, self._eos, self._pad = bos, eos, pad

    def forward(self,
                prev_tokens: Tensor,
                memory: Tensor,
                memory_padding_mask: Tensor,
                target_mask: Optional[Tensor] = None,
                prev_scores: Optional[Tensor] = None):
        """
        Decoding full-step sequence

        Args:
            prev_tokens: previous tokens or prefix of sequence
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(B, V)` where B is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(B)` where B is batch size

        Returns:
            - log probability of generated sequence
            - generated sequence
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        self._decoder.reset(mode)




modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file

class ShapePretrainingModel(EncoderDecoderModel):
    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None,
                 no_shape=False,
                 no_trans=False,
                 no_rotat=False):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         d_model=d_model,
                         share_embedding=share_embedding,
                         path=path)
        self._no_shape = no_shape
        self._no_trans = no_trans
        self._no_rotat = no_rotat
    
    def forward(self, 
                shape,
                shape_patches, 
                input_frag_idx,
                input_frag_idx_mask,
                input_frag_trans,
                input_frag_trans_mask,
                input_frag_r_mat,
                input_frag_r_mat_mask):
        memory, memory_padding_mask = self._encoder(src=shape_patches)
        if self._no_shape:
            memory = torch.zeros_like(memory)
        if self._no_trans:
            input_frag_trans = torch.zeros_like(input_frag_trans)
        if self._no_rotat:
            input_frag_r_mat = torch.zeros_like(input_frag_r_mat)
        logits, trans, r_mat = self._decoder(input_frag_idx=input_frag_idx,
                                             input_frag_trans=input_frag_trans,
                                             input_frag_r_mat=input_frag_r_mat,
                                             memory=memory,
                                             memory_padding_mask=memory_padding_mask)
        return (logits, trans, r_mat)

class ShapePretrainingCriterionNoRegression(CrossEntropy):
    def __init__(self, weight=None, logging_metric='acc', trans=1.0, rotat=1.0):
        super().__init__(weight=weight, logging_metric=logging_metric)
        self._nll = nn.NLLLoss(ignore_index=0)
        self._trans = trans
        self._rotat = rotat
    
    def compute_loss(self,
                     lprobs, 
                     output_frag_idx,
                     output_frag_idx_mask,
                     output_frag_trans,
                     output_frag_trans_mask,
                     output_frag_r_mat,
                     output_frag_r_mat_mask):
        predict_frag_idx = lprobs[0]
        predict_frag_trans = lprobs[1]
        predict_frag_r_mat = lprobs[2]
        
        if isinstance(predict_frag_idx, list):
            tmp_nll_loss, tmp_acc = [], []
            for i in range(len(predict_frag_idx)):
                curr_nll_loss, curr_logging_states = super().compute_loss(predict_frag_idx[i], output_frag_idx)
                tmp_nll_loss.append(curr_nll_loss)
                tmp_acc.append(curr_logging_states['acc'])
            nll_loss = sum(tmp_nll_loss) / len(tmp_nll_loss)
            logging_states = {
                'loss': nll_loss.data.item(),
                'acc': tmp_acc[-1] # use the prediction at last layer as the final prediction
            }
        else:
            nll_loss, logging_states = super().compute_loss(predict_frag_idx, output_frag_idx)
        
        if isinstance(predict_frag_trans, list):
            tmp_trans_nll_loss, tmp_trans_lprobs = [], []
            trans_target = output_frag_trans.view(-1)
            for i in range(len(predict_frag_trans)):
                curr_trans_lprobs = F.log_softmax(predict_frag_trans[i], dim=-1)
                curr_trans_lprobs = curr_trans_lprobs.view(-1, curr_trans_lprobs.size(-1))
                
                curr_trans_nll_loss = self._nll(curr_trans_lprobs, trans_target)
                curr_trans_nll_loss = self._trans * curr_trans_nll_loss

                tmp_trans_lprobs.append(curr_trans_lprobs)
                tmp_trans_nll_loss.append(curr_trans_nll_loss)
            trans_nll_loss = sum(tmp_trans_nll_loss) / len(tmp_trans_nll_loss)
            trans_lprobs = tmp_trans_lprobs[-1] # use the prediction at last layer as the final prediction
        else:
            trans_lprobs = F.log_softmax(predict_frag_trans, dim=-1)
            trans_lprobs = trans_lprobs.view(-1, trans_lprobs.size(-1))
            trans_target = output_frag_trans.view(-1)
            trans_nll_loss = self._nll(trans_lprobs, trans_target)
            trans_nll_loss = self._trans * trans_nll_loss

        if isinstance(predict_frag_r_mat, list):
            tmp_rotat_nll_loss, tmp_rotat_lprobs = [], []
            rotat_target = output_frag_r_mat.view(-1)
            for i in range(len(predict_frag_r_mat)):
                curr_rotat_lprobs = F.log_softmax(predict_frag_r_mat[i], dim=-1)
                curr_rotat_lprobs = curr_rotat_lprobs.view(-1, curr_rotat_lprobs.size(-1))

                curr_rotat_nll_loss = self._nll(curr_rotat_lprobs, rotat_target)
                curr_rotat_nll_loss = self._rotat * curr_rotat_nll_loss

                tmp_rotat_nll_loss.append(curr_rotat_nll_loss)
                tmp_rotat_lprobs.append(curr_rotat_lprobs)
            rotat_nll_loss = sum(tmp_rotat_nll_loss) / len(tmp_rotat_nll_loss)
            rotat_lprobs = tmp_rotat_lprobs[-1] # use the prediction at last layer as the final prediction
        else:
            rotat_lprobs = F.log_softmax(predict_frag_r_mat, dim=-1)
            rotat_lprobs = rotat_lprobs.view(-1, rotat_lprobs.size(-1))
            rotat_target = output_frag_r_mat.view(-1)
            rotat_nll_loss = self._nll(rotat_lprobs, rotat_target)
            rotat_nll_loss = self._rotat * rotat_nll_loss

        total_loss = nll_loss + trans_nll_loss + rotat_nll_loss

        logging_states['nll_trans'] = trans_nll_loss.item()
        logging_states['nll_rotat'] = rotat_nll_loss.item()

        # ------------------- fix the wrong acc here ---------------------
        if self._logging_metric == 'acc':
            if isinstance(predict_frag_idx, list):
                lprobs = F.log_softmax(predict_frag_idx[-1], dim=-1)
            else:
                lprobs = F.log_softmax(predict_frag_idx, dim=-1)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = output_frag_idx.view(-1)
            correct = ((lprobs.max(dim=-1)[1] == target) * output_frag_idx_mask.view(-1)).sum().data.item()
            tot = output_frag_idx_mask.sum().data.item()
            logging_states['acc'] = correct / tot

            correct = ((trans_lprobs.max(dim=-1)[1] == trans_target) * output_frag_trans_mask.view(-1)).sum().data.item()
            tot = output_frag_trans_mask.sum().data.item()
            logging_states['acc_trans'] = correct / tot

            correct = ((rotat_lprobs.max(dim=-1)[1] == rotat_target) * output_frag_r_mat_mask.view(-1)).sum().data.item()
            tot = output_frag_r_mat_mask.sum().data.item()
            logging_states['acc_rotat'] = correct / tot
        
        return total_loss, logging_states






class ShapePretrainingIteratorNoRegression(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build(self,
              embed, 
              special_tokens,
              trans_size,
              rotat_size):
        super().build(embed, special_tokens)

        self._trans_emb = Embedding(vocab_size=trans_size,
                                    d_model=embed.weight.shape[1])
        self._rotat_emb = Embedding(vocab_size=rotat_size,
                                    d_model=embed.weight.shape[1])
        
        self._logits_output_proj = nn.Linear(embed.weight.shape[1],
                                             embed.weight.shape[0],
                                             bias=False)
        self._logits_output_proj.weight = embed.weight
        self._trans_output_proj = nn.Linear(self._trans_emb.weight.shape[1],
                                            self._trans_emb.weight.shape[0],
                                            bias=False)
        self._trans_output_proj.weight = self._trans_emb.weight
        self._rotat_output_proj = nn.Linear(self._rotat_emb.weight.shape[1],
                                            self._rotat_emb.weight.shape[0],
                                            bias=False)
        self._rotat_output_proj.weight = self._rotat_emb.weight
    
    def _forward(self, logits, trans, r_mat, padding_mask):
        bz, sl = logits.size(0), logits.size(1)
        logits_pred = logits.argmax(-1)
        trans_pred = trans.argmax(-1)
        r_mat_pred = r_mat.argmax(-1)
        
        x = self._embed(logits_pred)
        x = x + self._trans_emb(trans_pred)
        x = x + self._rotat_emb(r_mat_pred)

        if self._embed_scale is not None:
            x = x * self._embed_scale
        if self._pos_embed is not None:
            pos = torch.arange(sl).unsqueeze(0).repeat(bz, 1).to(x.device)
            x = x + self._pos_embed(pos)
        if self._embed_norm is not None:
            x = self._embed_norm(x)
        x = self._embed_dropout(x)

        src_padding_mask = padding_mask
        x = x.transpose(0, 1)
        for layer in self._layers:
            x = layer(x, src_key_padding_mask=src_padding_mask)
        
        if self._norm is not None:
            x = self._norm(x)
        
        x = x.transpose(0, 1)
        logits = self._logits_output_proj(x)
        trans = self._trans_output_proj(x)
        r_mat = self._rotat_output_proj(x)

        return logits, trans, r_mat




class ShapePretrainingSearchForwardSamplingDockDedupIterativeNoRegression(SequenceSearch):
    def __init__(self, 
                 maxlen_coef=(1.2, 10), 
                 topk=1, 
                 ltopk=1, 
                 ttopk=1, 
                 rtopk=1,
                 ltopp=0.95,
                 ttopp=0.0,
                 rtopp=0.0,
                 ltemp=1.2,
                 ttemp=1.0,
                 rtemp=1.0,
                 num_return_sequence=2,
                 fnum_return_sequence=2,
                 keepdim=False,
                 for_protein_decode=False,
                 sampling_type='topp_independent'):
        super().__init__()

        self._maxlen_a, self._maxlen_b = maxlen_coef
        
        # topk sampling
        self._topk = topk
        self._ltopk = ltopk
        self._ttopk = ttopk
        self._rtopk = rtopk

        # topp sampling
        self._ttopp = ttopp
        self._ltopp = ltopp
        self._rtopp = rtopp
        
        self._num_return_sequence = num_return_sequence
        self._keepdim = keepdim
        self._sampling_type = sampling_type

        self._ltemp = ltemp
        self._ttemp = ttemp
        self._rtemp = rtemp

        self._fnum_return_sequence = fnum_return_sequence
        self._for_protein_decode = for_protein_decode
        if for_protein_decode:
            with open('/opt/tiger/shape_based_pretraining/data/vocab/vocab.h_nei_nof', 'rb') as fr:
                self._vocab_h_nei_nof = pickle.load(fr)
    
    def forward(self,
                units,
                memory,
                memory_padding_mask,
                target_mask=None,
                prev_scores=None):
        tokens = units[0]
        trans = units[1]
        rotat = units[2]
        if self._for_protein_decode:
            nof = units[4]

        bz, sl = tokens.size(0), tokens.size(1)
        tokens = tokens.unsqueeze(1).repeat(1, self._num_return_sequence, 1).view(bz * self._num_return_sequence, sl)
        trans = trans.unsqueeze(1).repeat(1, self._num_return_sequence, 1).view(bz * self._num_return_sequence, sl)
        rotat = rotat.unsqueeze(1).repeat(1, self._num_return_sequence, 1).view(bz * self._num_return_sequence, sl)

        # copy memory for 'num_return_sequence' times
        memory, memory_padding_mask = self._expand(memory, memory_padding_mask)

        scores = create_init_scores(tokens, memory) if prev_scores is None else prev_scores
        for _ in range(int(memory.size(0) * self._maxlen_a + self._maxlen_b)):
            logits, tlogits, rlogits = self._decoder(tokens, trans, rotat, memory, memory_padding_mask)
            
            if isinstance(logits, list):
                logits = logits[-1]
                tlogits = tlogits[-1]
                rlogits = rlogits[-1]

            logits = logits[:, -1, :]
            tlogits = tlogits[:, -1, :]
            rlogits = rlogits[:, -1, :]
            if target_mask is not None:
                logits = logits.masked_fill(target_mask, float('-inf'))
            
            logits = logits / self._ltemp
            tlogits = tlogits / self._ttemp
            rlogits = rlogits / self._rtemp
            
            if self._sampling_type == 'topk_joint':
                next_score, next_token, next_trans, next_rotat = self._sample_from_topk_joint(logits, tlogits, rlogits)
            elif self._sampling_type == 'topk_independent':
                next_score, next_token, next_trans, next_rotat = self._sample_from_topk_independent(logits, tlogits, rlogits)
            elif self._sampling_type == 'topp_independent':
                next_score, next_token, next_trans, next_rotat = self._sample_from_topp_independent(logits, tlogits, rlogits)
            elif self._sampling_type == 'topp_independent_for_protein_decode':
                assert self._for_protein_decode
                next_score, next_token, next_trans, next_rotat = self._sample_from_topp_independent_for_protein_decode(logits, tlogits, rlogits, nof)
            else:
                raise NotImplementedError
            
            eos_mask = next_token.eq(self._eos)
            scores = scores + next_score.masked_fill_(eos_mask, 0.).view(-1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            trans = torch.cat([trans, next_trans], dim=-1)
            rotat = torch.cat([rotat, next_rotat], dim=-1)
        
        scores = scores.view(bz, self._num_return_sequence, -1)
        tokens = tokens.view(bz, self._num_return_sequence, -1)
        trans = trans.view(bz, self._num_return_sequence, -1)
        rotat = rotat.view(bz, self._num_return_sequence, -1)

        tokens, trans, rotat = self._get_top_dock_dedup(tokens, trans, rotat)

        if not self._keepdim and self._num_return_sequence == 1:
            tokens = tokens.squeeze(dim=1)
            trans = trans.squeeze(dim=1)
            rotat = rotat.squeeze(dim=1)

        return scores, (tokens, trans, rotat)
    
    def _get_top_dock_dedup(self, tokens, trans, rotat):
        curr_len = tokens.size(2)
        dock_results = get_dock_fast_with_smiles(tokens.cpu().tolist()[0],
                                                 trans.cpu().tolist()[0],
                                                 rotat.cpu().tolist()[0],
                                                 '--PDBQT PATH FOR DOCKING--',
                                                 '--LIGAND PATH FOR CALCULATE CENTER--')
        idxs = []
        smis = set()
        sorted_dock_results = sorted(dock_results, key=lambda x: x[0], reverse=True)
        for _, idx, smi in sorted_dock_results:
            if len(idxs) == self._fnum_return_sequence:
                break
            if smi in smis:
                continue
            if smi == '':
                continue
            idxs.append(idx)
            smis.add(smi)
        idxs = torch.tensor(idxs).to(tokens.device).unsqueeze(0)
        top_tokens = tokens.gather(1, index=idxs.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_trans = trans.gather(1, index=idxs.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_rotat = rotat.gather(1, index=idxs.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        return top_tokens, top_trans, top_rotat
    
    def _get_top_dock_fast(self, tokens, trans, rotat):
        mol_num = tokens.size(1)
        curr_len = tokens.size(2)
        mol_dock = [float('-inf') for _ in range(mol_num)]
        dock_results = get_dock_fast(tokens.cpu().tolist()[0],
                                     trans.cpu().tolist()[0],
                                     rotat.cpu().tolist()[0],
                                     '--PDBQT PATH FOR DOCKING--',
                                     '--LIGAND PATH FOR CALCULATE CENTER--')
        for ds, idx in dock_results:
            mol_dock[idx] = ds
        mol_dock = torch.tensor(mol_dock).to(tokens.device).unsqueeze(0)
        _, idx = mol_dock.topk(self._fnum_return_sequence, dim=1)
        top_tokens = tokens.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_trans = trans.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        top_rotat = rotat.gather(1, index=idx.unsqueeze(dim=-1).repeat(1, 1, curr_len))
        return top_tokens, top_trans, top_rotat
    
    def _sample_from_topp_independent_for_protein_decode(self, logits, trans, r_mat, nof):
        position = trans.argmax(dim=-1) - 2
        flat_nof = nof.view(-1)
        nof_num = flat_nof[position]
        vocab_h_nei_nof = torch.tensor(self._vocab_h_nei_nof, dtype=torch.float).to(nof_num.device)
        match_score = nof_num.unsqueeze(-1) * vocab_h_nei_nof.unsqueeze(0)
        match_max = match_score.max(dim=-1)[0]
        match_min = match_score.min(dim=-1)[0]
        alpha = (match_score - match_min.unsqueeze(-1)) / (match_max - match_min + 1e-9).unsqueeze(-1)
        logits = alpha * logits + logits
        return self._sample_from_topp_independent(logits, trans, r_mat)

    def _sample_from_topp_independent(self, logits, trans, r_mat):
        logits = F.softmax(logits, dim=-1)
        t_logits = F.softmax(trans, dim=-1)
        r_logits = F.softmax(r_mat, dim=-1)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_t_logits, sorted_t_indices = torch.sort(t_logits, descending=True)
        sorted_r_logits, sorted_r_indices = torch.sort(r_logits, descending=True)

        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        cumulative_t_probs = torch.cumsum(sorted_t_logits, dim=-1)
        cumulative_r_probs = torch.cumsum(sorted_r_logits, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self._ltopp
        sorted_t_indices_to_remove = cumulative_t_probs > self._ttopp
        sorted_r_indices_to_remove = cumulative_r_probs > self._rtopp

        # make sure at least have one point to sample
        sorted_indices_to_remove[..., 0] = 0
        sorted_t_indices_to_remove[..., 0] = 0
        sorted_r_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        t_indices_to_remove = sorted_t_indices_to_remove.scatter(1, sorted_t_indices, sorted_t_indices_to_remove)
        r_indices_to_remove = sorted_r_indices_to_remove.scatter(1, sorted_r_indices, sorted_r_indices_to_remove)
        
        logits[indices_to_remove] = 0.0
        t_logits[t_indices_to_remove] = 0.0
        r_logits[r_indices_to_remove] = 0.0
        
        token_prob = logits / logits.sum(dim=-1, keepdim=True)
        trans_prob = t_logits / t_logits.sum(dim=-1, keepdim=True)
        rotat_prob = r_logits / r_logits.sum(dim=-1, keepdim=True)

        token_dist = D.Categorical(token_prob)
        trans_dist = D.Categorical(trans_prob)
        rotat_dist = D.Categorical(rotat_prob)

        next_token = token_dist.sample((1, )).permute(1, 0)
        next_trans = trans_dist.sample((1, )).permute(1, 0)
        next_rotat = rotat_dist.sample((1, )).permute(1, 0)

        next_token_score = logits.gather(-1, next_token)
        next_trans_score = t_logits.gather(-1, next_trans)
        next_rotat_score = r_logits.gather(-1, next_rotat)

        next_score = next_token_score * next_trans_score * next_rotat_score

        return next_score, next_token, next_trans, next_rotat

    def _sample_from_topk_independent(self, logits, trans, r_mat):
        logits = F.softmax(logits, dim=-1)
        t_logits = F.softmax(trans, dim=-1)
        r_logits = F.softmax(r_mat, dim=-1)

        topk_token_scores, topk_token = logits.topk(self._ltopk, dim=-1)
        topk_trans_scores, topk_trans = t_logits.topk(self._ttopk, dim=-1)
        topk_rotat_scores, topk_rotat = r_logits.topk(self._rtopk, dim=-1)

        token_prob = topk_token_scores / topk_token_scores.sum(dim=-1, keepdim=True)
        trans_prob = topk_trans_scores / topk_trans_scores.sum(dim=-1, keepdim=True)
        rotat_prob = topk_rotat_scores / topk_rotat_scores.sum(dim=-1, keepdim=True)

        token_dist = D.Categorical(token_prob)
        trans_dist = D.Categorical(trans_prob)
        rotat_dist = D.Categorical(rotat_prob)

        next_token_index = token_dist.sample((1, )).permute(1, 0)
        next_trans_index = trans_dist.sample((1, )).permute(1, 0)
        next_rotat_index = rotat_dist.sample((1, )).permute(1, 0)

        next_token = topk_token.gather(-1, next_token_index)
        next_trans = topk_trans.gather(-1, next_trans_index)
        next_rotat = topk_rotat.gather(-1, next_rotat_index)

        next_token_score = topk_token_scores.gather(-1, next_token_index)
        next_trans_score = topk_trans_scores.gather(-1, next_trans_index)
        next_rotat_score = topk_rotat_scores.gather(-1, next_rotat_index)

        next_score = next_token_score * next_trans_score * next_rotat_score

        return next_score, next_token, next_trans, next_rotat

    
    def _sample_from_topk_joint(self, logits, trans, r_mat):
        batch_size = logits.size(0)
        
        logits = F.softmax(logits, dim=-1)
        t_logits = F.softmax(trans, dim=-1)
        r_logits = F.softmax(r_mat, dim=-1)

        topk_token_scores, topk_token = logits.topk(self._ltopk, dim=-1)
        topk_trans_scores, topk_trans = t_logits.topk(self._ttopk, dim=-1)
        topk_rotat_scores, topk_rotat = r_logits.topk(self._rtopk, dim=-1)

        token_trans_scores = topk_token_scores.view(batch_size, self._ltopk, 1) * \
                             topk_trans_scores.view(batch_size, 1, self._ttopk)
        token_trans_rotat_scores = token_trans_scores.view(batch_size, self._ltopk * self._ttopk, 1) * \
                                   topk_rotat_scores.view(batch_size, 1, self._rtopk)
        
        next_token_trans_rotat_scores, next_token_trans_rotat = token_trans_rotat_scores.view(batch_size, -1).topk(self._topk, dim=-1)
        
        # prob = F.softmax(next_token_trans_rotat_scores, dim=-1)
        prob = next_token_trans_rotat_scores / next_token_trans_rotat_scores.sum(dim=-1, keepdim=True)
        dist = D.Categorical(prob)
        next_token_trans_rotat = dist.sample((1, ))
        next_token_trans_rotat = next_token_trans_rotat.permute(1, 0)

        next_rotat_index = next_token_trans_rotat % self._rtopk
        next_trans_index = ((next_token_trans_rotat - next_rotat_index) % (self._ttopk * self._rtopk)) // self._rtopk
        next_token_index = (next_token_trans_rotat - next_rotat_index - next_trans_index * self._rtopk) // (self._ttopk * self._rtopk)

        next_token = topk_token.gather(-1, next_token_index)
        next_trans = topk_trans.gather(-1, next_trans_index)
        next_rotat = topk_rotat.gather(-1, next_rotat_index)

        next_score = next_token_trans_rotat_scores.gather(-1, next_token_trans_rotat)

        return next_score, next_token, next_trans, next_rotat
    
    def _expand(self, memory, memory_padding_mask):
        batch_size, memory_size = memory_padding_mask.size()
        memory = memory.unsqueeze(dim=2).repeat(1, 1, self._num_return_sequence, 1)
        memory = memory.view(memory_size, batch_size * self._num_return_sequence, -1)
        memory_padding_mask = memory_padding_mask.unsqueeze(dim=1).repeat(1, self._num_return_sequence, 1)
        memory_padding_mask = memory_padding_mask.view(batch_size * self._num_return_sequence, memory_size)
        return memory, memory_padding_mask

from torch.nn import Module



modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file

import os
import importlib
import json
import logging
logger = logging.getLogger(__name__)

from typing import Dict, List
import json
import random

import numpy as np



def split_tgt_sequence(tgt, bos, eos):
    """
    Split gold target into previous tokens and prediction target.
    For examples in text, `[hello, world, !] -> [<bos>, hello, world, !], [hello, world, !, <eos>]`

    Args:
        tgt: target sequence
        bos: begin-of-sequence index
        eos: end-of-sequence index

    Returns:
        - previous tokens
        - prediction target
    """
    if len(tgt[0]) > 0 and tgt[0][0] == bos and tgt[0][-1] == eos:
        prev_tokens = [v[:-1] for v in tgt]
        tgt = [v[1:] for v in tgt]
    else:
        prev_tokens = [[bos] + v for v in tgt]
        tgt = [v + [eos] for v in tgt]
    return tgt, prev_tokens


def reorganize(samples: List[Dict]):
    """
    Transforming List[Dict] to Dict[List] by grouping with keys

    Args:
        - samples: a list of samples
    """
    samples_ = {key: [] for key in samples[0]}
    for sample in samples:
        for key, val in sample.items():
            samples_[key].append(val)
    return samples_


def count_sample_token(sample):
    """
    Count sample tokens

    Args:
        sample: a piece of samples

    Returns:
        - total token numbers
    """
    if isinstance(sample, str):
        return len(SPACE_NORMALIZER.split(sample))
    elif isinstance(sample, list):
        return sum([count_sample_token(s) for s in sample])
    elif isinstance(sample, Dict):
        return sum([count_sample_token(s) for s in sample.values()])
    else:
        return 1


def transform_data(key, data):
    """
    Transform data

    Args:
        key:
        data:

    Returns:

    """
    if isinstance(data[0], Dict):
        return transform_table(data)
    else:
        return {key: data}


def transform_table(table):
    """
    Unsqueeze keys aligning with values

    Args:
        table: table defining key-value pairs

    Returns:
        - unsqueezed key-value dict
    """
    keys, values = [], []
    for sample in table:
        ks, vs = [], []
        for k, vals in sample.items():
            ks.extend([k for _ in vals])
            vs.extend(vals)
        keys.append(ks)
        values.append(vs)
    return {'key': keys, 'value': values}


def mask_seq(seq: List, p: float, mask='<mask>'):
    """
    Randomly mask tokens in sequence

    Args:
        seq: original sequence
        p: mask probability
        mask: mask token

    Returns:
        - sequence with token mask
    """
    seq = [mask if random.random() < p else s for s in seq]
    return seq


def delete_token(seq: List, p: float):
    """
    Randomly drop tokens

    Args:
        seq: original sequence
        p: drop rate

    Returns:
        - sequence with randomly deleted tokens
    """
    seq = [s for s in seq if random.random() > p]
    return seq


def infill_text(seq: List, lam, mask='<mask>'):
    """
    Mask a segment in the sequence

    Args:
        seq: original sequence
        lam: possion lambda
        mask: mask token

    Returns:
        - a masked sequence
    """
    l = np.random.poisson(lam)
    l = min(l, len(seq))
    start = random.randint(0, len(seq) - l)
    end = start + l
    seq = seq[:start] + [mask] + seq[end:]
    return seq


def permute(seq: List):
    """
    Permute a sequence

    Args:
        seq: sequence to be shuffle

    Returns:
        - shuffled sequence
    """
    random.shuffle(seq)
    return seq


def rotate(seq: List):
    """
    Rotate a sequence

    Args:
        seq: a sequence

    Returns:
        - rotated sequence
    """
    idx = random.randint(0, len(seq) - 1)
    seq = seq[idx:] + seq[:idx]
    return seq


def possible_load_json(sample):
    """
    Callback for json data

    Args:
        sample: data in raw format

    Returns:
        sample (dict): a dict of samples consisting of parallel data of different sources
    """
    try:
        sample = json.loads(sample)
    except:
        pass
    finally:
        return sample


def possible_eval(x):
    """
    Eval a value if possible
    """
    try:
        return eval(x)
    except:
        return x
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict
import inspect


def echo(x):
    return x


def merge_states(exist, current, weight=None):
    """
    Merge a new dict into a historical one

    Args:
        exist: long-time dict
        current: dict info at current time
        weight: weight on current dict

    Returns:
        - a merge long-time dict
    """
    if not current:
        return exist
    for name, val in current.items():
        if name not in exist:
            exist[name] = val
        else:
            if weight is not None:
                exist[name] = exist[name] * (1 - weight) + val * weight
            else:
                exist[name] += val
    return exist



@contextmanager
def local_seed(seed):
    """
    Set local running context with a given seed, and recover the seed once exited.

    Args:
        seed: seed in local context
    """
    import torch
    state = torch.random.get_rng_state()
    env = Environment()
    if env.device == 'cuda':
        state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        if env.device == 'cuda':
            torch.cuda.random.set_rng_state(state_cuda)


deepcopy_on_ref = recursive(lambda x: x)


def search_key(d, key):
    if key in d:
        return d[key]
    else:
        for k, v in d.items():
            if isinstance(v, Dict):
                return search_key(v, key)
    return None

from io import IOBase, TextIOBase
from multiprocessing import Process
import os
import sys
import re
import time
import json
import subprocess
import random
import logging
logger = logging.getLogger(__name__)

import importlib
import json
import logging
logger = logging.getLogger(__name__)

@singleton
class Environment:
    """
    Environment is a running environment class.

    Args:
        profiling_window: profiling window size
        configs: configs for running tasks
        debug: running with debug information
        no_warning: do not output warning informations
        seed: initial seed for random and torch
        device: running device
        fp16: running with fp16
        no_progress_bar: do not show progress bar
        pb_interval: show progress bar with an interval
    """

    def __init__(self,
                 configs=None,
                 profiling_window: int = 0,
                 debug: bool = False,
                 no_warning: bool = False,
                 seed: int = 0,
                 device: str = None,
                 fp16: bool = False,
                 no_progress_bar: bool = False,
                 pb_interval: int = 1,
                 custom_libs: str = None):
        self.profiling_window = profiling_window
        self.configs = configs
        self.debug = debug
        self.no_warning = no_warning
        self.seed = seed
        self.fp16 = fp16
        self.no_progress_bar = no_progress_bar
        self.pb_interval = pb_interval

        self.distributed_world = 1
        self.rank = 0
        self.local_rank = 0
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        if self.device == 'cuda':
            self._init_cuda()

        self._init_log()
        self._init_seed()
        self._import_custom_lib(custom_libs)

    def _init_log(self):
        FORMAT = f'%(asctime)s ｜ %(levelname)s | %(name)s |{f" RANK {self.rank} | " if not self.is_master() else " "}%(message)s'
        logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
        if not self.is_master():
            logging.disable(logging.INFO)

    def _import_custom_lib(self, path):
        """
        Import library manually

        Args:
            path: external libraries split with `,`
        """
        if path:
            path = path.strip('\n')
            for line in path.split(','):
                logger.info(f'import module from {line}')
                line = line.replace('/', '.')
                importlib.import_module(line)

    def _init_cuda(self):
        """
        Initialize cuda device

        used on each worker.
        """
        if torch.cuda.device_count() > 1:
            import horovod.torch as hvd
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())
            self.rank = hvd.rank()
            self.local_rank = hvd.local_rank()
            self.distributed_world = hvd.size()
        torch.cuda.empty_cache()

    def _init_seed(self):
        """
        Initialize global seed
        """
        import random
        random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def is_master(self):
        """
        check the current process is the master process
        """
        return self.rank == 0


def build_env(*args, **kwargs):
    """
    Build environment
    """
    env = Environment(*args, **kwargs)
    logger.info('Create environment with \n{}\n'.format(json.dumps({
        'device': env.device,
        'fp16': env.fp16,
        'profiling_window': env.profiling_window,
        'debug': env.debug,
        'distributed_world': env.distributed_world,
        'rank': env.rank,
        'local_rank': env.local_rank,
        'no_progress_bar': env.no_progress_bar,
        'no_warning': env.no_warning,
        "pb_interval": env.pb_interval
    }, indent=4, sort_keys=True)))


def format_states(states):
    """
    Format logging states to prettify logging information

    Args:
        states: logging states

    Returns:
        - formated logging states
    """
    formated_states = {}
    for key, val in states.items():
        if isinstance(val, float):
            if val < 1e-3:
                val = '{:.4e}'.format(val)
            else:
                val = '{:.4f}'.format(val)
        formated_states[key] = val
    return formated_states


def str_pipes(states):
    """
    Make state dict into a string

    Args:
        states: state dict

    Returns:
        - state dict in string
    """
    return " | ".join('{} {}'.format(key, states[key]).strip() for key in states.keys())


def progress_bar(iterable, streaming=False, **kwargs):
    """
    Create progress bar for iterable object

    Args:
        iterable: iterable object
        streaming: iterable object does not have __len__ property

    Returns:
        - progress bar
    """
    env = Environment()
    if env.is_master() and not env.no_progress_bar:
        total = 0 if streaming else len(iterable)
        pb = tqdm(iterable, total=total, leave=False, mininterval=env.pb_interval, **kwargs) if total > 0 else \
            tqdm(iterable, leave=False, mininterval=env.pb_interval, **kwargs)
    else:
        pb = iterable
    return pb

SPACE_NORMALIZER = re.compile(r"\s+")
TEMP_IO_SAVE_PATH = ""


def init_io():
    global TEMP_IO_SAVE_PATH
    try:
        TEMP_IO_SAVE_PATH = os.path.join(os.getenv('HOME'), '.cache/uio/')
    except Exception:
        TEMP_IO_SAVE_PATH = os.path.join(os.getcwd(), '.cache_uio/')
    if not os.path.exists(TEMP_IO_SAVE_PATH):
        os.makedirs(TEMP_IO_SAVE_PATH, exist_ok=True)


def clear_cache():
    global TEMP_IO_SAVE_PATH
    output = subprocess.run('lsof +d {}'.format(TEMP_IO_SAVE_PATH).split(), capture_output=True)
    occupied = str(output.stdout, encoding='utf8').split('\n')
    occupied = set([filepath for filepath in occupied if filepath])
    for name in os.listdir(TEMP_IO_SAVE_PATH):
        filename = os.path.join(TEMP_IO_SAVE_PATH, name)
        if filename not in occupied:
            try:
                os.remove(filename)
            except:
                pass


init_io()


def _run_cmd(args_list):
    """
    run linux commands
    """
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return =  proc.returncode
    return s_return, s_output, s_err

def parse_single_path(path):
    """
    Parse path with regular expression

    Args:
        path: input path

    Returns:
        - parse path list
    """
    def _get_files(path):
        return [f for f in listdir(path, return_files=True, return_dirs=False)]

    if path.endswith('*'):
        path = path.split('/')
        pathdir, pathprefix = '/'.join(path[:-1]), path[-1][:-1]
        files = ['{}/{}'.format(pathdir, f) for f in _get_files(pathdir) if f.startswith(pathprefix)]
    elif isdir(path):
        files = ['{}/{}'.format(path, f) for f in _get_files(path)]
    else:
        files = [path]
    random.shuffle(files)
    return files


def parse_path(path):
    files = []
    for singlepath in path.strip().split(','):
        if singlepath:
            files += parse_single_path(singlepath)
    return files


def read_vocab(path):
    """
    Read a vocab

    Args:
        path: path to restore vocab

    Returns:
        - a dict of frequency table
    """
    freq = []
    with UniIO(path, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            line = SPACE_NORMALIZER.split(line)
            freq.append((' '.join(line[:-1]), int(line[-1])))
    return freq


def read_table(path):
    """
    Read a table

    Args:
        path: path to restore table

    Returns:
        - a dict of table
    """
    table = {}
    with UniIO(path, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            line = SPACE_NORMALIZER.split(line)
            table[' '.join(line[:-1])] = line[-1]
    return table


def read_list(path):
    """
    Read a list

    Args:
        path: path to restore list

    Returns:
        - a list
    """
    with UniIO(path, 'r') as fin:
        freq = [line.strip('\n') for line in fin]
    return freq


def jsonable(x):
    """
    Check if x is suit json.dumps
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def listdir(path, return_files=True, return_dirs=False, retry=5):
    """
    Given a path, return a list of files under this path

    :param path: directory
    :return: a list of files / dirs
    """
    def _listdir(path):
        retval = list()
        returncode = 1
        for i in range(retry):
            if path.startswith('hdfs:'):
                output = subprocess.run('hadoop fs -ls {}'.format(path).split(), capture_output=True)
                returncode = output.returncode
                output = output.stdout
                output = str(output, encoding='utf8').split('\n')
                getname = lambda x: x.split('/')[-1]
                if return_files:
                    retval += [getname(f) for f in output if f.startswith('-')]
                if return_dirs:
                    retval += [getname(f) for f in output if f.startswith('d')]
            else:
                output = subprocess.run('ls -A -H -l {}'.format(path).split(), capture_output=True)
                returncode = output.returncode
                output = output.stdout
                output = str(output, encoding='utf8').split('\n')
                getname = lambda x: x.split(' ')[-1]
                if return_files:
                    retval += [getname(f) for f in output if f.startswith('-')]
                if return_dirs:
                    retval += [getname(f) for f in output if f.startswith('d')]
            if returncode == 0:
                break
        if returncode != 0:
            logger.warning(f'fail to listdir {path}')
        return retval

    if path:
        return _listdir(path)
    else:
        raise ValueError


def isdir(path):
    """
    Check if a path if a directory

    :param path: path to check
    :return:
    """
    if path.startswith('hdfs:'):
        output = subprocess.run('hadoop fs -test -d {}'.format(path).split(), capture_output=True)
        return output.returncode == 0
    else:
        return os.path.isdir(path)


def wait_until_exist(path, timeout=10000):
    start = time.time()
    while True:
        if exists(path):
            return True
        if time.time() - start > timeout:
            logger.warning(f"timeout: {path} not found!")
            return False
        time.sleep(5)


def cp(src, tgt, retry=5, wait=False):
    """
    Copy a file from src to tgt

    :param src: source file / directory
    :param tgt: target file / directory
    :return:
    """
    def _cp(src, tgt):
        if not wait_until_exist(src):
            logger.info(f'timeout: {src} not found')
            return
        returncode = 1
        for i in range(retry):
            if exists(tgt):
                remove(tgt, wait=True)
            if src.startswith('hdfs:') and tgt.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-cp", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif src.startswith('hdfs:') and not tgt.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-get", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif not src.startswith('hdfs:') and tgt.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-put", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.run(["cp", src, tgt], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            returncode = output.returncode
            if returncode == 0:
                logger.info(f'successfully copy from {src} to {tgt}')
                break
        if returncode != 0:
            logger.warning(f'copy from {src} to {tgt} fails')

    env = Environment()
    if env.is_master():
        if wait:
            _cp(src, tgt)
        else:
            Process(target=_cp, args=(src, tgt)).start()


def mkdir(path, retry=5, wait=True):
    """
    Create a directory at path

    :param path: path to directory
    :return:
    """
    def _mkdir(path):
        returncode = 1
        for i in range(retry):
            if path.startswith('hdfs:'):
                output = subprocess.run(["hadoop", "fs", "-mkdir", "-p", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.run(["mkdir", "-p", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            returncode = output.returncode
            if returncode == 0:
                logger.info(f'successfully make directory: {path}')
                break
        if returncode != 0:
            logger.warning(f'mkdir {path} fails')

    env = Environment()
    if env.is_master() and path:
        if wait:
            _mkdir(path)
        else:
            Process(target=_mkdir, args=(path,)).start()


def remove(path, retry=5, wait=False):
    """
    Remove a directory or file

    :param path: path to remove
    :return:
    """
    def _remove(path):
        if exists(path):
            returncode = 1
            for i in range(retry):
                if path.startswith('hdfs:'):
                    output = subprocess.run(['hadoop', 'fs', '-rm', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    output = subprocess.run(['rm', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                returncode = output.returncode
                if returncode == 0:
                    logger.info(f'successfully remove file: {path}')
                    break
            if returncode != 0:
                logger.warning(f'remove file {path} fails')

    env = Environment()
    if env.is_master() and path:
        if wait:
            _remove(path)
        else:
            Process(target=_remove, args=(path,)).start()


def exists(path):
    """
    check if path exists

    :param path: path to check
    :return:
    """
    if path.startswith('hdfs:'):
        r = subprocess.run(['hadoop', 'fs', '-stat', path], capture_output=True)
        return True if r.returncode == 0 else False
    else:
        return os.path.exists(path)


def not_exist(paths):
    for p in paths:
        if not exists(p):
            return p
    return None


def remove_tree(path, retry=5, wait=True):
    """
    remove directory recursively

    :param path: path to remove
    :return
    """
    def _rmtree(path):
        returncode = 1
        for i in range(retry):
            if path.startswith('hdfs:'):
                output = subprocess.run(['hadoop', 'fs', '-rm', '-r', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                output = subprocess.run(['rm', '-r', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            returncode = output.returncode
            if returncode == 0:
                logger.info(f'successfully remove directory: {path}')
                break
        if returncode != 0:
            logger.warning(f'remove directory {path} fails')

    env = Environment()
    if env.is_master() and path:
        if wait:
            _rmtree(path)
        else:
            Process(target=_rmtree, args=(path,)).start()


def create_data_map(path):
    """
    read a data map from path
    """
    data_map = []
    with UniIO(path) as fin:
        data_position = 0
        for i, line in enumerate(fin):
            d = json.loads(line)
            token_num = d['token_num'] if 'token_num' in d else 1
            data_map.append((i, data_position, token_num))
            data_position += len(line)
    return data_map


def utf8len(s):
    """
    Get the byte number of the utf-8 sentence.
    """
    return len(s.encode('utf-8'))


def _InputFileOpen(path, mode='r', encoding='utf8', timeout=-1, poll_interval=0.1, *args, **kwargs):
    try:
        if path.startswith('hdfs:'):
            if 'localpath' in kwargs:
                localpath = kwargs['localpath']
            else:
                localpath = TEMP_IO_SAVE_PATH + re.sub(r'[^\w]', '', path)
            lockfilename = localpath + '.lock'  # Multiprocess may read the file; they share the same cached file; 
                                                # They need to wait until it is downloaded completely
            if (not os.path.exists(localpath)) or ('checkpoints' in localpath.lower()): # do not update dataset between epoch, but update checkpoints
                if not os.path.exists(lockfilename):  # acquire lock
                    fd = os.open(lockfilename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)  # lock
                    if os.path.exists(localpath):
                        os.remove(localpath)
                    p = subprocess.run("hadoop fs -get {} {}".format(path, localpath).split(),
                                    capture_output=True)
                    if p.returncode:
                        logger.warning(f'failed to open {path}, hadoop fs return code: {p.returncode}')
                    os.close(fd)
                    os.remove(lockfilename)  # release lock
                else:
                    start = time.time()
                    while True:  # Wait until the file is released (finished downloading)
                        if not os.path.exists(lockfilename):
                            break
                        if timeout >= 0 and time.time() - start > timeout:
                            logger.warning(f'failed to open {path}, file is locked, timeout')
                            break
                        time.sleep(poll_interval)
        else:
            localpath = path
        if 'b' in mode.lower():
            istream = open(localpath, mode=mode)
        else:
            istream = open(localpath, mode=mode, encoding=encoding)
        # logger.info(f'successfully open file: {path}')
        return istream
    except Exception as e:
        logger.warning(f'open file {path} fails: {e}')
        return None


class _InputStream(TextIOBase):
    """
    A InputSteam wrapper to tackle with multiple files input
    """
    def __init__(self, path, encoding='utf8'):
        super().__init__()
        self._paths = parse_path(path)
        _hash = hash(''.join(self._paths + [str(os.getpid())]))
        _hash &= sys.maxsize
        self._localpath = os.path.join(TEMP_IO_SAVE_PATH, str(_hash))
        self._encoding = encoding
        self._idx = -1
        self._fin = None
        self._next_file()
    
    def _next_file(self):
        if self._fin is not None:
            self._fin.close()
        self._idx += 1
        if 0 <= self._idx < len(self._paths):
            self._fin = _InputFileOpen(self._paths[self._idx], mode='r', encoding=self._encoding, localpath=self._localpath)
            if self._fin is None:
                self._next_file()
        else:
            raise StopIteration

    def reset(self):
        self._idx = -1
        self._next_file()

    def close(self):
        if self._fin is not None:
            self._fin.close()
        super().close()
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self._idx >= len(self._paths):
               raise IndexError
            return next(self._fin)
        except StopIteration:
            try:
                self._next_file()
                return self.__next__()
            except Exception as e:
                raise e
        except IndexError:
            raise StopIteration

    def readline(self, size=-1):
        if self._fin is None or self._fin.closed:
            return ''
        sample = self._fin.readline(size)
        if sample:
            return sample
        try:
            self._next_file()
            return self.readline(size)
        except StopIteration:
            return ''

    def readlines(self, hint=-1):
        retval = []
        total_size = 0
        while hint is None or hint <= 0 or total_size <= hint:
            line = self.readline()
            if line:
                retval.append(line)
                total_size += len(line)
            else:
                break
        return retval

    def read(self, size=-1):
        if self._fin is None or self._fin.closed:
            return ''
        if size == -1:
            buffer = ''
            while True:
                buffer += self._fin.read()
                try:
                    self._next_file()
                except StopIteration:
                    break
            return buffer
        else:
            buffer = ['' for i in range(size)]
            offset = 0
            while size > 0:
                filesize = self._size(self._fin)
                if filesize <= size:
                    buffer[offset : offset + filesize] = self._fin.read()
                    offset += filesize
                    size -= filesize
                    try:
                        self._next_file()
                    except StopIteration:
                        break
                else:
                    buffer[offset : ] = self._fin.read(size)
                    size = 0
            buffer = ''.join(buffer)
            return buffer
    
    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            if offset < 0:
                raise OSError(22, 'Invalid argument')
            self.reset()
            _offset = offset
            while offset > 0:
                size = self._size(self._fin)
                if offset <= size:
                    self._fin.seek(offset, os.SEEK_SET)
                    offset = 0
                else:
                    offset -= size
                    try:
                        self._next_file()
                    except StopIteration:
                        break 
            return _offset
        elif whence == os.SEEK_CUR:
            if offset:
                raise ValueError(f'invalid offset {offset}, offset must be zero')
            else:
                pass  # do nothing, according to TextIOBase.seek()
            return self.tell()
        elif whence == os.SEEK_END:
            if offset:
                raise ValueError(f'invalid offset {offset}, offset must be zero')
            else:
                while True:
                    try:
                        self._next_file()
                    except StopIteration:
                        break
            return self.tell()
        else:
            raise ValueError(f'invalid whence ({whence}, should be 0, 1 or 2)')
            
    def tell(self):
        return self._fin.tell()  # Not a proper implementation

    def _size(self, fin):
        cur = fin.tell()
        tail = fin.seek(0, os.SEEK_END)
        size = max(0, tail - cur)
        fin.seek(cur, os.SEEK_SET)
        return size


def _OutputFileOpen(path, localpath, mode='w', encoding='utf8'):
    try: 
        if path.startswith('hdfs:'):
            if not os.path.exists(TEMP_IO_SAVE_PATH):
                os.mkdir(TEMP_IO_SAVE_PATH)
        else:
            localpath = path
        if 'b' in mode.lower():
            ostream = open(localpath, mode=mode)
        else:
            ostream = open(localpath, mode=mode, encoding=encoding)
        return ostream
    except Exception as e:
        logger.warning(f'open file {path} fails: {e}')


class _OutputStream(TextIOBase):
    """
    OutputStream is an io wrapper to tackle with multiple kinds of path

    Args:
        path: output file path
    """
    def __init__(self, path, encoding='utf8'):
        super().__init__()
        self._path = path
        if self._path.startswith('hdfs:'):
            self._localpath = TEMP_IO_SAVE_PATH + re.sub(r'[^\w]', '', '{}_{}_w'.format(path, os.getpid()))
        else:
            self._localpath = path
        self._encoding = encoding
        self._fout = _OutputFileOpen(path, self._localpath, encoding=encoding)

    def reset(self):
        """
        Reset output stream
        """
        self._fout.seek(0)
    
    def close(self):
        """
        Close output stream
        """
        self._fout.close()
        if self._path.startswith('hdfs:'):
            cp(self._localpath, self._path, wait=True)
            wait_until_exist(self._path)
        super().close()

    def write(self, content):
        """
        Write to output stream

        Args:
            content: str to write
        """
        self._fout.write(content)
    
    def writelines(self, content):
        """
        Write to output InputStream

        Args:
            content: list of str
        """
        self._fout.writelines(content)

    def seek(self, offset, whence=os.SEEK_SET):
        """
        The same as TextIOBase.seek()
        """
        return self._fout.seek(offset, whence)

    def tell(self):
        """
        The same as TextIOBase.tell()
        """
        return self._fout.tell()


class _InputBytes(IOBase):
    """
    InputBytes is an io wrapper to tackle with multiple kinds of path

    Args:
        path: input file path
    """
    def __init__(self, path, mode='rb'):
        super().__init__()
        self._paths = parse_path(path)
        self._paths = sorted(self._paths)
        # print(self._paths)
        self._fins = [_InputFileOpen(path, mode=mode) for path in self._paths]
        self._fins = [item for item in self._fins if item is not None]
        self._sizes = [self._size(fin) for fin in self._fins]
        self._idx = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Fetch next line from file.  The line terminator is b'\n'.

        Returns:
            - next line
        """
        try:
            if self._idx >= len(self._fins):
                raise IndexError
            sample = next(self._fins[self._idx])
            return sample
        except StopIteration:
            self._idx += 1
            sample = self.__next__()
            return sample
        except IndexError:
            raise StopIteration

    def reset(self):
        """
        Reset input stream
        """
        self._idx = 0
        for fin in self._fins:
            fin.seek(0)

    def readline(self, size=-1):
        """
        Read the next line.  Return b'' at EOF.  The line terminator is b'\n'.

        Args:
            size: read at most `size` bytes

        Returns:
            - next line  
        """
        try:
            if size == 0:
                return b''
            if self._idx >= len(self._fins):
                raise StopIteration
            sample = self._fins[self._idx].readline(size)
            if sample:
                return sample
            self._idx += 1
            return self.readline(size)
        except StopIteration:
            return b''
    
    def readlines(self, hint=-1):
        """
        Read all lines and return in a list
        
        Args:
            hint: read at most `hint` bytes

        Returns:
            - list of lines
        """
        retval = []
        total_size = 0
        while hint is None or hint <= 0 or total_size <= hint:
            line = self.readline()
            if line:
                retval.append(line)
                total_size += len(line)
            else:
                break
        return retval

    def read(self, size=-1):
        """
        Read the rest of file

        Args:
            size: read at most `size` bytes

        Returns:
            - the rest of file
        """
        if size == -1:
            buffer = b''
            while self._idx < len(self._fins):
                buffer += self._fins[self._idx].read()
                self._idx += 1
            return buffer
        else:
            buffer = bytearray(size)
            offset = 0
            while self._idx < len(self._fins) and size > 0:
                filesize = self._size(self._fins[self._idx])
                if filesize <= size:
                    buffer[offset : offset + filesize] = self._fins[self._idx].read()
                    offset += filesize
                    self._idx += 1
                    size -= filesize
                else:
                    buffer[offset : ] = self._fins[self._idx].read(size)
                    size = 0
            buffer = bytes(buffer)
            return buffer
                            
    def _size(self, fin):
        # Given a file descriptor, calculate its size
        cur = fin.tell()
        tail = fin.seek(0, os.SEEK_END)
        size = max(0, tail - cur)
        fin.seek(cur, os.SEEK_SET)
        return size

    def tell(self):
        """
        Return the absolute current stream position

        Returns:
            - current stream position
        """
        position = 0
        if self._idx < len(self._fins):
            position += self._fins[self._idx].tell()
        for i in range(min(self._idx, len(self._fins))):
            position += self._sizes[i]
        return position

    def seek(self, offset, whence=os.SEEK_SET):
        """
        Change the stream position to the given byte offset.

        Args:
            offset: byte offset
            whence: Values for whence are SEEK_SET (0), SEEK_CUR (1) or SEEK_END (2)
        
        Returns:
            Stream position after seek
        """
        if whence == os.SEEK_SET:
            if offset < 0:
                raise OSError(22, 'Invalid argument')
            return self.seek(offset - self.tell(), whence=os.SEEK_CUR)
        if whence == os.SEEK_CUR:
            self._idx = max(0, min(len(self._fins) - 1, self._idx))
            while self._idx < len(self._fins) and offset > 0:
                filesize = self._size(self._fins[self._idx])
                if filesize < offset:
                    self._fins[self._idx].seek(0, os.SEEK_END)
                    self._idx += 1
                    offset -= filesize
                else:
                    self._fins[self._idx].seek(offset, os.SEEK_CUR)
                    offset = 0
            while self._idx >= 0 and offset < 0:
                filesize = self._fins[self._idx].tell()
                if offset + filesize < 0:
                    self._fins[self._idx].seek(0, os.SEEK_SET)
                    self._idx -= 1
                    offset += filesize
                else:
                    self._fins[self._idx].seek(offset, os.SEEK_CUR)
                    offset = 0
            self._idx = max(0, min(len(self._fins) - 1, self._idx))
            return self.tell()
        if whence == os.SEEK_END:
            for i in range(len(self._fins)):
                offset += self._sizes[i]
            return self.seek(offset, whence=os.SEEK_SET)
        raise ValueError(f'invalid whence ({whence}, should be 0, 1 or 2)')

    def close(self):
        """
        Close the input stream
        """
        for fin in self._fins:
            fin.close()
        super().close()


class _OutputBytes(IOBase):
    """
    OutputBytes is an io wrapper to tackle with multiple kinds of path

    Args:
        path: output file path
    """
    def __init__(self, path, mode='wb'):
        super().__init__()
        self._path = path
        self._localpath = TEMP_IO_SAVE_PATH + re.sub(r'[^\w]', '', '{}_{}_w'.format(path, os.getpid()))
        self._fout = _OutputFileOpen(path, self._localpath, mode=mode)

    def reset(self):
        """
        Reset output stream
        """
        self._fout.seek(0)
    
    def close(self):
        """
        Close output stream
        """
        self._fout.close()
        if self._path.startswith('hdfs:'):
            cp(self._localpath, self._path, wait=True)
            wait_until_exist(self._path)
        super().close()

    def write(self, content):
        """
        Write to output Stream

        Args:
            content: bytes to write
        """
        self._fout.write(content)

    def seek(self, offset, whence=os.SEEK_SET):
        """
        The same as IOBase.seek()
        """
        return self._fout.seek(offset, whence)

    def tell(self):
        """
        The same as IOBase.tell()        
        """
        return self._fout.tell()


class UniIO(_InputStream, _OutputStream, _InputBytes, _OutputBytes):
    """
    A universal IO with the same functions as python:open
    """
    def __init__(self, path, mode='r', encoding='utf8'):
        pass

    def __new__(cls, path, mode='r', encoding='utf8'):
        if 'r' in mode.lower():
            if 'b' in mode.lower():
                return _InputBytes(path, mode=mode)
            return _InputStream(path, encoding=encoding)
        elif 'w' in mode.lower():
            if 'b' in mode.lower():
                return _OutputBytes(path, mode=mode)
            return _OutputStream(path, encoding=encoding)
        logger.warning(f'Not support file mode: {mode}')
        raise ValueError

MODULE_REGISTRY = {}


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



modules_dir = os.path.dirname(__file__)
for file in os.listdir(modules_dir):
    path = os.path.join(modules_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file


class AbstractDecoder(Module):
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
        Build decoder with task instance
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Process forward of decoder.
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache.clear()
        self._mode = mode

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

class AbstractDecoderLayer(nn.Module):
    """
    AbstractDecoderLayer is an abstract class for decoder layers.
    """

    def __init__(self):
        super().__init__()
        self._cache = dict()
        self._mode = 'train'
        self._dummy_param = nn.Parameter(torch.empty(0))

    def reset(self, mode: str):
        """
        Reset encoder layer and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._cache: Dict[str, torch.Tensor] = {"prev": self._dummy_param}
        self._mode = mode

    def _update_cache(self, *args, **kwargs):
        """
        Update cache with current states
        """
        pass

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return self._cache

    def set_cache(self, cache: Dict[str, torch.Tensor]):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        self._cache = cache

class TransformerDecoderLayer(AbstractDecoderLayer):
    """
    TransformerDecoderLayer performs one layer of time-masked transformer operation,
    namely self-attention and feed-forward network.

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
                 attention_dropout=0.,
                 activation="relu",
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward=dim_feedforward, activation=activation)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Pass the inputs (and mask) through the decoder layer in training mode.

        Args:
            tgt: the sequence to the decoder layer (required).
                :math:`(T, B, D)`, where T is sequence length, B is batch size and D is feature dimension
            memory: the sequence from the last layer of the encoder (required).
                :math:`(M, B, D)`, where M is memory size, B is batch size and D is feature dimension
            tgt_mask: the mask for the tgt sequence (optional).
                :math:`(T, T)`, where T is sequence length.
            memory_mask: the mask for the memory sequence (optional).
                :math:`(M, M)`, where M is memory size.
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                :math: `(B, T)`, where B is batch size and T is sequence length.
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
                :math: `(B, M)`, where B is batch size and M is memory size.
        """
        if self._mode == 'infer':
            tgt = tgt[-1:]
            tgt_mask, tgt_key_padding_mask = None, None
        residual = tgt
        if self.normalize_before:
            tgt = self.self_attn_norm(tgt)
        prevs = self._update_cache(tgt) if self._mode == 'infer' else tgt
        tgt = self.self_attn(tgt, prevs, prevs, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.dropout1(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.self_attn_norm(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.multihead_attn_norm(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.dropout2(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.multihead_attn_norm(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.ffn_norm(tgt)
        tgt = self.ffn(tgt)
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.ffn_norm(tgt)
        return tgt

    def _update_cache(self, cur):
        """
        Update cache with current states

        Args:
            cur: current state
        """
        prev = torch.cat([self._cache['prev'], cur], dim=0) if 'prev' in self._cache else cur
        self._cache['prev'] = prev
        return prev
    
class TransformerDecoder(AbstractDecoder):
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
        normalize_before: use pre-norm fashion, default as post-norm.
            Pre-norm suit deep nets while post-norm achieve better results when nets are shallow.
        name: module name
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.,
                 activation='relu',
                 learn_pos=False,
                 normalize_before=False,
                 output_bias=False,
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
        self._learn_pos = learn_pos
        self._normalize_before = normalize_before
        self._output_bias = output_bias
        self._name = name
        self._embed_scale = d_model ** .5
        self._max_pos = max_pos
        self._share_layers = share_layers

        self._special_tokens = None
        self._embed, self._pos_embed, self._embed_dropout = None, None, None
        self._layer, self._layers = None, None
        self._norm = None
        self._out_proj = None
        self._out_proj_bias = None
        self._position_emb_post_mask = position_emb_post_mask

    def build(self,
              embed,
              special_tokens,
              out_proj):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
            out_proj: output projection. It is allowed to be initialized with embedding weight in model buildup.
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
        self._embed_dropout = nn.Dropout(self._dropout)
        if self._share_layers:
            self._layer = TransformerDecoderLayer(d_model=self._d_model,
                                                  nhead=self._n_head,
                                                  dim_feedforward=self._dim_feedforward,
                                                  dropout=self._dropout,
                                                  attention_dropout=self._attention_dropout,
                                                  activation=self._activation,
                                                  normalize_before=self._normalize_before)
            self._layers = [self._layer for _ in range(self._num_layers)]
        else:
            self._layers = nn.ModuleList([TransformerDecoderLayer(d_model=self._d_model,
                                                                  nhead=self._n_head,
                                                                  dim_feedforward=self._dim_feedforward,
                                                                  dropout=self._dropout,
                                                                  attention_dropout=self._attention_dropout,
                                                                  activation=self._activation,
                                                                  normalize_before=self._normalize_before)
                                          for _ in range(self._num_layers)])
        self._norm = nn.LayerNorm(self._d_model) if self._normalize_before else None
        self._out_proj = out_proj
        if self._output_bias:
            self._out_proj_bias = nn.Parameter(torch.zeros(out_proj.weight.size(0)))

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            tgt: previous tokens in tgt side.
              :math:`(N, L)` where N is the batch size, L is the target sequence length.
              E is the embedding dimension.
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.


        Returns:
            - estimated logits.
              :math:`(N, L, V)` where N is the batch size, L is the target sequence length,
              V is the vocabulary size.
        """

        x = self._embed(tgt) * self._embed_scale

        if self._pos_embed is not None:
            x = x + self._pos_embed(tgt)
        x = self._embed_dropout(x)

        x = x.transpose(0, 1)

        tgt_mask = create_time_mask(tgt)
        tgt_padding_mask = tgt.eq(self._special_tokens['pad'])
        for layer in self._layers:
            x = layer(tgt=x,
                      memory=memory,
                      tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_padding_mask,
                      memory_key_padding_mask=memory_padding_mask,)
        if self._norm is not None:
            x = self._norm(x)
        x = x.transpose(0, 1)
        logits = self._out_proj(x)
        if self._out_proj_bias is not None:
            logits = logits + self._out_proj_bias
        return logits

    def reset(self, mode='train'):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        for layer in self._layers:
            layer.reset(mode)

    def get_cache(self):
        """
        Retrieve inner cache

        Returns:
            - cached states as a Dict
        """
        return {i: layer.get_cache() for i, layer in enumerate(self._layers)}

    def set_cache(self, cache):
        """
        Set cache from outside

        Args:
            cache: cache dict from outside
        """
        for i, layer in enumerate(self._layers):
            layer.set_cache(cache[i])
            
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


def fuse_key_value(key, value, key_padding_mask, value_padding_mask, fusing):
    """
    Fuse key representation and value representation

    Args:
        key:
            :math:`(K, N, E)` where N is the batch size, K is the key number, E is the embedding size.
        value:
            :math:`(L, K, N, E)` where L is the value length, N is the batch size, K is the key number,
            E is the embedding size.
        key_padding_mask:
            :math:`(N, K)` where N is the batch size, K is the key number, E is the embedding size.`
        value_padding_mask:
            :math:`(N, K, L)` where N is the batch size, K is the key number, L is the value length size.`
        fusing: fusing type

    Returns:
        - output: fused representation for key-value pair
    """
    if fusing == 'max-pool-value':
        value, _ = value.max(dim=0)
        return key + value, key_padding_mask
    elif fusing == 'expand-key':
        key = key.unsqueeze(0)
        return key + value, value_padding_mask
    else:
        raise NotImplementedError


def create_init_scores(prev_tokens, tensor):
    """
    Create init scores in search algorithms

    Args:
        prev_tokens: previous token
        tensor: a type tensor

    Returns:
        - initial scores as zeros
    """
    batch_size = prev_tokens.size(0)
    prev_scores = torch.zeros(batch_size).type_as(tensor)
    return prev_scores


def create_upper_triangular_mask(tensor: Tensor):
    """
    Create upper triangular mask. It is usually used in auto-regressive model in training

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).type_as(tensor).bool()
    return mask.detach()


def create_max_segment_mask(tensor: Tensor, max_segment_length):
    """
    Create max-segment mask.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - max-segment mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = [[i <= j < i + max_segment_length for j in range(sz)] for i in range(sz)]
    mask = torch.BoolTensor(mask).type_as(tensor).bool()
    return mask


def create_time_mask(tensor: Tensor):
    """
    Create time mask. It is usually used in auto-regressive model in training.

    Args:
        tensor:
            :math: (N, T, *) where T is target dimension

    Returns:
        - upper triangular mask:
            :math:`(N, T)` where T is target dimension
    """
    sz = tensor.size(1)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).type_as(tensor).float()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.detach()


def sample_from_gaussian(mean, logvar):
    """
    Sample a vector from gaussian distribution

    Args:
        mean: mean of gaussian distribution
        logvar: log-variance of gaussian distribution

    Returns:
        - sampled vector from gassian distribution
    """
    std = torch.exp(0.5 * logvar)
    z = torch.randn_like(mean)
    z = z * std + mean
    return z


def mean_pooling(x, x_padding_mask):
    """
    Mean pooling on representation

    Args:
        x: feature matrix
            :math:`(T, N, E)', where T is sequence length, N is batch size and E is feature dimension.
        x_padding_mask:
            :math:`(N, T)`, where T is sequence length and N is batch size.

    Returns:
    """
    sql = torch.sum((~x_padding_mask).long(), -1).unsqueeze(-1) # [bsz, 1]
    return torch.sum(x * (~x_padding_mask).transpose(0, 1).unsqueeze(-1).float(), dim=0) / sql


def create_source_target_modality(d_model,
                                  src_vocab_size,
                                  tgt_vocab_size,
                                  src_padding_idx,
                                  tgt_padding_idx,
                                  share_embedding=None):
    """
    Create source and target modality (embedding)

    Args:
        d_model: model dimension
        src_vocab_size: vocabulary size at source side
        tgt_vocab_size: vocabulary size at target side
        src_padding_idx: padding_idx in source vocabulary
        tgt_padding_idx: padding_idx in target vocabulary
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.

    Returns:
        - source embedding
            :math:`(V_s, E)` where V_s is source vocabulary size and E is feature dimension.
        - target embedding
            :math:`(V_t, E)` where V_s is target vocabulary size and E is feature dimension.
        - target output projection
            :math:`(V_t, E)` where V_s is target vocabulary size and E is feature dimension.
    """


    src_embed = Embedding(vocab_size=src_vocab_size,
                          d_model=d_model,
                          padding_idx=src_padding_idx)
    if share_embedding == 'all':
        assert src_vocab_size == tgt_vocab_size, \
            'The sizes of source and target vocabulary must be equal when sharing all the embedding'
        assert src_padding_idx == tgt_padding_idx, \
            'The padding idx must be the same by sharing all the embedding'
        tgt_embed = src_embed
    else:
        tgt_embed = Embedding(vocab_size=tgt_vocab_size,
                              d_model=d_model,
                              padding_idx=tgt_padding_idx)
    if share_embedding in ['all', 'decoder-input-output']:
        tgt_out_proj = nn.Linear(tgt_embed.weight.shape[1],
                                 tgt_embed.weight.shape[0],
                                 bias=False)
        tgt_out_proj.weight = tgt_embed.weight
    else:
        tgt_out_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        nn.init.normal_(tgt_out_proj.weight, mean=0, std=d_model ** -0.5)

    return src_embed, tgt_embed, tgt_out_proj


def create_padding_mask_from_length(length, maxlen=None):
    """
    Transform a sequence length matrix to padding mask

    Args:
        length: sequence length matrix
            :math:`(N)` where N is batch size

    Returns:
        - padding mask indicating length
            :math:`(N, L)` where N is batch size and L is maximum length in `length`
    """
    bsz = length.size(0)
    if maxlen is None:
        maxlen = length.max()
    index = torch.arange(maxlen).long().unsqueeze(0).repeat(bsz, 1).to(length)
    padding_mask = index.ge(length.unsqueeze(1))
    return padding_mask


def uniform_assignment(src_padding_mask, tgt_padding_mask):
    """
    Compute uniform assignment matrix between source sequence and target sequence

    Args:
        src_padding_mask: padding mask at source side
            :math:`(N, S)` where N is batch size and S is source sequence length
        tgt_padding_mask: padding mask at target side
            :math:`(N, T)` where N is batch size and T is source sequence length

    Returns:
        - uniform assignment matrix:
            :math:`(N, T, S)` where N is batch size, T is source sequence length and S is source sequence length
    """
    src_length = (~src_padding_mask.bool()).sum(dim=-1)
    tgt_length = (~tgt_padding_mask.bool()).sum(dim=-1)
    max_trg_len = tgt_padding_mask.size(-1)
    steps = (src_length.float() - 1) / (tgt_length.float() - 1 + 1e-4)
    # max_trg_len
    index_t = new_arange(tgt_length, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long()
    index_t = index_t.masked_fill(tgt_padding_mask, 0)
    return index_t.detach()


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def create_sequence(padding_mask, idx, pad_id=None):
    """
    Create a sequence filled with an index

    Args:
        padding_mask: padding mask of target sequence
        idx: filled value
        pad_id: index of pad

    Returns:
        - a long tensor that is of the same shape as padding_mask and filled with idx
    """
    seq = padding_mask.long()
    seq = seq.masked_fill(~padding_mask, idx)
    if pad_id is not None:
        seq = seq.masked_fill(padding_mask, pad_id)
    return seq


def param_summary(model):
    """
    Compute the number of trainable/total parameters

    Args:
        model: a torch module

    Returns:
        - a tuple of number of (trainable, total) parameters
    """
    numel_train = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000
    numel_total = sum(p.numel() for p in model.parameters()) // 1000000
    return numel_train, numel_total

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


class ShapePretrainingDecoderIterativeNoRegression(TransformerDecoder):
    def __init__(self, 
                 *args, 
                 iterative_block, 
                 iterative_num=1, 
                 max_dist=10.0, 
                 grid_resolution=1.0, 
                 rotation_bin_direction=11,
                 rotation_bin_angle=24,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._iterative_block_config = iterative_block
        self._iterative_num = iterative_num
        self._box_size = ceil(2 * max_dist // grid_resolution + 1)
        self._rotation_bin_direction = rotation_bin_direction
        self.rotation_bin_angle = rotation_bin_angle
    
    def build(self,
              embed,
              special_tokens,
              out_proj):
        super().build(embed, special_tokens, out_proj)
        
        self._trans_emb = Embedding(vocab_size=self._box_size ** 3 + 2,
                                    d_model=self._d_model)
        self._rotat_emb = Embedding(vocab_size=self._rotation_bin_direction * self._rotation_bin_direction * 3 * (self.rotation_bin_angle - 1) + 1 + 1,
                                    d_model=self._d_model)
        
        self._trans_output_proj = nn.Linear(self._trans_emb.weight.shape[1],
                                            self._trans_emb.weight.shape[0],
                                            bias=False)
        self._trans_output_proj.weight = self._trans_emb.weight

        self._rotat_output_proj = nn.Linear(self._rotat_emb.weight.shape[1],
                                            self._rotat_emb.weight.shape[0],
                                            bias=False)
        self._rotat_output_proj.weight = self._rotat_emb.weight

        iterative_block_emb =  Embedding(vocab_size=embed.weight.shape[0], d_model=embed.weight.shape[1], padding_idx=embed.padding_idx)
        self._iterative_block = self._iterative_block_config
        self._iterative_block.build(iterative_block_emb, special_tokens, self._trans_emb.weight.shape[0], self._rotat_emb.weight.shape[0])
    
    def forward(self, 
                input_frag_idx,
                input_frag_trans,
                input_frag_r_mat,
                memory,
                memory_padding_mask):
        tgt = input_frag_idx
        
        x = self._embed(tgt)
        
        input_frag_trans = self._trans_emb(input_frag_trans)

        input_frag_r_mat = self._rotat_emb(input_frag_r_mat)

        x = x + input_frag_trans

        x = x + input_frag_r_mat

        x = x * self._embed_scale

        if self._pos_embed is not None:
            x = x + self._pos_embed(tgt)
        x = self._embed_dropout(x)
        
        x = x.transpose(0, 1)

        tgt_mask = create_time_mask(tgt)
        tgt_padding_mask = tgt.eq(self._special_tokens['pad'])
        for layer in self._layers:
            x = layer(tgt=x,
                      memory=memory,
                      tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_padding_mask,
                      memory_key_padding_mask=memory_padding_mask,)
        if self._norm is not None:
            x = self._norm(x)
        x = x.transpose(0, 1)
        logits = self._out_proj(x)
        if self._out_proj_bias is not None:
            logits = logits + self._out_proj_bias
        
        trans = self._trans_output_proj(x)
        r_mat = self._rotat_output_proj(x)

        ret_logits = [logits]
        ret_trans = [trans]
        ret_r_mat = [r_mat]

        if self._mode != 'infer':
            for _ in range(self._iterative_num):
                logits, trans, r_mat = self._iterative_block(logits, trans, r_mat, tgt_padding_mask)
                ret_logits.append(logits)
                ret_trans.append(trans)
                ret_r_mat.append(r_mat)

        return ret_logits, ret_trans, ret_r_mat

    def reset(self, mode='train'):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        for layer in self._layers:
            layer.reset(mode)
        self._iterative_block.reset(mode)

    def iterative_infer(self, frag_idx, frag_trans, frag_r_mat):
        assert self._mode == 'infer'

        bz, sl = frag_idx.shape[0], frag_idx.shape[1]

        padding_mask = []
        for s in frag_idx:
            cnt = 0
            for idx in s:
                if idx != self._special_tokens['eos']:
                    cnt += 1
                else:
                    cnt += 1 # include an EOS token
                    break
            curr_mask = frag_idx.new_ones(sl)
            curr_mask[:cnt] = 0.0
            padding_mask.append(curr_mask.bool())
        padding_mask = torch.stack(padding_mask, dim=0)
        
        logits = frag_idx.new_zeros((bz * sl, self._embed.weight.shape[0]))
        tmp = frag_idx.new_ones((bz * sl, 1))
        frag_idx = frag_idx.contiguous().view(bz * sl, 1)
        logits = logits.scatter(-1, frag_idx, tmp)
        logits = logits.view(bz, sl, -1)

        trans = frag_trans.new_zeros((bz * sl, self._trans_emb.weight.shape[0]))
        tmp = frag_trans.new_ones((bz * sl, 1))
        frag_trans = frag_trans.contiguous().view(bz * sl, 1)
        trans = trans.scatter(-1, frag_trans, tmp)
        trans = trans.view(bz, sl, -1)

        r_mat = frag_r_mat.new_zeros((bz * sl, self._rotat_emb.weight.shape[0]))
        tmp = frag_r_mat.new_ones((bz * sl, 1))
        frag_r_mat = frag_r_mat.contiguous().view(bz * sl, 1)
        r_mat = r_mat.scatter(-1, frag_r_mat, tmp)
        r_mat = r_mat.view(bz, sl, -1)

        for _ in range(self._iterative_num):
            logits, trans, r_mat = self._iterative_block(logits, trans, r_mat, padding_mask)
        
        return logits, trans, r_mat