"""
DESERT (DEep Shape-based Encoder for Retrosynthesis Transformers) module.
This module provides functionality for generating fragment sequences from molecular shapes
and encoding them for use with Synformer.
"""

from .inference import run_desert_inference, load_desert_model, generate_shape_patches, visualize_fragments
from .encoder import create_fragment_encoder, FragmentEncoder
__all__ = [
    'run_desert_inference',
    'load_desert_model',
    'generate_shape_patches',
    'visualize_fragments',
    'create_fragment_encoder',
    'FragmentEncoder',
]

