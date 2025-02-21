from .base import BaseEncoder, NoEncoder
from .graph import GraphEncoder
from .smiles import SMILESEncoder
from .shape import ShapePretrainingEncoder, ShapeEncoder

def get_encoder(encoder_type: str, cfg: dict) -> BaseEncoder:
    """Get encoder instance based on type and config."""
    if encoder_type == 'shape':
        return ShapeEncoder(**cfg)
    elif encoder_type == 'shape_pretrained':
        # Extract encoder config and pretrained path
        encoder_cfg = {k: v for k, v in cfg.items() if k != 'pretrained'}
        return ShapeEncoder.from_pretrained(cfg.pretrained, **encoder_cfg)
    elif encoder_type == "smiles":
        print("WARNING: WRONG ENCODER (SMILES) IS BEING RAN")
        return SMILESEncoder(**cfg)
    elif encoder_type == "graph":
        print("WARNING: WRONG ENCODER (GRAPH) IS BEING RAN")
        return GraphEncoder(**cfg)
    elif encoder_type == "none":
        print("SUCCESS: Using decoder-only mode with NoEncoder")
        return NoEncoder(**cfg)
    else:
        raise ValueError(f'Unknown encoder type: {encoder_type}')
