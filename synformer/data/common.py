import random
import numpy as np
import torch
from pytransform3d.rotations import quaternion_from_matrix
from math import ceil
from .utils.shape_utils import get_atom_stamp, get_shape, get_shape_patches, time_shift, get_grid_coords, get_rotation_bins
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pickle
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem

class ShapePretrainingTaskNoRegression:
    def __init__(self,
                 grid_resolution=0.5,
                 max_dist_stamp=4.0,
                 max_dist=6.75,
                 rotation_bin=24,
                 max_translation=1.0,
                 max_seq_len=20,
                 patch_size=4,
                 delta_input=False,
                 teacher_force_inference=False,
                 shape_noise_mu=0.0,
                 shape_noise_sigma=0.0,
                 rotation_bin_direction=11,
                 rotation_bin_angle=24,
                 vocab_path=None):
        
        self._grid_resolution = grid_resolution
        self._max_dist_stamp = max_dist_stamp
        self._max_dist = max_dist
        self._rotation_bin = rotation_bin
        self._max_translation = max_translation
        self._atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)
        self._max_seq_len = max_seq_len
        self._patch_size = patch_size
        self._delta_input = delta_input
        self._teacher_force_inference = teacher_force_inference
        self._box_size = ceil(2 * max_dist // grid_resolution + 1)
        self._rotation_bins = get_rotation_bins(rotation_bin_direction, rotation_bin_angle)
        self._shape_noise_mu = shape_noise_mu
        self._shape_noise_sigma = shape_noise_sigma

    def process_samples(self, samples, training=True):
        """Process a batch of samples."""
        self._training = training
        return self._collate(samples)

    def _collate(self, samples):
        shape = []
        shape_patches = []

        if not hasattr(self, '_infering') or (hasattr(self, '_infering') and self._teacher_force_inference):
            for sample in samples:
                curr_atom_stamp = get_atom_stamp(self._grid_resolution, 
                                               self._max_dist_stamp)
                curr_shape = get_shape(sample['mol'], 
                                     curr_atom_stamp, 
                                     self._grid_resolution, 
                                     self._max_dist)
                shape.append(curr_shape)

                curr_shape_patches = get_shape_patches(curr_shape, self._patch_size)
                curr_shape_patches = curr_shape_patches.reshape(curr_shape.shape[0] // self._patch_size,
                                                              curr_shape.shape[0] // self._patch_size,
                                                              curr_shape.shape[0] // self._patch_size, -1)
                curr_shape_patches = curr_shape_patches.reshape(-1, self._patch_size**3)
                shape_patches.append(curr_shape_patches)

            shape = torch.tensor(np.array(shape), dtype=torch.long)
            shape_patches = torch.tensor(np.array(shape_patches), dtype=torch.float)

            if not hasattr(self, '_infering'):
                batch = {
                    'net_input': {
                        'shape': shape,
                        'shape_patches': shape_patches
                    }
                }
            elif hasattr(self, '_infering') and self._teacher_force_inference:
                net_input = {
                    'encoder': (shape_patches,)
                }
                batch = {'net_input': net_input}
            else:
                raise Exception('please make sure [self._infering is False] or [(self._infering and self._teacher_force_inference) is True]!')

            return batch

PLACE_HOLDER_ATOM = 80     

class SingleSmilesProcessor:
    def __init__(self, vocab_path):
        with open(vocab_path, 'rb') as fr:
            self._vocab = pickle.load(fr)

    def generate_3d_structure(self, smiles: str) -> Chem.Mol:
        """Generate a 3D conformer for a given SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            raise ValueError(f"Failed to generate 3D conformer for {smiles}")

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception as e:
            raise ValueError(f"MMFF optimization failed ({e})")
        return mol

    def process_smiles(self, smiles):
        """Process a single SMILES string through the entire pipeline."""
        mol = self.generate_3d_structure(smiles)
        return {
            'mol': mol
        }

import enum
from collections.abc import Sequence
from typing import TypedDict
import torch
from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.chem.stack import Stack
from synformer.utils.image import draw_text, make_grid

vocab_path = 'data/vocab.pkl'

class TokenType(enum.IntEnum):
    END = 0
    START = 1
    REACTION = 2
    REACTANT = 3

class ProjectionData(TypedDict, total=False):
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    mol_seq: Sequence[Molecule]
    rxn_seq: Sequence[Reaction | None]
    shape: torch.Tensor
    shape_patches: torch.Tensor

class ProjectionBatch(TypedDict, total=False):
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    mol_seq: Sequence[Sequence[Molecule]]
    rxn_seq: Sequence[Sequence[Reaction | None]]
    shape: torch.Tensor
    shape_patches: torch.Tensor

def featurize_stack_actions(
    mol_idx_seq: Sequence[int | None],
    rxn_idx_seq: Sequence[int | None],
    end_token: bool,
    fpindex: FingerprintIndex,
) -> dict[str, torch.Tensor]:
    seq_len = len(mol_idx_seq) + 1  # Plus START token
    if end_token:
        seq_len += 1
    fp_dim = fpindex.fp_option.dim
    feats = {
        "token_types": torch.zeros([seq_len], dtype=torch.long),
        "rxn_indices": torch.zeros([seq_len], dtype=torch.long),
        "reactant_fps": torch.zeros([seq_len, fp_dim], dtype=torch.float),
        "token_padding_mask": torch.zeros([seq_len], dtype=torch.bool),
    }
    feats["token_types"][0] = TokenType.START
    for i, (mol_idx, rxn_idx) in enumerate(zip(mol_idx_seq, rxn_idx_seq), start=1):
        if rxn_idx is not None:
            feats["token_types"][i] = TokenType.REACTION
            feats["rxn_indices"][i] = rxn_idx
        elif mol_idx is not None:
            feats["token_types"][i] = TokenType.REACTANT
            _, mol_fp = fpindex[mol_idx]
            feats["reactant_fps"][i] = torch.from_numpy(mol_fp)
    return feats

def featurize_stack(stack: Stack, end_token: bool, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_idx_seq=stack.get_mol_idx_seq(),
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        fpindex=fpindex,
    )

def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
):
    atom_f, bond_f = product.featurize_simple()
    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
    )

    processor = SingleSmilesProcessor(vocab_path)
    processed_data = processor.process_smiles(product._smiles)
    
    encoder = ShapePretrainingTaskNoRegression(vocab_path=vocab_path)
    desert_batch = encoder.process_samples([processed_data], training=True)
    
    desert_data = {
        k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
        for k, v in desert_batch["net_input"].items()
    }
    
    data: "ProjectionData" = {
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "atoms": atom_f,
        "bonds": bond_f,
        "smiles": product.tokenize_csmiles(),
        "atom_padding_mask": torch.zeros([atom_f.size(0)], dtype=torch.bool),
        "token_types": stack_feats["token_types"],
        "rxn_indices": stack_feats["rxn_indices"],
        "reactant_fps": stack_feats["reactant_fps"],
        "token_padding_mask": stack_feats["token_padding_mask"],
        "shape": desert_data["shape"],
        "shape_patches": desert_data["shape_patches"]
    }

    return data

def draw_data(data: ProjectionData):
    im_list = [draw_text("START")]
    for m, r in zip(data["mol_seq"], data["rxn_seq"]):
        if r is not None:
            im_list.append(r.draw())
        else:
            im_list.append(m.draw())
    im_list.append(draw_text("END"))
    return make_grid(im_list)

def draw_batch(batch: ProjectionBatch):
    bsz = len(batch["mol_seq"])
    for b in range(bsz):
        im_list = [draw_text("START")]
        for m, r in zip(batch["mol_seq"][b], batch["rxn_seq"][b]):
            if r is not None:
                im_list.append(r.draw())
            else:
                im_list.append(m.draw())
        im_list.append(draw_text("END"))
        yield make_grid(im_list)