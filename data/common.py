import numpy as np
import torch
from .utils.shape_utils import get_atom_stamp, get_shape, get_shape_patches
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import enum
from collections.abc import Sequence
from typing import TypedDict
from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.chem.stack import Stack
from synformer.utils.image import draw_text, make_grid


def process_smiles(smiles: str) -> dict:
    """Generate a 3D conformer for a given SMILES."""
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
            
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
                raise ValueError(f"Charged sulfur atoms not supported by MMFF")
        
        mol = Chem.AddHs(mol)
        
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except ValueError as e:
            raise ValueError(f"Kekulization failed: {str(e)}")
        
        
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(f"Sanitization failed: {str(e)}")
        
            
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            raise ValueError(f"Failed to generate 3D conformer")

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception as e:
            raise ValueError(f"MMFF optimization failed")
        
        return {'mol': mol}
        
    except Exception as e:
        raise ValueError(f"Processing failed for {smiles}: {str(e)}")
    finally:
        pass


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
    mol: Chem.Mol,
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
) -> ProjectionData:
    """
    Create projection data including shape encoding for a molecule.
    """
    grid_resolution = 0.5
    max_dist_stamp = 4.0
    max_dist = 6.75
    patch_size = 4

    atom_f, bond_f = product.featurize_simple()
    
    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
    )

    curr_atom_stamp = get_atom_stamp(grid_resolution, max_dist_stamp)
    curr_shape = get_shape(mol, curr_atom_stamp, grid_resolution, max_dist)
    
    curr_shape_patches = get_shape_patches(curr_shape, patch_size)
    grid_size = curr_shape.shape[0] // patch_size
    curr_shape_patches = curr_shape_patches.reshape(grid_size, grid_size, grid_size, -1)
    curr_shape_patches = curr_shape_patches.reshape(-1, patch_size**3)
    
    shape_patches = torch.tensor(curr_shape_patches, dtype=torch.float)
    
    data: ProjectionData = {
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "atoms": atom_f,
        "bonds": bond_f,
        "smiles": product,#.tokenize_csmiles(),
        "atom_padding_mask": torch.zeros([atom_f.size(0)], dtype=torch.bool),
        "token_types": stack_feats["token_types"],
        "rxn_indices": stack_feats["rxn_indices"],
        "reactant_fps": stack_feats["reactant_fps"],
        "token_padding_mask": stack_feats["token_padding_mask"],
        "shape_patches": shape_patches
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