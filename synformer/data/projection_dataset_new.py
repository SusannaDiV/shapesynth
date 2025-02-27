import os
import pickle
import random
from typing import cast
from contextlib import contextmanager
import signal
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import threading
import time
from rdkit import RDLogger
from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.stack import create_stack_step_by_step
from synformer.utils.train import worker_init_fn

from .collate import (
    apply_collate,
    collate_1d_features,
    collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
    collate_2d_features
)
from .common import ProjectionBatch, ProjectionData, create_data
RDLogger.DisableLog('rdApp.*') #UNCOMMENT (ACTUALLY JUST COMMENT)


@contextmanager
def timeout(seconds, message="Operation timed out"):
    def signal_handler(signum, frame):
        raise TimeoutError(message)
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)


class Collater:
    def __init__(self, max_num_atoms: int = 96, max_smiles_len: int = 192, max_num_tokens: int = 24):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_smiles_len = max_smiles_len
        self.max_num_tokens = max_num_tokens

        self.spec_atoms = {
            "atoms": collate_tokens,
            "bonds": collate_2d_tokens,
            "atom_padding_mask": collate_padding_masks,
        }
        self.spec_smiles = {"smiles": collate_tokens}
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }
        self.spec_desert = {
            "shape_patches": collate_2d_features
        }

    def __call__(self, data_list: list[ProjectionData]) -> ProjectionBatch:
        data_list_t = cast(list[dict[str, torch.Tensor]], data_list)
        batch = {
            **apply_collate(self.spec_atoms, data_list_t, max_size=self.max_num_atoms),
            **apply_collate(self.spec_smiles, data_list_t, max_size=self.max_smiles_len),
            **apply_collate(self.spec_tokens, data_list_t, max_size=self.max_num_tokens),
            **apply_collate(self.spec_desert, data_list_t, max_size=self.max_smiles_len),
            "mol_seq": [d["mol_seq"] for d in data_list],
            "rxn_seq": [d["rxn_seq"] for d in data_list],
        }
        return cast(ProjectionBatch, batch)


def timeout_handler(smiles: str, func, args=(), timeout=1.3): #used to be 3, but let's see if 1 second is enough
    """Thread-based timeout handler that works with multiprocessing"""
    result = [None]
    error = [None]
    
    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Timeout for {smiles}") #UNCOMMENT
        return None
    if error[0] is not None:
        #print(f"Error processing {smiles}: {str(error[0])}")
        return None
    return result[0]


class ProjectionDataset(IterableDataset[ProjectionData]):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_num_atoms: int = 80,
        max_smiles_len: int = 192,
        max_num_reactions: int = 5,
        init_stack_weighted_ratio: float = 0.0,
        virtual_length: int = 65536,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length

    def _process_mol(self, mol, smiles: str) -> dict | None:
        """Process molecule without timeout"""
        if mol is None:
            return None
            
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
                return None
        
        mol = Chem.AddHs(mol)
        
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            Chem.SanitizeMol(mol)
            
            if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
                return None

            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            return {'mol': mol}
            
        except Exception:
            return None

    def _process_smiles(self, smiles: str) -> dict | None:
        """Generate a 3D conformer for a given SMILES."""
        # Early rejection criteria
        if any([
            len(smiles) > 180,  # Very long SMILES strings
            smiles.count('(') > 8,  # Too many branches
            smiles.count('@') > 6,  # Too many chiral centers
            smiles.count('1') + smiles.count('2') + smiles.count('3') > 6,  # Too many rings
        ]):
            #print(f"Early rejection of complex molecule: {smiles}")  # Optional debug
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        return timeout_handler(smiles, self._process_mol, args=(mol, smiles))

    def __len__(self) -> int:
        return self._virtual_length

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        random.seed(42 + worker_id) 
        
        while True:
            for stack in create_stack_step_by_step(
                self._reaction_matrix,
                max_num_reactions=self._max_num_reactions,
                max_num_atoms=self._max_num_atoms,
                init_stack_weighted_ratio=self._init_stack_weighted_ratio,
            ):
                try:
                    mol_seq_full = stack.mols
                    mol_idx_seq_full = stack.get_mol_idx_seq()
                    rxn_seq_full = stack.rxns
                    rxn_idx_seq_full = stack.get_rxn_idx_seq()
                    product = random.choice(list(stack.get_top()))


                    processed = self._process_smiles(product._smiles)
                    if processed is not None:
                        data = create_data(
                            product=product,
                            mol_seq=mol_seq_full,
                            mol=processed['mol'],
                            mol_idx_seq=mol_idx_seq_full,
                            rxn_seq=rxn_seq_full,
                            rxn_idx_seq=rxn_idx_seq_full,
                            fpindex=self._fpindex,
                        )
                        data["smiles"] = data["smiles"][: self._max_smiles_len]
                        yield data
                        break  # Found a valid molecule, move to next stack
                except Exception:
                    continue


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("The trainer is missing.")

        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        self.train_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.config.train.val_freq * self.batch_size,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.batch_size,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )


class ProjectionDataModuleForSample(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        self.train_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.config.train.val_freq * self.batch_size,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            fpindex=fpindex,
            virtual_length=self.batch_size,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )