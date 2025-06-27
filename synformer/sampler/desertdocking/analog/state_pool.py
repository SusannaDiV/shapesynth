import copy
import dataclasses
import itertools
import time
import os
import shutil
import subprocess
import threading
from collections.abc import Iterable
from functools import cached_property
from multiprocessing.synchronize import Lock
from dataclasses import dataclass

import pandas as pd
import torch
from tqdm.auto import tqdm
from rdkit import Chem

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import FingerprintOption, Molecule
from synformer.chem.stack import Stack
from synformer.data.collate import (
    apply_collate,
    collate_1d_features,
    collate_padding_masks,
    collate_tokens,
)
from synformer.data.common import TokenType, featurize_stack
from synformer.models.synformer import Synformer

@dataclass
class QVinaOption:
    """Options for QVina2 docking"""
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8
    num_modes: int = 1

def timeout_handler(smiles, func, args=(), timeout=3):
    """Thread-based timeout handler for molecule processing"""
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
        print(f"Timeout for {smiles}")
        return None
    if error[0] is not None:
        print(f"Error processing {smiles}: {str(error[0])}")
        return None
    return result[0]

def prepare_ligand_pdbqt(mol, obabel_path="obabel"):
    """
    Prepare a ligand for docking by converting it to PDBQT format using OpenBabel
    """
    try:
        import tempfile
        import uuid
        
        # Create unique filenames with absolute paths in the system temp directory
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        temp_mol_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.mol")
        temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.pdbqt")
        
        # Write the molecule to a temporary file
        Chem.MolToMolFile(mol, temp_mol_file)
        
        # Convert to PDBQT using OpenBabel
        cmd = [
            obabel_path,
            "-imol", temp_mol_file,
            "-opdbqt", "-O", temp_pdbqt_file,
            "--partialcharge", "gasteiger",
            "--gen3d", "best"
        ]
        
        # Run OpenBabel
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Check if the conversion was successful
        if process.returncode != 0:
            print(f"Error converting molecule to PDBQT: {process.stderr}")
            return None
        
        # Read the PDBQT file
        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()
            
        # Clean up temporary files
        try:
            os.remove(temp_mol_file)
            os.remove(temp_pdbqt_file)
        except:
            pass
        
        # Check if the PDBQT file is valid (contains ATOM or HETATM lines)
        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            print("Error: Generated PDBQT file does not contain valid atom entries")
            return None
            
        return pdbqt_content
    except Exception as e:
        print(f"Error preparing ligand: {str(e)}")
        # Clean up temporary files
        try:
            if os.path.exists(temp_mol_file):
                os.remove(temp_mol_file)
            if os.path.exists(temp_pdbqt_file):
                os.remove(temp_pdbqt_file)
        except:
            pass
        return None

def dock_best_molecule(mol, receptor_path="/home/luost_local/sdivita/synformer/experiments/sbdd/receptor.pdbqt"):
    """Dock the molecule against ADRB2 target"""
    try:
        import tempfile
        import uuid
        
        # Get SMILES for logging
        smiles = Chem.MolToSmiles(mol)
        print(f"Docking molecule: {smiles}")
        
        # Create unique ID for temporary files
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        
        # ADRB2 docking parameters
        center = [-9.845024108886719, -4.321293354034424, 39.35286331176758]
        box_size = [11.208, 9.997, 14.994]
        
        # Find qvina path - check if it's in the current directory or bin subdirectory
        qvina_path = "bin/qvina2.1"
        if not os.path.exists(qvina_path):
            # Try to find it in the PATH
            qvina_path = shutil.which("qvina2.1")
            if qvina_path is None:
                print("Error: QVina2 executable not found")
                return None
        
        obabel_path = shutil.which("obabel")
        
        if obabel_path is None:
            print("Error: OpenBabel (obabel) not found in PATH")
            return None
            
        # Check if receptor file exists
        if not os.path.exists(receptor_path):
            print(f"Error: Receptor file not found at {receptor_path}")
            return None
            
        # Prepare ligand
        ligand_pdbqt = prepare_ligand_pdbqt(mol, obabel_path)
        if ligand_pdbqt is None:
            print("Failed to prepare ligand for docking")
            return None
            
        # Write ligand to temporary file
        temp_ligand_file = os.path.join(temp_dir, f"temp_ligand_dock_{unique_id}.pdbqt")
        with open(temp_ligand_file, "w") as f:
            f.write(ligand_pdbqt)
            
        # Set up QVina options
        options = QVinaOption(
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            size_x=box_size[0],
            size_y=box_size[1],
            size_z=box_size[2]
        )
        
        # Create QVina command
        output_file = os.path.join(temp_dir, f"temp_ligand_dock_out_{unique_id}.pdbqt")
        cmd = [
            qvina_path,
            "--receptor", receptor_path,
            "--ligand", temp_ligand_file,
            "--center_x", str(options.center_x),
            "--center_y", str(options.center_y),
            "--center_z", str(options.center_z),
            "--size_x", str(options.size_x),
            "--size_y", str(options.size_y),
            "--size_z", str(options.size_z),
            "--exhaustiveness", str(options.exhaustiveness),
            "--num_modes", str(options.num_modes),
            "--out", output_file
        ]
        
        # Run QVina with timeout
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False
            )
        except subprocess.TimeoutExpired:
            print("Docking timed out after 5 minutes")
            return None
            
        # Parse output to get docking score
        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        try:
                            score = float(line.split()[3])
                            print(f"Docking score for {smiles}: {score}")
                            break
                        except (IndexError, ValueError):
                            pass
                            
        # Clean up temporary files
        try:
            if os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
            
        return score
        
    except Exception as e:
        print(f"Error during docking: {str(e)}")
        # Clean up temporary files
        try:
            if 'temp_ligand_file' in locals() and os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if 'output_file' in locals() and os.path.exists(output_file):
                os.remove(output_file)
        except:
            pass
        return None

@dataclasses.dataclass
class State:
    stack: Stack = dataclasses.field(default_factory=Stack)
    scores: list[float] = dataclasses.field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(self.scores)

    def featurize(self, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
        feats = featurize_stack(self.stack, end_token=False, fpindex=fpindex)
        return feats


@dataclasses.dataclass
class _ProductInfo:
    molecule: Molecule
    stack: Stack


class TimeLimit:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._start = time.time()

    def exceeded(self) -> bool:
        if self._seconds <= 0:
            return False
        return time.time() - self._start > self._seconds

    def check(self):
        if self.exceeded():
            raise TimeoutError()


class StatePool:
    def __init__(
        self,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        mol: Molecule,
        model: Synformer,
        factor: int = 16,
        max_active_states: int = 256,
        sort_by_score: bool = True,
    ) -> None:
        super().__init__()
        self._fpindex = fpindex
        self._rxn_matrix = rxn_matrix

        self._model = model
        self._mol = mol
        device = next(iter(model.parameters())).device
        atoms, bonds = mol.featurize_simple()
        self._atoms = atoms[None].to(device)
        self._bonds = bonds[None].to(device)
        num_atoms = atoms.size(0)
        self._atom_padding_mask = torch.zeros([1, num_atoms], dtype=torch.bool, device=device)

        # Store both the original SMILES string and tokenized version
        self._smiles_str = mol.smiles  # Original SMILES string
        
        # Check if we're using DESERT encoder
        if hasattr(model, 'encoder_type') and model.encoder_type == "desert":
            # For DESERT, we'll use the string directly in the encode method
            self._smiles = None  # Not needed for DESERT
        else:
            # For other encoders, tokenize the SMILES
            smiles_tokens = mol.tokenize_csmiles()
            self._smiles = smiles_tokens[None].to(device)

        self._factor = factor
        self._max_active_states = max_active_states
        self._sort_by_score = sort_by_score

        self._active: list[State] = [State()]
        self._finished: list[State] = []
        self._aborted: list[State] = []
        
        # Add a list to store good molecules found during evolution
        self._good_molecules: list[dict] = []

    @cached_property
    def device(self) -> torch.device:
        return self._atoms.device

    @cached_property
    def code(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            # Create the batch dictionary based on encoder type
            if hasattr(self._model, 'encoder_type') and self._model.encoder_type == "desert":
                batch = {
                    "atoms": self._atoms,
                    "bonds": self._bonds,
                    "atom_padding_mask": self._atom_padding_mask,
                    "smiles_str": self._smiles_str,  # Pass the original SMILES string
                }
                print(f"StatePool: Using DESERT encoder for molecule: {self._smiles_str}")
            else:
                batch = {
                    "atoms": self._atoms,
                    "bonds": self._bonds,
                    "atom_padding_mask": self._atom_padding_mask,
                    "smiles": self._smiles,  # Pass the tokenized SMILES
                }
            
            try:
                code, code_padding_mask, encoder_loss_dict = self._model.encode(batch)
                
                # Handle different return types from encode
                if hasattr(code, 'code') and hasattr(code, 'code_padding_mask'):
                    # If code is an object with code and code_padding_mask attributes
                    print(f"StatePool: Received object with code attribute, shape: {code.code.shape}")
                    return code.code, code.code_padding_mask
                
                print(f"StatePool: Encoder returned code with shape: {code.shape}, padding mask shape: {code_padding_mask.shape}")
                return code, code_padding_mask
            except Exception as e:
                print(f"StatePool: Error in encode method: {str(e)}")
                raise

    def _sort_states(self) -> None:
        if self._sort_by_score:
            self._active.sort(key=lambda s: s.score, reverse=True)
        self._active = self._active[: self._max_active_states]

    def _collate(self, feat_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }
        return apply_collate(spec_tokens, feat_list, feat_list[0]["token_types"].size(0))

    def evolve(
        self,
        gpu_lock: Lock | None = None,
        show_pbar: bool = False,
        time_limit: TimeLimit | None = None,
    ) -> None:
        if len(self._active) == 0:
            return
        feat_list = [
            featurize_stack(
                state.stack,
                end_token=False,
                fpindex=self._fpindex,
            )
            for state in self._active
        ]

        if gpu_lock is not None:
            gpu_lock.acquire()

        feat = {k: v.to(self.device) for k, v in self._collate(feat_list).items()}

        code, code_padding_mask = self.code
        code_size = list(code.size())
        code_size[0] = len(feat_list)
        code = code.expand(code_size)
        mask_size = list(code_padding_mask.size())
        mask_size[0] = len(feat_list)
        code_padding_mask = code_padding_mask.expand(mask_size)

        result = self._model.predict(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=feat["token_types"],
            rxn_indices=feat["rxn_indices"],
            reactant_fps=feat["reactant_fps"],
            rxn_matrix=self._rxn_matrix,
            fpindex=self._fpindex,
            topk=self._factor,
            result_device=torch.device("cpu"),
        )

        if gpu_lock is not None:
            gpu_lock.release()

        n = code.size(0)
        m = self._factor
        nm_iter: Iterable[tuple[int, int]] = itertools.product(range(n), range(m))
        if show_pbar:
            nm_iter = tqdm(nm_iter, total=n * m, desc="evolve", dynamic_ncols=True)

        best_token = result.best_token()
        top_reactants = result.top_reactants(topk=m)
        top_reactions = result.top_reactions(topk=m, rxn_matrix=self._rxn_matrix)

        next: list[State] = []
        for i, j in nm_iter:
            if time_limit is not None and time_limit.exceeded():
                break

            tok_next = best_token[i]
            base_state = self._active[i]
            if tok_next == TokenType.END:
                self._finished.append(base_state)

            elif tok_next == TokenType.REACTANT:
                reactant, mol_idx, score = top_reactants[i][j]
                new_state = copy.deepcopy(base_state)
                new_state.stack.push_mol(reactant, mol_idx)
                new_state.scores.append(score)
                next.append(new_state)

            elif tok_next == TokenType.REACTION:
                reaction, rxn_idx, score = top_reactions[i][j]
                new_state = copy.deepcopy(base_state)
                success = new_state.stack.push_rxn(reaction, rxn_idx, product_limit=None)
                if success:
                    # Get the top molecules from the stack
                    top_mols = new_state.stack.get_top()
                    
                    # Calculate Tanimoto similarity
                    tanimoto_scores = [self._mol.sim(m, fp_option=FingerprintOption.rdkit()) for m in top_mols]
                    max_tanimoto = max(tanimoto_scores) if tanimoto_scores else 0.0
                    
                    # Get RDKit molecule for docking
                    rdkit_mols = []
                    mol_smiles_list = []
                    for m in top_mols:
                        try:
                            # Use the molecule's smiles to create an RDKit molecule
                            rdkit_mol = Chem.MolFromSmiles(m.smiles)
                            if rdkit_mol is not None:
                                rdkit_mols.append(rdkit_mol)
                                mol_smiles_list.append(m.smiles)
                                print(f"Processing molecule: {m.smiles}")
                        except Exception as e:
                            print(f"Error converting molecule to RDKit: {str(e)}")
                    
                    # Try to dock each molecule and get the best docking score
                    docking_scores = []
                    for idx, rdkit_mol in enumerate(rdkit_mols):
                        if rdkit_mol is not None:
                            docking_score = dock_best_molecule(rdkit_mol)
                            if docking_score is not None:
                                # Docking scores are negative (lower is better)
                                # Normalize to 0-1 range where 1 is best
                                # Assuming docking scores typically range from -12 to 0
                                normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                                docking_scores.append((normalized_docking_score, docking_score, idx))
                                print(f"Normalized docking score: {normalized_docking_score:.4f}")
                    
                    # Calculate combined score: 90% docking + 10% Tanimoto
                    if docking_scores:
                        # Sort by normalized docking score (highest first)
                        docking_scores.sort(reverse=True)
                        max_docking, raw_docking, best_idx = docking_scores[0]
                        best_smiles = mol_smiles_list[best_idx]
                        combined_score = 0.9 * max_docking + 0.1 * max_tanimoto
                        print(f"Combined score: {combined_score:.4f} (Docking: {max_docking:.4f}, Tanimoto: {max_tanimoto:.4f})")
                        
                        # Save good molecules directly to our results list
                        if combined_score >= 0.5:  # Threshold for "good" molecules
                            self._good_molecules.append({
                                "target": self._mol.smiles,
                                "smiles": best_smiles,
                                "tanimoto_score": max_tanimoto,
                                "docking_score": raw_docking,
                                "score": combined_score,
                                "synthesis": new_state.stack.get_action_string(),
                                "num_steps": new_state.stack.count_reactions(),
                            })
                            print(f"Saved molecule with combined score: {combined_score:.4f}")
                    else:
                        # If docking failed, use only Tanimoto
                        combined_score = max_tanimoto
                        print(f"Using only Tanimoto score: {max_tanimoto:.4f} (docking failed)")
                    
                    new_state.scores.append(combined_score)
                    next.append(new_state)
                else:
                    self._aborted.append(new_state)

            else:
                self._aborted.append(base_state)

        del self._active
        self._active = next
        self._sort_states()

    def get_products(self) -> Iterable[_ProductInfo]:
        visited: set[Molecule] = set()
        for state in self._finished:
            for mol in state.stack.get_top():
                if mol in visited:
                    continue
                yield _ProductInfo(mol, state.stack)
                visited.add(mol)
        yield from []

    def get_dataframe(self, num_calc_extra_metrics: int = 10) -> pd.DataFrame:
        rows: list[dict[str, str | float]] = []
        smiles_to_mol: dict[str, Molecule] = {}
        
        # First, add all the good molecules we found during evolution
        rows.extend(self._good_molecules)
        
        # Then process the finished states
        for product in self.get_products():
            # Skip if we already have this molecule in our results
            if any(r["smiles"] == product.molecule.smiles for r in rows):
                continue
                
            # Get RDKit molecule for docking
            rdkit_mol = None
            try:
                rdkit_mol = Chem.MolFromSmiles(product.molecule.smiles)
                print(f"\nEvaluating final product: {product.molecule.smiles}")
            except Exception as e:
                print(f"Error converting molecule to RDKit: {str(e)}")
            
            # Calculate Tanimoto similarity
            tanimoto_score = self._mol.sim(product.molecule, FingerprintOption.morgan_for_tanimoto_similarity())
            print(f"Tanimoto similarity: {tanimoto_score:.4f}")
            
            # Get docking score
            docking_score = None
            if rdkit_mol is not None:
                docking_score = dock_best_molecule(rdkit_mol)
                if docking_score is not None:
                    print(f"Raw docking score: {docking_score:.4f}")
            
            # Calculate combined score
            if docking_score is not None:
                # Normalize docking score (lower is better, typically -12 to 0)
                normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                combined_score = 0.9 * normalized_docking_score + 0.1 * tanimoto_score
                print(f"Combined score: {combined_score:.4f} (Normalized docking: {normalized_docking_score:.4f}, Tanimoto: {tanimoto_score:.4f})")
            else:
                combined_score = tanimoto_score
                print(f"Using only Tanimoto score: {tanimoto_score:.4f} (docking failed)")
            
            rows.append(
                {
                    "target": self._mol.smiles,
                    "smiles": product.molecule.smiles,
                    "tanimoto_score": tanimoto_score,
                    "docking_score": docking_score,
                    "score": combined_score,  # Combined score
                    "synthesis": product.stack.get_action_string(),
                    "num_steps": product.stack.count_reactions(),
                }
            )
            smiles_to_mol[product.molecule.smiles] = product.molecule
            
        # Sort all results by score
        rows.sort(key=lambda r: r["score"], reverse=True)
        
        # Calculate extra metrics for top molecules
        for row in rows[:num_calc_extra_metrics]:
            if row["smiles"] in smiles_to_mol:
                mol = smiles_to_mol[str(row["smiles"])]
                row["scf_sim"] = self._mol.scaffold.tanimoto_similarity(
                    mol.scaffold,
                    fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
                )
                row["pharm2d_sim"] = self._mol.dice_similarity(mol, fp_option=FingerprintOption.gobbi_pharm2d())
                row["rdkit_sim"] = self._mol.tanimoto_similarity(mol, fp_option=FingerprintOption.rdkit())

        df = pd.DataFrame(rows)
        return df

    def print_stats(self) -> None:
        print(f"Active: {len(self._active)}")
        print(f"Finished: {len(self._finished)}")
        print(f"Aborted: {len(self._aborted)}")


class StatePoolWithShapePatches(StatePool):
    def __init__(
        self,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        mol: Molecule,
        model: Synformer,
        shape_patches: torch.Tensor,
        factor: int = 16,
        max_active_states: int = 256,
        sort_by_score: bool = True,
    ) -> None:
        super().__init__(fpindex, rxn_matrix, mol, model, factor, max_active_states, sort_by_score)
        
        # Store shape patches
        device = next(iter(model.parameters())).device
        self._shape_patches = shape_patches.unsqueeze(0).to(device)
    
    @cached_property
    def code(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            code, code_padding_mask, encoder_loss_dict = self._model.encode(
                {
                    "atoms": self._atoms,
                    "bonds": self._bonds,
                    "atom_padding_mask": self._atom_padding_mask,
                    "smiles": self._smiles,
                    "shape_patches": self._shape_patches,
                }
            )
            return code, code_padding_mask
