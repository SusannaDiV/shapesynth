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
from rdkit.Chem import QED

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
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8
    num_modes: int = 1

def timeout_handler(smiles, func, args=(), timeout=3):
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
    try:
        import tempfile
        import uuid
        
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        temp_mol_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.mol")
        temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.pdbqt")
        
        Chem.MolToMolFile(mol, temp_mol_file)
        
        cmd = [
            obabel_path,
            "-imol", temp_mol_file,
            "-opdbqt", "-O", temp_pdbqt_file,
            "--partialcharge", "gasteiger",
            "--gen3d", "best"
        ]
        
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            print(f"Error converting molecule to PDBQT: {process.stderr}")
            return None
        
        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()
            
        try:
            os.remove(temp_mol_file)
            os.remove(temp_pdbqt_file)
        except:
            pass
        
        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            print("Error: Generated PDBQT file does not contain valid atom entries")
            return None
            
        return pdbqt_content
    except Exception as e:
        print(f"Error preparing ligand: {str(e)}")
        try:
            if os.path.exists(temp_mol_file):
                os.remove(temp_mol_file)
            if os.path.exists(temp_pdbqt_file):
                os.remove(temp_pdbqt_file)
        except:
            pass
        return None

def dock_best_molecule(mol, receptor_path, receptor_center):
    try:
        import tempfile
        import uuid
        
        smiles = Chem.MolToSmiles(mol)
        print(f"Docking molecule: {smiles}")
        
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        
        center = receptor_center
        box_size = [22.5, 22.5, 22.5]

        qvina_path = "bin/qvina2.1"
        if not os.path.exists(qvina_path):
            qvina_path = shutil.which("qvina2.1")
            if qvina_path is None:
                print("Error: QVina2 executable not found")
                return None
        
        obabel_path = shutil.which("obabel")
        
        if obabel_path is None:
            print("Error: OpenBabel (obabel) not found in PATH")
            return None
            
        if not os.path.exists(receptor_path):
            print(f"Error: Receptor file not found at {receptor_path}")
            return None
            
        ligand_pdbqt = prepare_ligand_pdbqt(mol, obabel_path)
        if ligand_pdbqt is None:
            print("Failed to prepare ligand for docking")
            return None
            
        temp_ligand_file = os.path.join(temp_dir, f"temp_ligand_dock_{unique_id}.pdbqt")
        with open(temp_ligand_file, "w") as f:
            f.write(ligand_pdbqt)
            
        options = QVinaOption(
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            size_x=box_size[0],
            size_y=box_size[1],
            size_z=box_size[2]
        )
        
        output_file = os.path.join(temp_dir, f"temp_ligand_dock_out_{unique_id}.pdbqt")
        cmd = [
            str(qvina_path),
            "--receptor", str(receptor_path),
            "--ligand", str(temp_ligand_file),
            "--center_x", str(options.center_x),
            "--center_y", str(options.center_y),
            "--center_z", str(options.center_z),
            "--size_x", str(options.size_x),
            "--size_y", str(options.size_y),
            "--size_z", str(options.size_z),
            "--exhaustiveness", str(options.exhaustiveness),
            "--num_modes", str(options.num_modes),
            "--out", str(output_file)
        ]
        
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
                check=False
            )
        except subprocess.TimeoutExpired:
            print("Docking timed out after 5 minutes")
            return None
            
        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        try:
                            score = float(line.split()[3])
                            print(f"Docking score for {smiles}: {score}")
                            
                            sdf_output = output_file.replace('.pdbqt', '.sdf')
                            obabel_cmd = [
                                obabel_path,
                                "-ipdbqt", output_file,
                                "-osdf", "-O", sdf_output,
                                "--addhydrogens"
                            ]
                            subprocess.run(obabel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                            
                            if os.path.exists(sdf_output):
                                receptor_dir = os.path.dirname(receptor_path)
                                receptor_name = os.path.splitext(os.path.basename(receptor_path))[0]
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                final_sdf_path = os.path.join(receptor_dir, f"{receptor_name}_docked_groupof53_{smiles.replace('/', '_')}_{score:.2f}_{timestamp}.sdf")
                                shutil.copy2(sdf_output, final_sdf_path)
                                print(f"Saved docked pose to: {final_sdf_path}")
                            
                            break
                        except (IndexError, ValueError):
                            pass
                            
        try:
            if os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(sdf_output):
                os.remove(sdf_output)
        except:
            pass
            
        return score
        
    except Exception as e:
        print(f"Error during docking: {str(e)}")
        try:
            if 'temp_ligand_file' in locals() and os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if 'output_file' in locals() and os.path.exists(output_file):
                os.remove(output_file)
            if 'sdf_output' in locals() and os.path.exists(sdf_output):
                os.remove(sdf_output)
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
        factor: int = 32,
        max_active_states: int = 256,
        sort_by_score: bool = True,
        shape_patches_path: str = None,
        receptor_path: str = None,
        receptor_center: list = None,
        mixture_weight: float | None = None,
    ) -> None:
        self._fpindex = fpindex
        self._rxn_matrix = rxn_matrix
        self._mol = mol
        self._model = model
        self._factor = factor
        self._max_active_states = max_active_states
        self._sort_by_score = sort_by_score
        self._shape_patches_path = shape_patches_path
        self._receptor_path = receptor_path
        self._receptor_center = receptor_center
        self._mixture_weight = mixture_weight
        
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
            # Create batch with shape_patches_path
            self._batch = {
                'smiles_str': self._smiles_str,
                'shape_patches_path': self._shape_patches_path
            }
        else:
            # For other encoders, tokenize the SMILES
            smiles_tokens = mol.tokenize_csmiles()
            self._smiles = smiles_tokens[None].to(device)
            self._batch = {
                'smiles_tokens': self._smiles,
                'smiles_padding_mask': torch.zeros_like(self._smiles, dtype=torch.bool, device=self._smiles.device)
            }

        self._active: list[State] = [State()]
        self._finished: list[State] = []
        self._aborted: list[State] = []
        
        # Add a list to store good molecules found during evolution
        self._good_molecules: list[dict] = []
        
        self._sampled = set()
        self._smiles2idx = {mol.smiles: -1}

    @cached_property
    def device(self) -> torch.device:
        return self._atoms.device

    @cached_property
    def code(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Call the encode method with the prepared batch
        try:
            code, code_padding_mask, _ = self._model.encode(
                self._batch, 
                mixture_weight=self._mixture_weight
            )
            return code, code_padding_mask
        except Exception as e:
            print(f"Error in encode method with mixture_weight={self._mixture_weight}: {str(e)}")
            print("Trying with mixture_weight=0.5")
            try:
                code, code_padding_mask, _ = self._model.encode(
                    self._batch,
                    mixture_weight=0.5
                )
                return code, code_padding_mask
            except Exception as e2:
                print(f"Error in encode method on second attempt with mixture_weight=0.5: {str(e2)}")
                # Fallback
                device = next(iter(self._model.parameters())).device
                return torch.zeros((1, 1, 768), device=device), torch.zeros((1, 1), dtype=torch.bool, device=device)

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
                    
                    # Filter out short SMILES strings
                    min_smiles_len = 24  # Length of "CC(C#CCCCOC(=O)Cl)CN"
                    filtered_top_mols = []
                    for m in top_mols:
                        if len(m.smiles) > min_smiles_len:
                            filtered_top_mols.append(m)
                        else:
                            print(f"Skipping short SMILES: {m.smiles} (len: {len(m.smiles)})")
                    top_mols = filtered_top_mols

                    # Calculate QED score
                    qed_scores = []
                    for m in top_mols:
                        try:
                            rdkit_mol = Chem.MolFromSmiles(m.smiles)
                            if rdkit_mol is not None:
                                qed_value = QED.qed(rdkit_mol)
                                qed_scores.append(qed_value)
                                print(f"QED score for {m.smiles}: {qed_value:.4f}") # Print QED
                        except Exception as e:
                            print(f"Error calculating QED for {m.smiles}: {str(e)}")
                    
                    max_qed = max(qed_scores) if qed_scores else 0.0
                    
                    # Get RDKit molecule for docking
                    rdkit_mols = []
                    mol_smiles_list = []
                    for m in top_mols:
                        try:
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
                        if rdkit_mol is not None and self._receptor_path and self._receptor_center:
                            docking_score = dock_best_molecule(rdkit_mol, self._receptor_path, self._receptor_center)
                            if docking_score is not None:
                                normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                                docking_scores.append((normalized_docking_score, docking_score, idx))
                                print(f"Normalized docking score: {normalized_docking_score:.4f}")
                    
                    # Score is based exclusively on docking. QED is calculated and printed but does not affect the score.
                    if docking_scores:
                        # Sort by normalized docking score (highest first)
                        docking_scores.sort(reverse=True)
                        max_docking, raw_docking, best_idx = docking_scores[0]
                        best_smiles = mol_smiles_list[best_idx]
                        
                        combined_score = max_docking # Score is now exclusively the best normalized docking score
                        print(f"Docking-based score for dataframe: {combined_score:.4f} (Raw docking: {raw_docking:.4f}, QED: {max_qed:.4f})")
                        
                        # Save good molecules to our results list
                        if combined_score >= 0.619:  # Threshold for "good" molecules based on docking
                            self._good_molecules.append({
                                "target": self._mol.smiles,
                                "smiles": best_smiles,
                                "qed_score": max_qed, # Store QED
                                "docking_score": raw_docking,
                                "score": combined_score, # Docking-based score
                                "synthesis": new_state.stack.get_action_string(),
                                "num_steps": new_state.stack.count_reactions(),
                            })
                            print(f"Saved molecule with docking-based score: {combined_score:.4f}")
                        
                        # Only continue reaction pathways that have good docking scores
                        if combined_score >= 0.1:
                            new_state.scores.append(combined_score)
                            next.append(new_state)
                            print(f"Continuing reaction pathway with docking-based score: {combined_score:.4f}")
                        else:
                            # Also save molecules that are aborted but have a good docking score
                            if raw_docking < -7.2 and combined_score < 0.619:
                                self._good_molecules.append({
                                    "target": self._mol.smiles,
                                    "smiles": best_smiles,
                                    "qed_score": max_qed,
                                    "docking_score": raw_docking,
                                    "score": combined_score,
                                    "synthesis": new_state.stack.get_action_string(),
                                    "num_steps": new_state.stack.count_reactions(),
                                })
                                print(f"Saved aborted molecule with good docking score: {raw_docking:.4f}")
                            self._aborted.append(new_state)
                            print(f"Aborting reaction pathway with low docking-based score: {combined_score:.4f}")
                    else:
                        # If docking failed, the score is low, and pathway is likely aborted.
                        combined_score = 0.0 
                        print(f"Docking failed for dataframe. Score set to {combined_score:.4f}. (QED: {max_qed:.4f})")
                        
                        # Optionally, still save molecule if QED is high, but with a score of 0 or low docking score
                        # For now, we will not save it if docking fails, as score is docking-exclusive.
                        # if max_qed > 0.8: # Example: save if QED is very high despite docking failure
                        #     self._good_molecules.append({
                        #         "target": self._mol.smiles,
                        #         "smiles": top_mols[0].smiles if top_mols else "N/A",
                        #         "qed_score": max_qed,
                        #         "docking_score": None,
                        #         "score": 0.0, # Reflects docking failure
                        #         "synthesis": new_state.stack.get_action_string(),
                        #         "num_steps": new_state.stack.count_reactions(),
                        #     })
                        #     print(f"Saved molecule based on high QED ({max_qed:.4f}) despite docking failure.")

                        self._aborted.append(new_state)
                        print(f"Aborting reaction pathway due to docking failure (score: {combined_score:.4f}).")
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
            
            # Calculate QED score
            qed_score = 0.0
            if rdkit_mol is not None:
                try:
                    qed_score = QED.qed(rdkit_mol)
                    print(f"QED score: {qed_score:.4f}")
                except Exception as e:
                    print(f"Error calculating QED: {str(e)}")
            
            # Get docking score
            docking_score = None
            if rdkit_mol is not None and self._receptor_path and self._receptor_center:
                docking_score = dock_best_molecule(rdkit_mol, self._receptor_path, self._receptor_center)
                if docking_score is not None:
                    print(f"Raw docking score: {docking_score:.4f}")
            
            # Calculate combined score
            if docking_score is not None:
                # Normalize docking score (lower is better, typically -12 to 0)
                normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                combined_score = normalized_docking_score # Score is now exclusively the normalized docking score
                print(f"Docking-based score for dataframe: {combined_score:.4f} (Raw docking: {docking_score:.4f}, QED: {qed_score:.4f})")
            else:
                combined_score = 0.0 # Docking failed, so score is 0
                print(f"Docking failed for dataframe. Score set to {combined_score:.4f}. (QED: {qed_score:.4f})")
            
            rows.append(
                {
                    "target": self._mol.smiles,
                    "smiles": product.molecule.smiles,
                    "qed_score": qed_score, # Store QED
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
        factor: int = 32,
        max_active_states: int = 256,
        sort_by_score: bool = True,
        mixture_weight: float | None = None,
    ) -> None:
        super().__init__(fpindex, rxn_matrix, mol, model, factor, max_active_states, sort_by_score, mixture_weight=mixture_weight)
        
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
                },
                mixture_weight=self._mixture_weight
            )
            return code, code_padding_mask
