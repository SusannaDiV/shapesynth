# System imports
import os
import random
import numpy as np
import joblib
import tdc
import shutil
import subprocess
import threading
import pathlib
import pandas as pd
from time import time
from dataclasses import dataclass
import urllib.request
import tempfile
from tqdm import tqdm  # Import tqdm for progress bars

# Scientific computing
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# RDKit imports
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import BRICS, AllChem, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

# Visualization imports
import py3Dmol
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Docking imports
from meeko import MoleculePreparation
from vina import Vina

# Synformer imports
from synformer.chem.mol import Molecule
from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles

# Local imports
import crossover as co
import mutate as mu
import yaml
from joblib import delayed
from joblib import Parallel

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
rdBase.DisableLog("rdApp.error")

# Import all necessary functions from crossover_shape.py
from crossover_shape import (
    process_mol,
    process_smiles,
    calculate_usr_descriptors,
    calculate_usr_similarity,
    calculate_shape_moments,
    calculate_gaussian_overlap,
    save_fingerprints_to_file,
    reconstruct_shape_from_fingerprint,
    visualize_fingerprint_shape,
    visualize_molecules,
    identify_pharmacophore_features,
    get_atom_sigma,
    generate_pharmacophore_fingerprints,
    save_pharmacophore_fingerprints,
    load_pharmacophore_fingerprints,
    visualize_pharmacophore_fingerprints,
    blend_pharmacophore_fingerprints,
    align_pharmacophore_grids,
    visualize_all_pharmacophore_fingerprints,
    generate_valid_child_molecule,
    fragment_molecule_with_dummy_atoms,
    connect_fragments_at_dummy_atoms,
    generate_and_optimize_conformers,
    select_best_conformer,
    evaluate_docking,
    evaluate_synformer_molecules,
    dock_molecule as cs_dock_molecule  # Rename imported dock_molecule to avoid conflict
)

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

def prepare_ligand_pdbqt(mol, obabel_path="obabel", temp_id=None):
    """
    Prepare a ligand for docking by converting it to PDBQT format using OpenBabel
    
    Args:
        mol: RDKit molecule
        obabel_path: Path to OpenBabel executable
        temp_id: Optional identifier for creating unique temporary files
        
    Returns:
        pdbqt_content: String containing PDBQT file content
    """
    try:
        # Generate unique temporary file names using process ID and optional identifier
        import os
        import tempfile
        
        # Create a temporary directory for this process
        temp_dir = tempfile.mkdtemp()
        
        # Use unique filenames including PID and an optional identifier
        pid = os.getpid()
        if temp_id is None:
            temp_id = ''
        else:
            temp_id = f"_{temp_id}"
        
        temp_mol_file = os.path.join(temp_dir, f"temp_ligand_pid{pid}{temp_id}.mol")
        temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_pid{pid}{temp_id}.pdbqt")
        
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
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {str(e)}")
        
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
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temporary files: {str(cleanup_error)}")
        return None

def dock_best_molecule(mol, receptor_path="/home/luost_local/sdivita/synformer/experiments/sbdd/receptor.pdbqt"):
    """Dock the molecule against ADRB2 target"""
    try:
        # ADRB2 docking parameters
        center = [-9.845024108886719, -4.321293354034424, 39.35286331176758]
        box_size = [11.208, 9.997, 14.994]
        qvina_path = "bin/qvina2.1"
        obabel_path = shutil.which("obabel")
        
        if obabel_path is None:
            print("Error: OpenBabel (obabel) not found in PATH")
            return None
            
        # Check if receptor file exists
        if not os.path.exists(receptor_path):
            print(f"Error: Receptor file not found at {receptor_path}")
            return None
            
        # Use the cs_dock_molecule function directly with process-specific temp files
        return cs_dock_molecule(mol, receptor_path, center, box_size, qvina_path, obabel_path)
        
    except Exception as e:
        print(f"Error during docking: {str(e)}")
        return None

def sanitize(mol_list):
    new_mol_list = []
    smiles_set = set()
    for mol in mol_list:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles is not None and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    new_mol_list.append(mol)
            except ValueError:
                print("bad smiles")
    return new_mol_list


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")
        self.last_log = 0
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.output_dir, "results.yaml")
        else:
            output_file_path = os.path.join(self.output_dir, "results_" + suffix + ".yaml")

        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[
                        : self.max_oracle_calls
                    ]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)

        print(
            f"{n_calls}/{self.max_oracle_calls} | "
            f"avg_top1: {avg_top1:.3f} | "
            f"avg_top10: {avg_top10:.3f} | "
            f"avg_top100: {avg_top100:.3f} | "
            f"avg_sa: {avg_sa:.3f} | "
            f"div: {diversity_top100:.3f}"
        )

        # try:
        print(
            {
                "avg_top1": avg_top1,
                "avg_top10": avg_top10,
                "avg_top100": avg_top100,
                "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
                "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
                "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
                "avg_sa": avg_sa,
                "diversity_top100": diversity_top100,
                "n_oracle": n_calls,
            }
        )

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if isinstance(smiles_lst, list):
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


MINIMUM = 1e-10


def make_mating_pool(population_mol: list[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate, receptor_path, center, box_size, pharmacophore_info, qvina_path="bin/qvina2.1", obabel_path="obabel"):
    """
    Perform crossover between two parent molecules using fragment-based approach.
    Returns the child if it improves on both parents' docking scores, otherwise returns the better parent.
    Logs the chosen molecule's information to crossover_log.txt.
    
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
        receptor_path: Path to receptor PDBQT file
        center: Docking box center coordinates [x, y, z]
        box_size: Docking box dimensions [x, y, z]
        pharmacophore_info: Dictionary containing pre-calculated pharmacophore information (unused)
        qvina_path: Path to QVina2 executable
        obabel_path: Path to OpenBabel executable
    Returns:
        RDKit Mol with best docking score
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    
    try:
        # Get parent SMILES
        parent_a_smiles = Chem.MolToSmiles(parent_a)
        parent_b_smiles = Chem.MolToSmiles(parent_b)
        
        # Dock parent molecules to get their scores (but don't print yet)
        parent_a_score = cs_dock_molecule(parent_a, receptor_path, center, box_size, qvina_path, obabel_path)
        parent_b_score = cs_dock_molecule(parent_b, receptor_path, center, box_size, qvina_path, obabel_path)
        
        # Generate a valid child molecule
        print("\nGenerating a valid child molecule...")
        child_mol = generate_valid_child_molecule(parent_a, parent_b)
        
        if child_mol:
            # Run Synformer on the child molecule to optimize it
            print("\nRunning Synformer on the child molecule...")
            child_smiles = Chem.MolToSmiles(child_mol)
            synformer_smiles_list = projection([child_smiles])
            
            if synformer_smiles_list and len(synformer_smiles_list) > 0:
                # Use the first (best) Synformer result
                synformer_child_smiles = synformer_smiles_list[0]
                print(f"Original child: {child_smiles}")
                print(f"Synformer optimized child: {synformer_child_smiles}")
                
                # Convert Synformer SMILES to molecule
                synformer_child_mol = Chem.MolFromSmiles(synformer_child_smiles)
                
                if synformer_child_mol:
                    # Select best conformer based on docking score using the Synformer-optimized child
                    print("\nSelecting best conformer based on docking for Synformer-optimized child...")
                    best_mol, best_score = select_best_conformer(
                        synformer_child_mol, parent_a, parent_b,
                        receptor_path, center, box_size,
                        qvina_path, obabel_path
                    )
                    
                    if best_mol is not None:
                        # Now print all results together
                        print("\nDocking Results:")
                        print(f"Parent A ({parent_a_smiles}): {parent_a_score:.2f} kcal/mol")
                        print(f"Parent B ({parent_b_smiles}): {parent_b_score:.2f} kcal/mol")
                        print(f"Synformer Child ({Chem.MolToSmiles(best_mol)}): {best_score:.2f} kcal/mol")
                        print(f"Improvement vs Parent A: {parent_a_score - best_score:.2f} kcal/mol")
                        print(f"Improvement vs Parent B: {parent_b_score - best_score:.2f} kcal/mol")
                        
                        # Log the results
                        with open("crossover_log.txt", "a") as f:
                            f.write("\n=== Crossover Result ===\n")
                            f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                            f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                            f.write(f"Original Child: {child_smiles}\n")
                            f.write(f"Synformer Child: {synformer_child_smiles}\n")
                            
                            # If child improves on both parents, return it
                            if best_score < parent_a_score and best_score < parent_b_score:
                                print("Synformer child improves on both parents, keeping child")
                                f.write(f"Selected: NEW SYNFORMER CHILD\n")
                                f.write(f"SMILES: {Chem.MolToSmiles(best_mol)}\n")
                                f.write(f"Score: {best_score:.2f} kcal/mol\n")
                                f.write(f"Improvement vs Parent A: {parent_a_score - best_score:.2f} kcal/mol\n")
                                f.write(f"Improvement vs Parent B: {parent_b_score - best_score:.2f} kcal/mol\n")
                                return mu.mutate(best_mol, mutation_rate)
                            else:
                                # Otherwise return the better parent
                                if parent_a_score <= parent_b_score:
                                    print("Synformer child does not improve on both parents, keeping Parent A")
                                    f.write(f"Selected: PARENT A\n")
                                    f.write(f"SMILES: {parent_a_smiles}\n")
                                    f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                                    return parent_a
                                else:
                                    print("Synformer child does not improve on both parents, keeping Parent B")
                                    f.write(f"Selected: PARENT B\n")
                                    f.write(f"SMILES: {parent_b_smiles}\n")
                                    f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                                    return parent_b
                else:
                    print("Failed to create molecule from Synformer SMILES")
            else:
                print("Synformer did not return any results for the child molecule")
        
        # If child generation or Synformer optimization failed, return better parent
        print("Crossover or Synformer optimization failed, returning better parent")
        with open("crossover_log.txt", "a") as f:
            f.write("\n=== Crossover or Synformer Failed ===\n")
            f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
            f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
            if parent_a_score <= parent_b_score:
                f.write(f"Selected: PARENT A\n")
                f.write(f"SMILES: {parent_a_smiles}\n")
                f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                return parent_a
            else:
                f.write(f"Selected: PARENT B\n")
                f.write(f"SMILES: {parent_b_smiles}\n")
                f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                return parent_b
        
    except Exception as e:
        print(f"Error in reproduce: {str(e)}")
        # In case of error, return better parent and log the error
        with open("crossover_log.txt", "a") as f:
            f.write("\n=== Crossover Error ===\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
            f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
            if parent_a_score <= parent_b_score:
                f.write(f"Selected: PARENT A\n")
                f.write(f"SMILES: {parent_a_smiles}\n")
                f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                return parent_a
            else:
                f.write(f"Selected: PARENT B\n")
                f.write(f"SMILES: {parent_b_smiles}\n")
                f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                return parent_b


def projection(smiles_list, model_path="/home/luost_local/sdivita/synformer/data/trained_weights/sf_ed_default.ckpt"):
    input = [Molecule(s) for s in smiles_list]
    result_df = run_parallel_sampling_return_smiles(
        input=input,
        model_path=model_path,
        search_width=24,
        exhaustiveness=64,
        num_gpus=4,
        num_workers_per_gpu=2,
        task_qsize=0,
        result_qsize=0,
        time_limit=180,
        sort_by_scores=True,
    )
    result_df.drop_duplicates(subset="target", inplace=True, keep="first")
    return result_df.smiles.to_list()


def process_initial_population(smiles_list, receptor_path, center, box_size, qvina_path="bin/qvina2.1", obabel_path="obabel"):
    """
    Process initial population:
    1. Generate 3 conformers for each molecule
    2. Dock each conformer and keep the best one
    3. Calculate GSO and pharmacophore features for the best conformer
    
    Args:
        smiles_list: List of SMILES strings
        receptor_path: Path to receptor PDBQT file
        center: Docking box center coordinates [x, y, z]
        box_size: Docking box dimensions [x, y, z]
        qvina_path: Path to QVina2 executable
        obabel_path: Path to OpenBabel executable
        
    Returns:
        List of tuples (mol, best_score, overall_grid, feature_grids, grid_info)
    """
    print("\n=== Processing Initial Population ===")
    results = []
    
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Processing molecules"):
        print(f"\nProcessing molecule {i+1}/{len(smiles_list)}: {smiles}")
        
        # Convert SMILES to RDKit mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Failed to create molecule from SMILES")
            continue
            
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3 conformers
        conformers = []
        try:
            # Remove existing conformers
            while mol.GetNumConformers() > 0:
                mol.RemoveConformer(0)
                
            # Generate new conformers
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 0
            
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=3,
                params=params
            )
            
            print(f"Generated {len(conformer_ids)} conformers")
            
            # Optimize each conformer
            for conf_id in conformer_ids:
                # Create a copy of the molecule with just this conformer
                conf_mol = Chem.Mol(mol)
                conf_mol.RemoveAllConformers()
                conf_mol.AddConformer(mol.GetConformer(conf_id))
                
                # Optimize geometry
                props = AllChem.MMFFGetMoleculeProperties(conf_mol)
                ff = AllChem.MMFFGetMoleculeForceField(conf_mol, props)
                if ff:
                    ff.Minimize(maxIts=200)
                    conformers.append(conf_mol)
                    
            print(f"Successfully optimized {len(conformers)} conformers")
            
            # Dock each conformer
            best_score = float('inf')
            best_conformer = None
            
            for j, conf_mol in enumerate(conformers):
                print(f"Docking conformer {j+1}/{len(conformers)}")
                score = cs_dock_molecule(conf_mol, receptor_path, center, box_size, qvina_path, obabel_path)
                
                if score is not None and score < best_score:
                    best_score = score
                    best_conformer = conf_mol
                    print(f"New best score: {best_score:.2f} kcal/mol")
            
            if best_conformer is not None:
                # Calculate pharmacophore features for best conformer
                print("Calculating pharmacophore features for best conformer")
                overall_grid, feature_grids, feature_atoms, grid_info = generate_pharmacophore_fingerprints(best_conformer)
                
                results.append((best_conformer, best_score, overall_grid, feature_grids, grid_info))
                print(f"Successfully processed molecule {i+1}")
            else:
                print(f"No valid docking poses found for molecule {i+1}")
                
        except Exception as e:
            print(f"Error processing molecule {i+1}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(results)} molecules out of {len(smiles_list)}")
    return results


def process_synformer_molecules(smiles_list, receptor_path, center, box_size, qvina_path="bin/qvina2.1", obabel_path="obabel"):
    """
    Process Synformer-generated molecules:
    1. Generate 3 conformers for each molecule
    2. Dock each conformer and keep the best one
    3. Calculate GSO and pharmacophore features for the best conformer
    
    Similar to process_initial_population but specifically for Synformer output
    """
    print("\n=== Processing Synformer-Generated Molecules ===")
    results = []
    
    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Processing Synformer molecules"):
        print(f"\nProcessing Synformer molecule {i+1}/{len(smiles_list)}: {smiles}")
        
        # Convert SMILES to RDKit mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Failed to create molecule from SMILES")
            continue
            
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3 conformers
        conformers = []
        try:
            # Remove existing conformers
            while mol.GetNumConformers() > 0:
                mol.RemoveConformer(0)
                
            # Generate new conformers
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 0
            
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=3,
                params=params
            )
            
            print(f"Generated {len(conformer_ids)} conformers")
            
            # Optimize each conformer
            for conf_id in conformer_ids:
                # Create a copy of the molecule with just this conformer
                conf_mol = Chem.Mol(mol)
                conf_mol.RemoveAllConformers()
                conf_mol.AddConformer(mol.GetConformer(conf_id))
                
                # Optimize geometry
                props = AllChem.MMFFGetMoleculeProperties(conf_mol)
                ff = AllChem.MMFFGetMoleculeForceField(conf_mol, props)
                if ff:
                    ff.Minimize(maxIts=200)
                    conformers.append(conf_mol)
                    
            print(f"Successfully optimized {len(conformers)} conformers")
            
            # Dock each conformer
            best_score = float('inf')
            best_conformer = None
            
            for j, conf_mol in enumerate(conformers):
                print(f"Docking conformer {j+1}/{len(conformers)}")
                score = cs_dock_molecule(conf_mol, receptor_path, center, box_size, qvina_path, obabel_path)
                
                if score is not None and score < best_score:
                    best_score = score
                    best_conformer = conf_mol
                    print(f"New best score: {best_score:.2f} kcal/mol")
            
            if best_conformer is not None:
                # Calculate pharmacophore features for best conformer
                print("Calculating pharmacophore features for best conformer")
                overall_grid, feature_grids, feature_atoms, grid_info = generate_pharmacophore_fingerprints(best_conformer)
                
                results.append((best_conformer, best_score, overall_grid, feature_grids, grid_info))
                print(f"Successfully processed Synformer molecule {i+1}")
            else:
                print(f"No valid docking poses found for Synformer molecule {i+1}")
                
        except Exception as e:
            print(f"Error processing Synformer molecule {i+1}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(results)} Synformer molecules out of {len(smiles_list)}")
    return results


def save_population_results(results, filename="initial_population.pkl"):
    """
    Save population results to a pickle file
    
    Args:
        results: List of tuples (mol, score, overall_grid, feature_grids, grid_info)
        filename: Output file name
    """
    # Convert results to a serializable format
    serializable_results = []
    for mol, score, overall_grid, feature_grids, grid_info in results:
        # Convert molecule to SMILES and save conformer info
        smiles = Chem.MolToSmiles(mol)
        mol_block = Chem.MolToMolBlock(mol)  # This preserves 3D coordinates
        
        result_dict = {
            'smiles': smiles,
            'mol_block': mol_block,
            'score': score,
            'overall_grid': overall_grid,
            'feature_grids': feature_grids,
            'grid_info': grid_info
        }
        serializable_results.append(result_dict)
    
    # Save to file
    with open(filename, 'wb') as f:
        import pickle
        pickle.dump(serializable_results, f)
    print(f"Saved population results to {filename}")

def load_population_results(filename="initial_population.pkl"):
    """
    Load population results from a pickle file
    
    Args:
        filename: Input file name
    Returns:
        List of tuples (mol, score, overall_grid, feature_grids, grid_info)
    """
    with open(filename, 'rb') as f:
        import pickle
        serializable_results = pickle.load(f)
    
    # Convert back to original format
    results = []
    for result_dict in serializable_results:
        # Convert SMILES and mol block back to RDKit mol with 3D coordinates
        mol = Chem.MolFromMolBlock(result_dict['mol_block'])
        if mol is None:
            print(f"Warning: Failed to load molecule {result_dict['smiles']}")
            continue
            
        results.append((
            mol,
            result_dict['score'],
            result_dict['overall_grid'],
            result_dict['feature_grids'],
            result_dict['grid_info']
        ))
    
    print(f"Loaded {len(results)} molecules from {filename}")
    return results

def cs_dock_molecule(mol, receptor_path, center, box_size, qvina_path="bin/qvina2.1", obabel_path="obabel", temp_id=None):
    """
    Dock a molecule using QVina2
    
    Args:
        mol: RDKit molecule
        receptor_path: Path to receptor PDBQT file
        center: List of [x, y, z] coordinates for docking box center
        box_size: List of [x, y, z] dimensions for docking box
        qvina_path: Path to QVina2 executable
        obabel_path: Path to OpenBabel executable
        temp_id: Optional identifier for creating unique temporary files
        
    Returns:
        score: Best docking score (float) or None if docking failed
    """
    import os
    import tempfile
    
    # Generate a unique ID for this call
    pid = os.getpid()
    if temp_id is None:
        import uuid
        temp_id = str(uuid.uuid4())[:8]
    
    # Create a temporary directory for this docking run
    temp_dir = tempfile.mkdtemp()
    
    # Use unique filenames
    temp_ligand_file = os.path.join(temp_dir, f"dock_ligand_{pid}_{temp_id}.pdbqt")
    output_file = os.path.join(temp_dir, f"dock_output_{pid}_{temp_id}.pdbqt")
    
    try:
        # Prepare ligand
        ligand_pdbqt = prepare_ligand_pdbqt(mol, obabel_path, temp_id)
        if ligand_pdbqt is None:
            print("Failed to prepare ligand for docking")
            return None
            
        # Write ligand to temporary file
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
            print(f"Docking timed out after 5 minutes for molecule in process {pid}")
            return None
            
        # Parse output to get docking score
        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        try:
                            score = float(line.split()[3])
                            break
                        except (IndexError, ValueError):
                            pass
                            
        # Clean up temporary files
        try:
            if os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {str(e)}")
            
        return score
        
    except Exception as e:
        print(f"Error during docking: {str(e)}")
        # Clean up temporary files
        try:
            if os.path.exists(temp_ligand_file):
                os.remove(temp_ligand_file)
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temporary files: {str(cleanup_error)}")
        return None

# Function to generate a valid child molecule and return parent information
def generate_child_with_parents(parent_a, parent_b, receptor_path, center, box_size, qvina_path="bin/qvina2.1", obabel_path="obabel"):
    try:
        # Get parent SMILES
        parent_a_smiles = Chem.MolToSmiles(parent_a)
        parent_b_smiles = Chem.MolToSmiles(parent_b)
        
        # Dock parent molecules to get their scores
        parent_a_score = cs_dock_molecule(parent_a, receptor_path, center, box_size, qvina_path, obabel_path)
        parent_b_score = cs_dock_molecule(parent_b, receptor_path, center, box_size, qvina_path, obabel_path)
        
        # Generate a valid child molecule
        child_mol = generate_valid_child_molecule(parent_a, parent_b)
        
        if child_mol:
            child_smiles = Chem.MolToSmiles(child_mol)
            return {
                'child_mol': child_mol,
                'child_smiles': child_smiles,
                'parent_a': parent_a,
                'parent_b': parent_b,
                'parent_a_smiles': parent_a_smiles,
                'parent_b_smiles': parent_b_smiles,
                'parent_a_score': parent_a_score,
                'parent_b_score': parent_b_score,
                'success': True
            }
        else:
            return {
                'parent_a': parent_a,
                'parent_b': parent_b,
                'parent_a_smiles': parent_a_smiles,
                'parent_b_smiles': parent_b_smiles,
                'parent_a_score': parent_a_score,
                'parent_b_score': parent_b_score,
                'success': False
            }
    except Exception as e:
        print(f"Error generating child: {str(e)}")
        return {
            'parent_a': parent_a,
            'parent_b': parent_b,
            'parent_a_smiles': parent_a_smiles if 'parent_a_smiles' in locals() else None,
            'parent_b_smiles': parent_b_smiles if 'parent_b_smiles' in locals() else None,
            'parent_a_score': parent_a_score if 'parent_a_score' in locals() else None,
            'parent_b_score': parent_b_score if 'parent_b_score' in locals() else None,
            'success': False,
            'error': str(e)
        }

# Function to process a Synformer-optimized child
def process_synformer_child(child_info, synformer_smiles, receptor_path, center, box_size, mutation_rate, qvina_path="bin/qvina2.1", obabel_path="obabel"):
    """
    Process a Synformer-optimized child molecule:
    1. Generate conformers for the Synformer-optimized molecule
    2. Dock each conformer and select the best one
    3. Compare with parent scores and return the child if it improves, otherwise return the better parent
    
    Args:
        child_info: Dictionary containing child and parent information
        synformer_smiles: SMILES string of the Synformer-optimized child
        receptor_path: Path to receptor PDBQT file
        center: Docking box center coordinates [x, y, z]
        box_size: Docking box dimensions [x, y, z]
        mutation_rate: Rate of mutation to apply if child is selected
        qvina_path: Path to QVina2 executable
        obabel_path: Path to OpenBabel executable
        
    Returns:
        RDKit Mol of the selected molecule (child or better parent)
    """
    try:
        log_file = "crossover_log.txt"
        parent_a_smiles = child_info['parent_a_smiles']
        parent_b_smiles = child_info['parent_b_smiles']
        parent_a_score = child_info['parent_a_score']
        parent_b_score = child_info['parent_b_score']
        original_child_smiles = child_info['child_smiles']
        
        # Convert Synformer SMILES to molecule
        synformer_child_mol = Chem.MolFromSmiles(synformer_smiles)
        
        if synformer_child_mol is None:
            print(f"Failed to create molecule from Synformer SMILES: {synformer_smiles}")
            # Return better parent
            if parent_a_score <= parent_b_score:
                with open(log_file, "a") as f:
                    f.write("\n=== Synformer Conversion Failed ===\n")
                    f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                    f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                    f.write(f"Original Child: {original_child_smiles}\n")
                    f.write(f"Synformer Failed: {synformer_smiles}\n")
                    f.write(f"Selected: PARENT A\n")
                    f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                return child_info['parent_a']
            else:
                with open(log_file, "a") as f:
                    f.write("\n=== Synformer Conversion Failed ===\n")
                    f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                    f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                    f.write(f"Original Child: {original_child_smiles}\n")
                    f.write(f"Synformer Failed: {synformer_smiles}\n")
                    f.write(f"Selected: PARENT B\n")
                    f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                return child_info['parent_b']
        
        # Add hydrogens
        synformer_child_mol = Chem.AddHs(synformer_child_mol)
        
        # Generate and optimize 3 conformers
        conformers = []
        try:
            # Remove existing conformers
            while synformer_child_mol.GetNumConformers() > 0:
                synformer_child_mol.RemoveConformer(0)
                
            # Generate new conformers
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 0
            
            conformer_ids = AllChem.EmbedMultipleConfs(
                synformer_child_mol,
                numConfs=3,
                params=params
            )
            
            if len(conformer_ids) == 0:
                print(f"Failed to generate any conformers for {synformer_smiles}")
                # Return better parent
                if parent_a_score <= parent_b_score:
                    with open(log_file, "a") as f:
                        f.write("\n=== Conformer Generation Failed ===\n")
                        f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                        f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                        f.write(f"Original Child: {original_child_smiles}\n")
                        f.write(f"Synformer Child: {synformer_smiles}\n")
                        f.write(f"Selected: PARENT A\n")
                        f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                    return child_info['parent_a']
                else:
                    with open(log_file, "a") as f:
                        f.write("\n=== Conformer Generation Failed ===\n")
                        f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                        f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                        f.write(f"Original Child: {original_child_smiles}\n")
                        f.write(f"Synformer Child: {synformer_smiles}\n")
                        f.write(f"Selected: PARENT B\n")
                        f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                    return child_info['parent_b']
            
            # Optimize each conformer
            for conf_id in conformer_ids:
                # Create a copy of the molecule with just this conformer
                conf_mol = Chem.Mol(synformer_child_mol)
                conf_mol.RemoveAllConformers()
                conf_mol.AddConformer(synformer_child_mol.GetConformer(conf_id))
                
                # Optimize geometry
                props = AllChem.MMFFGetMoleculeProperties(conf_mol)
                ff = AllChem.MMFFGetMoleculeForceField(conf_mol, props)
                if ff:
                    ff.Minimize(maxIts=200)
                    conformers.append(conf_mol)
            
            if len(conformers) == 0:
                print(f"Failed to optimize any conformers for {synformer_smiles}")
                # Return better parent
                if parent_a_score <= parent_b_score:
                    with open(log_file, "a") as f:
                        f.write("\n=== Conformer Optimization Failed ===\n")
                        f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                        f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                        f.write(f"Original Child: {original_child_smiles}\n")
                        f.write(f"Synformer Child: {synformer_smiles}\n")
                        f.write(f"Selected: PARENT A\n")
                        f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                    return child_info['parent_a']
                else:
                    with open(log_file, "a") as f:
                        f.write("\n=== Conformer Optimization Failed ===\n")
                        f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                        f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                        f.write(f"Original Child: {original_child_smiles}\n")
                        f.write(f"Synformer Child: {synformer_smiles}\n")
                        f.write(f"Selected: PARENT B\n")
                        f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                    return child_info['parent_b']
            
            # Dock each conformer
            best_score = float('inf')
            best_conformer = None
            
            for conf_mol in conformers:
                score = cs_dock_molecule(conf_mol, receptor_path, center, box_size, qvina_path, obabel_path)
                
                if score is not None and score < best_score:
                    best_score = score
                    best_conformer = conf_mol
            
            if best_conformer is not None:
                # Log the results
                with open(log_file, "a") as f:
                    f.write("\n=== Crossover Result ===\n")
                    f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                    f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                    f.write(f"Original Child: {original_child_smiles}\n")
                    f.write(f"Synformer Child: {synformer_smiles}\n")
                    f.write(f"Child Docking Score: {best_score:.2f} kcal/mol\n")
                    
                    # If child improves on both parents, return it
                    if best_score < parent_a_score and best_score < parent_b_score:
                        print(f"Synformer child improves on both parents: {best_score:.2f} vs {parent_a_score:.2f}/{parent_b_score:.2f}")
                        f.write(f"Selected: NEW SYNFORMER CHILD\n")
                        f.write(f"SMILES: {Chem.MolToSmiles(best_conformer)}\n")
                        f.write(f"Score: {best_score:.2f} kcal/mol\n")
                        f.write(f"Improvement vs Parent A: {parent_a_score - best_score:.2f} kcal/mol\n")
                        f.write(f"Improvement vs Parent B: {parent_b_score - best_score:.2f} kcal/mol\n")
                        return mu.mutate(best_conformer, mutation_rate)
                    else:
                        # Otherwise return the better parent
                        if parent_a_score <= parent_b_score:
                            print(f"Synformer child does not improve on both parents, keeping Parent A: {parent_a_score:.2f}")
                            f.write(f"Selected: PARENT A\n")
                            f.write(f"SMILES: {parent_a_smiles}\n")
                            f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                            return child_info['parent_a']
                        else:
                            print(f"Synformer child does not improve on both parents, keeping Parent B: {parent_b_score:.2f}")
                            f.write(f"Selected: PARENT B\n")
                            f.write(f"SMILES: {parent_b_smiles}\n")
                            f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                            return child_info['parent_b']
            else:
                print(f"Failed to dock any conformers for {synformer_smiles}")
                # Return better parent
                if parent_a_score <= parent_b_score:
                    with open(log_file, "a") as f:
                        f.write("\n=== Docking Failed ===\n")
                        f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                        f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                        f.write(f"Original Child: {original_child_smiles}\n")
                        f.write(f"Synformer Child: {synformer_smiles}\n")
                        f.write(f"Selected: PARENT A\n")
                        f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                    return child_info['parent_a']
                else:
                    with open(log_file, "a") as f:
                        f.write("\n=== Docking Failed ===\n")
                        f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                        f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                        f.write(f"Original Child: {original_child_smiles}\n")
                        f.write(f"Synformer Child: {synformer_smiles}\n")
                        f.write(f"Selected: PARENT B\n")
                        f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                    return child_info['parent_b']
        except Exception as e:
            print(f"Error processing conformers: {str(e)}")
            # Return better parent on error
            if parent_a_score <= parent_b_score:
                with open(log_file, "a") as f:
                    f.write("\n=== Processing Error ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                    f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                    f.write(f"Original Child: {original_child_smiles}\n")
                    f.write(f"Synformer Child: {synformer_smiles}\n")
                    f.write(f"Selected: PARENT A\n")
                    f.write(f"Score: {parent_a_score:.2f} kcal/mol\n")
                return child_info['parent_a']
            else:
                with open(log_file, "a") as f:
                    f.write("\n=== Processing Error ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Parent A: {parent_a_smiles} ({parent_a_score:.2f} kcal/mol)\n")
                    f.write(f"Parent B: {parent_b_smiles} ({parent_b_score:.2f} kcal/mol)\n")
                    f.write(f"Original Child: {original_child_smiles}\n")
                    f.write(f"Synformer Child: {synformer_smiles}\n")
                    f.write(f"Selected: PARENT B\n")
                    f.write(f"Score: {parent_b_score:.2f} kcal/mol\n")
                return child_info['parent_b']
    except Exception as e:
        print(f"Error in process_synformer_child: {str(e)}")
        # In case of error, try to return better parent if we have scores
        try:
            if parent_a_score <= parent_b_score:
                return child_info['parent_a']
            else:
                return child_info['parent_b']
        except:
            # If all else fails, return parent_a
            return child_info['parent_a']

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os
    from datetime import datetime

    # Initialize crossover log file with timestamp
    log_file = "crossover_log.txt"
    with open(log_file, "w") as f:
        f.write(f"=== Crossover Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    # Ensure log file is writable
    if not os.access(log_file, os.W_OK):
        print(f"Warning: {log_file} is not writable")
        try:
            os.chmod(log_file, 0o666)
            print(f"Changed permissions of {log_file} to be writable")
        except Exception as e:
            print(f"Error making log file writable: {str(e)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=str, default="qed")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default="initial_population.pkl",
                       help="File to load initial population from")
    parser.add_argument("--save_checkpoint", type=str, default="initial_population.pkl",
                       help="File to save initial population to")
    parser.add_argument("--synformer_checkpoint", type=str, default="synformer_population.pkl",
                       help="File to save/load Synformer-processed population")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of children to process in a batch")
    parser.add_argument("--n_jobs", type=int, default=4,
                       help="Number of parallel jobs for docking")
    args = parser.parse_args()

    config = {
        "population_size": 100,
        "offspring_size": 100,
        "mutation_rate": 0.1,
    }

    oracle = Oracle()
    oracle.assign_evaluator(tdc.Oracle(name=args.oracle))

    # Set up paths
    receptor_path = "/home/luost_local/sdivita/synformer/experiments/sbdd/receptor.pdbqt"
    center = [-9.845024108886719, -4.321293354034424, 39.35286331176758]
    box_size = [11.208, 9.997, 14.994]

    print("\n=== STEP 1: INITIAL POPULATION SETUP ===")
    # Try to load initial population from checkpoint
    initial_population_results = None
    if os.path.exists(args.load_checkpoint):
        print(f"Loading initial population from {args.load_checkpoint}")
        try:
            initial_population_results = load_population_results(args.load_checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            initial_population_results = None

    # If loading failed or file doesn't exist, process initial population
    if initial_population_results is None:
        print("Processing initial population from scratch...")
        # Load ADRB2 dataset
        df = pd.read_csv("/home/luost_local/sdivita/synformer/experiments/sbdd/pocket2mol.csv")
        adrb2_smiles = df[df['receptor'] == 'ADRB2']['smiles'].tolist()[:100]  # Take first 100 SMILES
        print(f"Loaded {len(adrb2_smiles)} ADRB2 molecules as starting population")
        
        # Process initial population
        initial_population_results = process_initial_population(
            adrb2_smiles,
            receptor_path,
            center,
            box_size
        )
        
        # Save results to checkpoint
        if args.save_checkpoint:
            save_population_results(initial_population_results, args.save_checkpoint)
    
    # Use processed molecules as starting population
    population_mol = [result[0] for result in initial_population_results]  # Get molecules
    population_scores = [result[1] for result in initial_population_results]  # Get docking scores
    
    print(f"Initial population size: {len(population_mol)}")

    # STEP 2: SYNFORMER PROJECTION OF INITIAL POPULATION
    print("\n=== STEP 2: SYNFORMER PROJECTION OF INITIAL POPULATION ===")
    synformer_results = None
    
    # Try to load Synformer results from checkpoint
    if os.path.exists(args.synformer_checkpoint):
        print(f"Loading Synformer-processed population from {args.synformer_checkpoint}")
        try:
            synformer_results = load_population_results(args.synformer_checkpoint)
        except Exception as e:
            print(f"Error loading Synformer checkpoint: {str(e)}")
            synformer_results = None
    
    # If loading failed or file doesn't exist, process with Synformer
    if synformer_results is None:
        print("Processing initial population with Synformer...")
        # Project through Synformer
        population_smiles = [Chem.MolToSmiles(mol) for mol in population_mol]
        print("Running Synformer on initial population...")
        synformer_smiles = projection(population_smiles)
        
        # Process Synformer-generated molecules
        print("\nProcessing Synformer-generated molecules...")
        synformer_results = process_synformer_molecules(
            synformer_smiles,
            receptor_path,
            center,
            box_size
        )
        
        # Save Synformer results to checkpoint
        if args.synformer_checkpoint:
            save_population_results(synformer_results, args.synformer_checkpoint)
    
    # Update population with processed Synformer molecules
    population_mol = [result[0] for result in synformer_results]  # Get molecules with best conformers
    population_scores = [result[1] for result in synformer_results]  # Get best docking scores
    
    # Create a lookup dictionary for molecule scores
    molecule_score_lookup = {Chem.MolToSmiles(result[0]): result[1] for result in synformer_results}

    print(f"Synformer population size: {len(population_mol)}")

    # STEP 3: GENETIC ALGORITHM WITH DOCKING-GUIDED CROSSOVER
    print("\n=== STEP 3: GENETIC ALGORITHM WITH DOCKING-GUIDED CROSSOVER ===")
    patience = 0
    generation = 0

    while True:
        generation += 1
        print(f"\n=== Generation {generation} ===")
        
        # Log generation to crossover_log.txt
        with open("crossover_log.txt", "a") as f:
            f.write(f"\n\n=== Starting Generation {generation} ===\n")
        
        if len(oracle) > 100:
            oracle.sort_buffer()
            old_score = np.mean([item[1][0] for item in list(oracle.mol_buffer.items())[:100]])
        else:
            old_score = 0

        # STEP 3.1: IMPROVED BATCH DOCKING-GUIDED CROSSOVER
        print("\nPerforming optimized batch crossover...")
        mating_pool = make_mating_pool(population_mol, population_scores, config["population_size"])
        
        # Step 1: Generate all child molecules in parallel
        print(f"Generating {config['offspring_size']} child molecules in parallel...")
        parent_pairs = [(random.choice(mating_pool), random.choice(mating_pool)) 
                       for _ in range(config["offspring_size"])]
        
        # First get scores for all parent molecules
        print("Getting parent molecule scores...")
        all_parents = []
        for parent_a, parent_b in parent_pairs:
            if parent_a not in all_parents:
                all_parents.append(parent_a)
            if parent_b not in all_parents:
                all_parents.append(parent_b)
        
        parent_smiles_to_mol = {Chem.MolToSmiles(mol): mol for mol in all_parents}
        
        # First try to get scores from lookup, only dock new molecules
        parent_dock_scores = {}
        new_parents_to_dock = []
        
        print("Looking up cached docking scores...")
        for mol in all_parents:
            smiles = Chem.MolToSmiles(mol)
            if smiles in molecule_score_lookup:
                parent_dock_scores[smiles] = molecule_score_lookup[smiles]
            else:
                new_parents_to_dock.append(mol)
        
        if new_parents_to_dock:
            print(f"Docking {len(new_parents_to_dock)} new parent molecules...")
            new_parent_scores = Parallel(n_jobs=args.n_jobs)(
                delayed(cs_dock_molecule)(
                    mol, receptor_path, center, box_size
                ) for mol in tqdm(new_parents_to_dock, desc="Docking new parents")
            )
            
            # Add new scores to both dictionaries
            for mol, score in zip(new_parents_to_dock, new_parent_scores):
                if score is not None:
                    smiles = Chem.MolToSmiles(mol)
                    parent_dock_scores[smiles] = score
                    molecule_score_lookup[smiles] = score
        else:
            print("All parent molecules have cached docking scores")
        
        # Generate all children in parallel with progress bar
        print(f"Generating child molecules...")
        child_mols = Parallel(n_jobs=args.n_jobs)(
            delayed(generate_child_with_parents)(parent_a, parent_b, receptor_path, center, box_size)
            for parent_a, parent_b in tqdm(parent_pairs, desc="Generating children")
        )
        
        # Filter out None values and get parent information
        valid_children = []
        for i, child_info in enumerate(child_mols):
            if child_info['success']:
                valid_children.append(child_info)
        
        print(f"Successfully generated {len(valid_children)} valid children")
        
        # Step 2: Run Synformer on all children at once
        if valid_children:
            print(f"Running Synformer on all {len(valid_children)} valid children at once...")
            all_child_smiles = [child_info['child_smiles'] for child_info in valid_children]
            
            # Add progress display for Synformer
            with tqdm(total=1, desc="Synformer projection") as pbar:
                synformer_smiles_list = projection(all_child_smiles)
                pbar.update(1)
            
            # Match Synformer results with children
            if len(synformer_smiles_list) != len(valid_children):
                print(f"Warning: Synformer returned {len(synformer_smiles_list)} results for {len(valid_children)} children")
                # If we got fewer results than expected, use only those we got
                valid_children = valid_children[:len(synformer_smiles_list)]
            
            # Step 3: Process each Synformer-optimized child in parallel
            print("Processing all Synformer-optimized children in parallel...")
            child_results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_synformer_child)(
                    child_info, 
                    synformer_smiles,
                    receptor_path,
                    center, 
                    box_size,
                    config["mutation_rate"]
                ) for child_info, synformer_smiles in tqdm(zip(valid_children, synformer_smiles_list), 
                                                          total=len(valid_children), 
                                                          desc="Processing Synformer children")
            )
            
            # Filter out None values
            offspring_mol = [mol for mol in child_results if mol is not None]
            
            # For any failed children, use the better parent
            for i, result in enumerate(child_results):
                if result is None and i < len(valid_children):
                    child_info = valid_children[i]
                    if child_info['parent_a_score'] <= child_info['parent_b_score']:
                        offspring_mol.append(child_info['parent_a'])
                    else:
                        offspring_mol.append(child_info['parent_b'])
        else:
            # If no valid children were generated, just add parents to offspring
            print("No valid children generated, using parents")
            offspring_mol = []
            for parent_a, parent_b in parent_pairs:
                parent_a_smiles = Chem.MolToSmiles(parent_a)
                parent_b_smiles = Chem.MolToSmiles(parent_b)
                
                parent_a_score = parent_dock_scores.get(parent_a_smiles)
                parent_b_score = parent_dock_scores.get(parent_b_smiles)
                
                if parent_a_score is not None and parent_b_score is not None:
                    if parent_a_score <= parent_b_score:
                        offspring_mol.append(parent_a)
                    else:
                        offspring_mol.append(parent_b)
                elif parent_a_score is not None:
                    offspring_mol.append(parent_a)
                elif parent_b_score is not None:
                    offspring_mol.append(parent_b)
                else:
                    # If both parents failed to dock, just add parent_a
                    offspring_mol.append(parent_a)

        # STEP 3.4: EVALUATE AND SELECT
        print("\nEvaluating population with Oracle...")
        # Score with Oracle (QED)
        oracle_scores = oracle([Chem.MolToSmiles(mol) for mol in tqdm(population_mol, desc="Evaluating QED")])
        
        # Get docking scores for molecules
        docking_scores = []
        for mol in tqdm(population_mol, desc="Collecting docking scores"):
            smiles = Chem.MolToSmiles(mol)
            # Check if we already have a docking score
            found = False
            for result in synformer_results:
                if Chem.MolToSmiles(result[0]) == smiles:
                    docking_scores.append(result[1])
                    found = True
                    break
            
            # If not found, use a high (bad) score
            if not found:
                docking_scores.append(100.0)
        
        # Combine scores and molecules, sort by Oracle score (QED)
        population_tuples = list(zip(oracle_scores, population_mol, docking_scores))
        population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
        
        # Unpack the sorted tuples
        oracle_scores = [t[0] for t in population_tuples]
        population_mol = [t[1] for t in population_tuples]
        population_scores = [t[2] for t in population_tuples]  # These are docking scores

        # STEP 3.5: PRINT STATISTICS
        print("\nGeneration Statistics:")
        print(f"Best QED score: {max(oracle_scores):.3f}")
        print(f"Average QED score: {np.mean(oracle_scores):.3f}")
        print(f"Best docking score: {min(population_scores):.2f} kcal/mol")
        print(f"Average docking score: {np.mean(population_scores):.2f} kcal/mol")
        print(f"Population size: {len(population_mol)}")
        
        # Print scores for molecule with best QED
        best_qed_idx = oracle_scores.index(max(oracle_scores))
        print(f"\nMolecule with best QED:")
        print(f"SMILES: {Chem.MolToSmiles(population_mol[best_qed_idx])}")
        print(f"QED score: {oracle_scores[best_qed_idx]:.3f}")
        print(f"Docking score: {population_scores[best_qed_idx]:.2f} kcal/mol")

        # Print scores for molecule with best docking
        best_dock_idx = population_scores.index(min(population_scores))  # min because lower docking score is better
        print(f"\nMolecule with best docking score:")
        print(f"SMILES: {Chem.MolToSmiles(population_mol[best_dock_idx])}")
        print(f"Docking score: {population_scores[best_dock_idx]:.2f} kcal/mol")
        print(f"QED score: {oracle_scores[best_dock_idx]:.3f}")

        # STEP 3.6: CHECK STOPPING CRITERIA
        if len(oracle) > 100:
            oracle.sort_buffer()
            new_score = np.mean([item[1][0] for item in list(oracle.mol_buffer.items())[:100]])
            if (new_score - old_score) < 1e-3:
                patience += 1
                if patience >= 5:
                    oracle.log_intermediate(finish=True)
                    print("Convergence criteria met, stopping...")
                    
                    # Get the best molecule by QED score
                    oracle.sort_buffer()
                    best_mol_smiles = list(oracle.mol_buffer.keys())[0]  # First molecule has highest QED
                    best_mol = Chem.MolFromSmiles(best_mol_smiles)
                    best_qed = list(oracle.mol_buffer.values())[0][0]  # First value is QED score
                    
                    # Dock against ADRB2
                    print("\n=== Docking Best QED Molecule Against ADRB2 ===")
                    print(f"Best molecule SMILES: {best_mol_smiles}")
                    print(f"QED score: {best_qed:.3f}")
                    
                    docking_score = cs_dock_molecule(best_mol)
                    if docking_score is not None:
                        print(f"ADRB2 docking score: {docking_score:.2f} kcal/mol")
                    else:
                        print("Failed to dock molecule against ADRB2")
                    
                    # Print summary of best performing child from last generation
                    print("\n=== Final Generation Summary ===")
                    best_child_score = min(population_scores)  # Lower docking score is better
                    best_child_idx = population_scores.index(best_child_score)
                    best_child_mol = population_mol[best_child_idx]
                    best_child_smiles = Chem.MolToSmiles(best_child_mol)
                    print(f"Best child molecule SMILES: {best_child_smiles}")
                    print(f"Best child docking score: {best_child_score:.2f} kcal/mol")
                    
                    break
            else:
                patience = 0

            old_score = new_score

        if oracle.finish:
            break

        oracle.save_result(suffix=args.name)