import os
import torch
import pickle
import csv # Added for CSV output
import numpy as np # Added for run_desert_inference
import torch.nn as nn # Added for load_desert_model
import argparse
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED # Added for QED calculation
import math # Added for get_rotation_bins
import subprocess # Added for docking
import tempfile # Added for docking
import uuid # Added for docking
import shutil # Added for docking
import pandas as pd # Added for reading receptor centers CSV

# Attempt to import run_desert_inference. 
# This assumes 'synformer' is in PYTHONPATH or installed.
try:
    from synformer.models.desert.inference import run_desert_inference
except ImportError:
    print("Error: Could not import 'run_desert_inference' from 'synformer.models.desert.inference'.")
    print("Please ensure the 'synformer' package is correctly installed and in your PYTHONPATH.")
    exit(1)

# --- BEGIN: Imports from synformer/models/desert/inference.py ---
try:
    from synformer.data.common import process_smiles 
    from synformer.models.desert.fulldecoder import ShapePretrainingDecoderIterativeNoRegression
    from synformer.data.utils.shape_utils import bin_to_grid_coords, grid_coords_to_real_coords, bin_to_rotation_mat
except ImportError as e:
    print(f"Error importing Synformer modules: {e}")
    print("Please ensure Synformer is installed and in your PYTHONPATH.")
    exit(1)
# --- END: Imports from synformer/models/desert/inference.py ---

# --- Configuration ---
# The script will expect these paths to be set by the user or via command-line arguments.
# Default shape patches directory as requested by the user.
DEFAULT_SHAPES_BASE_DIR = "/workspace/data/TestCrossDocked2020/sample_shapes_ablation_study/" 
DEFAULT_DESERT_MODEL_PATH = "/workspace/data/desert/1WW_30W_5048064.pt"  # Placeholder - MUST be provided by user
DEFAULT_VOCAB_PATH = "/workspace/data/desert/vocab.pkl"         # Placeholder - MUST be provided by user
OUTPUT_CSV_FILE = "/workspace/fragments_qed_docking.csv" # Updated CSV filename
DEFAULT_SMILES_STRING = "C"      # Placeholder - MUST be provided by user
ROTATION_BIN_PATH = "/workspace/data/desert/rotation_bin.pkl" # Placeholder - User MUST provide path to rotation_bin.pkl
# --- NEW Docking Configuration ---
# DEFAULT_RECEPTOR_PATH = "/path/to/your/receptor.pdbqt" # Placeholder - USER MUST UPDATE <-- REMOVE
# DEFAULT_RECEPTOR_CENTER = [0.0, 0.0, 0.0] # Placeholder - USER MUST UPDATE [x, y, z] <-- REMOVE
RECEPTOR_INFO_CSV_PATH = "/workspace/data/TestCrossDocked2020/receptors/test.csv"
RECEPTORS_DIR_PATH = "/workspace/data/TestCrossDocked2020/receptors/"
# --- End Configuration ---

# --- BEGIN: rotation_matrix function (copied from synformer/data/utils/tfbio_data.py) ---
# This function is a dependency for get_rotation_bins_local
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
# --- END: rotation_matrix function ---

# --- BEGIN: get_rotation_bins_local function (copied from synformer/models/desert/shape_utils.py) ---
# Depends on rotation_matrix (copied above) and math.pi, numpy
def get_rotation_bins_local(sp, rp):
    mid = sp // 2
    sr = 1.0 / sp

    face1 = []
    for y in range(sp):
        for z in range(sp):
            face1.append(np.array([0.5, (y - mid) * sr, (z - mid) * sr]))
    face2 = []
    for x in range(sp):
        for y in range(sp):
            face2.append(np.array([(x - mid) * sr, (y - mid) * sr, 0.5]))
    face3 = []
    for x in range(sp):
        for z in range(sp):
            face3.append(np.array([(x - mid) * sr, 0.5, (z - mid) * sr]))
    
    face_point = face1 + face2 + face3
    
    # Use the local rotation_matrix function
    rotation_mat_bin = [rotation_matrix(np.array((1, 1, 1)), 0)] 
    for p in face_point:
        for t in range(1, rp):
            axis = p
            theta = t * math.pi / (rp / 2) # Use math.pi
            rotation_mat_bin.append(rotation_matrix(axis, theta)) # Use local rotation_matrix
    rotation_mat_bin = np.stack(rotation_mat_bin, axis=0)

    return rotation_mat_bin
# --- END: get_rotation_bins_local function ---

# --- BEGIN: Docking functions (copied from excellent_checkadaptertraining.py) ---
import dataclasses # Ensure dataclasses is imported for QVinaOption

@dataclasses.dataclass
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

def prepare_ligand_pdbqt_local(mol, obabel_path="obabel"): # Renamed
    """
    Prepare a ligand for docking by converting it to PDBQT format using OpenBabel
    """
    # ... (Contents of prepare_ligand_pdbqt from excellent_checkadaptertraining.py) ...
    # Create unique filenames with absolute paths in the system temp directory
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()
    temp_mol_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.mol")
    temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.pdbqt")

    try:
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

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False # Important: check=False to handle errors manually
        )

        if process.returncode != 0:
            print(f"Error converting molecule to PDBQT: {process.stderr}")
            return None

        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()

        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            print("Error: Generated PDBQT file does not contain valid atom entries")
            return None
            
        return pdbqt_content
    except Exception as e:
        print(f"Error preparing ligand: {str(e)}")
        return None
    finally:
        # Clean up temporary files
        for f_path in [temp_mol_file, temp_pdbqt_file]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                    print(f"Warning: Could not remove temporary file {f_path}")


def dock_best_molecule_local(mol, receptor_path, receptor_center): # Renamed
    """Dock the molecule against receptor target"""
    # ... (Contents of dock_best_molecule from excellent_checkadaptertraining.py, 
    #      making sure to call prepare_ligand_pdbqt_local) ...
    temp_ligand_file = None
    output_file = None
    try:
        smiles = Chem.MolToSmiles(mol)
        print(f"  Attempting docking for: {smiles}")

        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        
        center = receptor_center
        box_size = [22.5, 22.5, 22.5]

        qvina_path = shutil.which("qvina2.1")
        if qvina_path is None:
            local_qvina_path = os.path.join(os.getcwd(), "bin", "qvina2.1") # Check ./bin/ relative to script
            if os.path.exists(local_qvina_path):
                 qvina_path = local_qvina_path
            else:
                # Fallback to a common Synformer path
                user_specific_qvina_path = "/workspace/synformer/bin/qvina2.1" 
                if os.path.exists(user_specific_qvina_path):
                    qvina_path = user_specific_qvina_path
                else:
                    print("    Error: QVina2 executable (qvina2.1) not found in PATH, ./bin/, or /workspace/synformer/bin/.")
                    return None
        
        obabel_path = shutil.which("obabel")
        if obabel_path is None:
            print("    Error: OpenBabel (obabel) not found in PATH.")
            return None
            
        if not os.path.exists(receptor_path):
            print(f"    Error: Receptor file not found at {receptor_path}")
            return None
            
        ligand_pdbqt = prepare_ligand_pdbqt_local(mol, obabel_path) # Call renamed helper
        if ligand_pdbqt is None:
            print("    Failed to prepare ligand for docking.")
            return None
            
        temp_ligand_file = os.path.join(temp_dir, f"temp_ligand_dock_{unique_id}.pdbqt")
        with open(temp_ligand_file, "w") as f:
            f.write(ligand_pdbqt)
            
        options = QVinaOption(
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=box_size[0], size_y=box_size[1], size_z=box_size[2]
        )
        
        output_file = os.path.join(temp_dir, f"temp_ligand_dock_out_{unique_id}.pdbqt")
        cmd = [
            qvina_path, "--receptor", receptor_path, "--ligand", temp_ligand_file,
            "--center_x", str(options.center_x), "--center_y", str(options.center_y), "--center_z", str(options.center_z),
            "--size_x", str(options.size_x), "--size_y", str(options.size_y), "--size_z", str(options.size_z),
            "--exhaustiveness", str(options.exhaustiveness), "--num_modes", str(options.num_modes),
            "--out", output_file
        ]
        
        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300, check=False
        )
        
        if process.returncode != 0:
            print(f"    QVina2 execution error for {smiles}: {process.stderr}")
            # return None # Don't return None immediately, try to parse if output file exists

        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line:
                        try:
                            score = float(line.split()[3])
                            print(f"    Docking score for {smiles}: {score}")
                            break
                        except (IndexError, ValueError):
                            pass
        if score is None and process.returncode != 0 : # If score is still None AND qvina failed
             print(f"    No docking score found and QVina2 reported an error for {smiles}.")
             return None
        elif score is None:
             print(f"    No docking score found in output for {smiles}, but QVina2 ran without error code.")


        return score
        
    except subprocess.TimeoutExpired:
        print(f"    Docking timed out for {smiles}")
        return None
    except Exception as e:
        print(f"    Error during docking for {smiles}: {str(e)}")
        return None
    finally:
        for f_path in [temp_ligand_file, output_file]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                     print(f"    Warning: Could not remove temporary file {f_path}")

# --- END: Docking functions ---

# --- BEGIN: Globals and Functions copied/adapted from shape_utils.py ---
# Global variables for shape utilities
SHAPE_UTILS_VOCAB = None
SHAPE_UTILS_VOCAB_R = None
SHAPE_UTILS_ROTATION_BIN = None

def shape_utils_trans(x, y, z): # Copied from shape_utils.py (used by get_3d_frags indirectly)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [x, y, z]
    return translation_matrix

# bin_to_grid_coords, grid_coords_to_real_coords, bin_to_rotation_mat are already imported
# from synformer.data.utils.shape_utils

# Renamed from get_3d_frags_local to avoid confusion with the one in shape_utils if it were also present
# This version uses the global SHAPE_UTILS_VOCAB etc. defined in this script.
def _get_3d_frags_local(frags_sequence):
    ret_frags = []
    if SHAPE_UTILS_VOCAB is None or SHAPE_UTILS_VOCAB_R is None or SHAPE_UTILS_ROTATION_BIN is None:
        print("Error: Vocab or rotation_bin for _get_3d_frags_local not initialized."); return ret_frags
        
    for unit in frags_sequence:
        idx, tr_bin, rm_bin = unit
        key = SHAPE_UTILS_VOCAB_R.get(idx)
        if key is None or key in ['UNK', 'BOS', 'BOB', 'EOB', 'PAD', 'EOS']:
            if key == 'EOS': break # Stop processing at EOS for this sequence
            continue

        frag_mol_template_container = SHAPE_UTILS_VOCAB.get(key)
        frag_mol_template = None
        if frag_mol_template_container:
            if hasattr(frag_mol_template_container, 'GetConformer'): # Direct Mol object
                frag_mol_template = frag_mol_template_container
            elif isinstance(frag_mol_template_container, (list, tuple)) and len(frag_mol_template_container) > 0 and hasattr(frag_mol_template_container[0], 'GetConformer'):
                frag_mol_template = frag_mol_template_container[0] # Mol object is the first element
        
        if frag_mol_template is None:
            # print(f"Warning: Fragment template for key '{key}' (ID: {idx}) not found or not a Mol object."); 
            continue

        frag = copy.deepcopy(frag_mol_template)
        conformer = frag.GetConformer() 
        if conformer is None: 
            AllChem.Compute2DCoords(frag)
            if AllChem.EmbedMolecule(frag, AllChem.ETKDG()) == -1: 
                 continue
            conformer = frag.GetConformer()
            if conformer is None: continue 

        grid_coords = bin_to_grid_coords(tr_bin, 28)
        tr_real_coords = grid_coords_to_real_coords(grid_coords, 28, 0.5)
        rm_matrix = bin_to_rotation_mat(rm_bin, SHAPE_UTILS_ROTATION_BIN)

        if tr_real_coords is None or rm_matrix is None:
            continue

        transform_matrix_rot = np.eye(4)
        transform_matrix_rot[:3,:3] = rm_matrix
        Chem.rdMolTransforms.TransformConformer(conformer, transform_matrix_rot)

        transform_matrix_trans = np.eye(4)
        transform_matrix_trans[0,3] = tr_real_coords[0]
        transform_matrix_trans[1,3] = tr_real_coords[1]
        transform_matrix_trans[2,3] = tr_real_coords[2]
        Chem.rdMolTransforms.TransformConformer(conformer, transform_matrix_trans)
        
        ret_frags.append(frag)
    return ret_frags

# Renamed from get_star_info_local to avoid conflict and indicate it's a robust helper
def _get_star_info_robust_local(current_frags): # Was get_star_info_local
    star_info_list = []
    for f_idx, frag_mol in enumerate(current_frags):
        if frag_mol is None: continue 
        con = frag_mol.GetConformer()
        if con is None: continue 
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbours = atom.GetNeighbors()
                if not neighbours: continue
                assert len(neighbours) == 1, f"Star atom {atom.GetIdx()} in frag {f_idx} has {len(neighbours)} neighbors, expected 1"
                nei_idx = neighbours[0].GetIdx()
                atom_idx = atom.GetIdx()
                try:
                    atom_pos_rdk = con.GetAtomPosition(atom_idx)
                    nei_pos_rdk = con.GetAtomPosition(nei_idx)
                    atom_pos = np.array([atom_pos_rdk.x, atom_pos_rdk.y, atom_pos_rdk.z])
                    nei_pos = np.array([nei_pos_rdk.x, nei_pos_rdk.y, nei_pos_rdk.z])
                except RuntimeError: continue
                if np.isinf(atom_pos).any() or np.isinf(nei_pos).any(): continue
                star_info_list.append({
                    'f_idx': f_idx, 'atom_idx': atom_idx, 'nei_idx': nei_idx,
                    'atom_pos': atom_pos, 'nei_pos': nei_pos,
                    # Store the RDKit atom objects or unique identifiers for breakpoints if needed for failed_connections key
                    'breakpoint_id_self': (f_idx, atom_idx) # Unique ID for this breakpoint
                })
    return star_info_list

# Helper function for distance calculation
def distance(pos1, pos2): return np.sqrt(np.sum((pos1 - pos2)**2))

def connect_fragments_local(frags_list_input):
    if not frags_list_input: return None
    current_frags_list = [f for f in copy.deepcopy(frags_list_input) if f is not None]
    if not current_frags_list: return None

    # Assuming distance() is globally available or defined if not part of this scope before.
    # It was defined as a nested function previously, but Algorithm 2 implies it as a utility.
    # For this edit, we assume it's accessible. If not, it needs to be defined globally or passed.
    
    failed_connection_breakpoint_pairs = set()

    while True: 
        made_a_connection_in_this_pass = False
        if len(current_frags_list) <= 1:
            break 

        all_star_info = _get_star_info_robust_local(current_frags_list) # Uses the robust helper
        if len(all_star_info) <= 1:
            break 

        potential_connections = []
        for i in range(len(all_star_info)):
            for j in range(i + 1, len(all_star_info)):
                star1_data = all_star_info[i]
                star2_data = all_star_info[j]

                if star1_data['f_idx'] == star2_data['f_idx']:
                    continue

                bp_id1 = star1_data['breakpoint_id_self']
                bp_id2 = star2_data['breakpoint_id_self']
                current_pair_key = tuple(sorted((bp_id1, bp_id2)))

                if current_pair_key in failed_connection_breakpoint_pairs:
                    continue 
                # Assuming distance function is defined globally or accessible
                dist_val = distance(star1_data['atom_pos'], star2_data['nei_pos']) + \
                           distance(star1_data['nei_pos'], star2_data['atom_pos'])
                
                potential_connections.append({
                    'dist': dist_val,
                    'star1_data': star1_data,
                    'star2_data': star2_data,
                    'pair_key': current_pair_key
                })
        
        if not potential_connections:
            break 

        potential_connections.sort(key=lambda x: x['dist'])

        for attempt in potential_connections:
            star1_data_to_connect = attempt['star1_data']
            star2_data_to_connect = attempt['star2_data']
            
            # Robustness: ensure f_idx are valid for the current list size
            if not (star1_data_to_connect['f_idx'] < len(current_frags_list) and \
                    star2_data_to_connect['f_idx'] < len(current_frags_list)):
                print(f"    Warning: Stale fragment index during connection attempt. Skipping.")
                failed_connection_breakpoint_pairs.add(attempt['pair_key'])
                continue

            mol_obj1 = current_frags_list[star1_data_to_connect['f_idx']]
            mol_obj2 = current_frags_list[star2_data_to_connect['f_idx']]
            
            # Use the renamed _connectMols_helper_local
            connected_mol = _connectMols_helper_local(mol_obj1, mol_obj2, 
                                              star1_data_to_connect['atom_idx'], 
                                              star2_data_to_connect['atom_idx'])

            if connected_mol:
                print(f"    Successfully connected fragment index {star1_data_to_connect['f_idx']} with {star2_data_to_connect['f_idx']}")
                
                indices_to_remove_from_list = sorted([star1_data_to_connect['f_idx'], star2_data_to_connect['f_idx']], reverse=True)
                
                temp_list = []
                for k_idx, frag in enumerate(current_frags_list):
                    if k_idx not in indices_to_remove_from_list:
                        temp_list.append(frag)
                temp_list.append(connected_mol)
                current_frags_list = temp_list
                
                failed_connection_breakpoint_pairs.clear() 
                made_a_connection_in_this_pass = True
                break 
            else:
                print(f"    Connection failed between breakpoint {star1_data_to_connect['breakpoint_id_self']} and {star2_data_to_connect['breakpoint_id_self']}. Marking as failed for this pass.")
                failed_connection_breakpoint_pairs.add(attempt['pair_key'])
        
        if not made_a_connection_in_this_pass:
            break

    if not current_frags_list: return None
    if len(current_frags_list) > 1:
        print(f"  Multiple ({len(current_frags_list)}) disconnected components remain. Selecting the largest.")
        current_frags_list.sort(key=lambda m: m.GetNumAtoms() if m else 0, reverse=True)
    
    final_mol = current_frags_list[0]
    if final_mol is None: return None

    rw_final_mol = Chem.RWMol(final_mol)
    needs_another_pass = True
    passes = 0
    max_passes = 10

    while needs_another_pass and passes < max_passes:
        needs_another_pass = False
        passes += 1
        atoms_to_replace_this_pass = []
        for atom in rw_final_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atoms_to_replace_this_pass.append(atom.GetIdx())
        
        if atoms_to_replace_this_pass:
            for atom_idx in sorted(atoms_to_replace_this_pass, reverse=True):
                try:
                    atom_to_check = rw_final_mol.GetAtomWithIdx(atom_idx)
                    if atom_to_check and atom_to_check.GetAtomicNum() == 0:
                        rw_final_mol.ReplaceAtom(atom_idx, Chem.Atom(6)) 
                        needs_another_pass = True 
                except RuntimeError: 
                    pass 
            if needs_another_pass: 
                try:
                    Chem.SanitizeMol(rw_final_mol) 
                except Exception as e_sanitize_cap_pass:
                    print(f"    Warning: Sanitization during capping pass {passes} failed: {e_sanitize_cap_pass}.")
                    final_mol = rw_final_mol.GetMol() 
                    return final_mol 
        else:
            pass

    final_mol = rw_final_mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol) 
    except Exception as e_sanitize_final:
        print(f"    Warning: Final sanitization of capped molecule failed: {e_sanitize_final}")
        pass 
            
    return final_mol

# This is the renamed helper function, was connectMols_local
def _connectMols_helper_local(mol1, mol2, atom_idx_mol1_star, atom_idx_mol2_star):
    try:
        # Check if mol1 and mol2 are valid Mol objects
        if not mol1 or not mol2: return None

        # Get the atom in mol1 that is a neighbor of the dummy atom (atom_idx_mol1_star)
        dummy_atom_mol1 = mol1.GetAtomWithIdx(atom_idx_mol1_star)
        # Ensure it's a dummy atom (atomic number 0)
        if dummy_atom_mol1.GetAtomicNum() != 0:
            # print(f"Warning: Atom at {atom_idx_mol1_star} in mol1 is not a dummy atom.")
            return None
        neighbors_mol1 = dummy_atom_mol1.GetNeighbors()
        if not neighbors_mol1:
            # print(f"Warning: Dummy atom at {atom_idx_mol1_star} in mol1 has no neighbors.")
            return None
        attach_atom_idx_mol1 = neighbors_mol1[0].GetIdx()

        # Get the atom in mol2 that is a neighbor of the dummy atom (atom_idx_mol2_star)
        dummy_atom_mol2 = mol2.GetAtomWithIdx(atom_idx_mol2_star)
        # Ensure it's a dummy atom (atomic number 0)
        if dummy_atom_mol2.GetAtomicNum() != 0:
            # print(f"Warning: Atom at {atom_idx_mol2_star} in mol2 is not a dummy atom.")
            return None
        neighbors_mol2 = dummy_atom_mol2.GetNeighbors()
        if not neighbors_mol2:
            # print(f"Warning: Dummy atom at {atom_idx_mol2_star} in mol2 has no neighbors.")
            return None
        attach_atom_idx_mol2 = neighbors_mol2[0].GetIdx()

        # Create an editable version of mol1 (this will be our combined molecule)
        combo_mol = Chem.RWMol(mol1)

        # Add atoms from mol2 to combo_mol, keeping track of new indices
        mol2_atom_map = {} # Maps original mol2 atom index to new index in combo_mol
        for atom_mol2 in mol2.GetAtoms():
            new_idx = combo_mol.AddAtom(atom_mol2)
            mol2_atom_map[atom_mol2.GetIdx()] = new_idx
        
        # Add bonds from mol2 to combo_mol, using mapped atom indices
        for bond_mol2 in mol2.GetBonds():
            begin_atom_orig_idx = bond_mol2.GetBeginAtomIdx()
            end_atom_orig_idx = bond_mol2.GetEndAtomIdx()
            combo_mol.AddBond(mol2_atom_map[begin_atom_orig_idx], 
                              mol2_atom_map[end_atom_orig_idx], 
                              bond_mol2.GetBondType())

        # New index for the attachment atom from mol2 in combo_mol
        new_attach_atom_idx_mol2 = mol2_atom_map[attach_atom_idx_mol2]

        # Add a single bond between the attachment atom from mol1 (its index is unchanged)
        # and the new attachment atom index from mol2
        combo_mol.AddBond(attach_atom_idx_mol1, new_attach_atom_idx_mol2, Chem.BondType.SINGLE)
        
        # Identify the final indices of the dummy atoms in combo_mol
        # Dummy atom from mol1 keeps its original index: atom_idx_mol1_star
        # Dummy atom from mol2 gets its mapped index: mol2_atom_map[atom_idx_mol2_star]
        dummy1_final_idx = atom_idx_mol1_star
        dummy2_final_idx = mol2_atom_map[atom_idx_mol2_star]
        
        # Remove the dummy atoms. Must remove the one with the larger index first to avoid index shifts affecting the other.
        indices_to_remove = sorted([dummy1_final_idx, dummy2_final_idx], reverse=True)
        
        for idx_to_remove in indices_to_remove:
             # Sanity check before removal
            if idx_to_remove < combo_mol.GetNumAtoms() and combo_mol.GetAtomWithIdx(idx_to_remove).GetAtomicNum() == 0:
                combo_mol.RemoveAtom(idx_to_remove)
            else:
                # This indicates a logic error or unexpected input if a non-dummy or out-of-bounds atom is targeted
                # print(f"Warning: Problem removing dummy atom at index {idx_to_remove}. Atom may not exist or not be a dummy.")
                # Depending on desired robustness, could return None here or try to continue.
                # For now, if a dummy atom cannot be confirmed and removed, it's safer to indicate connection failure.
                return None

        # Get the final molecule from the RWMol
        result_mol = combo_mol.GetMol()
        
        # Sanitize the resulting molecule
        Chem.SanitizeMol(result_mol)
        return result_mol

    except Exception as e:
        # print(f"Error connecting molecules in _connectMols_helper_local: {e}")
        # For debugging: import traceback; traceback.print_exc()
        return None

# --- END: Copied and modified functions from shape_utils.py ---

# --- BEGIN: Copied and modified load_desert_model ---
def load_desert_model(model_path, vocab_path_for_model_load, device='cuda'):
    print(f"Loading DESERT model from {model_path}...")
    
    with open(vocab_path_for_model_load, 'rb') as f:
        model_vocab = pickle.load(f)
    
    src_vocab_size = len(model_vocab)
    tgt_vocab_size = len(model_vocab)
    pad_token_idx = model_vocab.get('PAD', [None, None, 0])[2]

    src_special_tokens = {'pad': pad_token_idx}
    tgt_special_tokens = {'pad': pad_token_idx}

    d_model = 1024
    tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_special_tokens['pad'])
    
    class CustomShapePretrainingModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, shape_patches, input_frag_idx, input_frag_idx_mask, 
                   input_frag_trans, input_frag_trans_mask, input_frag_r_mat, input_frag_r_mat_mask, 
                   memory=None, memory_padding_mask=None, **kwargs):
            if memory is None:
                fixed_seq_len_for_memory = 32 
                batch_size_for_memory = 1 
                current_device = next(self.decoder.parameters()).device
                memory = torch.zeros((fixed_seq_len_for_memory, batch_size_for_memory, d_model), device=current_device)
                memory_padding_mask = torch.zeros((batch_size_for_memory, fixed_seq_len_for_memory), dtype=torch.bool, device=current_device)
            
            logits, trans, r_mat = self.decoder(
                input_frag_idx=input_frag_idx,
                input_frag_trans=input_frag_trans,
                input_frag_r_mat=input_frag_r_mat,
                memory=memory,
                memory_padding_mask=memory_padding_mask
            )
            return logits, trans, r_mat
            
    class IterativeBlock:
        def __init__(self, config):
            self.config = config
            self._mode = 'train'
        def build(self, embed, special_tokens, trans_vocab_size, rot_vocab_size):
            self.embed = embed
            self.special_tokens = special_tokens
            self.trans_vocab_size = trans_vocab_size
            self.rot_vocab_size = rot_vocab_size
        def __call__(self, logits, trans, r_mat, padding_mask):
            return logits, trans, r_mat
        def reset(self, mode):
            self._mode = mode
            
    iterative_block = IterativeBlock({
        'num_layers': 3, 'd_model': 1024, 'n_head': 8, 'dim_feedforward': 4096,
        'dropout': 0.1, 'activation': 'relu', 'learn_pos': True
    })
    
    decoder = ShapePretrainingDecoderIterativeNoRegression(
        num_layers=12, d_model=1024, n_head=8, dim_feedforward=4096,
        dropout=0.1, activation='relu', learn_pos=True, iterative_num=1,
        max_dist=6.75, grid_resolution=0.5, rotation_bin_direction=11,
        rotation_bin_angle=24, iterative_block=iterative_block
    ).to(device)
    
    decoder.build(embed=tgt_embed.to(device), special_tokens=tgt_special_tokens, 
                  out_proj=nn.Linear(d_model, tgt_vocab_size, bias=False).to(device))

    model = CustomShapePretrainingModel(encoder=None, decoder=decoder).to(device)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model' in state_dict: 
        state_dict = state_dict['model']
    
    decoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_decoder.'): 
            decoder_key = k.replace('_decoder.', '')
            decoder_state_dict[decoder_key] = v
        elif k.startswith('decoder.'): 
            decoder_key = k.replace('decoder.', '')
            decoder_state_dict[decoder_key] = v
            
    if not decoder_state_dict and state_dict:
         print("Warning: Decoder state_dict might not be prefixed. Using all keys from checkpoint for decoder.")
         decoder_state_dict = state_dict

    try:
        model.decoder.load_state_dict(decoder_state_dict, strict=False)
    except RuntimeError as e:
        print(f"Error loading decoder state_dict: {e}")
        print("Keys in loaded checkpoint (after prefix removal attempt):", list(decoder_state_dict.keys()))
        print("Keys in model's decoder (model.decoder.state_dict()):", list(model.decoder.state_dict().keys()))
        raise
        
    model.eval()
    print("DESERT model loaded successfully.")
    return model
# --- END: Copied and modified load_desert_model ---

# --- BEGIN: Copied and modified run_desert_inference ---
def run_desert_inference_local(
    loaded_model, 
    bos_token_id, 
    eos_token_id, 
    shape_patches_path, 
    device, 
    smiles_str, 
    max_length=50 
    ):
    
    if not os.path.exists(shape_patches_path):
        raise FileNotFoundError(f"Shape patches file not found: {shape_patches_path}")
        
    with open(shape_patches_path, 'rb') as f:
        shape_data = pickle.load(f)
        if isinstance(shape_data, (list, tuple)) and len(shape_data) > 0:
            shape_patches_raw = shape_data[0]
        else:
            shape_patches_raw = shape_data

    if isinstance(shape_patches_raw, np.ndarray):
        shape_patches = torch.tensor(shape_patches_raw, dtype=torch.float, device=device)
    elif torch.is_tensor(shape_patches_raw):
        shape_patches = shape_patches_raw.float().to(device)
    else:
        raise TypeError(f"Shape patches from {shape_patches_path} are of unexpected type: {type(shape_patches_raw)}")

    if shape_patches.ndim == 2:
        shape_patches = shape_patches.unsqueeze(0)

    batch_size = 1
    input_frag_idx = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    input_frag_trans = torch.zeros((batch_size, 1), dtype=torch.long, device=device) 
    input_frag_r_mat = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    all_pred_idx = [bos_token_id]
    all_pred_trans = [0]
    all_pred_rot = [0]
    
    d_model_for_memory = loaded_model.decoder.d_model if hasattr(loaded_model, 'decoder') and hasattr(loaded_model.decoder, 'd_model') else 1024
    seq_len_for_memory = 32 
    
    memory = torch.zeros((seq_len_for_memory, batch_size, d_model_for_memory), device=device)
    memory_padding_mask = torch.zeros((batch_size, seq_len_for_memory), dtype=torch.bool, device=device)
    
    for step in range(max_length):
        input_frag_idx_mask = torch.zeros_like(input_frag_idx, dtype=torch.bool, device=device)
        input_frag_trans_mask = torch.zeros_like(input_frag_trans, dtype=torch.bool, device=device)
        input_frag_r_mat_mask = torch.zeros_like(input_frag_r_mat, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            outputs = loaded_model(
                shape_patches=shape_patches, 
                input_frag_idx=input_frag_idx,
                input_frag_idx_mask=input_frag_idx_mask,
                input_frag_trans=input_frag_trans,
                input_frag_trans_mask=input_frag_trans_mask,
                input_frag_r_mat=input_frag_r_mat,
                input_frag_r_mat_mask=input_frag_r_mat_mask,
                memory=memory,
                memory_padding_mask=memory_padding_mask
            )
            logits, trans, r_mat = outputs
            
            if isinstance(logits, list): logits = logits[-1]
            if isinstance(trans, list): trans = trans[-1]
            if isinstance(r_mat, list): r_mat = r_mat[-1]
            
            pred_idx = torch.argmax(logits[:, -1:], dim=-1)
            probs_trans = torch.softmax(trans[:, -1:] / 1.2, dim=-1)
            pred_trans = torch.multinomial(probs_trans.view(-1, probs_trans.size(-1)), 1)
            pred_rot = torch.argmax(r_mat[:, -1:], dim=-1)
            
            next_idx = pred_idx[0, 0].item()
            next_trans = pred_trans[0, 0].item()
            next_rot = pred_rot[0, 0].item()
            
            all_pred_idx.append(next_idx)
            all_pred_trans.append(next_trans)
            all_pred_rot.append(next_rot)
            
            if eos_token_id is not None and next_idx == eos_token_id:
                break
                
            input_frag_idx = torch.cat([input_frag_idx, pred_idx], dim=1)
            input_frag_trans = torch.cat([input_frag_trans, pred_trans.view(batch_size, 1)], dim=1)
            input_frag_r_mat = torch.cat([input_frag_r_mat, pred_rot.view(batch_size, 1)], dim=1)
    
    sequence = []
    for i in range(1, len(all_pred_idx)):
        sequence.append((all_pred_idx[i], all_pred_trans[i], all_pred_rot[i]))
    
    return [sequence]
# --- END: Copied and modified run_desert_inference ---

def main_batch_inference():
    global SHAPE_UTILS_VOCAB, SHAPE_UTILS_VOCAB_R, SHAPE_UTILS_ROTATION_BIN # Allow modification of globals
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Receptor Center Information --- 
    receptor_centers_df = None
    if os.path.exists(RECEPTOR_INFO_CSV_PATH):
        try:
            receptor_centers_df = pd.read_csv(RECEPTOR_INFO_CSV_PATH)
            receptor_centers_df.set_index('pdb', inplace=True) # Set pdb id as index for easy lookup
            print(f"Successfully loaded receptor center information from {RECEPTOR_INFO_CSV_PATH}")
        except Exception as e_csv:
            print(f"Error loading receptor center CSV {RECEPTOR_INFO_CSV_PATH}: {e_csv}. Docking will be skipped if centers cannot be found.")
    else:
        print(f"Receptor center CSV not found at {RECEPTOR_INFO_CSV_PATH}. Docking will be skipped if centers cannot be found.")

    # --- START: Ensure rotation_bin.pkl exists or generate it ---
    if not os.path.exists(ROTATION_BIN_PATH):
        print(f"Rotation bin file not found at {ROTATION_BIN_PATH}. Generating it...")
        try:
            # Define default parameters for rotation bin generation
            sp_param = 11 
            rp_param = 24
            print(f"Generating rotation_bin.pkl with sp={sp_param}, rp={rp_param}...")
            generated_rotation_bin = get_rotation_bins_local(sp=sp_param, rp=rp_param)
            with open(ROTATION_BIN_PATH, 'wb') as f_rot_bin:
                pickle.dump(generated_rotation_bin, f_rot_bin)
            print(f"Successfully generated and saved rotation_bin.pkl to {ROTATION_BIN_PATH}")
        except Exception as e_rot_gen:
            print(f"Error generating rotation_bin.pkl: {e_rot_gen}")
            import traceback; traceback.print_exc()
            print("Cannot proceed without rotation_bin.pkl. Exiting.")
        return
    else:
        print(f"Found existing rotation_bin.pkl at {ROTATION_BIN_PATH}")
    # --- END: Ensure rotation_bin.pkl exists or generate it ---
    
    # --- Initialize shape_utils globals (including loading the now guaranteed rotation_bin) ---
    if not os.path.exists(DEFAULT_VOCAB_PATH):
        print(f"Error: Vocab file for shape_utils not found at {DEFAULT_VOCAB_PATH}"); return
    if not os.path.exists(ROTATION_BIN_PATH): # Should exist by now, but double check
        print(f"Critical Error: Rotation bin file still not found at {ROTATION_BIN_PATH} after attempting generation."); return

    try:
        print(f"Loading shape_utils vocabulary from: {DEFAULT_VOCAB_PATH}")
        with open(DEFAULT_VOCAB_PATH, 'rb') as f_vocab_su:
            SHAPE_UTILS_VOCAB = pickle.load(f_vocab_su)
        SHAPE_UTILS_VOCAB_R = {v_item[2]: k_item for k_item, v_item in SHAPE_UTILS_VOCAB.items() if isinstance(v_item, (list, tuple)) and len(v_item) >=3}
        print("Shape_utils vocabulary loaded.")

        print(f"Loading shape_utils rotation bin from: {ROTATION_BIN_PATH}")
        with open(ROTATION_BIN_PATH, 'rb') as f_rot_bin_su:
            SHAPE_UTILS_ROTATION_BIN = pickle.load(f_rot_bin_su)
        print("Shape_utils rotation bin loaded.")
    except Exception as e_init_su:
        print(f"Error initializing shape_utils globals: {e_init_su}")
        import traceback; traceback.print_exc()
        return
    # --- End Initialize shape_utils globals ---

    if not os.path.exists(DEFAULT_DESERT_MODEL_PATH):
        print(f"Error: DESERT model file not found: {DEFAULT_DESERT_MODEL_PATH}"); return
    if not os.path.isdir(DEFAULT_SHAPES_BASE_DIR):
        print(f"Error: Shapes directory not found: {DEFAULT_SHAPES_BASE_DIR}"); return
    
    model_dir_vocab_path = os.path.join(os.path.dirname(DEFAULT_DESERT_MODEL_PATH), "vocab.pkl")
    if not os.path.exists(model_dir_vocab_path):
         print(f"Error: Vocabulary file for model loading not found at {model_dir_vocab_path}"); return

    print(f"Processing shapes from: {DEFAULT_SHAPES_BASE_DIR}")
    print(f"Using SMILES string: {DEFAULT_SMILES_STRING}")
    print(f"Using DESERT model: {DEFAULT_DESERT_MODEL_PATH}")

    try:
        desert_model = load_desert_model(DEFAULT_DESERT_MODEL_PATH, model_dir_vocab_path, device)
        
        id_to_token = {}
        bos_token_id, eos_token_id = None, None
        vocab_for_reporting_path = DEFAULT_VOCAB_PATH if (DEFAULT_VOCAB_PATH and os.path.exists(DEFAULT_VOCAB_PATH)) else model_dir_vocab_path
        
        if os.path.exists(vocab_for_reporting_path):
            print(f"Loading vocabulary for reporting/tokens from: {vocab_for_reporting_path}")
            with open(vocab_for_reporting_path, 'rb') as f:
                reporting_vocab = pickle.load(f)
            for token, vocab_item in reporting_vocab.items():
                    if isinstance(vocab_item, (list, tuple)) and len(vocab_item) >= 3:
                        idx = vocab_item[2]
                        id_to_token[idx] = token
                    if token == 'BOS': bos_token_id = idx
                    if token == 'EOS': eos_token_id = idx
            if id_to_token: print(f"Successfully loaded reporting vocabulary.")
            if bos_token_id is None: print("Warning: BOS token ID not found in loaded vocab.")
        else:
            print(f"Error: No vocabulary file found at {DEFAULT_VOCAB_PATH} or {model_dir_vocab_path} for BOS/EOS tokens.")
            return

        if bos_token_id is None:
            print("Error: BOS token ID could not be determined. Inference cannot proceed.")
            return

    except Exception as e:
        print(f"Error loading DESERT model or vocabulary: {e}")
        import traceback; traceback.print_exc(); return
    
    results_data = []
    processed_files_count = 0
    successful_inferences = 0
    max_observed_fragments = 0 # To determine CSV columns dynamically

    for root, _, files in os.walk(DEFAULT_SHAPES_BASE_DIR):
        for filename in files:
            if not (filename.endswith(".pkl") or filename.endswith(".patch")):
                continue
            
            shape_file_full_path = os.path.join(root, filename)
            print(f"\nProcessing shape_file: {shape_file_full_path}")
            processed_files_count += 1
            
            # --- Determine Receptor Path and Center Dynamically ---
            current_receptor_path = None
            current_receptor_center = None
            try:
                # Extract receptor ID from shape filename, e.g., "4aaw_A" from "/path/to/4aaw_A_shapes.pkl"
                shape_basename = os.path.basename(shape_file_full_path)
                receptor_id_from_shape = shape_basename.replace("_shapes.pkl", "").replace(".patch","") # Handle both extensions
                
                current_receptor_path = os.path.join(RECEPTORS_DIR_PATH, f"{receptor_id_from_shape}.pdbqt")
                
                if receptor_centers_df is not None and receptor_id_from_shape in receptor_centers_df.index:
                    center_info = receptor_centers_df.loc[receptor_id_from_shape]
                    current_receptor_center = [float(center_info['c1']), float(center_info['c2']), float(center_info['c3'])]
                else:
                    print(f"    Center information for receptor ID '{receptor_id_from_shape}' not found in {RECEPTOR_INFO_CSV_PATH}.")

            except Exception as e_receptor_info:
                print(f"    Error determining receptor path/center for {shape_file_full_path}: {e_receptor_info}")
            # --- End Determine Receptor Path and Center ---

            current_result_entry = {
                'shape_file_path': shape_file_full_path, 
                'smiles_used': DEFAULT_SMILES_STRING,
                'fragments': [],
                'connected_smiles': '',
                'qed_score': None, # New field
                'docking_score': None # New field
            }

            try:
                desert_sequences = run_desert_inference_local(
                    loaded_model=desert_model,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    shape_patches_path=shape_file_full_path,
                    device=device,
                    smiles_str=DEFAULT_SMILES_STRING,
                    max_length=50 
                )

                if desert_sequences and desert_sequences[0]:
                    desert_sequence = desert_sequences[0] 
                    current_result_entry['status'] = "Success"
                    current_result_entry['num_fragments'] = len(desert_sequence)
                    if len(desert_sequence) > max_observed_fragments:
                        max_observed_fragments = len(desert_sequence)
                    successful_inferences += 1
                    # print(f"  Successfully ran DESERT inference. Received {len(desert_sequence)} fragments.")
                    
                    for frag_idx, fragment_data in enumerate(desert_sequence):
                        if len(fragment_data) == 3:
                            f_id, f_trans, f_rot = fragment_data
                            f_name = id_to_token.get(f_id, f"ID:{f_id}")
                            current_result_entry['fragments'].append({
                                'id': f_id,
                                'name': f_name,
                                'translation': f_trans,
                                'rotation': f_rot
                            })
                        else:
                            print(f"    Fragment {frag_idx+1}: Unexpected data format: {fragment_data}")
                    
                    # --- Connect Fragments ---
                    rdkit_frags_list = _get_3d_frags_local(desert_sequence)
                    if rdkit_frags_list:
                        connected_mol = connect_fragments_local(rdkit_frags_list)
                        if connected_mol:
                            try:
                                current_result_entry['connected_smiles'] = Chem.MolToSmiles(connected_mol)
                                print(f"  Connected SMILES: {current_result_entry['connected_smiles']}")

                                # Calculate QED
                                try:
                                    current_result_entry['qed_score'] = QED.qed(connected_mol)
                                    print(f"  QED Score: {current_result_entry['qed_score']:.3f}")
                                except Exception as e_qed:
                                    print(f"  Error calculating QED: {e_qed}")
                                    current_result_entry['qed_score'] = "ErrorQED"
                                
                                # Perform Docking
                                # Ensure receptor path and center are valid before attempting docking
                                if current_receptor_path and os.path.exists(current_receptor_path) and current_receptor_center:
                                    print(f"  Receptor: {current_receptor_path}, Center: {current_receptor_center}")
                                    docking_score = dock_best_molecule_local(connected_mol, current_receptor_path, current_receptor_center)
                                    if docking_score is not None:
                                        current_result_entry['docking_score'] = docking_score
                                    else:
                                        current_result_entry['docking_score'] = "DockingFailed"
                                        print(f"  Docking failed for {current_result_entry['connected_smiles']}")
                                else:
                                    print("  Skipping docking: Receptor path or center not configured or receptor file not found.")
                                    current_result_entry['docking_score'] = "DockingSkipped"

                            except Exception as e_smiles:
                                print(f"  Error converting connected mol to SMILES: {e_smiles}")
                                current_result_entry['connected_smiles'] = "ErrorConvertingToSmiles"
                        else:
                            print("  Fragment connection resulted in None.")
                            current_result_entry['connected_smiles'] = "ConnectionFailed"
                    else:
                        print("  _get_3d_frags_local returned no RDKit fragments to connect.")
                        current_result_entry['connected_smiles'] = "No3DFrags"
                else:
                    current_result_entry['status'] = "No sequences returned"
                    current_result_entry['num_fragments'] = 0
                    print(f"  DESERT inference returned no sequences for {shape_file_full_path}.")

            except FileNotFoundError as e:
                current_result_entry['status'] = f"Error: File not found - {e}"; print(f"  Error: {e}")
                current_result_entry['num_fragments'] = 0
            except Exception as e:
                current_result_entry['status'] = f"Error: {type(e).__name__} - {e}"; print(f"  Error: {e}")
                current_result_entry['num_fragments'] = 0
            results_data.append(current_result_entry)

    # --- Prepare CSV headers based on max_observed_fragments ---
    csv_fieldnames = ['shape_file_path', 'smiles_used', 'status', 'num_fragments', 'connected_smiles', 'qed_score', 'docking_score']
    for i in range(1, max_observed_fragments + 1):
        csv_fieldnames.extend([f'fragment_{i}_id', f'fragment_{i}_name', f'fragment_{i}_translation', f'fragment_{i}_rotation'])

    # --- Write results to CSV ---
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames, extrasaction='ignore')
            writer.writeheader()
            for entry in results_data:
                row_to_write = {k: entry.get(k, '') for k in ['shape_file_path', 'smiles_used', 'status', 'num_fragments', 'connected_smiles', 'qed_score', 'docking_score']}
                for i, frag_detail in enumerate(entry.get('fragments', [])):
                    if (i + 1) > max_observed_fragments: 
                        break 
                    row_to_write[f'fragment_{i+1}_id'] = frag_detail['id']
                    row_to_write[f'fragment_{i+1}_name'] = frag_detail['name']
                    row_to_write[f'fragment_{i+1}_translation'] = frag_detail['translation']
                    row_to_write[f'fragment_{i+1}_rotation'] = frag_detail['rotation']
                writer.writerow(row_to_write)
        print(f"\nResults saved to {OUTPUT_CSV_FILE}")
    except IOError:
        print(f"Error: Could not write results to {OUTPUT_CSV_FILE}")

    print(f"\n--- Summary ---")
    print(f"Total shape files processed: {processed_files_count}")
    print(f"Successful DESERT inferences: {successful_inferences}")
    print(f"Finished processing.")

    # --- BEGIN: Print Fragment Details for Successful Inferences (all fragments) ---
    print("\n--- Generated Fragment Details for Successful Inferences ---")
    for result in results_data:
        if result.get('status') == "Success":
            print(f"\nShape File: {result['shape_file_path']}")
            print(f"  SMILES Used: {result['smiles_used']}")
            print(f"  Number of Fragments Generated: {result['num_fragments']}")
            if result.get('fragments'):
                for i, frag_detail in enumerate(result['fragments']):
                    print(f"    Fragment {i+1}: ID={frag_detail['id']} ({frag_detail['name']}), Translation={frag_detail['translation']}, Rotation={frag_detail['rotation']}")
            elif result['num_fragments'] > 0:
                 print("    (Fragments were generated but details not found in results_data structure)") # Should not happen
            else:
                print("    No fragments were generated for this entry.")
            print(f"  Connected SMILES: {result.get('connected_smiles', 'N/A')}")
            qed_val = result.get('qed_score')
            dock_val = result.get('docking_score')
            print(f"  QED Score: {qed_val if isinstance(qed_val, (int, float)) else (f'{qed_val:.3f}' if qed_val else 'N/A') if qed_val not in [None, 'ErrorQED', 'DockingSkipped', 'DockingFailed'] else qed_val if qed_val else 'N/A'}")
            print(f"  Docking Score: {dock_val if isinstance(dock_val, (int, float)) else (f'{dock_val:.3f}' if dock_val else 'N/A') if dock_val not in [None, 'ErrorQED', 'DockingSkipped', 'DockingFailed'] else dock_val if dock_val else 'N/A'}")
    print("--- End of Fragment Details ---")
    # --- END: Print Fragment Details for Successful Inferences ---

if __name__ == "__main__":
    main_batch_inference() 