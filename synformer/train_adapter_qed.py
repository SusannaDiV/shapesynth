import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json # For parsing latent string
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm # For progress bars
from sklearn.model_selection import train_test_split # Added for train/test split
import os # Added for directory creation
import subprocess # For docking
import tempfile   # For docking
import uuid       # For docking
import shutil     # For docking
import dataclasses # For docking

# --- Synformer/DESERT components (kept for potential inference, not used in this training loop) ---
from synformer.models.decoder import Decoder
from synformer.models.classifier_head import ClassifierHead
from synformer.models.fingerprint_head import get_fingerprint_head
from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.data.common import TokenType
from synformer.chem.stack import Stack
from synformer.chem.mol import Molecule
from omegaconf import OmegaConf

# Configuration
SMILES_CHECKPOINT_PATH = "/workspace/data/processed/sf_ed_default.ckpt" # For potential inference later
RXN_MATRIX_PATH = "/workspace/data/processed/comp_2048/matrix.pkl" # For potential inference later
FPINDEX_PATH = "/workspace/data/processed/comp_2048/fpindex.pkl" # For potential inference later
CSV_PATH = "/workspace/fragments_qed_docking_with_latents.csv" # Adjusted for clarity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QED_IMPROVEMENT_THRESHOLD = 0.1 # Minimum QED increase to consider a target "better"
MIN_QED_FOR_TARGET_SEARCH = 0.0 # Optimization: only try to find better targets for z_i if QED(z_i) > this value. Can be 0.0 to include all.
MAX_QED_CONSIDERED = 0.95 # Optimization: if a z_i already has QED > this, consider its delta_z target as zero.
DOCKING_IMPROVEMENT_THRESHOLD = 0.5 # How much better docking score should be to trigger a shift if QED is already good. Assumes lower is better.
RECEPTORS_DIR_PATH = "/workspace/data/TestCrossDocked2020/receptors/"
RECEPTOR_INFO_CSV_PATH = "/workspace/data/TestCrossDocked2020/receptors/test.csv"

# --- Adapter Network ---
class Adapter(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, z):
        return self.network(z)

# --- Dataset ---
class OracleShiftDataset(Dataset):
    def __init__(self, dataframe, qed_improvement_threshold=0.1, min_qed_for_target_search=0.0, max_qed_considered=0.95):
        print(f"Initializing dataset from DataFrame with {len(dataframe)} rows...")
        # df = pd.read_csv(csv_path)
        df = dataframe # Use the passed DataFrame
        df = df.dropna(subset=['latent_representation', 'qed_score'])
        df = df[df['latent_representation'].apply(lambda x: isinstance(x, str) and '[' in x)]

        raw_latents = []
        raw_qeds = []
        raw_docking_scores = [] 
        valid_indices = []
        raw_receptor_ids = [] # Added to store receptor_ids

        print("Parsing initial latents, QED scores, docking scores, and receptor IDs...")
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Parsing DataFrame"):
            try:
                latent_list = json.loads(row['latent_representation'])
                qed_score = float(row['qed_score'])
                
                try:
                    docking_score = float(row['docking_score'])
                except (ValueError, TypeError):
                    docking_score = float('inf') 
                
                # Attempt to get receptor_id; default to None if not present or not a string
                receptor_id = str(row.get('receptor_id', None)) if isinstance(row.get('receptor_id'), str) else None
                if not receptor_id and 'pdb' in df.columns: # Fallback to 'pdb' column if 'receptor_id' is missing/invalid
                    receptor_id = str(row.get('pdb', None)) if isinstance(row.get('pdb'), str) else None

                if isinstance(latent_list, list) and all(isinstance(x, (int, float)) for x in latent_list):
                    raw_latents.append(torch.tensor(latent_list, dtype=torch.float32))
                    raw_qeds.append(qed_score)
                    raw_docking_scores.append(docking_score) 
                    raw_receptor_ids.append(receptor_id) # Store receptor_id
                    valid_indices.append(idx)
                else:
                    pass 
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                pass 
        
        if not raw_latents:
            raise ValueError("No valid raw latents loaded. Check DataFrame for 'latent_representation', 'qed_score', 'docking_score', and 'receptor_id'/'pdb'.")

        print(f"Successfully parsed {len(raw_latents)} initial latents, QEDs, docking scores, and receptor_ids.")
        self.input_latents = []
        self.target_delta_zs = []
        self.original_qeds_for_input_latents = [] 
        self.original_docking_scores_for_input_latents = [] 
        self.receptor_ids_for_input_latents = [] # Added to store receptor_ids for dataset items

        print("Identifying oracle target based on QED > 0.5 and best docking score...")
        z_oracle_target = None
        qed_oracle_target = -1.0
        docking_oracle_target = float('inf')

        potential_oracle_candidates = []
        for i in range(len(raw_latents)):
            if raw_qeds[i] > 0.5 and raw_docking_scores[i] != float('inf'):
                potential_oracle_candidates.append({
                    'latent': raw_latents[i],
                    'qed': raw_qeds[i],
                    'docking': raw_docking_scores[i]
                })
        
        if potential_oracle_candidates:
            # Sort by docking score (ascending, as lower is better)
            potential_oracle_candidates.sort(key=lambda x: x['docking'])
            best_oracle_candidate = potential_oracle_candidates[0]
            z_oracle_target = best_oracle_candidate['latent']
            qed_oracle_target = best_oracle_candidate['qed']
            docking_oracle_target = best_oracle_candidate['docking']
            print(f"Identified Oracle Target: QED={qed_oracle_target:.4f}, Docking Score={docking_oracle_target:.4f}")
        else:
            print("Warning: No suitable oracle target found in the dataset (QED > 0.5 and valid docking score). Adapter will learn zero shifts.")

        print("Calculating target oracle shifts (Δz) based on new criteria...")
        for i in tqdm(range(len(raw_latents)), desc="Calculating Shifts"):
            current_z = raw_latents[i]
            current_qed = raw_qeds[i]
            current_docking = raw_docking_scores[i]
            current_receptor_id = raw_receptor_ids[i]
            
            self.input_latents.append(current_z)
            self.original_qeds_for_input_latents.append(current_qed)
            self.original_docking_scores_for_input_latents.append(current_docking)
            self.receptor_ids_for_input_latents.append(current_receptor_id) # Store receptor_id for the dataset item

            target_delta_z = torch.zeros_like(current_z)

            if z_oracle_target is not None:
                perform_shift = False
                if current_qed <= 0.5:  # Condition 1: Current QED is bad
                    perform_shift = True
                    # print(f"Debug: Shift for sample {i} due to low QED ({current_qed:.2f})")
                elif current_docking == float('inf'): # Condition 2: Current docking score is invalid
                    perform_shift = True
                    # print(f"Debug: Shift for sample {i} due to invalid docking score")
                elif docking_oracle_target < current_docking - DOCKING_IMPROVEMENT_THRESHOLD: # Condition 3: QED is OK, but docking can improve significantly
                    perform_shift = True
                    # print(f"Debug: Shift for sample {i} due to docking improvement ({current_docking:.2f} -> {docking_oracle_target:.2f})")
                
                if perform_shift:
                    target_delta_z = z_oracle_target - current_z
            
            self.target_delta_zs.append(target_delta_z)

        if not self.input_latents:
            raise ValueError("Dataset initialization failed: No input latents after processing.")
        
        num_zero_deltas = sum(1 for dz in self.target_delta_zs if torch.all(dz == 0).item())
        print(f"Finished calculating oracle shifts. Total samples: {len(self.input_latents)}")
        print(f"Number of samples with zero target Δz: {num_zero_deltas} (either already good QED or no better global target found)")


    def __len__(self):
        return len(self.input_latents)

    def __getitem__(self, idx):
        return {
            "latent_input": self.input_latents[idx],
            "target_delta_z": self.target_delta_zs[idx],
            "original_qed": self.original_qeds_for_input_latents[idx],
            "original_docking": self.original_docking_scores_for_input_latents[idx],
            "receptor_id": self.receptor_ids_for_input_latents[idx] # Add receptor_id to dataset item
        }


# --- Docking Utilities (copied and adapted) ---
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

def prepare_ligand_pdbqt_local(mol, obabel_path="obabel"):
    """
    Prepare a ligand for docking by converting it to PDBQT format using OpenBabel
    """
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()
    temp_mol_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.mol")
    temp_pdbqt_file = os.path.join(temp_dir, f"temp_ligand_{unique_id}.pdbqt")

    try:
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
            print(f"    Error converting molecule to PDBQT: {process.stderr}")
            return None
        with open(temp_pdbqt_file, "r") as f:
            pdbqt_content = f.read()
        if "ATOM" not in pdbqt_content and "HETATM" not in pdbqt_content:
            print("    Error: Generated PDBQT file does not contain valid atom entries")
            return None
        return pdbqt_content
    except Exception as e:
        print(f"    Error preparing ligand: {str(e)}")
        return None
    finally:
        for f_path in [temp_mol_file, temp_pdbqt_file]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError:
                    print(f"    Warning: Could not remove temporary file {f_path}")

def dock_best_molecule_local(mol, receptor_pdbqt_path, receptor_center_coords):
    """Dock the molecule against receptor target"""
    temp_ligand_file = None
    output_file = None
    try:
        smiles = Chem.MolToSmiles(mol)
        # print(f"  Attempting docking for: {smiles}") # Can be verbose

        unique_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.gettempdir()
        
        qvina_path = shutil.which("qvina2.1") or shutil.which("qvina2") # Try common names
        if qvina_path is None:
            # Basic fallback check, can be made more robust
            local_qvina_path = "./bin/qvina2.1" 
            if os.path.exists(local_qvina_path):
                 qvina_path = local_qvina_path
            else:
                print("    Error: QVina2 executable (qvina2.1 or qvina2) not found in PATH or ./bin/.")
                return None
        
        obabel_path = shutil.which("obabel")
        if obabel_path is None:
            print("    Error: OpenBabel (obabel) not found in PATH.")
            return None
            
        if not os.path.exists(receptor_pdbqt_path):
            print(f"    Error: Receptor file not found at {receptor_pdbqt_path}")
            return None
            
        ligand_pdbqt = prepare_ligand_pdbqt_local(mol, obabel_path)
        if ligand_pdbqt is None:
            # print("    Failed to prepare ligand for docking.") # prepare_ligand_pdbqt_local prints errors
            return None
            
        temp_ligand_file = os.path.join(temp_dir, f"temp_ligand_dock_{unique_id}.pdbqt")
        with open(temp_ligand_file, "w") as f:
            f.write(ligand_pdbqt)
            
        options = QVinaOption(
            center_x=receptor_center_coords[0], center_y=receptor_center_coords[1], center_z=receptor_center_coords[2]
        )
        
        output_file = os.path.join(temp_dir, f"temp_ligand_dock_out_{unique_id}.pdbqt")
        cmd = [
            qvina_path, "--receptor", receptor_pdbqt_path, "--ligand", temp_ligand_file,
            "--center_x", str(options.center_x), "--center_y", str(options.center_y), "--center_z", str(options.center_z),
            "--size_x", str(options.size_x), "--size_y", str(options.size_y), "--size_z", str(options.size_z),
            "--exhaustiveness", str(options.exhaustiveness), "--num_modes", str(options.num_modes),
            "--out", output_file
        ]
        
        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, check=False # Reduced timeout for eval script
        )
        
        score = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT" in line and line.split()[3]:
                        try:
                            score = float(line.split()[3])
                            # print(f"    Docking score for {smiles}: {score}") # Can be verbose
                            break
                        except (IndexError, ValueError):
                            pass # Continue if score parsing fails for a line
        
        if score is None and process.returncode != 0:
             print(f"    No docking score found AND QVina2 error for {smiles}: {process.stderr if process.stderr else 'Unknown error'}")
             return None # Return None if no score and qvina failed
        elif score is None:
             # print(f"    No docking score found in output for {smiles}, QVina2 return code: {process.returncode}")
             pass # Allow to proceed if qvina ran but no score parsed, will return None or score. Better to return None here too.
             return None

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
# --- End Docking Utilities ---

# --- Decoder and Generation Functions (Restored for Evaluation) ---
# These functions (load_decoder_models, generate_smiles_from_latent, decode_batch, calculate_qed_batch)
# are kept for completeness if inference capabilities were to be added to this script later,
# or if a more complex Δz calculation in the dataset required them.
# They are NOT called during the supervised adapter training.

def load_decoder_models(checkpoint_path, rxn_matrix_path, fpindex_path, device):
    """Loads the Synformer decoder, heads, and other necessary components."""
    full_model_checkpoint = torch.load(checkpoint_path, map_location=device)
    config = OmegaConf.create(full_model_checkpoint['hyper_parameters']['config'])

    # Determine decoder params from checkpoint (as in excellent_checkadaptertraining.py)
    decoder_params_detected = {}
    for k_ckpt in full_model_checkpoint['state_dict'].keys():
        if k_ckpt.startswith('model.decoder.'):
            parts = k_ckpt.split('.')
            if len(parts) > 2:
                if parts[2] == 'dec' and parts[3] == 'layers' and len(parts) > 4:
                    layer_num = int(parts[4])
                    if 'num_layers' not in decoder_params_detected or layer_num + 1 > decoder_params_detected['num_layers']:
                        decoder_params_detected['num_layers'] = layer_num + 1
                if parts[2] == 'pe_dec' and parts[3] == 'pe':
                    pe_shape = full_model_checkpoint['state_dict'][k_ckpt].shape
                    decoder_params_detected['pe_max_len'] = pe_shape[1]

    decoder = Decoder(
        d_model=config.model.decoder.d_model, # Use config value
        nhead=config.model.decoder.nhead,
        dim_feedforward=config.model.decoder.dim_feedforward,
        num_layers=decoder_params_detected.get('num_layers', config.model.decoder.num_layers),
        pe_max_len=decoder_params_detected.get('pe_max_len', config.model.decoder.pe_max_len),
        output_norm=config.model.decoder.output_norm,
        fingerprint_dim=config.model.decoder.fingerprint_dim,
        num_reaction_classes=config.model.decoder.num_reaction_classes
    )
    decoder_state_dict = {k.replace('model.decoder.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.decoder.')}
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device).eval()

    token_head = ClassifierHead(config.model.decoder.d_model, max(TokenType) + 1)
    token_head_state_dict = {k.replace('model.token_head.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.token_head.')}
    token_head.load_state_dict(token_head_state_dict)
    token_head.to(device).eval()

    reaction_head = ClassifierHead(config.model.decoder.d_model, config.model.decoder.num_reaction_classes)
    reaction_head_state_dict = {k.replace('model.reaction_head.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.reaction_head.')}
    reaction_head.load_state_dict(reaction_head_state_dict)
    reaction_head.to(device).eval()
    
    fingerprint_head = get_fingerprint_head(config.model.fingerprint_head_type, config.model.fingerprint_head)
    fingerprint_head_state_dict = {k.replace('model.fingerprint_head.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.fingerprint_head.')}
    fingerprint_head.load_state_dict(fingerprint_head_state_dict)
    fingerprint_head.to(device).eval()

    rxn_matrix = pickle.load(open(rxn_matrix_path, 'rb'))
    fpindex = pickle.load(open(fpindex_path, 'rb'))

    return {
        "decoder": decoder, "token_head": token_head, "reaction_head": reaction_head,
        "fingerprint_head": fingerprint_head, "rxn_matrix": rxn_matrix, "fpindex": fpindex,
        "config": config
    }

def generate_smiles_from_latent(z_input_single, models, max_steps=24, temperature=0.1):
    """Generates a single SMILES string from a single z_prime vector."""
    decoder = models["decoder"]
    token_head = models["token_head"]
    reaction_head = models["reaction_head"]
    fingerprint_head = models["fingerprint_head"]
    rxn_matrix = models["rxn_matrix"]
    fpindex = models["fpindex"]
    device = z_input_single.device

    # z_prime_single is [D_latent]. Decoder expects code: [1, seq_len_code, D_latent]
    # Here, seq_len_code = 1 as z_prime is a pooled representation.
    code_input = z_input_single.unsqueeze(0).unsqueeze(0) # -> [1, 1, D_latent]
    code_padding_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

    stack = Stack()
    token_types = torch.tensor([[TokenType.START.value]], dtype=torch.long, device=device)
    rxn_indices = torch.zeros_like(token_types)
    
    # fp_dim should be accessed safely
    fp_dim = getattr(fpindex.fp_option, 'dim', 2048) # Default to 2048 if not found
    reactant_fps = torch.zeros((1, 1, fp_dim), dtype=torch.float32, device=device)
    
    generated_smiles = None

    with torch.no_grad():
        for _ in range(max_steps):
            current_seq_len = token_types.shape[1]
            # Ensure reactant_fps has the right shape for the current token_types
            if reactant_fps.shape[1] != current_seq_len:
                new_fps_shape = (1, current_seq_len, fp_dim)
                if reactant_fps.numel() == 0 and current_seq_len > 0 : # Initial step if reactant_fps starts empty for seq_len 1
                     reactant_fps = torch.zeros(new_fps_shape, dtype=torch.float32, device=device)
                elif reactant_fps.shape[1] < current_seq_len: # Append zeros
                    padding_needed = current_seq_len - reactant_fps.shape[1]
                    padding_tensor = torch.zeros((1, padding_needed, fp_dim), dtype=torch.float32, device=device)
                    reactant_fps = torch.cat([reactant_fps, padding_tensor], dim=1)
                elif reactant_fps.shape[1] > current_seq_len: # Truncate
                     reactant_fps = reactant_fps[:, :current_seq_len, :]


            h = decoder(
                code=code_input,
                code_padding_mask=code_padding_mask,
                token_types=token_types,
                rxn_indices=rxn_indices,
                reactant_fps=reactant_fps,
                token_padding_mask=None # Assuming no padding in generated sequence for now
            )
            h_next = h[:, -1]  # (bsz=1, h_dim)

            token_logits = token_head.predict(h_next)
            token_probs = torch.nn.functional.softmax(token_logits / temperature, dim=-1)
            token_sampled = torch.multinomial(token_probs, num_samples=1)
            
            token_val = token_sampled.item()

            if token_val == TokenType.END.value:
                break

            # Reaction processing (simplified)
            reaction_logits = reaction_head.predict(h_next)[..., :len(rxn_matrix.reactions)]
            reaction_probs = torch.nn.functional.softmax(reaction_logits / temperature, dim=-1)
            rxn_idx_next = torch.multinomial(reaction_probs, num_samples=1)[..., 0]

            if token_val == TokenType.REACTANT.value:
                retrieved_reactants = fingerprint_head.retrieve_reactants(h_next, fpindex, topk=4) # Removed mask
                if retrieved_reactants.reactants.size == 0: # No reactants found
                    token_val = TokenType.END.value # Force end
                    token_sampled[0,0] = TokenType.END.value
                    # print("Warning: No reactants retrieved, forcing END token.") # Optional debug
                else:
                    fp_scores = torch.from_numpy(1.0 / (retrieved_reactants.distance + 1e-4)).reshape(1, -1).to(device)
                    fp_probs = torch.nn.functional.softmax(fp_scores / temperature, dim=-1)
                    fp_idx_next = torch.multinomial(fp_probs, num_samples=1)[..., 0]
                    
                    reactant_mol = retrieved_reactants.reactants[0, fp_idx_next.item()]
                    reactant_idx = retrieved_reactants.indices[0, fp_idx_next.item()]
                    stack.push_mol(reactant_mol, reactant_idx)
                    
                    fp_retrieved = torch.from_numpy(retrieved_reactants.fingerprint_retrieved[0, fp_idx_next]).unsqueeze(0).unsqueeze(1).to(device)
                    
                    if reactant_fps.shape[1] == token_types.shape[1] : # About to concat, ensure reactant_fps is ready
                         reactant_fps = torch.cat([reactant_fps, fp_retrieved], dim=1)
                    # else: Handled by check at start of loop. This case should not be common if logic is correct.
                    #    print(f"Warning: reactant_fps shape {reactant_fps.shape} mismatch with token_types {token_types.shape} before appending fp_retrieved.")


            elif token_val == TokenType.REACTION.value:
                reaction = rxn_matrix.reactions[rxn_idx_next.item()]
                success = stack.push_rxn(reaction, rxn_idx_next.item())
                if not success:
                    token_val = TokenType.END.value # Force end on fail
                    token_sampled[0,0] = TokenType.END.value


            token_types = torch.cat([token_types, token_sampled], dim=1)
            rxn_indices = torch.cat([rxn_indices, rxn_idx_next.unsqueeze(1)], dim=1)
            
            if token_val == TokenType.END.value: # Check again if forced end
                break

    final_mols = stack.get_top()
    if final_mols:
        generated_smiles = list(final_mols)[0].smiles
    return generated_smiles


def decode_batch(z_batch, models, max_steps=24, temperature=0.1):
    """Decodes a batch of z_prime vectors into SMILES strings."""
    all_smiles = [generate_smiles_from_latent(z_batch[i], models, max_steps, temperature) for i in range(z_batch.shape[0])]
    return [s if s else "" for s in all_smiles]

def calculate_qed_from_smiles(smiles_string):
    """Calculates QED for a single SMILES string."""
    if smiles_string:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return QED.qed(mol)
    return 0.0

# --- Evaluation Function ---
def evaluate_adapter(adapter, dataset, decoder_models, device, num_samples_to_evaluate=10, receptor_centers_df=None):
    print(f"\n--- Starting Evaluation (on {num_samples_to_evaluate} samples) ---")
    adapter.eval() # Set adapter to evaluation mode

    if len(dataset) == 0:
        print("Evaluation dataset is empty.")
        return

    num_samples = min(num_samples_to_evaluate, len(dataset))
    
    original_qeds_csv_list = [] 
    baseline_decoded_qeds_list = [] 
    adapted_decoded_qeds_list = [] 
    original_docking_csv_list = [] 
    evaluation_details = [] 
    num_adapted_better_than_baseline = 0 # Counter for QED improvements
    
    # For decoded docking scores
    baseline_decoded_docking_scores_list = []
    adapted_decoded_docking_scores_list = []
    num_adapted_docking_better_than_baseline = 0

    eval_indices = np.random.choice(len(dataset), num_samples, replace=False) if num_samples < len(dataset) else np.arange(len(dataset))

    for i in tqdm(range(num_samples), desc="Evaluating Adapter"):
        idx = eval_indices[i]
        sample_data = dataset[idx]
        z_original = sample_data["latent_input"].to(device)
        original_qed_from_csv = sample_data["original_qed"] 
        original_docking_from_csv = sample_data["original_docking"] 
        receptor_id_for_sample = sample_data["receptor_id"]
        original_qeds_csv_list.append(original_qed_from_csv)
        original_docking_csv_list.append(original_docking_from_csv) 

        # --- Prepare for potential docking for this sample ---
        current_receptor_pdbqt_path = None
        current_receptor_center = None
        can_dock_this_sample = False

        if receptor_id_for_sample and receptor_centers_df is not None and receptor_id_for_sample in receptor_centers_df.index:
            current_receptor_pdbqt_path = os.path.join(RECEPTORS_DIR_PATH, f"{receptor_id_for_sample}.pdbqt")
            if os.path.exists(current_receptor_pdbqt_path):
                try:
                    center_info = receptor_centers_df.loc[receptor_id_for_sample]
                    current_receptor_center = [float(center_info['c1']), float(center_info['c2']), float(center_info['c3'])]
                    can_dock_this_sample = True
                except KeyError as e:
                    print(f"    Warning: Docking center key error for {receptor_id_for_sample}: {e}. Will skip docking.")
                except ValueError as e:
                    print(f"    Warning: Docking center coordinate error for {receptor_id_for_sample}: {e}. Will skip docking.")
            else:
                print(f"    Warning: Receptor PDBQT file not found for {receptor_id_for_sample} at {current_receptor_pdbqt_path}. Will skip docking.")
        # else:
            # print(f"    Info: Skipping docking for sample with receptor_id '{receptor_id_for_sample}' (receptor info missing or CSV not loaded).")
        # --- End Prepare for docking ---

        # 1. Baseline: Decode original latent & attempt docking
        smiles_baseline = generate_smiles_from_latent(z_original, decoder_models)
        qed_baseline_decoded = calculate_qed_from_smiles(smiles_baseline)
        baseline_decoded_qeds_list.append(qed_baseline_decoded)
        
        docking_score_baseline_decoded = float('nan')
        if smiles_baseline and can_dock_this_sample:
            mol_baseline = Chem.MolFromSmiles(smiles_baseline)
            if mol_baseline:
                docking_score_baseline_decoded = dock_best_molecule_local(mol_baseline, current_receptor_pdbqt_path, current_receptor_center)
                if docking_score_baseline_decoded is None: docking_score_baseline_decoded = float('nan') # Ensure NaN if docking returns None
        baseline_decoded_docking_scores_list.append(docking_score_baseline_decoded)

        # 2. Adapted: Apply adapter, then decode & attempt docking
        with torch.no_grad(): 
            predicted_delta_z = adapter(z_original.unsqueeze(0)) 
        z_prime = z_original + predicted_delta_z.squeeze(0) 
        
        smiles_adapted = generate_smiles_from_latent(z_prime, decoder_models)
        qed_adapted_decoded = calculate_qed_from_smiles(smiles_adapted)
        adapted_decoded_qeds_list.append(qed_adapted_decoded)

        docking_score_adapted_decoded = float('nan')
        if smiles_adapted and can_dock_this_sample:
            mol_adapted = Chem.MolFromSmiles(smiles_adapted)
            if mol_adapted:
                docking_score_adapted_decoded = dock_best_molecule_local(mol_adapted, current_receptor_pdbqt_path, current_receptor_center)
                if docking_score_adapted_decoded is None: docking_score_adapted_decoded = float('nan') # Ensure NaN
        adapted_decoded_docking_scores_list.append(docking_score_adapted_decoded)

        # Increment QED counter if adapted is better 
        if qed_adapted_decoded > qed_baseline_decoded:
            num_adapted_better_than_baseline += 1
        
        # Increment docking counter if adapted is better (lower score)
        if not np.isnan(docking_score_adapted_decoded) and not np.isnan(docking_score_baseline_decoded):
            if docking_score_adapted_decoded < docking_score_baseline_decoded:
                num_adapted_docking_better_than_baseline += 1

        # Store details for printing later or for all samples
        evaluation_details.append({
            "original_z_idx": idx, # if you want to map back to original df index
            "original_qed_csv": original_qed_from_csv,
            "original_docking_csv": original_docking_from_csv if original_docking_from_csv != float('inf') else 'N/A',
            "smiles_baseline": smiles_baseline if smiles_baseline else 'N/A',
            "qed_baseline_decoded": qed_baseline_decoded,
            "docking_baseline_decoded": docking_score_baseline_decoded if not np.isnan(docking_score_baseline_decoded) else 'N/A',
            "smiles_adapted": smiles_adapted if smiles_adapted else 'N/A',
            "qed_adapted_decoded": qed_adapted_decoded,
            "docking_adapted_decoded": docking_score_adapted_decoded if not np.isnan(docking_score_adapted_decoded) else 'N/A',
            "receptor_id": receptor_id_for_sample
        })

        if i < 5: # Print first few examples immediately during evaluation
            print(f"\n  Eval Sample {i+1} (Dataset Index: {idx}, Receptor: {receptor_id_for_sample if receptor_id_for_sample else 'N/A'}):")
            print(f"    Original QED (from CSV): {original_qed_from_csv:.4f} | Original Docking (from CSV): {original_docking_from_csv if original_docking_from_csv != float('inf') else 'N/A'}")
            print(f"    Baseline Decoded SMILES: {smiles_baseline if smiles_baseline else 'N/A'} | QED: {qed_baseline_decoded:.4f} | Decoded Docking: {docking_score_baseline_decoded if not np.isnan(docking_score_baseline_decoded) else 'N/A'}")
            print(f"    Adapted Decoded SMILES:  {smiles_adapted if smiles_adapted else 'N/A'} | QED: {qed_adapted_decoded:.4f} | Decoded Docking: {docking_score_adapted_decoded if not np.isnan(docking_score_adapted_decoded) else 'N/A'}")

    avg_original_qed_csv = np.mean(original_qeds_csv_list) if original_qeds_csv_list else 0
    avg_baseline_decoded_qed = np.mean(baseline_decoded_qeds_list) if baseline_decoded_qeds_list else 0
    avg_adapted_decoded_qed = np.mean(adapted_decoded_qeds_list) if adapted_decoded_qeds_list else 0
    
    valid_original_docking_scores = [s for s in original_docking_csv_list if s != float('inf')]
    avg_original_docking_csv = np.mean(valid_original_docking_scores) if valid_original_docking_scores else float('nan') 

    # Calculate averages for decoded docking scores, ignoring NaNs
    avg_baseline_decoded_docking = np.nanmean(baseline_decoded_docking_scores_list) if baseline_decoded_docking_scores_list else float('nan')
    avg_adapted_decoded_docking = np.nanmean(adapted_decoded_docking_scores_list) if adapted_decoded_docking_scores_list else float('nan')

    print("\nEvaluation Summary:")
    print(f"  Avg QED from CSV (for selected samples): {avg_original_qed_csv:.4f}")
    print(f"  Avg Docking Score from CSV (for selected samples): {avg_original_docking_csv:.4f}")
    print(f"  Avg QED from decoding original latents:  {avg_baseline_decoded_qed:.4f}")
    print(f"  Avg Docking Score from decoding original latents: {avg_baseline_decoded_docking:.4f}")
    print(f"  Avg QED from decoding adapted latents:   {avg_adapted_decoded_qed:.4f}")
    print(f"  Avg Docking Score from decoding adapted latents:  {avg_adapted_decoded_docking:.4f}")
    
    improvement_over_baseline_qed = avg_adapted_decoded_qed - avg_baseline_decoded_qed
    print(f"  Avg QED improvement over baseline decoding:    {improvement_over_baseline_qed:+.4f}")
    print(f"  Number of adapted SMILES with QED > baseline QED: {num_adapted_better_than_baseline}/{num_samples}")
    
    # Docking improvement (lower is better, so baseline - adapted)
    improvement_over_baseline_docking = avg_baseline_decoded_docking - avg_adapted_decoded_docking if not (np.isnan(avg_baseline_decoded_docking) or np.isnan(avg_adapted_decoded_docking)) else float('nan')
    print(f"  Avg Docking Score improvement over baseline decoding: {improvement_over_baseline_docking:+.4f} (positive means adapted is better/lower)")
    print(f"  Number of adapted SMILES with Docking < baseline Docking: {num_adapted_docking_better_than_baseline}/{num_samples} (where docking was possible for both)")

    print("\n--- Examples of Evaluated Generated SMILES (Adapted Latents) ---")
    num_to_print = min(num_samples, 5) # Print up to 5 examples, or all if less than 5 evaluated
    for i in range(num_to_print):
        detail = evaluation_details[i]
        print(f"  Example {i+1} (Receptor: {detail.get('receptor_id', 'N/A')}, Original CSV QED: {detail['original_qed_csv']:.4f}, Original CSV Docking: {detail.get('original_docking_csv', 'N/A')}):")
        print(f"    Baseline SMILES: {detail['smiles_baseline']} (QED: {detail['qed_baseline_decoded']:.4f}, Decoded Docking: {detail['docking_baseline_decoded']})")
        print(f"    Adapted SMILES:  {detail['smiles_adapted']} (QED: {detail['qed_adapted_decoded']:.4f}, Decoded Docking: {detail['docking_adapted_decoded']})")
        # QED comparison printout
        if detail['qed_adapted_decoded'] > detail['qed_baseline_decoded']:
            print("      -> QED Improved!")
        elif detail['qed_adapted_decoded'] < detail['qed_baseline_decoded']:
            print("      -> QED Decreased.")
        else:
            print("      -> QED Unchanged.")

    adapter.train() # Set adapter back to training mode if further training is planned

# --- Main Training Loop ---
def main():
    num_epochs = 20 # Increased epochs for supervised learning
    batch_size = 32 
    learning_rate = 1e-4
    latent_dim = 768 # Default, will be inferred
    num_samples_for_eval = 10 # Number of samples to use in evaluation after training

    # Load data with oracle shifts
    # Make sure CSV_PATH is correct relative to where script is run
    # If train_adapter_qed.py is in /workspace/synformer/synformer,
    # and csv is also there, CSV_PATH="fragments_qed_docking_with_latents.csv" is fine.
    # If csv is in /workspace, then CSV_PATH="/workspace/fragments_qed_docking_with_latents.csv"
    # For now, assuming it's in the same directory as the script or /workspace.
    # Let's try to be more robust, assuming it's in /workspace, as per previous file structure.
    csv_full_path = "/workspace/fragments_qed_docking_with_latents.csv"
    train_split_ratio = 0.8 # 80% for training, 20% for testing
    
    print(f"Attempting to load and split dataset from: {csv_full_path}")
    try:
        full_df = pd.read_csv(csv_full_path)
        if full_df.empty:
            print("Error: The loaded CSV file is empty.")
            return
        
        # Drop rows with missing essential data before splitting
        full_df_cleaned = full_df.dropna(subset=['latent_representation', 'qed_score'])
        full_df_cleaned = full_df_cleaned[full_df_cleaned['latent_representation'].apply(lambda x: isinstance(x, str) and '[' in x)]

        if full_df_cleaned.empty:
            print("Error: No valid data rows remaining after cleaning the CSV. Check 'latent_representation' and 'qed_score' columns.")
            return

        train_df, test_df = train_test_split(full_df_cleaned, train_size=train_split_ratio, random_state=42, shuffle=True)
        print(f"Data split: {len(train_df)} training samples, {len(test_df)} test samples.")

        train_dataset = OracleShiftDataset(
            train_df, 
            qed_improvement_threshold=QED_IMPROVEMENT_THRESHOLD,
            min_qed_for_target_search=MIN_QED_FOR_TARGET_SEARCH,
            max_qed_considered=MAX_QED_CONSIDERED
            )
        test_dataset = OracleShiftDataset(
            test_df, 
            qed_improvement_threshold=QED_IMPROVEMENT_THRESHOLD, # Usually, thresholds are consistent
            min_qed_for_target_search=MIN_QED_FOR_TARGET_SEARCH,
            max_qed_considered=MAX_QED_CONSIDERED
            )
            
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_full_path}. Please check the path.")
        # print(f"If your CSV is in the same directory as the script, change csv_full_path to '{CSV_PATH}'")
        return
    except ValueError as ve:
        print(f"Error initializing dataset from DataFrame: {ve}")
        return


    if len(train_dataset) == 0:
        print("Training dataset is empty after processing. Exiting.")
        return
    if len(test_dataset) == 0:
        print("Warning: Test dataset is empty after processing. Evaluation might not be meaningful.")
        # Depending on requirements, one might choose to exit or continue without evaluation.
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_loader will not be used for batching in evaluate_adapter as it processes samples individually for now
    
    # Infer latent_dim from the first item of the train_dataset
    if len(train_dataset) > 0:
        first_item = train_dataset[0]["latent_input"]
        if first_item is not None:
            latent_dim = first_item.shape[0]
        print(f"Inferred latent dimension: {latent_dim}")
    else:
        print(f"Using default latent dimension: {latent_dim}. Could not infer from training dataset.")
        # return # Critical if latent_dim cannot be inferred
    # else: # Should be caught by the empty train_dataset check above
    #     print("Cannot infer latent_dim as training dataset is empty.")
    #     return

    # Initialize adapter
    adapter = Adapter(latent_dim).to(DEVICE)
    optimizer = optim.Adam(adapter.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Load decoder models for evaluation phase
    print("Loading decoder models for evaluation...")
    try:
        decoder_models_for_eval = load_decoder_models(SMILES_CHECKPOINT_PATH, RXN_MATRIX_PATH, FPINDEX_PATH, DEVICE)
        print("Decoder models for evaluation loaded successfully.")
    except Exception as e:
        print(f"Error loading decoder models: {e}. Evaluation will be skipped.")
        decoder_models_for_eval = None

    # --- Load Receptor Center Information for Docking --- 
    receptor_centers_df = None
    if os.path.exists(RECEPTOR_INFO_CSV_PATH):
        try:
            receptor_centers_df = pd.read_csv(RECEPTOR_INFO_CSV_PATH)
            # Assuming the CSV has a 'pdb' column for receptor IDs, matching 'receptor_id' from the main data CSV
            receptor_centers_df.set_index('pdb', inplace=True) 
            print(f"Successfully loaded receptor center information from {RECEPTOR_INFO_CSV_PATH}")
        except Exception as e_csv:
            print(f"Error loading receptor center CSV {RECEPTOR_INFO_CSV_PATH}: {e_csv}. Docking for decoded molecules will be skipped.")
            receptor_centers_df = None # Ensure it's None on error
    else:
        print(f"Receptor center CSV not found at {RECEPTOR_INFO_CSV_PATH}. Docking for decoded molecules will be skipped.")

    print(f"Starting supervised training for adapter on {DEVICE}...")
    for epoch in range(num_epochs):
        adapter.train() # Ensure adapter is in training mode
        epoch_loss = 0.0
        processed_batches = 0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            z_input = batch_data['latent_input'].to(DEVICE)
            target_dz = batch_data['target_delta_z'].to(DEVICE)

            # Predict delta_z
            predicted_dz = adapter(z_input)
            
            # Compute MSE loss
            loss = loss_fn(predicted_dz, target_dz)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            processed_batches += 1
            
            # Minimal logging per batch to reduce verbosity with tqdm
            if batch_idx % (len(train_loader) // 4 + 1) == 0 and batch_idx > 0 : # Log ~4 times per epoch
                 tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)-1} - Current Batch Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss / processed_batches if processed_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_epoch_loss:.6f}")
        
        # Optional: Save model checkpoint
        # torch.save(adapter.state_dict(), f"adapter_epoch_{epoch+1}.pt")

    print("Adapter training finished.")

    # Save the trained adapter model
    adapter_save_dir = "/workspace/synformer/adapter"
    os.makedirs(adapter_save_dir, exist_ok=True)
    adapter_save_path = os.path.join(adapter_save_dir, "adapter_model.pkl")
    try:
        with open(adapter_save_path, 'wb') as f:
            pickle.dump(adapter.state_dict(), f)
        print(f"Trained adapter state_dict saved to {adapter_save_path}")
    except Exception as e:
        print(f"Error saving adapter model: {e}")

    # Example of how to use it at inference (conceptual)
    # z_new_fragment = ... load or compute ...
    # z_new_fragment_tensor = torch.tensor(z_new_fragment, dtype=torch.float32).to(DEVICE)
    # predicted_shift = adapter(z_new_fragment_tensor.unsqueeze(0)) # Add batch dim
    # z_prime = z_new_fragment_tensor + predicted_shift.squeeze(0) # Remove batch dim
    # Now decode z_prime using the kept decoder functions

    # Perform evaluation on the TEST dataset
    if decoder_models_for_eval and len(test_dataset) > 0:
        evaluate_adapter(adapter, test_dataset, decoder_models_for_eval, DEVICE, 
                         num_samples_to_evaluate=min(num_samples_for_eval, len(test_dataset)),
                         receptor_centers_df=receptor_centers_df) # Pass receptor_centers_df
    elif len(test_dataset) == 0:
        print("Skipping evaluation as the test dataset is empty.")
    else:
        print("Skipping evaluation as decoder models could not be loaded.")

if __name__ == "__main__":
    main() 