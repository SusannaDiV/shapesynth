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
        valid_indices = []

        print("Parsing initial latents and QED scores...")
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Parsing DataFrame"):
            try:
                latent_list = json.loads(row['latent_representation'])
                qed_score = float(row['qed_score'])
                # Ensure latent is a flat list of numbers (e.g. 768 numbers)
                if isinstance(latent_list, list) and all(isinstance(x, (int, float)) for x in latent_list):
                    raw_latents.append(torch.tensor(latent_list, dtype=torch.float32))
                    raw_qeds.append(qed_score)
                    valid_indices.append(idx)
                else:
                    # print(f"Skipping row {idx} due to unexpected latent format: {type(latent_list)}")
                    pass

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                # print(f"Skipping row {idx} due to parsing error: {e}")
                pass
        
        if not raw_latents:
            raise ValueError("No valid raw latents loaded. Check DataFrame format and content of 'latent_representation' and 'qed_score'.")

        print(f"Successfully parsed {len(raw_latents)} initial latents and QEDs.")
        self.input_latents = []
        self.target_delta_zs = []
        self.original_qeds_for_input_latents = [] # Store original QED for evaluation reference

        print("Calculating target oracle shifts (Δz)...")
        # Find the global best QED and its corresponding latent (among valid ones)
        best_global_qed = -1.0
        best_global_latent = None
        if raw_qeds:
            best_global_qed_idx = np.argmax(raw_qeds)
            best_global_qed = raw_qeds[best_global_qed_idx]
            best_global_latent = raw_latents[best_global_qed_idx]
            print(f"Global best QED in dataset: {best_global_qed:.4f}")

        for i in tqdm(range(len(raw_latents)), desc="Calculating Shifts"):
            current_z = raw_latents[i]
            current_qed = raw_qeds[i]
            self.input_latents.append(current_z)
            self.original_qeds_for_input_latents.append(current_qed)

            target_delta_z = torch.zeros_like(current_z) # Default to zero shift

            if current_qed < max_qed_considered and current_qed >= min_qed_for_target_search :
                best_candidate_target_z = None
                highest_qed_found_for_current_z = current_qed

                # Option 1: Shift towards the global best if it's significantly better
                if best_global_latent is not None and best_global_qed > current_qed + qed_improvement_threshold:
                    best_candidate_target_z = best_global_latent
                    highest_qed_found_for_current_z = best_global_qed # not strictly necessary to update this here
                
                # Option 2: (Could be added) Search for other local improvements.
                # For simplicity, we primarily use the global best as the main target if it's an improvement.
                # If not using global best or want more diverse targets, one could iterate through all raw_latents[j]
                # for j != i, check if raw_qeds[j] > current_qed + threshold, and pick the best among those.
                # For now, the logic is simpler: aim for global best if it's a good jump.

                if best_candidate_target_z is not None:
                    target_delta_z = best_candidate_target_z - current_z
            
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
            "original_qed": self.original_qeds_for_input_latents[idx]
        }


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
def evaluate_adapter(adapter, dataset, decoder_models, device, num_samples_to_evaluate=10):
    print(f"\n--- Starting Evaluation (on {num_samples_to_evaluate} samples) ---")
    adapter.eval() # Set adapter to evaluation mode

    if len(dataset) == 0:
        print("Evaluation dataset is empty.")
        return

    num_samples = min(num_samples_to_evaluate, len(dataset))
    
    original_qeds_csv_list = [] # Renamed to avoid conflict
    baseline_decoded_qeds_list = [] # Renamed
    adapted_decoded_qeds_list = [] # Renamed
    evaluation_details = [] # To store tuples for printing later
    num_adapted_better_than_baseline = 0 # Counter for improvements

    # Use a DataLoader for easier batching if num_samples_to_evaluate is large, or just iterate for small N
    eval_indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i in tqdm(range(num_samples), desc="Evaluating Adapter"):
        idx = eval_indices[i]
        sample_data = dataset[idx]
        z_original = sample_data["latent_input"].to(device)
        original_qed_from_csv = sample_data["original_qed"] # This is a float
        original_qeds_csv_list.append(original_qed_from_csv)

        # 1. Baseline: Decode original latent
        smiles_baseline = generate_smiles_from_latent(z_original, decoder_models)
        qed_baseline_decoded = calculate_qed_from_smiles(smiles_baseline)
        baseline_decoded_qeds_list.append(qed_baseline_decoded)

        # 2. Adapted: Apply adapter, then decode
        with torch.no_grad(): # No gradients needed for adapter during eval
            predicted_delta_z = adapter(z_original.unsqueeze(0)) # Add batch dim
        z_prime = z_original + predicted_delta_z.squeeze(0) # Remove batch dim
        
        smiles_adapted = generate_smiles_from_latent(z_prime, decoder_models)
        qed_adapted_decoded = calculate_qed_from_smiles(smiles_adapted)
        adapted_decoded_qeds_list.append(qed_adapted_decoded)

        # Increment counter if adapted is better
        if qed_adapted_decoded > qed_baseline_decoded:
            num_adapted_better_than_baseline += 1

        # Store details for printing later or for all samples
        evaluation_details.append({
            "original_z_idx": idx, # if you want to map back to original df index
            "original_qed_csv": original_qed_from_csv,
            "smiles_baseline": smiles_baseline if smiles_baseline else 'N/A',
            "qed_baseline_decoded": qed_baseline_decoded,
            "smiles_adapted": smiles_adapted if smiles_adapted else 'N/A',
            "qed_adapted_decoded": qed_adapted_decoded
        })

        if i < 5: # Print first few examples immediately during evaluation
            print(f"\n  Eval Sample {i+1} (Dataset Index: {idx}):")
            print(f"    Original QED (from CSV): {original_qed_from_csv:.4f}")
            print(f"    Baseline Decoded SMILES: {smiles_baseline if smiles_baseline else 'N/A'} | QED: {qed_baseline_decoded:.4f}")
            print(f"    Adapted Decoded SMILES:  {smiles_adapted if smiles_adapted else 'N/A'} | QED: {qed_adapted_decoded:.4f}")

    avg_original_qed_csv = np.mean(original_qeds_csv_list) if original_qeds_csv_list else 0
    avg_baseline_decoded_qed = np.mean(baseline_decoded_qeds_list) if baseline_decoded_qeds_list else 0
    avg_adapted_decoded_qed = np.mean(adapted_decoded_qeds_list) if adapted_decoded_qeds_list else 0

    print("\nEvaluation Summary:")
    print(f"  Avg QED from CSV (for selected samples): {avg_original_qed_csv:.4f}")
    print(f"  Avg QED from decoding original latents:  {avg_baseline_decoded_qed:.4f}")
    print(f"  Avg QED from decoding adapted latents:   {avg_adapted_decoded_qed:.4f}")
    
    improvement_over_baseline = avg_adapted_decoded_qed - avg_baseline_decoded_qed
    print(f"  Avg improvement over baseline decoding:    {improvement_over_baseline:+.4f}")
    print(f"  Number of adapted SMILES with QED > baseline QED: {num_adapted_better_than_baseline}/{num_samples}")

    print("\n--- Examples of Evaluated Generated SMILES (Adapted Latents) ---")
    num_to_print = min(num_samples, 5) # Print up to 5 examples, or all if less than 5 evaluated
    for i in range(num_to_print):
        detail = evaluation_details[i]
        print(f"  Example {i+1} (Original CSV QED: {detail['original_qed_csv']:.4f}):")
        print(f"    Baseline SMILES: {detail['smiles_baseline']} (QED: {detail['qed_baseline_decoded']:.4f})")
        print(f"    Adapted SMILES:  {detail['smiles_adapted']} (QED: {detail['qed_adapted_decoded']:.4f})")
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
            return # Critical if latent_dim cannot be inferred
    else: # Should be caught by the empty train_dataset check above
        print("Cannot infer latent_dim as training dataset is empty.")
        return

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
        evaluate_adapter(adapter, test_dataset, decoder_models_for_eval, DEVICE, num_samples_to_evaluate=min(num_samples_for_eval, len(test_dataset)))
    elif len(test_dataset) == 0:
        print("Skipping evaluation as the test dataset is empty.")
    else:
        print("Skipping evaluation as decoder models could not be loaded.")

if __name__ == "__main__":
    main() 