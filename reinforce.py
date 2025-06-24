import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal # For sampling delta_z
from torch.utils.tensorboard import SummaryWriter # Added for TensorBoard
import numpy as np
import pickle
import json # For parsing latent string
from rdkit import Chem
from rdkit.Chem import QED

# --- Synformer/DESERT components (to be copied or adapted from excellent_checkadaptertraining.py) ---
# Placeholder for imports from Synformer - these will be extensive
from synformer.models.decoder import Decoder # Assuming this path is correct or will be made correct
from synformer.models.classifier_head import ClassifierHead
from synformer.models.fingerprint_head import get_fingerprint_head
from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.data.common import TokenType
from synformer.chem.stack import Stack # For generation
from synformer.chem.mol import Molecule # For generation
from omegaconf import OmegaConf

# Configuration (paths might need adjustment)
SMILES_CHECKPOINT_PATH = "/workspace/data/processed/sf_ed_default.ckpt"
RXN_MATRIX_PATH = "/workspace/data/processed/comp_2048/matrix.pkl"
FPINDEX_PATH = "/workspace/data/processed/comp_2048/fpindex.pkl"
CSV_PATH = "fragments_qed_docking_with_latents.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -20  # Minimum value for log_std to prevent numerical instability
LOG_STD_MAX = 2    # Maximum value for log_std
TENSORBOARD_LOG_DIR = "runs/qed_adapter_reinforce" # For TensorBoard logs

# --- Adapter Network (Modified for REINFORCE) ---
class Adapter(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        self.fc_mu = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.fc_log_std = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z):
        mu = self.fc_mu(z)
        log_std = self.fc_log_std(z)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # Clamp log_std for stability
        std = torch.exp(log_std)
        
        # Create a Normal distribution
        dist = Normal(mu, std)
        
        # Sample delta_z
        # Using rsample() allows for reparameterization trick if we were doing VAE-like things,
        # but for REINFORCE, simple sample() is fine. However, rsample() is generally preferred.
        delta_z_sampled = dist.rsample() 
        
        # Calculate log probability of the sample
        log_prob = dist.log_prob(delta_z_sampled).sum(dim=-1) # Sum over latent dimensions
        
        return delta_z_sampled, log_prob, mu, std

# --- Dataset ---
class QEDLatentDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Filter out rows where latent_representation or qed_score is NaN or invalid
        df = df.dropna(subset=['latent_representation', 'qed_score'])
        df = df[df['latent_representation'].apply(lambda x: isinstance(x, str) and '[' in x)] # Basic check for list-like string

        self.latents = []
        self.qeds = []

        for _, row in df.iterrows():
            try:
                # Latent representation is stored as a string of a list
                latent_str = row['latent_representation']
                latent_list = json.loads(latent_str)
                self.latents.append(torch.tensor(latent_list, dtype=torch.float32))
                self.qeds.append(torch.tensor(float(row['qed_score']), dtype=torch.float32))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Skipping row due to parsing error: {e} for latent '{row['latent_representation']}' or QED '{row['qed_score']}'")
                continue
        
        if not self.latents:
            raise ValueError("No valid data loaded from CSV. Check CSV format and content.")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {"latent": self.latents[idx], "qed_original": self.qeds[idx]}

# --- Decoder and Generation Functions ---
def load_decoder_models(checkpoint_path, rxn_matrix_path, fpindex_path, device):
    """Loads the Synformer decoder, heads, and other necessary components."""
    full_model_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    ).to(device)
    decoder_state_dict = {k.replace('model.decoder.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.decoder.')}
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()

    token_head = ClassifierHead(config.model.decoder.d_model, max(TokenType) + 1).to(device)
    token_head_state_dict = {k.replace('model.token_head.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.token_head.')}
    token_head.load_state_dict(token_head_state_dict)
    token_head.eval()

    reaction_head = ClassifierHead(config.model.decoder.d_model, config.model.decoder.num_reaction_classes).to(device)
    reaction_head_state_dict = {k.replace('model.reaction_head.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.reaction_head.')}
    reaction_head.load_state_dict(reaction_head_state_dict)
    reaction_head.eval()
    
    fingerprint_head = get_fingerprint_head(config.model.fingerprint_head_type, config.model.fingerprint_head).to(device)
    fingerprint_head_state_dict = {k.replace('model.fingerprint_head.', ''): v for k, v in full_model_checkpoint['state_dict'].items() if k.startswith('model.fingerprint_head.')}
    fingerprint_head.load_state_dict(fingerprint_head_state_dict)
    fingerprint_head.eval()

    rxn_matrix = pickle.load(open(rxn_matrix_path, 'rb'))
    fpindex = pickle.load(open(fpindex_path, 'rb'))

    return {
        "decoder": decoder, "token_head": token_head, "reaction_head": reaction_head,
        "fingerprint_head": fingerprint_head, "rxn_matrix": rxn_matrix, "fpindex": fpindex,
        "config": config
    }

def generate_smiles_from_latent(z_prime_single, models, max_steps=24, temperature=0.1):
    """Generates a single SMILES string from a single z_prime vector."""
    decoder = models["decoder"]
    token_head = models["token_head"]
    reaction_head = models["reaction_head"]
    fingerprint_head = models["fingerprint_head"]
    rxn_matrix = models["rxn_matrix"]
    fpindex = models["fpindex"]
    device = z_prime_single.device

    # z_prime_single is [D_latent]. Decoder expects code: [1, seq_len_code, D_latent]
    # Here, seq_len_code = 1 as z_prime is a pooled representation.
    code_input = z_prime_single.unsqueeze(0).unsqueeze(0) # -> [1, 1, D_latent]
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


def decode_batch(z_prime_batch, models, max_steps=24, temperature=0.1):
    """Decodes a batch of z_prime vectors into SMILES strings."""
    all_smiles = []
    for i in range(z_prime_batch.shape[0]):
        z_single = z_prime_batch[i]
        smiles = generate_smiles_from_latent(z_single, models, max_steps, temperature)
        all_smiles.append(smiles if smiles else "") # Append empty string if generation failed
    return all_smiles

def calculate_qed_batch(smiles_list):
    """Calculates QED for a list of SMILES strings."""
    qeds = []
    for smiles in smiles_list:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                qeds.append(QED.qed(mol))
            else:
                qeds.append(0.0) # Invalid SMILES
        else:
            qeds.append(0.0) # Empty SMILES
    return torch.tensor(qeds, dtype=torch.float32)


# --- Main Training Loop (Modified for REINFORCE) ---
def main():
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR) # Initialize TensorBoard writer

    num_epochs = 20 # Increased epochs for RL
    batch_size = 16 # Or from args
    learning_rate = 1e-5 # RL often requires smaller learning rates
    latent_dim = 768 # Should match the output of FragmentEncoder / latents in CSV
    gamma = 0.99 # Discount factor for future rewards (not strictly used here as we get immediate QED)
    # For REINFORCE, often an entropy bonus is added to the loss to encourage exploration
    entropy_coefficient = 0.01 

    # Load data
    print(f"Loading dataset from {CSV_PATH}...")
    try:
        dataset = QEDLatentDataset(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_PATH}. Make sure it's in the execution directory or provide an absolute path.")
        writer.close()
        return
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        writer.close()
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Set num_workers=0 for easier debugging initially
    
    # Infer latent_dim from the first item if not hardcoded
    if hasattr(dataset, 'latents') and dataset.latents:
        latent_dim = dataset.latents[0].shape[0]
        print(f"Inferred latent dimension: {latent_dim}")
    else:
        print(f"Critical error: Dataset empty or latent dim could not be inferred.")
        # Potentially exit if dataset is empty and dim not confirmed. Here, it's handled by QEDLatentDataset raising error.


    # Load decoder models
    print("Loading decoder models...")
    # Ensure synformer modules are in PYTHONPATH or adjust paths
    # For now, assuming they are accessible
    decoder_models = load_decoder_models(SMILES_CHECKPOINT_PATH, RXN_MATRIX_PATH, FPINDEX_PATH, DEVICE)
    print("Decoder models loaded.")

    # Initialize adapter
    adapter = Adapter(latent_dim).to(DEVICE)
    optimizer = optim.Adam(adapter.parameters(), lr=learning_rate)
    # No explicit loss_fn like MSE for REINFORCE, loss is constructed directly

    print(f"Starting REINFORCE training on {DEVICE}...")
    print(f"TensorBoard logs will be saved to: {writer.log_dir}")
    
    # For moving average baseline
    moving_avg_reward = 0.0
    first_batch_for_baseline = True

    for epoch in range(num_epochs):
        total_epoch_loss = 0.0
        total_qed_reward = 0.0
        total_epoch_entropy = 0.0 # For logging average epoch entropy
        processed_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            z = batch['latent'].to(DEVICE)
            
            # Get delta_z_sampled and log_prob from the adapter
            delta_z_sampled, log_probs, mu, std = adapter(z) # adapter now returns log_prob
            z_prime = z + delta_z_sampled

            smiles_decoded = decode_batch(z_prime, decoder_models)
            
            # Rewards are the QED scores
            rewards = calculate_qed_batch(smiles_decoded).to(DEVICE)
            
            # Baseline: subtract the moving average of rewards (or batch mean for simplicity initially)
            if first_batch_for_baseline and epoch == 0: # Initialize moving average with first batch's mean
                 moving_avg_reward = rewards.mean().item()
                 first_batch_for_baseline = False
            
            advantages = rewards - moving_avg_reward
            
            # Update moving average baseline (simple exponential moving average)
            moving_avg_reward = 0.99 * moving_avg_reward + 0.01 * rewards.mean().item()


            # REINFORCE loss: - (log_probs * advantages).mean()
            # We want to maximize rewards, so we minimize the negative of this.
            policy_loss = -(log_probs * advantages.detach()).mean() # Detach advantages as they shouldn't contribute to policy gradient directly

            # Entropy bonus to encourage exploration (optional but often helpful)
            # dist = Normal(mu, std) # Recreate dist for entropy, or pass std from adapter
            entropy = Normal(mu, std).entropy().mean() # Mean entropy over batch and latent_dim
            
            loss = policy_loss - entropy_coefficient * entropy

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0) # Optional: gradient clipping
            optimizer.step()

            total_epoch_loss += loss.item()
            total_qed_reward += rewards.mean().item()
            total_epoch_entropy += entropy.item() # Accumulate entropy
            processed_batches += 1
            
            if batch_idx % 1 == 0: 
                 print(f"Epoch {epoch}/{num_epochs-1} - Batch {batch_idx}/{len(dataloader)-1} - Loss: {loss.item():.4f} - Avg QED: {rewards.mean().item():.4f} - Baseline: {moving_avg_reward:.4f} - Entropy: {entropy.item():.4f}")
                 # Batch-level TensorBoard logging (optional, can be verbose)
                 # global_step = epoch * len(dataloader) + batch_idx
                 # writer.add_scalar('Loss/Batch', loss.item(), global_step)
                 # writer.add_scalar('Reward/Avg_QED_Batch', rewards.mean().item(), global_step)
                 # writer.add_scalar('Policy/Entropy_Batch', entropy.item(), global_step)
                 # writer.add_scalar('Policy/Baseline_Batch', moving_avg_reward, global_step)

        avg_epoch_loss = total_epoch_loss / processed_batches if processed_batches > 0 else 0
        avg_epoch_qed = total_qed_reward / processed_batches if processed_batches > 0 else 0
        avg_epoch_entropy = total_epoch_entropy / processed_batches if processed_batches > 0 else 0
        
        print(f"Epoch {epoch} Summary - Avg Loss: {avg_epoch_loss:.4f} - Avg QED Reward: {avg_epoch_qed:.4f} - Avg Entropy: {avg_epoch_entropy:.4f}")
        
        # Log epoch-level summaries to TensorBoard
        writer.add_scalar('Loss/Epoch_Avg', avg_epoch_loss, epoch)
        writer.add_scalar('Reward/Epoch_Avg_QED', avg_epoch_qed, epoch)
        writer.add_scalar('Policy/Epoch_Avg_Entropy', avg_epoch_entropy, epoch)
        writer.add_scalar('Policy/Epoch_Baseline_End', moving_avg_reward, epoch)

        avg_epoch_loss = epoch_loss / processed_batches if processed_batches > 0 else 0
        print(f"Epoch {epoch} - Average Loss: {avg_epoch_loss:.4f}")
        # Potentially save model checkpoints here

if __name__ == "__main__":
    # This is a placeholder for potentially needing to adjust PYTHONPATH
    # to find synformer modules if they are not installed in the environment.
    # import sys
    # sys.path.append('/path/to/your/synformer_project_root') # Example
    main() 