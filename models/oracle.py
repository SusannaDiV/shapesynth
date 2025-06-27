import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from pmnet_appl.tacogfn_reward import TacoGFN_Proxy
import logging
from typing import Optional, Dict, List, Tuple, Union
from synformer.models.synformer import Synformer
from synformer.data.common import ProjectionBatch

logger = logging.getLogger(__name__)

your_data_list = [...]

LATENT_DIM = len(your_data_list[0]['latent'])

class DockingDataset(Dataset):
    def __init__(self, data_list: List[Dict]):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'latent': torch.tensor(item['latent'], dtype=torch.float32),
            'pocket_id': item['pocket_id'],
            'smiles': item['smiles'],
            'score': torch.tensor(item['score'], dtype=torch.float32)
        }

dataset = DockingDataset(your_data_list)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class LatentOptimizer(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, latent_vec: torch.Tensor) -> torch.Tensor:
        delta = self.model(latent_vec)
        return latent_vec + delta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LatentOptimizer(LATENT_DIM).to(device)

proxy = TacoGFN_Proxy.load(
    scoring_fn='QVina', 
    train_dataset='ZINCDock15M', 
    cache_db='train', 
    device=device
)

def decode_latent(latent_vec: torch.Tensor) -> str:
    raise NotImplementedError('Replace decode_latent with your decoder')

optimizer = Adam(model.parameters(), lr=1e-4)
lambda_reg = 0.1
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        latent = batch['latent'].to(device)
        pocket_ids = batch['pocket_id']

        optimized_latent = model(latent)

        decoded_smiles = []
        for vec in optimized_latent:
            decoded_smiles.append(decode_latent(vec))

        proxy_scores = []
        for pid, smi in zip(pocket_ids, decoded_smiles):
            score = proxy.scoring(pid, smi)
            proxy_scores.append(score)
        proxy_scores = torch.tensor(proxy_scores, dtype=torch.float32, device=device)

        reg_term = ((optimized_latent - latent) ** 2).mean()
        loss = -proxy_scores.mean() + lambda_reg * reg_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * latent.size(0)

    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'latent_optimizer.pth')
print('Training complete. Model saved to latent_optimizer.pth')

class DockingOracle(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        latent_dim: Optional[int] = None,
        pretrained_weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        synformer_model: Optional[Synformer] = None,
        optimization_steps: int = 10,
        step_size: float = 0.1,
        regularization_weight: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.synformer = synformer_model
        
        self.optimization_steps = optimization_steps
        self.step_size = step_size
        self.regularization_weight = regularization_weight
        
        self.network = None
        
        self.proxy = TacoGFN_Proxy.load(
            scoring_fn='QVina',
            train_dataset='CrossDocked2020', 
            cache_db='train',
            device=self.device
        )
        
        if pretrained_weights_path is not None:
            self.load_pretrained(pretrained_weights_path)
            
        self.to(self.device)

    def init_network(self, latent_dim: int):
        if self.network is not None:
            return
            
        layers = []
        dims = [latent_dim] + [self.hidden_dim] * (self.num_layers - 1) + [1]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            
        layers = layers[:-2]
        
        self.network = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            self.init_network(latent_vectors.shape[1])
            
        return self.network(latent_vectors)

    def optimize_latent(
        self, 
        latent_vector: torch.Tensor,
        target_pocket_id: str,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[float]]]:
        if num_steps is None:
            num_steps = self.optimization_steps
            
        latent = latent_vector.clone().detach().to(self.device)
        latent.requires_grad_(True)
        
        best_latent = latent.clone()
        best_score = float('inf')
        
        score_trajectory = []
        
        for step in range(num_steps):
            pred_score = self(latent)
            
            reg_loss = self.regularization_weight * torch.norm(latent - latent_vector, dim=1).mean()
            
            loss = pred_score.mean() + reg_loss
            
            loss.backward()
            
            with torch.no_grad():
                latent_grad = latent.grad.clone()
                latent -= self.step_size * latent_grad
                
                try:
                    smiles = self.decode_latents(latent.detach())
                    if smiles[0]:
                        actual_score = self.proxy.scoring(target_pocket_id, smiles[0])
                        score_trajectory.append(actual_score)
                        
                        if actual_score < best_score:
                            best_score = actual_score
                            best_latent = latent.clone()
                except Exception as e:
                    logger.warning(f"Failed to validate at step {step}: {e}")
                
            latent.grad.zero_()
            
        if return_trajectory:
            return best_latent.detach(), score_trajectory
        return best_latent.detach()

    def batch_optimize_latents(
        self,
        latent_vectors: torch.Tensor,
        target_pocket_ids: List[str],
        **kwargs
    ) -> torch.Tensor:
        optimized_latents = []
        for latent, pocket_id in zip(latent_vectors, target_pocket_ids):
            opt_latent = self.optimize_latent(
                latent.unsqueeze(0),
                target_pocket_id=pocket_id,
                **kwargs
            )
            optimized_latents.append(opt_latent)
        return torch.cat(optimized_latents, dim=0)

    def decode_latents(self, latent_vectors: torch.Tensor) -> List[str]:
        if self.synformer is None:
            raise ValueError("Synformer model not provided - cannot decode latents")
            
        if not hasattr(self.synformer, 'rxn_matrix'):
            raise ValueError("Synformer model missing rxn_matrix - ensure it's properly initialized")
        if not hasattr(self.synformer, 'fpindex'):
            raise ValueError("Synformer model missing fpindex - ensure it's properly initialized")

        batch = {
            'code': latent_vectors,
            'code_padding_mask': torch.zeros(latent_vectors.size(0), 1, dtype=torch.bool, device=latent_vectors.device)
        }

        try:
            with torch.no_grad():
                result = self.synformer.generate_without_stack(
                    batch=ProjectionBatch(**batch),
                    rxn_matrix=self.synformer.rxn_matrix,
                    fpindex=self.synformer.fpindex,
                    max_len=32,
                    temperature_token=0.1,
                    temperature_reaction=0.1,
                    temperature_reactant=0.1
                )

                smiles_list = []
                for stack in result.build():
                    if stack is None or stack.is_empty():
                        logger.warning("Empty molecule stack generated")
                        smiles_list.append("")
                        continue
                        
                    try:
                        mol = stack.get_product()
                        if mol is None:
                            logger.warning("No product molecule in stack")
                            smiles_list.append("")
                            continue
                            
                        smiles = mol.to_smiles()
                        if not smiles:
                            logger.warning("Empty SMILES string generated")
                            smiles_list.append("")
                            continue
                            
                        smiles_list.append(smiles)
                    except Exception as e:
                        logger.warning(f"Failed to convert stack to SMILES: {str(e)}")
                        smiles_list.append("")

                return smiles_list
                
        except Exception as e:
            logger.error(f"Error during molecule generation: {str(e)}")
            return [""] * latent_vectors.size(0)
            
    def train_model(
        self,
        train_data: List[Dict],
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        validation_split: float = 0.1,
        save_path: Optional[str] = None
    ):
        if self.synformer is None:
            raise ValueError("Synformer model required for training")
            
        required_keys = {'latent', 'pocket_id'}
        if not all(required_keys.issubset(d.keys()) for d in train_data):
            raise ValueError(f"Training data must contain all required keys: {required_keys}")
        
        num_val = int(len(train_data) * validation_split)
        train_data, val_data = train_data[:-num_val], train_data[-num_val:]
        
        train_dataset = DockingDataset(train_data)
        val_dataset = DockingDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            num_valid_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                latents = batch['latent'].to(self.device)
                pocket_ids = batch['pocket_id']
                
                try:
                    generated_smiles = self.decode_latents(latents)
                    
                    actual_scores = []
                    valid_indices = []
                    
                    for idx, (smi, pid) in enumerate(zip(generated_smiles, pocket_ids)):
                        if not smi:
                            continue
                        try:
                            score = self.proxy.scoring(pid, smi)
                            actual_scores.append(score)
                            valid_indices.append(idx)
                        except Exception as e:
                            logger.warning(f"Failed to score molecule: {str(e)}")
                    
                    if not actual_scores:
                        continue
                        
                    actual_scores = torch.tensor(actual_scores, device=self.device)
                    valid_latents = latents[valid_indices]
                    
                    pred_scores = self(valid_latents).squeeze()
                    
                    loss = criterion(pred_scores, actual_scores)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_valid_batches += 1
                    
                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = train_loss / num_valid_batches if num_valid_batches > 0 else float('inf')
                        logger.info(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}:')
                        logger.info(f'  Loss: {avg_loss:.4f}')
                        logger.info(f'  Valid Molecules: {len(actual_scores)}/{len(latents)}')
                        
                except Exception as e:
                    logger.error(f"Failed to process batch: {str(e)}")
                    continue
            
            if num_valid_batches > 0:
                train_loss /= num_valid_batches
            
            self.eval()
            val_loss = 0.0
            num_valid_val = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    latents = batch['latent'].to(self.device)
                    pocket_ids = batch['pocket_id']
                    
                    try:
                        generated_smiles = self.decode_latents(latents)
                        actual_scores = []
                        valid_indices = []
                        
                        for idx, (smi, pid) in enumerate(zip(generated_smiles, pocket_ids)):
                            if not smi:
                                continue
                            try:
                                score = self.proxy.scoring(pid, smi)
                                actual_scores.append(score)
                                valid_indices.append(idx)
                            except Exception as e:
                                continue
                        
                        if not actual_scores:
                            continue
                            
                        actual_scores = torch.tensor(actual_scores, device=self.device)
                        valid_latents = latents[valid_indices]
                        
                        pred_scores = self(valid_latents).squeeze()
                        loss = criterion(pred_scores, actual_scores)
                        
                        val_loss += loss.item()
                        num_valid_val += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process validation batch: {str(e)}")
                        continue
            
            if num_valid_val > 0:
                val_loss /= num_valid_val
                
                logger.info(f'Epoch {epoch+1}/{num_epochs}:')
                logger.info(f'  Train Loss: {train_loss:.4f}')
                logger.info(f'  Val Loss: {val_loss:.4f}')
                
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_path)
                    logger.info(f'  Saved new best model to {save_path}')

    def load_pretrained(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(state_dict)
        logger.info(f'Loaded pretrained weights from {weights_path}')
