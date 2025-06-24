import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from synformer.models.synformer import Synformer
from synformer.data.desert import DesertSequence
from pmnet_appl.tacogfn_reward import TacoGFN_Proxy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_synformer_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Synformer:
    """Load pretrained Synformer model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = Synformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model

def encode_smiles_to_latent(
    model: Synformer,
    smiles_list: List[str],
    device: torch.device
) -> torch.Tensor:
    """
    Encode SMILES strings to latent vectors using fragment encoder
    
    Args:
        model: Pretrained Synformer model
        smiles_list: List of SMILES strings
        device: torch device
    
    Returns:
        Tensor of latent vectors (n_samples, latent_dim)
    """
    latents = []
    
    for smiles in tqdm(smiles_list, desc="Encoding SMILES"):
        try:
            # Convert SMILES to DESERT sequence
            desert_seq = DesertSequence.from_smiles(smiles)
            if desert_seq is None:
                logger.warning(f"Failed to convert SMILES to DESERT: {smiles}")
                continue
                
            # Encode sequence
            with torch.no_grad():
                embeddings = model.encoder.encode_desert_sequence(desert_seq, device=device)
                latents.append(embeddings.cpu())
                
        except Exception as e:
            logger.warning(f"Failed to encode SMILES {smiles}: {str(e)}")
            continue
            
    if not latents:
        raise ValueError("No SMILES could be encoded successfully")
        
    return torch.stack(latents)

def get_docking_scores(
    proxy: TacoGFN_Proxy,
    pocket_ids: List[str],
    smiles_list: List[str]
) -> List[float]:
    """Get docking scores for SMILES-pocket pairs"""
    scores = []
    
    for pid, smi in tqdm(zip(pocket_ids, smiles_list), desc="Computing docking scores"):
        try:
            score = proxy.scoring(pid, smi)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Failed to score {smi} with {pid}: {str(e)}")
            scores.append(None)
            
    return scores

def prepare_docking_dataset(
    smiles_file: str,
    pocket_ids_file: str,
    synformer_checkpoint: str,
    output_file: str,
    device: Optional[torch.device] = None,
    cache_db: str = "train"
) -> None:
    """
    Prepare dataset for docking oracle training
    
    Args:
        smiles_file: Path to file containing SMILES strings (one per line)
        pocket_ids_file: Path to file containing pocket IDs (one per line)
        synformer_checkpoint: Path to pretrained Synformer checkpoint
        output_file: Path to save prepared dataset
        device: torch device
        cache_db: Cache database for TacoGFN proxy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load inputs
    with open(smiles_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    with open(pocket_ids_file) as f:
        pocket_ids = [line.strip() for line in f if line.strip()]
        
    if len(smiles_list) != len(pocket_ids):
        raise ValueError(f"Number of SMILES ({len(smiles_list)}) must match number of pocket IDs ({len(pocket_ids)})")
        
    # Load models
    logger.info("Loading models...")
    synformer = load_synformer_model(synformer_checkpoint, device)
    proxy = TacoGFN_Proxy.load(
        scoring_fn='QVina',
        train_dataset='CrossDocked2020',
        cache_db=cache_db,
        device=device
    )
    
    # Generate latents
    logger.info("Encoding SMILES to latents...")
    latents = encode_smiles_to_latent(synformer, smiles_list, device)
    
    # Get docking scores
    logger.info("Computing docking scores...")
    scores = get_docking_scores(proxy, pocket_ids, smiles_list)
    
    # Prepare dataset
    dataset = []
    for smi, pid, lat, score in zip(smiles_list, pocket_ids, latents, scores):
        if score is not None:  # Only include valid examples
            dataset.append({
                'smiles': smi,
                'pocket_id': pid,
                'latent': lat.numpy().tolist(),
                'score': score
            })
    
    # Save dataset
    import json
    with open(output_file, 'w') as f:
        json.dump(dataset, f)
    
    logger.info(f"Saved {len(dataset)} examples to {output_file}")
    logger.info(f"Example data point: {dataset[0]}")

def main():
    parser = argparse.ArgumentParser(description="Prepare docking oracle training data")
    parser.add_argument("--smiles", required=True, help="Path to SMILES file")
    parser.add_argument("--pockets", required=True, help="Path to pocket IDs file")
    parser.add_argument("--checkpoint", required=True, help="Path to Synformer checkpoint")
    parser.add_argument("--output", required=True, help="Path to save dataset")
    parser.add_argument("--cache-db", default="train", help="Cache database for TacoGFN")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    prepare_docking_dataset(
        smiles_file=args.smiles,
        pocket_ids_file=args.pockets,
        synformer_checkpoint=args.checkpoint,
        output_file=args.output,
        device=device,
        cache_db=args.cache_db
    )

if __name__ == "__main__":
    main() 