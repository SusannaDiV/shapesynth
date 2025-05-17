import torch
import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from synformer.models.synformer import Synformer
from synformer.data.common import ProjectionBatch
from synformer.models.desert.inference import run_desert_inference
from synformer.models.desert.encoder import create_fragment_encoder
from pmnet_appl.tacogfn_reward import TacoGFN_Proxy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_synformer_model(
    checkpoint_path: str,
    decoder_checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Synformer:
    """Load pretrained Synformer model with separate decoder checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load base model
    model = Synformer.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Load pretrained decoder
    model.load_pretrained_decoder(decoder_checkpoint_path, device)
    
    model.to(device)
    return model

def encode_smiles_to_latent(
    smiles_list: List[str],
    desert_model_path: str,
    vocab_path: str,
    shape_patches_path: str,
    device: torch.device
) -> Tuple[torch.Tensor, List[str], List[int]]:
    """
    Encode SMILES strings to latent vectors using DESERT and fragment encoder
    
    Args:
        smiles_list: List of SMILES strings
        desert_model_path: Path to DESERT model
        vocab_path: Path to vocabulary file
        shape_patches_path: Path to shape patches
        device: torch device
    
    Returns:
        Tuple of:
        - Tensor of latent vectors (n_samples, latent_dim)
        - List of valid SMILES strings
        - List of indices of valid SMILES
    """
    latents = []
    valid_smiles = []
    valid_indices = []
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Create fragment encoder
    encoder = create_fragment_encoder(vocab_path=vocab_path, device=device)
    
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Encoding SMILES")):
        try:
            # Run DESERT inference
            logger.info(f"Running DESERT inference for SMILES: {smiles}")
            desert_sequences = run_desert_inference(
                smiles, 
                desert_model_path, 
                shape_patches_path, 
                device=device
            )
            desert_sequence = desert_sequences[0]  # Take first sequence
            
            # Encode sequence using fragment encoder
            with torch.no_grad():
                embeddings = encoder.encode_desert_sequence(desert_sequence, device=device)
                latents.append(embeddings.cpu())
                valid_smiles.append(smiles)
                valid_indices.append(idx)
                
        except Exception as e:
            logger.warning(f"Failed to encode SMILES {smiles}: {str(e)}")
            continue
            
    if not latents:
        raise ValueError("No SMILES could be encoded successfully")
        
    return torch.stack(latents), valid_smiles, valid_indices

def run_synformer_decoder(
    model: Synformer,
    latents: torch.Tensor,
    device: torch.device
) -> List[str]:
    """
    Run Synformer decoder on latent vectors to generate SMILES
    
    Args:
        model: Pretrained Synformer model
        latents: Tensor of latent vectors
        device: torch device
    
    Returns:
        List of generated SMILES strings
    """
    # Create batch for decoder
    batch = {
        'code': latents.to(device),
        'code_padding_mask': torch.zeros(latents.size(0), 1, dtype=torch.bool, device=device)
    }
    
    # Generate molecules using Synformer
    with torch.no_grad():
        result = model.generate_without_stack(
            batch=ProjectionBatch(**batch),
            rxn_matrix=model.rxn_matrix,
            fpindex=model.fpindex,
            max_len=32,
            temperature_token=0.1,
            temperature_reaction=0.1,
            temperature_reactant=0.1
        )
        
        # Convert generation results to SMILES
        generated_smiles = []
        for stack in result.build():
            if stack is None or stack.is_empty():
                generated_smiles.append("")
                continue
                
            try:
                mol = stack.get_product()
                if mol is None:
                    generated_smiles.append("")
                    continue
                    
                smiles = mol.to_smiles()
                if not smiles:
                    generated_smiles.append("")
                    continue
                    
                generated_smiles.append(smiles)
            except Exception as e:
                logger.warning(f"Failed to convert stack to SMILES: {str(e)}")
                generated_smiles.append("")
                
        return generated_smiles

def get_docking_scores(
    proxy: TacoGFN_Proxy,
    pocket_ids: List[str],
    smiles_list: List[str]
) -> List[float]:
    """Get docking scores for SMILES-pocket pairs"""
    scores = []
    
    for pid, smi in tqdm(zip(pocket_ids, smiles_list), desc="Computing docking scores"):
        if not smi:  # Skip empty SMILES
            scores.append(None)
            continue
            
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
    decoder_checkpoint: str,
    desert_model_path: str,
    vocab_path: str,
    shape_patches_path: str,
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
        decoder_checkpoint: Path to pretrained decoder checkpoint
        desert_model_path: Path to DESERT model
        vocab_path: Path to vocabulary file
        shape_patches_path: Path to shape patches
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
    synformer = load_synformer_model(synformer_checkpoint, decoder_checkpoint, device)
    proxy = TacoGFN_Proxy.load(
        scoring_fn='QVina',
        train_dataset='CrossDocked2020',
        cache_db=cache_db,
        device=device
    )
    
    # Generate latents and track valid SMILES using DESERT
    logger.info("Encoding SMILES to latents using DESERT...")
    latents, valid_smiles, valid_indices = encode_smiles_to_latent(
        smiles_list,
        desert_model_path,
        vocab_path,
        shape_patches_path,
        device
    )
    valid_pocket_ids = [pocket_ids[i] for i in valid_indices]
    
    # Run Synformer decoder
    logger.info("Running Synformer decoder...")
    generated_smiles = run_synformer_decoder(synformer, latents, device)
    
    # Get docking scores for generated molecules
    logger.info("Computing docking scores...")
    scores = get_docking_scores(proxy, valid_pocket_ids, generated_smiles)
    
    # Prepare dataset
    dataset = []
    for orig_smi, gen_smi, pid, lat, score in zip(valid_smiles, generated_smiles, valid_pocket_ids, latents, scores):
        if score is not None and gen_smi:  # Only include valid examples
            dataset.append({
                'original_smiles': orig_smi,
                'generated_smiles': gen_smi,
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
    parser.add_argument("--decoder-checkpoint", required=True, help="Path to pretrained decoder checkpoint")
    parser.add_argument("--desert-model", required=True, help="Path to DESERT model")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary file")
    parser.add_argument("--shape-patches", required=True, help="Path to shape patches")
    parser.add_argument("--output", required=True, help="Path to save dataset")
    parser.add_argument("--cache-db", default="train", help="Cache database for TacoGFN")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    prepare_docking_dataset(
        smiles_file=args.smiles,
        pocket_ids_file=args.pockets,
        synformer_checkpoint=args.checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        desert_model_path=args.desert_model,
        vocab_path=args.vocab,
        shape_patches_path=args.shape_patches,
        output_file=args.output,
        device=device,
        cache_db=args.cache_db
    )

if __name__ == "__main__":
    main() 