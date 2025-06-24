import pandas as pd
import torch
import numpy as np
import pickle
from excellent_checkadaptertraining import create_fragment_encoder

def parse_fragments_from_row(row):
    """Parse fragment data from a CSV row into the format expected by the encoder"""
    fragments = []
    
    # Extract fragment data - start from fragment_1 columns
    i = 1
    while f'fragment_{i}_id' in row and pd.notna(row[f'fragment_{i}_id']):
        fragment_id = int(row[f'fragment_{i}_id'])
        translation = int(row[f'fragment_{i}_translation'])
        rotation = int(row[f'fragment_{i}_rotation'])
        
        fragments.append((fragment_id, translation, rotation))
        i += 1
    
    return fragments

def compute_latent_representation(fragments, encoder):
    """Compute latent representation for a fragment sequence"""
    if not fragments:
        return None
    
    try:
        # Encode the fragment sequence
        encoder_output = encoder.encode_desert_sequence(fragments, device='cpu')
        
        # Get the pooled representation (mean over sequence length, excluding padding)
        embeddings = encoder_output.code  # Shape: [1, seq_len, d_model]
        padding_mask = encoder_output.code_padding_mask  # Shape: [1, seq_len]
        
        # Create a mask for non-padded positions
        valid_mask = ~padding_mask  # True for valid positions
        
        if valid_mask.sum() == 0:
            return None
        
        # Compute mean pooling over valid positions
        valid_embeddings = embeddings * valid_mask.unsqueeze(-1).float()
        pooled_embedding = valid_embeddings.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).float()
        
        # Return as numpy array
        return pooled_embedding.squeeze(0).detach().numpy()
        
    except Exception as e:
        print(f"Error computing latent representation: {e}")
        return None

def main():
    # Configuration
    vocab_path = "/workspace/data/desert/vocab.pkl"
    csv_file = "fragments_qed_docking.csv"
    output_file = "fragments_qed_docking_with_latents.csv"
    
    print("Loading fragment encoder...")
    encoder = create_fragment_encoder(
        vocab_path=vocab_path,
        embedding_dim=768,
        device='cpu',
        mixture_weight=1.0
    )
    encoder.eval()
    
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Processing {len(df)} rows...")
    latent_representations = []
    
    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)}")
        
        # Parse fragments from the row
        fragments = parse_fragments_from_row(row)
        
        if fragments:
            # Compute latent representation
            latent = compute_latent_representation(fragments, encoder)
            
            if latent is not None:
                # Convert to list for JSON serialization
                latent_representations.append(latent.tolist())
                print(f"  Generated latent of shape: {latent.shape}")
            else:
                latent_representations.append(None)
                print("  Failed to generate latent")
        else:
            latent_representations.append(None)
            print("  No fragments found")
    
    # Add latent representations to the dataframe
    df['latent_representation'] = latent_representations
    
    # Also add some summary statistics about the latents
    latent_dims = []
    latent_norms = []
    latent_means = []
    latent_stds = []
    
    for latent in latent_representations:
        if latent is not None:
            latent_array = np.array(latent)
            latent_dims.append(len(latent_array))
            latent_norms.append(float(np.linalg.norm(latent_array)))
            latent_means.append(float(np.mean(latent_array)))
            latent_stds.append(float(np.std(latent_array)))
        else:
            latent_dims.append(None)
            latent_norms.append(None)
            latent_means.append(None)
            latent_stds.append(None)
    
    df['latent_dim'] = latent_dims
    df['latent_norm'] = latent_norms
    df['latent_mean'] = latent_means
    df['latent_std'] = latent_stds
    
    # Save the updated dataframe
    print(f"Saving updated CSV to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    valid_latents = [l for l in latent_representations if l is not None]
    print(f"\nSummary:")
    print(f"Total rows processed: {len(df)}")
    print(f"Valid latent representations: {len(valid_latents)}")
    print(f"Failed representations: {len(df) - len(valid_latents)}")
    
    if valid_latents:
        latent_array = np.array(valid_latents)
        print(f"Latent dimension: {latent_array.shape[1]}")
        print(f"Average latent norm: {np.mean([np.linalg.norm(l) for l in valid_latents]):.4f}")
        print(f"Average latent mean: {np.mean([np.mean(l) for l in valid_latents]):.4f}")
        print(f"Average latent std: {np.mean([np.std(l) for l in valid_latents]):.4f}")

if __name__ == "__main__":
    main() 