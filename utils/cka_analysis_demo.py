#!/usr/bin/env python3
"""
Demo script for running CKA analysis on adapter ablation study.

This script demonstrates how to use the CKA analysis functionality
to compare DESERT latents with SynFormer decoder inputs across different variants.

Usage:
    python synformer/utils/cka_analysis_demo.py --config path/to/config.yaml --smiles "CCO"
"""

import argparse
import os
import sys
from pathlib import Path

# Add synformer to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from synformer.models.synformer import Synformer


def run_cka_analysis_demo(config_path, smiles, native_latents_path=None, save_plots=True):
    """
    Run CKA analysis demo for adapter ablation study.
    
    Args:
        config_path: Path to model configuration
        smiles: SMILES string to analyze
        native_latents_path: Optional path to native training latents
        save_plots: Whether to save plots
    """
    
    print("="*80)
    print("CKA ANALYSIS DEMO FOR ADAPTER ABLATION STUDY")
    print("="*80)
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    cfg.encoder_type = "desert"  # Ensure we're using DESERT
    cfg.enable_cka_analysis = True
    
    # Create output directory
    output_dir = "./cka_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing SMILES: {smiles}")
    print(f"Output directory: {output_dir}")
    
    # Test different variants
    variants_to_test = [
        {"name": "No Adapter", "no_adapter": True, "minimal_processing": False},
        {"name": "Minimal Processing", "no_adapter": False, "minimal_processing": True},
        {"name": "Full Fragment Encoder (w=0.5)", "no_adapter": False, "minimal_processing": False, "mixture_weight": 0.5},
        {"name": "Full Fragment Encoder (w=0.8)", "no_adapter": False, "minimal_processing": False, "mixture_weight": 0.8},
        {"name": "Full Fragment Encoder (w=1.0)", "no_adapter": False, "minimal_processing": False, "mixture_weight": 1.0},
    ]
    
    all_cka_scores = {}
    
    for i, variant in enumerate(variants_to_test):
        print(f"\n{'-'*60}")
        print(f"TESTING VARIANT {i+1}/{len(variants_to_test)}: {variant['name']}")
        print(f"{'-'*60}")
        
        # Update config for this variant
        cfg.no_adapter = variant.get("no_adapter", False)
        cfg.minimal_processing = variant.get("minimal_processing", False)
        
        # Create model
        model = Synformer(cfg)
        
        # Load native latents if provided
        if native_latents_path and os.path.exists(native_latents_path):
            model.load_native_latents(native_latents_path)
        
        # Create batch for encoding
        batch = {
            'smiles_str': smiles,
            'shape_patches_path': cfg.encoder.get('shape_patches_path', None)
        }
        
        # Set mixture weight if specified
        mixture_weight = variant.get("mixture_weight", None)
        
        # Run encoding (this will compute CKA scores automatically)
        with torch.no_grad():
            try:
                code, code_padding_mask, encoder_loss_dict = model.encode(batch, mixture_weight=mixture_weight)
                print(f"✓ Encoding successful for {variant['name']}")
                
                # Store CKA scores from this variant
                if model.cka_scores:
                    # Use the variant name as key
                    variant_key = list(model.cka_scores.keys())[-1]  # Get the most recent entry
                    all_cka_scores[variant['name']] = model.cka_scores[variant_key]
                    
                    print(f"CKA Scores for {variant['name']}:")
                    for comparison, score in model.cka_scores[variant_key].items():
                        print(f"  {comparison}: {score:.4f}")
                
            except Exception as e:
                print(f"✗ Error in {variant['name']}: {str(e)}")
                continue
    
    print(f"\n{'='*80}")
    print("FINAL CKA ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    # Print summary
    if all_cka_scores:
        print("\nSummary of CKA Scores:")
        print("-" * 50)
        for variant_name, scores in all_cka_scores.items():
            print(f"\n{variant_name}:")
            for comparison, score in scores.items():
                print(f"  {comparison}: {score:.4f}")
        
        # Create temporary model for plotting
        temp_model = Synformer(cfg)
        temp_model.cka_scores = all_cka_scores
        
        # Generate plots
        if save_plots:
            plot_dir = os.path.join(output_dir, "cka_plots")
            temp_model.analyze_cka_scores(save_plots=True, plot_dir=plot_dir)
        else:
            temp_model.analyze_cka_scores(save_plots=False)
        
        # Save CKA scores to file
        scores_file = os.path.join(output_dir, "cka_scores.txt")
        with open(scores_file, 'w') as f:
            f.write("CKA Analysis Results\n")
            f.write("="*50 + "\n")
            f.write(f"SMILES: {smiles}\n\n")
            
            for variant_name, scores in all_cka_scores.items():
                f.write(f"{variant_name}:\n")
                for comparison, score in scores.items():
                    f.write(f"  {comparison}: {score:.4f}\n")
                f.write("\n")
        
        print(f"\nCKA scores saved to: {scores_file}")
        
    else:
        print("No CKA scores were computed successfully.")
    
    print(f"\nCKA analysis complete. Results saved to: {output_dir}")
    return all_cka_scores


def main():
    parser = argparse.ArgumentParser(description="Run CKA analysis for adapter ablation study")
    parser.add_argument("--config", required=True, help="Path to model configuration file")
    parser.add_argument("--smiles", required=True, help="SMILES string to analyze")
    parser.add_argument("--native-latents", help="Path to native training latents file")
    parser.add_argument("--no-plots", action="store_true", help="Don't save plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    run_cka_analysis_demo(
        config_path=args.config,
        smiles=args.smiles,
        native_latents_path=args.native_latents,
        save_plots=not args.no_plots
    )


if __name__ == "__main__":
    main() 