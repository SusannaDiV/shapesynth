#!/usr/bin/env python3
"""
Auto CKA Analysis Integration

This script can be called automatically after model runs to generate CKA analysis plots.
It integrates seamlessly with existing code.

Usage:
    # At the end of your existing script, add:
    from synformer.utils.auto_cka_analysis import auto_cka_analysis
    auto_cka_analysis(model, smiles="your_smiles_here")
"""

import os
import time
from typing import Optional, Dict, Any

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def auto_cka_analysis(
    model, 
    smiles: str = "CCO", 
    output_dir: str = "./auto_cka_results",
    native_latents_path: Optional[str] = None,
    variants_to_test: Optional[list] = None
) -> Dict[str, Any]:
    """
    Automatically run CKA analysis on a SynFormer model.
    
    Args:
        model: SynFormer model instance
        smiles: SMILES string to analyze
        output_dir: Directory to save results
        native_latents_path: Optional path to native latents
        variants_to_test: List of variant configs to test
    
    Returns:
        Dictionary of CKA scores
    """
    
    print("\n" + "="*60)
    print("AUTO CKA ANALYSIS")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f"cka_session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Default variants if not provided
    if variants_to_test is None:
        variants_to_test = [
            {"name": "Current Model", "settings": {}}
        ]
    
    # Check if CKA analysis is enabled
    if not hasattr(model, 'enable_cka_analysis') or not model.enable_cka_analysis:
        print("CKA analysis not enabled on this model. Enabling...")
        model.enable_cka_analysis = True
        model.cka_scores = {}
    
    # Load native latents if provided
    if native_latents_path and os.path.exists(native_latents_path):
        model.load_native_latents(native_latents_path)
        print(f"Loaded native latents from: {native_latents_path}")
    
    print(f"Analyzing SMILES: {smiles}")
    print(f"Session directory: {session_dir}")
    
    all_results = {}
    
    # Test the current model configuration
    try:
        print(f"\nTesting current model configuration...")
        
        # Create batch
        batch = {
            'smiles_str': smiles,
            'shape_patches_path': getattr(model, 'shape_patches_path', None)
        }
        
        # Run encoding
        with torch.no_grad():
            code, code_padding_mask, _ = model.encode(batch)
            
        # Get CKA scores
        if model.cka_scores:
            latest_scores = list(model.cka_scores.values())[-1]
            all_results["Current Model"] = latest_scores
            
            print("CKA Scores:")
            for comparison, score in latest_scores.items():
                print(f"  {comparison}: {score:.4f}")
        else:
            print("No CKA scores computed")
            
    except Exception as e:
        print(f"Error in CKA analysis: {str(e)}")
        return {}
    
    # Generate summary plot if we have results
    if all_results:
        try:
            plot_path = os.path.join(session_dir, "cka_summary.png")
            create_quick_cka_plot(all_results, plot_path)
            print(f"CKA plot saved to: {plot_path}")
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
    
    # Save detailed results
    results_file = os.path.join(session_dir, "cka_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"CKA Analysis Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"SMILES: {smiles}\n")
        f.write("="*40 + "\n\n")
        
        for variant_name, scores in all_results.items():
            f.write(f"{variant_name}:\n")
            for comparison, score in scores.items():
                f.write(f"  {comparison}: {score:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to: {results_file}")
    print("Auto CKA analysis complete!")
    
    return all_results


def create_quick_cka_plot(cka_results: Dict[str, Dict[str, float]], save_path: str):
    """Create a quick summary plot of CKA results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    variants = list(cka_results.keys())
    desert_scores = [cka_results[v].get('desert_vs_decoder', 0) for v in variants]
    native_scores = [cka_results[v].get('blended_vs_native', 0) for v in variants]
    
    # Plot 1: DESERT vs Decoder
    bars1 = ax1.bar(variants, desert_scores, color='skyblue', alpha=0.7)
    ax1.set_title('CKA: DESERT vs Decoder Input', fontweight='bold')
    ax1.set_ylabel('CKA Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars1, desert_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Blended vs Native (if available)
    if any(score > 0 for score in native_scores):
        bars2 = ax2.bar(variants, native_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('CKA: Blended vs Native', fontweight='bold')
        ax2.set_ylabel('CKA Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars2, native_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Native Latents\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('CKA: Blended vs Native', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Convenience function for easy integration
def quick_cka_check(model, smiles="CCO"):
    """Quick CKA check - minimal output for debugging."""
    if hasattr(model, 'cka_scores') and model.cka_scores:
        print("\nQuick CKA Check:")
        for variant, scores in model.cka_scores.items():
            for comparison, score in scores.items():
                print(f"  {variant} - {comparison}: {score:.3f}")
    else:
        print("No CKA scores available")


if __name__ == "__main__":
    print("This is a utility module. Import and use auto_cka_analysis() function.")
    print("Example usage:")
    print("  from synformer.utils.auto_cka_analysis import auto_cka_analysis")
    print("  auto_cka_analysis(model, smiles='CCO')") 