# CKA Analysis for Adapter Ablation Study

This document explains how to use the Center Kernel Alignment (CKA) analysis functionality to evaluate different adapter variants in the SynFormer DESERT integration.

## Overview

The CKA analysis computes similarity scores between:
1. **DESERT latents vs SynFormer decoder inputs** - How well the adapter preserves DESERT's spatial information
2. **Blended latents vs Native training latents** - How similar the adapted representations are to native SynFormer representations

## Installation Requirements

```bash
pip install matplotlib seaborn scikit-learn scipy
```

## Quick Start

### 1. Basic Usage

```python
from synformer.models.synformer import Synformer
from omegaconf import OmegaConf

# Load your model configuration
cfg = OmegaConf.load("path/to/config.yaml")
cfg.encoder_type = "desert"
cfg.enable_cka_analysis = True

# Create model
model = Synformer(cfg)

# Create batch for analysis
batch = {
    'smiles_str': "CCO",  # Your SMILES string
    'shape_patches_path': "path/to/shape_patches.h5"
}

# Run encoding (CKA computed automatically)
code, code_padding_mask, _ = model.encode(batch, mixture_weight=0.8)

# Analyze results
model.analyze_cka_scores(save_plots=True, plot_dir="./cka_results")
```

### 2. Using the Demo Script

```bash
python synformer/utils/cka_analysis_demo.py \
    --config config.yaml \
    --smiles "CCO" \
    --native-latents native_latents.pt
```

## Variants Tested

### 1. No Adapter (`no_adapter=True`)
- Raw DESERT output reshaped to match decoder dimensions
- **Hypothesis**: Low CKA with decoder inputs, preserves raw DESERT structure

### 2. Minimal Processing (`minimal_processing=True`) 
- Basic embedding lookup and MLP processing
- **Hypothesis**: Medium CKA, some adaptation but limited processing

### 3. Full Fragment Encoder (various mixture weights)
- Complete spatial encoding with learnable parameters
- **Mixture weights**: 0.0 (no spatial info) to 1.0 (full spatial info)
- **Hypothesis**: High CKA with good spatial preservation

## Understanding CKA Scores

### CKA Score Interpretation:
- **0.0**: No similarity/correlation
- **0.5**: Moderate similarity
- **1.0**: Perfect similarity/correlation

### Expected Patterns:
1. **DESERT vs Decoder**: 
   - No Adapter: Low scores (0.1-0.3)
   - Minimal Processing: Medium scores (0.3-0.6)
   - Full Encoder: High scores (0.6-0.9)

2. **Blended vs Native**:
   - Higher scores indicate better integration with SynFormer
   - Very high scores (>0.9) might indicate loss of DESERT information

## Advanced Usage

### Comparing Multiple Molecules

```python
molecules = ["CCO", "CCC", "c1ccccc1"]
all_results = {}

for smiles in molecules:
    batch = {'smiles_str': smiles, 'shape_patches_path': shape_patches_path}
    code, _, _ = model.encode(batch)
    all_results[smiles] = model.cka_scores.copy()
    model.cka_scores.clear()  # Reset for next molecule

# Aggregate analysis
analyze_multiple_molecules(all_results)
```

### Saving and Loading Native Latents

```python
# Save native latents from a baseline model
baseline_model = load_baseline_model()
baseline_code, _, _ = baseline_model.encode(batch)
torch.save(baseline_code.squeeze(0).cpu(), "native_latents.pt")

# Load for comparison
model.load_native_latents("native_latents.pt")
```

### Custom CKA Analysis

```python
from synformer.models.synformer import compute_cka

# Compute custom CKA between any two representations
X = torch.randn(100, 768)  # First representation
Y = torch.randn(100, 512)  # Second representation

linear_cka = compute_cka(X, Y, kernel='linear')
rbf_cka = compute_cka(X, Y, kernel='rbf')

print(f"Linear CKA: {linear_cka:.4f}")
print(f"RBF CKA: {rbf_cka:.4f}")
```

## Output Files

### 1. CKA Plots
- `cka_comparison.png`: Bar chart comparing variants
- `cka_heatmap.png`: Pairwise similarity heatmap

### 2. Data Files
- `cka_scores.txt`: Text summary of all scores
- `native_latents.pt`: Saved native representations

## Troubleshooting

### Common Issues:

1. **"No CKA scores available"**
   - Ensure `enable_cka_analysis=True` in config
   - Check that DESERT encoder is working correctly

2. **CKA scores all zero**
   - Check tensor shapes are compatible
   - Verify representations aren't all zeros

3. **Memory errors with large molecules**
   - Reduce batch size or use shorter sequences
   - Consider using RBF kernel with lower gamma

### Debug Mode:

```python
# Enable detailed CKA logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check intermediate representations
print(f"DESERT latents shape: {model.desert_latents.shape}")
print(f"Decoder input shape: {code.shape}")
```

## Interpretation Guidelines

### Good Adapter Indicators:
1. **Moderate to high DESERT→Decoder CKA** (0.4-0.8)
   - Shows spatial information is preserved
2. **Reasonable Blended→Native CKA** (0.3-0.7)  
   - Shows compatibility with SynFormer
3. **Progressive improvement** across variants
   - No Adapter < Minimal < Full Encoder

### Red Flags:
1. **Very low all CKA scores** (<0.1)
   - Adapter may be losing information
2. **Perfect CKA** (>0.95)
   - May indicate trivial mappings
3. **Inconsistent scores** across molecules
   - Adapter may not generalize well

## Example Analysis Workflow

```python
# 1. Test multiple variants
variants = [
    {"no_adapter": True},
    {"minimal_processing": True}, 
    {"mixture_weight": 0.5},
    {"mixture_weight": 0.8},
    {"mixture_weight": 1.0}
]

results = {}
for variant in variants:
    model = create_model_with_variant(variant)
    code, _, _ = model.encode(batch, **variant)
    results[str(variant)] = model.cka_scores

# 2. Analyze trends
analyze_cka_trends(results)

# 3. Select best variant
best_variant = select_optimal_variant(results)
```

This CKA analysis framework provides quantitative metrics to guide adapter design decisions and validate that spatial information from DESERT is being effectively preserved and utilized by the SynFormer decoder. 