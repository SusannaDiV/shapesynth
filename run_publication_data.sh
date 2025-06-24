#!/bin/bash

start_time=$(date +%s)

# echo "Starting 100 runs for 1a2g_A..."

# for i in $(seq 1 100); do
#   python -m synformer.sampler.analog.cli \
#     --model-path /workspace/data/processed/sf_ed_default.ckpt \
#     --input /workspace/data/one.csv \
#     --output /workspace/publication_data/1a2g_A/1a2g_A_${i}.csv \
#     --use-desert \
#     --desert-model-path /workspace/data/desert/1WW_30W_5048064.pt \
#     --vocab-path /workspace/data/desert/vocab.pkl \
#     --smiles-checkpoint-path /workspace/data/processed/sf_ed_default.ckpt \
#     --receptor-path /workspace/data/1a2g_A.pdbqt \
#     --receptor-center -22.543 25.267 -3.522 \
#     --shape-patches-path /workspace/data/1a2g_A_shapes.pkl
# done
#5d7n_D_shapes.pkl
# echo "Finished 1a2g_A. Starting 100 runs for 3tym_A..."

for i in $(seq 1 100); do
  python -m synformer.sampler.analog.cli \
    --model-path /workspace/data/processed/sf_ed_default.ckpt \
    --input /workspace/data/one.csv \
    --output /workspace/publication_data/1phk_A/1phk_A_${i}.csv \
    --use-desert \
    --desert-model-path /workspace/data/desert/1WW_30W_5048064.pt \
    --vocab-path /workspace/data/desert/vocab.pkl \
    --smiles-checkpoint-path /workspace/data/processed/sf_ed_default.ckpt \
    --receptor-path /workspace/data/TestCrossDocked2020/receptors/1phk_A.pdbqt \
    --receptor-center 11.282 16.992 14.096 \
    --shape-patches-path /workspace/data/TestCrossDocked2020/sample_shapes_ablation_study/1phk_A_shapes.pkl
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "All runs completed in $elapsed seconds (~$((elapsed / 60)) minutes and $((elapsed % 60)) seconds)."
