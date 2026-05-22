#!/bin/bash
set -e

CONFIG="./config/fgvc-aves-tiny/swinv2_tiny512.json"
SEEDS=(1 7 99)

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /c/ProgramData/anaconda3/envs/myenv/python.exe \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed"
    echo "Finished seed=$seed"
done

echo "All seeds done."