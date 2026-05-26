#!/bin/bash

CONFIG="./config/fgvc-aves-tiny/swinv2_tiny.json"
SEEDS=(1)

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /c/ProgramData/anaconda3/envs/myenv/python.exe \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed" \
        # --ckpt_path ./wandb_logs/identify/gdhte74a/checkpoints/last.ckpt
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."