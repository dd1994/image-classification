#!/bin/bash

CONFIG="./config/large/swinv2_all.json"
SEEDS=(42)

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    /c/ProgramData/anaconda3/envs/myenv/python.exe \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed" \
        --ckpt_path "./wandb_logs/identify/x84v3ram/checkpoints/swinv2-all-step=step=14276.ckpt"
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."