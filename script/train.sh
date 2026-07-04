#!/bin/bash

CONFIG="./config/tiny/eva02_tiny.json"
SEEDS=(1)

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    /c/ProgramData/anaconda3/envs/myenv/python.exe -X faulthandler \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed"
        # --ckpt_path "./wandb_logs/identify/dzjyvnkn/checkpoints/last.ckpt"
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."
