#!/bin/bash

CONFIG="./config/large/eva02_all_part2.json"
SEEDS=(100)

mkdir -p ./logs

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    PYTHONUNBUFFERED=1 /c/ProgramData/anaconda3/envs/myenv/python.exe -X faulthandler \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed" \
        --ckpt_path "./wandb_logs/identify/9w26s7hq/checkpoints/last.ckpt" \
        > "./logs/train_seed_${seed}.log" 2>&1
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."
