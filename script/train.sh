#!/bin/bash

CONFIG="./config\large\eva02_all.json"
SEEDS=(42)

mkdir -p ./logs

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    PYTHONUNBUFFERED=1 /c/ProgramData/anaconda3/envs/myenv/python.exe -X faulthandler \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed" \
        --ckpt_path "./wandb_logs/identify/el5plj39/checkpoints/eva02-all-epoch=02-step=19397.ckpt" \
        > "./logs/train_seed_${seed}.log" 2>&1
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."
