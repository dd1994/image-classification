#!/bin/bash

CONFIG="./config/large/eva02_all_part2.json"
SEEDS=(102)

mkdir -p ./logs

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    PYTHONUNBUFFERED=1 /c/ProgramData/anaconda3/envs/myenv/python.exe -X faulthandler \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed" \
        --ckpt_path "./wandb_logs/identify/7ua132n2/checkpoints/eva02-all-epoch=09-val/acc_top1=0.8033.ckpt" \
        > "./logs/train_seed_${seed}.log" 2>&1
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."
