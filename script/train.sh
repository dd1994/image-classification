#!/bin/bash

CONFIG="./config/large/swinv2_all.json"
SEEDS=(42)

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Training with seed=$seed"
    echo "=========================================="
    /c/ProgramData/anaconda3/envs/myenv/python.exe -X faulthandler \
        ./script/train.py fit \
        --config "$CONFIG" \
        --seed_everything="$seed" #\
        # --ckpt_path "./wandb_logs/identify/tvh3wila/checkpoints/swinv2-all-epoch=01-step=27311.ckpt"
    echo "Finished seed=$seed"
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
done

echo "All seeds done."