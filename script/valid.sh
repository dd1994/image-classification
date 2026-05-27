#!/bin/bash

CONFIG="./config/fgvc-aves-tiny/swinv2_tiny512.json"
CHECKPOINT="./wandb_logs/identify/eoz9k2h9/checkpoints/last.ckpt"

echo "=========================================="
echo "Validating with Grad-CAM ensemble"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /c/ProgramData/anaconda3/envs/myenv/python.exe \
    ./script/validate_ensemble.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --batch-size 96
echo "Done."
