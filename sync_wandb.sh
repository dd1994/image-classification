#!/bin/bash
# Sync the most recent wandb offline run to cloud

source /c/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate myenv

LATEST=$(ls -t wandb_logs/wandb/offline-run-* 2>/dev/null | head -1 | sed 's/:$//')

if [ -z "$LATEST" ]; then
  echo "No offline wandb runs found."
  exit 1
fi

echo "Syncing: $LATEST"
python -m wandb sync "$LATEST"