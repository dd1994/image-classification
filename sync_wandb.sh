#!/bin/bash
# Sync wandb offline runs to cloud
# Usage: ./sync_wandb.sh [run_id]
#   run_id: optional, specific wandb run id to sync
#   If not provided, syncs the most recent run with changes

source /c/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate myenv

if [ -n "$1" ]; then
  MATCH=$(ls -t wandb_logs/wandb/offline-run-* 2>/dev/null | grep "$1" | head -1 | sed 's/:$//')

  if [ -z "$MATCH" ]; then
    echo "No offline run found containing: $1"
    exit 1
  fi

  echo "Syncing run: $MATCH"
  python -m wandb sync "$MATCH"
else
  LATEST=$(ls -t wandb_logs/wandb/offline-run-* 2>/dev/null | head -1 | sed 's/:$//')

  if [ -z "$LATEST" ]; then
    echo "No offline wandb runs found."
    exit 1
  fi

  echo "Syncing most recent run:"
  echo "$LATEST"
  python -m wandb sync "$LATEST"
fi