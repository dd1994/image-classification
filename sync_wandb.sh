#!/bin/bash
# Sync the two most recent wandb offline runs to cloud

source /c/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate myenv

LATEST=$(ls -t wandb_logs/wandb/offline-run-* 2>/dev/null | head -2 | sed 's/:$//')

if [ -z "$LATEST" ]; then
  echo "No offline wandb runs found."
  exit 1
fi

echo "Syncing the two most recent runs:"
echo "$LATEST"
echo ""
for RUN in $LATEST; do
  echo "Syncing: $RUN"
  python -m wandb sync "$RUN"
done