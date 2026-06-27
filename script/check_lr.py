"""
Extract recent LR values from the latest wandb run's lr-AdamW log.
Reads only the tail of the .wandb file for efficiency.
"""
import re
import os
import glob

wandb_root = r'D:\image-classification\wandb_logs\wandb'

# Find latest run directory
run_dirs = sorted(glob.glob(os.path.join(wandb_root, 'offline-run-*')), key=os.path.getmtime)
if not run_dirs:
    print("LR: no wandb run found")
    exit(0)

latest_dir = run_dirs[-1]
run_name = os.path.basename(latest_dir).split('-')[-1]
wandb_file = os.path.join(latest_dir, f'run-{run_name}.wandb')

if not os.path.exists(wandb_file):
    print(f"LR: wandb file not found for {run_name}")
    exit(0)

# Read last 32MB only (LR data typically in tail of file)
file_size = os.path.getsize(wandb_file)
read_size = min(32 * 1024 * 1024, file_size)

with open(wandb_file, 'rb') as f:
    if file_size > read_size:
        f.seek(file_size - read_size)
    data = f.read()

# Extract lr-AdamW + global_step pairs
pattern = re.compile(
    rb'lr-AdamW...([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)'
    rb'.{0,200}'
    rb'global_step...([0-9]+)',
    re.DOTALL
)

seen = {}
LR_MIN_SANE = 1e-10  # discard spurious binary-noise matches
for m in pattern.finditer(data):
    lr = float(m.group(1))
    step = int(m.group(2))
    if LR_MIN_SANE < lr < 1.0:
        seen[step] = lr  # overwrite: later entries in tail are newer

if not seen:
    # Fallback: try train/lr
    pattern2 = re.compile(
        rb'train/lr...([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)'
        rb'.{0,200}'
        rb'global_step...([0-9]+)',
        re.DOTALL
    )
    for m in pattern2.finditer(data):
        lr = float(m.group(1))
        step = int(m.group(2))
        if LR_MIN_SANE < lr < 1.0:
            seen[step] = lr

if not seen:
    print("LR: no LR data in tail (try full file scan)")
    exit(0)

sorted_steps = sorted(seen.keys())
latest_step = sorted_steps[-1]
latest_lr = seen[latest_step]

# Also get the earliest in this tail for trend
first_step = sorted_steps[0]
first_lr = seen[first_step]

# Determine phase: if LR is increasing, warmup; decreasing, cosine
mid_step = sorted_steps[len(sorted_steps)//2]
mid_lr = seen[mid_step]
trend = "warmup" if latest_lr >= first_lr else "cosine"

# Estimate progress
max_lr = max(seen.values())
min_lr = min(seen.values())

print(f"LR: {latest_lr:.6e} @ step {latest_step} [{trend}]")

if trend == "warmup":
    # Estimate warmup progress
    start_lr = 3e-7  # 3e-4 * 0.001
    target_lr = 3e-4
    progress = (latest_lr - start_lr) / (target_lr - start_lr) * 100 if target_lr > start_lr else 0
    print(f"     warmup {progress:.1f}% ({start_lr:.1e} -> {target_lr:.1e}), step delta={latest_lr - first_lr:+.2e}")
else:
    # Cosine phase
    eta_min = 1e-6
    progress = (max_lr - latest_lr) / (max_lr - eta_min) * 100 if max_lr > eta_min else 0
    print(f"     cosine {progress:.1f}% ({max_lr:.1e} -> {eta_min:.1e}), step delta={latest_lr - first_lr:+.2e}")
