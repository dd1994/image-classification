# -*- coding: utf-8 -*-
"""
Extract LR values from the latest wandb run.
Reads per-group lr-AdamW data and maps pg indices to model layers.
"""
import re
import os
import glob

wandb_root = r'D:\image-classification\wandb_logs\wandb'

# ── Find latest run ────────────────────────────────────────────────
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

# ── Read file ──────────────────────────────────────────────────────
file_size = os.path.getsize(wandb_file)
with open(wandb_file, 'rb') as f:
    data = f.read()

LR_MIN_SANE = 1e-10

# ── Strategy: find the last global_step, then collect all lr-AdamW/pg{N}
#    values in its vicinity. This avoids the problem of matching each
#    pg value to a step individually.
# ────────────────────────────────────────────────────────────────────

# 1) Collect all (position, step) for trainer/global_step
step_positions = []
for m in re.finditer(rb'trainer/global_step.{0,10}?(\d+)', data):
    step = int(m.group(1))
    step_positions.append((m.start(), step))

if not step_positions:
    # Fallback: plain global_step
    for m in re.finditer(rb'global_step.{0,10}?(\d+)', data):
        step = int(m.group(1))
        step_positions.append((m.start(), step))

if not step_positions:
    print("LR: no global_step found in wandb file")
    exit(0)

# 2) Get the latest step and its position
last_pos, latest_step = step_positions[-1]

# 3) Collect all lr-AdamW/pg{N} values in a window around the last global_step
#    The LR data for one step is typically logged within ~10KB before the global_step
SEARCH_WINDOW = 20 * 1024  # 20KB
window_start = max(0, last_pos - SEARCH_WINDOW)
window = data[window_start:last_pos + SEARCH_WINDOW]

# Parse lr-AdamW/pg{N}:<float>
pg_data = {}  # pg_index -> lr_value
for m in re.finditer(rb'lr-AdamW/pg(\d+).{0,10}?([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)', window):
    pg = int(m.group(1))
    lr = float(m.group(2))
    if LR_MIN_SANE < lr < 1.0:
        pg_data[pg] = lr  # keep latest (closest to global_step)

if not pg_data:
    print("LR: no lr-AdamW data near latest global_step")
    exit(0)

# ── Layer mapping for EVA02 base (12 blocks) ───────────────────────
#   depth 0:  patch_embed + cls_token + pos_embed + block.0
#   depth 1:  block.1   ...   depth 11: block.11 + fc_norm
#   depth 12: head / ArcFace
#   lr_scale = 0.85^(11-d) for d≤11, 1.0 for d=12
#   param groups split further by weight_decay (2e-5 vs 0)

HEAD_LR_MULT = 1.0

# ── Output ─────────────────────────────────────────────────────────
sorted_pgs = sorted(pg_data.keys())
base_lr_target = 3e-4
lr_bottom = pg_data[sorted_pgs[0]]   # smallest lr_scale (deepest block)
lr_top = pg_data[sorted_pgs[-1]]     # largest lr_scale (head)
base_lr_current = lr_top / HEAD_LR_MULT  # head group = base_lr * head_lr_mult

# Warmup progress
warmup_start = base_lr_target * 0.001  # 3e-7
warmup_progress = (base_lr_current - warmup_start) / (base_lr_target - warmup_start) * 100

print(f"Run: {run_name}  |  Step: {latest_step}  |  Phase: warmup ({warmup_progress:.1f}%)")
print(f"Base LR: {base_lr_current:.6e}  (target: {base_lr_target:.0e})")
print(f"  bottom (block.0): {lr_bottom:.6e}  |  top (head): {lr_top:.6e}")

if warmup_progress < 0.1:
    print(f"[WARN] Warmup at start -- LR barely above {warmup_start:.1e}")
elif warmup_progress < 99.9:
    print(f"[OK] Warmup in progress: {warmup_start:.1e} -> {base_lr_target:.1e}")
else:
    print(f"[DONE] Warmup complete, entering cosine decay")
