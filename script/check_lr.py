# -*- coding: utf-8 -*-
"""
Extract LR, train loss, and train accuracy from the latest wandb run.
Reads per-group lr-AdamW data + train/loss & train/acc from .wandb binary.
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
with open(wandb_file, 'rb') as f:
    data = f.read()

LR_MIN_SANE = 1e-10

# ══════════════════════════════════════════════════════════════════════
# 1. Global step
# ══════════════════════════════════════════════════════════════════════

step_positions = []
for m in re.finditer(rb'trainer/global_step.{0,10}?(\d+)', data):
    step = int(m.group(1))
    step_positions.append((m.start(), step))

if not step_positions:
    for m in re.finditer(rb'global_step.{0,10}?(\d+)', data):
        step = int(m.group(1))
        step_positions.append((m.start(), step))

if not step_positions:
    print("LR: no global_step found in wandb file")
    exit(0)

last_pos, latest_step = step_positions[-1]

# ══════════════════════════════════════════════════════════════════════
# 2. Learning rate (lr-AdamW/pg{N})
# ══════════════════════════════════════════════════════════════════════

LR_WINDOW = 20 * 1024  # 20KB
window_start = max(0, last_pos - LR_WINDOW)
window = data[window_start:last_pos + LR_WINDOW]

pg_data = {}
for m in re.finditer(rb'lr-AdamW/pg(\d+).{0,10}?([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)', window):
    pg = int(m.group(1))
    lr = float(m.group(2))
    if LR_MIN_SANE < lr < 1.0:
        pg_data[pg] = lr

# ══════════════════════════════════════════════════════════════════════
# 3. Train loss & accuracy
# ══════════════════════════════════════════════════════════════════════

def extract_metric(pattern, data, byte_positions_only=False):
    """Extract (byte_position, value) tuples; deduplicate by proximity."""
    entries = []
    seen = set()  # (int_position_bucket, rounded_value) for dedup
    for m in re.finditer(pattern, data):
        val = float(m.group(1))
        pos = m.start()
        # Bucket nearby positions to dedup protobuf duplicates
        bucket = pos // 500
        key = (bucket, round(val, 6))
        if key in seen:
            continue
        seen.add(key)
        entries.append((pos, val))
    entries.sort(key=lambda x: x[0])
    return entries

# Train loss: float in [0.1, 100]
loss_entries = []
for m in re.finditer(rb'train/loss.{0,20}?([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)', data):
    val = float(m.group(1))
    if 0.1 < val < 100:
        pos = m.start()
        bucket = pos // 500
        key = (bucket, round(val, 6))
        if key not in {e[0] for e in loss_entries[-1:]} or len(loss_entries) == 0:
            loss_entries.append((pos, val))

# Better dedup approach
loss_entries_raw = []
for m in re.finditer(rb'train/loss.{0,20}?([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)', data):
    val = float(m.group(1))
    if 0.1 < val < 100:
        loss_entries_raw.append((m.start(), val))

# Dedup by position proximity + value equality
loss_entries = []
for pos, val in loss_entries_raw:
    if loss_entries and abs(pos - loss_entries[-1][0]) < 200 and abs(val - loss_entries[-1][1]) < 0.001:
        continue  # duplicate
    loss_entries.append((pos, val))

# Train acc: float in [0, 1]
acc_entries_raw = []
for m in re.finditer(rb'train/acc.{0,20}?([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)', data):
    val = float(m.group(1))
    if 0 < val <= 1:
        acc_entries_raw.append((m.start(), val))

acc_entries = []
for pos, val in acc_entries_raw:
    if acc_entries and abs(pos - acc_entries[-1][0]) < 200 and abs(val - acc_entries[-1][1]) < 0.001:
        continue
    acc_entries.append((pos, val))

# Pair loss and acc by position proximity (they're logged together in training_step)
raw_pairs = []
for acc_pos, acc_val in acc_entries:
    best_loss = None
    best_dist = float('inf')
    for loss_pos, loss_val in loss_entries:
        dist = abs(acc_pos - loss_pos)
        if dist < 500 and dist < best_dist:
            best_dist = dist
            best_loss = (loss_pos, loss_val)
    if best_loss:
        raw_pairs.append((acc_pos, best_loss[1], acc_val))

# Dedup: discard pairs with same (loss, acc) within ~2KB
paired_entries = []
for pos, loss_val, acc_val in sorted(raw_pairs, key=lambda x: x[0]):
    if paired_entries:
        last = paired_entries[-1]
        if (abs(pos - last['pos']) < 2000 and
            abs(loss_val - last['loss']) < 0.001 and
            abs(acc_val - last['acc']) < 0.001):
            continue
    paired_entries.append({'pos': pos, 'loss': loss_val, 'acc': acc_val})

# ══════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════

# ── LR section ─────────────────────────────────────────────────────
if pg_data:
    sorted_pgs = sorted(pg_data.keys())
    lr_bottom = pg_data[sorted_pgs[0]]
    lr_top = pg_data[sorted_pgs[-1]]

    print(f"Run: {run_name}  |  Step: {latest_step}")
    print(f"Base LR: {lr_top:.6e}")
    print(f"  bottom (block.0): {lr_bottom:.6e}  |  top (head): {lr_top:.6e}")
else:
    print(f"LR: no lr-AdamW data near latest global_step")
    base_lr_current = None

# ── Train loss & acc section ────────────────────────────────────────
print()
if paired_entries:
    N_AVG = 30
    recent = paired_entries[-N_AVG:]
    avg_loss = sum(e['loss'] for e in recent) / len(recent)
    avg_acc = sum(e['acc'] for e in recent) / len(recent)
    print(f"Train  (avg of last {len(recent)} logged points, total {len(paired_entries)}):")
    print(f"  loss={avg_loss:.4f}  acc={avg_acc:.4f}")
else:
    print("Train loss/acc: no data found")
