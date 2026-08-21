# -*- coding: utf-8 -*-
"""Peek latest val/train top1 & top3 acc from the live wandb offline file."""
import re, os, glob

root = r'D:\image-classification\wandb_logs\wandb'
dirs = sorted(glob.glob(os.path.join(root, 'offline-run-*')), key=os.path.getmtime)
d = dirs[-1]
name = os.path.basename(d).split('-')[-1]
f = os.path.join(d, f'run-{name}.wandb')

with open(f, 'rb') as fp:
    data = fp.read()

print('run:', name, '| bytes:', len(data))

def tail_vals(pattern, lo=None, hi=None, max_n=8):
    out = []
    for m in re.finditer(pattern, data):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if lo is not None and not (lo < v <= hi):
            continue
        out.append(v)
    return out[-max_n:]

pat = lambda k: rb'' + k.encode() + rb'.{0,24}?([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)'

for k in ['val/acc_top3', 'val/acc_top1', 'train/acc']:
    if k.startswith('val/'):
        vals = tail_vals(pat(k), 0, 1)
    else:
        vals = tail_vals(pat(k), 0, 1)
    print(f'{k}: {vals}')
