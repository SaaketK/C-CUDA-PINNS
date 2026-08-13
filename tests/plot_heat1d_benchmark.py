#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]/'src/tests/results/heat1d'
R.mkdir(parents=True,exist_ok=True)
plt.figure(figsize=(8,5))
for p in sorted(R.glob('*_loss.csv')):
 with p.open() as f: x=list(csv.DictReader(f))
 if x: plt.semilogy([int(r['step']) for r in x],[float(r['physics_loss']) for r in x],label=p.stem.replace('_loss',''))
plt.xlabel('Training step');plt.ylabel('Physics loss');plt.legend();plt.tight_layout();plt.savefig(R/'loss_comparison.png',dpi=160);plt.close()
m=[json.loads(p.read_text()) for p in sorted(R.glob('*_metrics.json'))]
if m:
 plt.figure(figsize=(8,5));plt.bar([f"{x['implementation']}\n{x['device']}" for x in m],[x['milliseconds_per_step'] for x in m]);plt.ylabel('Milliseconds per training step');plt.tight_layout();plt.savefig(R/'timing_comparison.png',dpi=160)
