"""Track and visualise training-curve evolution across runs, live or after the fact.

Parses the epoch lines that the mice_behavior training scripts already print, so it works on
RUNNING jobs (no instrumentation, no wandb, no restart needed) as well as finished ones.

Produces results/vision/mice/frame/run_tracking.png with three panels -- monitor AP, train
loss, val loss -- one line per run, plus a printed table of where each run actually stopped
improving. That last part is the point: it exposes how many epochs were spent after the LR
schedule floored, which is compute spent for almost no gain.

Usage:
    python scripts/mice_behavior/track_runs.py                       # every recent run
    python scripts/mice_behavior/track_runs.py --pattern 'ctxk*'     # subset
    python scripts/mice_behavior/track_runs.py --watch 120           # refresh every 120s
"""
import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path('logs')
OUT_PNG = Path('results/vision/mice/frame/run_tracking.png')

# train_patchgrid_online.py: "epoch 3/20 loss=.. val_loss=.. macro_ap=.. nt=.. nn=.. lr=.. (12.3s)"
# train_online_aug.py:       "epoch 3/20 loss=.. monitor_ap=.. lr=.. (12.3s)"
EPOCH_RE = re.compile(
    r'epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)(?:\s+val_loss=([\d.]+))?'
    r'\s+(?:macro_ap|monitor_ap)=([\d.]+)(?:\s+nt=([\d.]+)\s+nn=([\d.]+))?'
    r'.*?lr=([\d.e+-]+).*?\(([\d.]+)s\)')


def parse(path: Path):
    ep = []
    try:
        txt = path.read_text(errors='ignore')
    except OSError:
        return None
    label = None
    m = re.search(r'context_k=(\d+)', txt)
    if m:
        label = f'k={m.group(1)}'
    m = re.search(r'augment=(\w+)', txt)
    if m:
        label = f'{label or ""} aug={m.group(1)}'.strip()
    for line in txt.splitlines():
        mm = EPOCH_RE.search(line)
        if mm:
            g = mm.groups()
            ep.append(dict(epoch=int(g[0]), n_epochs=int(g[1]), loss=float(g[2]),
                           val_loss=float(g[3]) if g[3] else np.nan, ap=float(g[4]),
                           nt=float(g[5]) if g[5] else np.nan,
                           nn=float(g[6]) if g[6] else np.nan,
                           lr=float(g[7]), secs=float(g[8])))
    if not ep:
        return None
    fv = re.search(r'FULL-VAL macro AP:\s*([\d.]+)', txt)
    return dict(job=path.stem.split('_')[-1], label=label or path.stem,
                epochs=ep, full_val=float(fv.group(1)) if fv else None,
                running=('Done.' not in txt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', default='*')
    p.add_argument('--watch', type=int, default=0, help='seconds between refreshes; 0 = once')
    args = p.parse_args()

    while True:
        runs = []
        for f in sorted(LOG_DIR.glob('mice_patchgrid_online_*.out')) + sorted(LOG_DIR.glob('online_aug_*.out')):
            r = parse(f)
            if r and (args.pattern == '*' or re.search(args.pattern.replace('*', '.*'), r['label'])):
                runs.append(r)
        runs = [r for r in runs if len(r['epochs']) >= 1]
        if not runs:
            print('no runs with epoch lines found')
            return

        print(f'\n{"run":<22} {"job":>9} {"eps":>4} {"best AP":>8} {"@ep":>4} '
              f'{"gain after LR floor":>20} {"s/ep":>6} {"full-val":>9}')
        for r in sorted(runs, key=lambda x: x['label']):
            e = r['epochs']
            aps = [x['ap'] for x in e]
            best_i = int(np.argmax(aps))
            lrs = [x['lr'] for x in e]
            floor = min(lrs)
            first_floor = next((i for i, x in enumerate(lrs) if x <= floor * 1.001), len(e) - 1)
            gain = aps[-1] - aps[first_floor]
            frac = f'{gain:+.4f} ({100*gain/max(aps[first_floor],1e-9):+.1f}%)'
            print(f'{r["label"]:<22} {r["job"]:>9} {len(e):>4} {max(aps):>8.4f} {best_i+1:>4} '
                  f'{frac:>20} {np.mean([x["secs"] for x in e]):>6.1f} '
                  f'{r["full_val"] if r["full_val"] else "-":>9}'
                  f'{"  [RUNNING]" if r["running"] else ""}')
            print(f'{"":22} -> LR floored at epoch {first_floor+1}/{len(e)}; '
                  f'{len(e)-first_floor-1} epochs spent after that')

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        cmap = plt.get_cmap('tab10')
        for i, r in enumerate(sorted(runs, key=lambda x: x['label'])):
            e = r['epochs']
            x = [q['epoch'] for q in e]
            c = cmap(i % 10)
            lab = f'{r["label"]} ({r["job"]})'
            axes[0].plot(x, [q['ap'] for q in e], marker='o', ms=3, color=c, label=lab)
            axes[1].plot(x, [q['loss'] for q in e], marker='o', ms=3, color=c, label=lab)
            axes[2].plot(x, [q['val_loss'] for q in e], marker='o', ms=3, color=c, label=lab)
            lrs = [q['lr'] for q in e]
            fl = next((j for j, v in enumerate(lrs) if v <= min(lrs) * 1.001), None)
            if fl is not None:
                axes[0].axvline(x[fl], color=c, ls=':', alpha=.5)
        axes[0].set_title('monitor AP (dotted = LR floor)'); axes[0].set_xlabel('epoch')
        axes[1].set_title('train loss'); axes[1].set_xlabel('epoch')
        axes[2].set_title('val loss'); axes[2].set_xlabel('epoch')
        for a in axes:
            a.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        plt.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_PNG, dpi=110)
        plt.close()
        print(f'\nsaved {OUT_PNG}')

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
