"""
Exploratory ground-truth plot for mice/v1: k=20 example frames for each of the
four (Y_nn, Y_nt) label combinations, laid out as a 2x2 contingency table of
panels (rows = nn=0/1, cols = nt=0/1).

No model is involved -- these are raw annotations, meant for eyeballing what
each label combination actually looks like on the 512x512 standardized frames.

Sampling maximizes diversity: 20 *distinct* observations per cell (all four
cells have >=28 annotated observations available), one random qualifying frame
each, so a single long bout can never fill a panel.

Y_np (the third, non-symmetrical nose-nose behavior) is orthogonal to the nn/nt
grid, so tiles where it is also positive are flagged -- "nn=0 & nt=0" is not
the same thing as "no behavior at all".

Usage:
    python scripts/mice_behavior/explore_label_combos.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATASET_DIR = Path('./dataset')
ANNOTATIONS_CSV = DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
OUT_PATH = Path('./results/vision/mice/frame/_figures/label_combo_examples.png')
SEED = 42
K = 20
N_COLS = 5                      # tile grid within one panel: N_COLS x (K // N_COLS)
COMBOS = [(0, 0), (0, 1), (1, 0), (1, 1)]   # (nn, nt), row-major over the 2x2
PANEL_COLOR = {(0, 0): '#888888', (0, 1): '#1f77b4', (1, 0): '#d62728', (1, 1): '#9467bd'}


def sample_cell(df, nn, nt, rng):
    """Pick K frames of the (nn, nt) cell from K distinct observations."""
    cell = df[(df.Y_nn == nn) & (df.Y_nt == nt)]
    obs = np.sort(cell.observation_id.unique())
    take = rng.choice(obs, size=min(K, len(obs)), replace=False)
    rows = []
    for o in take:
        g = cell[cell.observation_id == o]
        rows.append(g.iloc[rng.integers(len(g))])
    if len(rows) < K:                       # fewer observations than K: top up anywhere
        remaining = cell.drop(index=[r.name for r in rows])
        extra = rng.choice(remaining.index, size=min(K - len(rows), len(remaining)), replace=False)
        rows += [cell.loc[i] for i in extra]
    return pd.DataFrame(rows), len(cell), cell.observation_id.nunique()


def main():
    df = pd.read_csv(
        ANNOTATIONS_CSV,
        usecols=['observation_id', 'frame_idx', 'frame_path', 'Y_nn', 'Y_np', 'Y_nt'],
        low_memory=False,
    ).dropna(subset=['Y_nn', 'Y_np', 'Y_nt'])
    for c in ('Y_nn', 'Y_np', 'Y_nt'):
        df[c] = df[c].astype(int)
    n_annotated = len(df)
    print(f'{n_annotated} annotated frames over {df.observation_id.nunique()} observations')

    n_rows = K // N_COLS
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(2.05 * N_COLS * 2, 2.25 * n_rows * 2))
    outer = fig.add_gridspec(2, 2, hspace=0.13, wspace=0.06, top=0.90, bottom=0.02, left=0.04, right=0.99)

    for pi, (nn, nt) in enumerate(COMBOS):
        picks, n_cell, n_obs_cell = sample_cell(df, nn, nt, rng)
        pct = 100 * n_cell / n_annotated
        print(f'nn={nn} nt={nt}: {n_cell} frames ({pct:.3f}%), {n_obs_cell} obs -> sampled {len(picks)}')

        cell_gs = outer[pi // 2, pi % 2].subgridspec(n_rows, N_COLS, hspace=0.16, wspace=0.03)
        color = PANEL_COLOR[(nn, nt)]

        for j in range(K):
            ax = fig.add_subplot(cell_gs[j // N_COLS, j % N_COLS])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j >= len(picks):
                ax.axis('off')
                continue
            r = picks.iloc[j]
            try:
                ax.imshow(Image.open(DATASET_DIR / r.frame_path))
            except Exception:
                ax.text(0.5, 0.5, '(image not found)', ha='center', va='center', fontsize=6,
                        transform=ax.transAxes)
            tag = f'{r.observation_id}  #{r.frame_idx}'
            if r.Y_np == 1:
                tag += '  [np=1]'
            ax.set_title(tag, fontsize=6, color='#b8860b' if r.Y_np == 1 else 'black', pad=2)

        # panel frame + header, drawn in the outer cell's own coordinates
        hdr = fig.add_subplot(outer[pi // 2, pi % 2], zorder=-1)
        hdr.set_xticks([]); hdr.set_yticks([])
        hdr.patch.set_alpha(0)
        for s in hdr.spines.values():
            s.set_color(color); s.set_linewidth(2)
        np_share = 100 * picks.Y_np.mean() if len(picks) else 0.0
        hdr.set_title(
            f'nn={nn}, nt={nt}   —   {n_cell:,} frames ({pct:.2f}% of annotated), '
            f'{n_obs_cell} observations   |   Y_np=1 in {np_share:.0f}% of shown',
            fontsize=12, fontweight='bold', color=color, pad=22,
        )

    fig.suptitle(
        f'mice/v1 — k={K} random frames per (nn, nt) label combination\n'
        f'nn = symmetrical nose-nose · nt = non-symmetrical nose-tail · '
        f'one frame per distinct observation · seed={SEED}',
        fontsize=15, y=0.975,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=110)
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
