"""
Evaluate the foreground SAEs (ECI mice v1) against the first full-crop SAE (ep20).

1. Token level (held-out pools, foreground tokens): FVE / L0 / dead per prefix, from metrics.json.
2. Cross-seed decoder stability (s0 vs s1), src.eci.sae.decoder_stability.
3. Frame level firing rate on the held-out pools at 1 fps (every frame of the foreground token
   store in those pools; a frame with no foreground patch counts as not firing): per latent, the
   fraction of frames where it fires on at least one foreground patch; median over latents per
   prefix. ep20: codes_max > 0 on the same rows.
4. Behaviour check (EVALUATION ONLY; labels never used for training or for the mask): on every
   5 fps frame of the annotated held-out videos, for Y_nn > 0, Y_np > 0, Y_nt > 0, the AUROC of
   each latent's per-frame max (over foreground patches for fg; over all 256 crop patches for
   ep20); best latent per prefix, and the top-5 latents.

Output: dataset/mice/v1/eci/fg448/eval_fg_vs_ep20.json

Usage: python scripts/eci/eval_sae_fg.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.foreground import FgTokenStore  # noqa: E402
from src.eci.sae import decoder_stability, load_sae  # noqa: E402
sys.path.insert(0, str(REPO / 'scripts/eci'))
from train_sae_fg import row_pools  # noqa: E402

EVENTS = ('Y_nn', 'Y_np', 'Y_nt')


def auroc_cols(X, y):
    """AUROC of every column of X (n, m) for binary y, ties averaged."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    out = np.empty(X.shape[1])
    for a in range(0, X.shape[1], 128):
        r = rankdata(np.asarray(X[:, a:a + 128], dtype=np.float32), axis=0)
        out[a:a + 128] = (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def auroc_table(X, labels, prefixes):
    res = {}
    for ev in EVENTS:
        y = labels[ev] > 0
        au = auroc_cols(X, y)
        res[ev] = {'n_pos': int(y.sum()), 'n': int(len(y)),
                   'best_per_prefix': {str(m): {'latent': int(np.argmax(au[:m])), 'auroc': float(au[:m].max())}
                                       for m in prefixes},
                   'top5': [{'latent': int(j), 'auroc': float(au[j])} for j in np.argsort(-au)[:5]]}
    return res


@torch.no_grad()
def frame_fire_fg(sae_path, store, is_val_row, device='cuda'):
    """Per-latent frame-level firing fraction over held-out 1 fps frames."""
    sae, norm, _ = load_sae(sae_path, device)
    frows, n_fg = store.frames()
    vframes = frows[is_val_row[frows]]
    index = {int(r): i for i, r in enumerate(vframes)}
    fired = torch.zeros(len(vframes), sae.n_latents, dtype=torch.float32, device=device)
    for s in range(len(store.dirs)):
        rows = store.row(s)
        sel = np.nonzero(is_val_row[rows])[0]
        if not len(sel):
            continue
        tok = store.tokens(s)
        fi_all = np.array([index[int(r)] for r in rows[sel]])
        for a in range(0, len(sel), 262144):
            idx = sel[a:a + 262144]
            z = sae.encode(norm(torch.from_numpy(np.asarray(tok[idx])).to(device)), mode='threshold') > 0
            fi = torch.from_numpy(fi_all[a:a + 262144]).to(device)
            fired.index_reduce_(0, fi, z.float(), 'amax', include_self=True)
    return (fired > 0).float().mean(0).cpu().numpy(), vframes


def main():
    p = argparse.ArgumentParser()
    ds = REPO / 'dataset/mice/v1'
    p.add_argument('--sae-dir', default=str(ds / 'eci/sae'))
    p.add_argument('--fg', default='matryoshka_btk_1024_k16_fg448_s0,matryoshka_btk_1024_k16_fg448_s1')
    p.add_argument('--ref', default='matryoshka_btk_1024_k16_ep20_s0')
    p.add_argument('--tokens-dir', default=str(ds / 'eci/train_tokens/dinov2_base_l-1_fg448_fps1'))
    p.add_argument('--eval-codes', default=str(ds / 'eci/fg448/eval_codes'))
    p.add_argument('--out', default=str(ds / 'eci/fg448/eval_fg_vs_ep20.json'))
    args = p.parse_args()
    fg = args.fg.split(',')
    sae_dir = Path(args.sae_dir)
    prefixes = [128, 256, 512, 1024]
    res = {'fg': fg, 'ref': args.ref}

    # 1. token-level metrics
    res['token_level'] = {}
    for s in fg + [args.ref]:
        m = json.loads((sae_dir / s / 'metrics.json').read_text())
        res['token_level'][s] = {k: {kk: v[kk] for kk in ('fve', 'l0_per_token', 'dead_frac', 'n_dead')}
                                 for k, v in m['val_threshold']['prefixes'].items()}
        res['token_level'][s]['n_val_tokens'] = m.get('n_val_tokens')
        res['token_level'][s]['n_steps'] = m.get('n_steps') or len(m['history']) and m['history'][-1]['step'] + 1
    # 2. stability
    W = [load_sae(sae_dir / s / 'sae.pt')[0].W_dec.detach() for s in fg[:2]]
    res['stability_s0_vs_s1'] = decoder_stability(W[0], W[1], prefixes)

    # 3. frame-level firing (held-out pools, 1 fps rows of the foreground store)
    val_pools = json.loads((sae_dir / fg[0] / 'metrics.json').read_text())['val_pools']
    codes, names = row_pools(REPO / 'dataset', REPO / 'data')
    is_val_row = np.isin(np.array(names)[codes], val_pools)
    store = FgTokenStore(args.tokens_dir)
    res['frame_firing'] = {}
    for s in fg:
        rate, vframes = frame_fire_fg(sae_dir / s / 'sae.pt', store, is_val_row)
        res['frame_firing'][s] = {str(m): {'median': float(np.median(rate[:m])), 'mean': float(rate[:m].mean()),
                                           'frac_latents_gt_90pct': float((rate[:m] > 0.9).mean())} for m in prefixes}
    ref_max = np.load(ds / 'eci/codes' / args.ref / 'codes_max.npy', mmap_mode='r')
    rate = (np.asarray(ref_max[np.sort(vframes)]) > 0).mean(0)
    res['frame_firing'][args.ref] = {str(m): {'median': float(np.median(rate[:m])), 'mean': float(rate[:m].mean()),
                                              'frac_latents_gt_90pct': float((rate[:m] > 0.9).mean())} for m in prefixes}
    res['frame_firing']['n_frames'] = int(len(vframes))

    # 4. behaviour AUROC on annotated held-out videos, 5 fps
    parts = [np.load(f) for f in sorted(Path(args.eval_codes).glob('task_*.npz')) if '.tmp' not in f.name]
    rows = np.concatenate([z['rows'] for z in parts])
    order = np.argsort(rows)
    rows = rows[order]
    labels = pd.read_csv(ds / 'annotations.csv', usecols=list(EVENTS)).iloc[rows].reset_index(drop=True)
    res['auroc'] = {'n_frames': int(len(rows))}
    for s in fg:
        X = np.concatenate([z[f'codes_max_{s}'] for z in parts])[order]
        res['auroc'][s] = auroc_table(X, labels, prefixes)
    res['auroc'][args.ref] = auroc_table(np.asarray(ref_max[rows]), labels, prefixes)
    n_fg = np.concatenate([z['n_fg'] for z in parts])[order]
    res['auroc']['n_fg_eval_frames'] = {'median': float(np.median(n_fg)), 'zero': int((n_fg == 0).sum())}
    Path(args.out).write_text(json.dumps(res, indent=1))

    # print summary
    for s, t in res['token_level'].items():
        print(s, {m: (round(v['fve'], 4), round(v['l0_per_token'], 2), round(v['dead_frac'], 4))
                  for m, v in t.items() if m.isdigit()})
    print('stability', json.dumps(res['stability_s0_vs_s1']))
    for s, t in res['frame_firing'].items():
        print('frame firing', s, t)
    for s in fg + [args.ref]:
        for ev in EVENTS:
            t = res['auroc'][s][ev]
            print(s, ev, t['n_pos'], {m: round(v['auroc'], 3) for m, v in t['best_per_prefix'].items()},
                  [(d['latent'], round(d['auroc'], 3)) for d in t['top5']])


if __name__ == '__main__':
    main()
