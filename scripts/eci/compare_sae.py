"""
Compare ECI SAEs (e.g. v1 = 64 frames/obs vs v2 = 1 fps) on the SAME held-out tokens:
the 1 fps frames of the validation pools (pool split of train_sae.py, identical for all).

Per SAE and Matryoshka prefix:
    val          FVE / L0 per token / dead latents on all held-out 1 fps frames
    rare moment  FVE on the patch tokens of annotated-event frames (Y_nn>0, Y_np>0, Y_nt>0,
                 from annotations.csv) vs an equal-size random sample of held-out frames.
                 'fve' uses the subset's own mean; 'fve_vs_val_var' = 1 - MSE / (variance of
                 all held-out tokens), comparable across subsets.
    detection    single-latent AUROC of the max-pooled frame code for Y_e>0 on the annotated
                 held-out frames: 'selected' = the latent (among the first m) with the best
                 AUROC on the annotated TRAIN-pool 1 fps frames, evaluated on held-out;
                 'oracle' = best held-out AUROC among the first m (optimistic).
Labels are used only here, never for training. Also: decoder stability per prefix for
each pair given in --pairs.

Output: dataset/mice/v1/eci/diagnostics/sae_compare_<name>.json + printed tables.

Usage:
    python scripts/eci/compare_sae.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.sae import decoder_stability, evaluate_sae, load_sae  # noqa: E402
from src.eci.token_store import TokenStore  # noqa: E402
sys.path.insert(0, str(REPO / 'scripts/eci'))
from train_sae import pool_split  # noqa: E402

EVENTS = ('Y_nn', 'Y_np', 'Y_nt')


@torch.no_grad()
def max_codes(sae, norm, tok, n_patches=256, batch_frames=128):
    """(n_frames*P, d) or (n_frames, P, d) fp16 tokens (CPU) -> (n_frames, m) float16 max-pooled codes."""
    dev = sae.W_dec.device
    tok = tok.reshape(-1, n_patches, tok.shape[-1])
    out = torch.empty((tok.shape[0], sae.n_latents), dtype=torch.float16)
    for f0 in range(0, tok.shape[0], batch_frames):
        x = torch.as_tensor(tok[f0:f0 + batch_frames]).to(dev)
        B = x.shape[0]
        z = sae.encode(norm(x.reshape(B * n_patches, -1)), mode='threshold').view(B, n_patches, -1)
        out[f0:f0 + B] = z.amax(1).half().cpu()
    return out.numpy()


def auroc_cols(scores, y):
    """AUROC of every column of scores (n, m) for binary y (n,), ties -> average rank."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    r = rankdata(scores.astype(np.float32), axis=0)
    return (r[y].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    root = REPO / 'dataset/mice/v1/eci'
    p = argparse.ArgumentParser()
    p.add_argument('--tokens-dir', default=str(root / 'train_tokens/dinov2_base_l-1_fps1'))
    p.add_argument('--saes', default='matryoshka_btk_1024_k16_ep20_s0,matryoshka_btk_1024_k16_ep20_s1,'
                                     'matryoshka_btk_1024_k16_fps1_s0,matryoshka_btk_1024_k16_fps1_s1')
    p.add_argument('--pairs', default='matryoshka_btk_1024_k16_ep20_s0:matryoshka_btk_1024_k16_ep20_s1,'
                                      'matryoshka_btk_1024_k16_fps1_s0:matryoshka_btk_1024_k16_fps1_s1,'
                                      'matryoshka_btk_1024_k16_ep20_s0:matryoshka_btk_1024_k16_fps1_s0')
    p.add_argument('--n-random', type=int, default=2000, help='random held-out frames for the rare-moment baseline')
    p.add_argument('--name', default='v1_vs_fps1')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    store = TokenStore(args.tokens_dir)
    meta = store.meta
    val_pools = pool_split(meta, 4, 0)
    is_val = meta.pool.isin(val_pools).values
    names = args.saes.split(',')
    saes = {n: load_sae(root / 'sae' / n / 'sae.pt', dev) for n in names}
    for n, (_, _, ck) in saes.items():
        assert sorted(ck['val_pools']) == sorted(val_pools), (n, ck['val_pools'], val_pools)
    print(f'val pools {val_pools} (identical in all {len(names)} SAE checkpoints)', flush=True)

    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=list(EVENTS))
    lab = ann.iloc[meta.row_idx.values].reset_index(drop=True)
    annotated = lab[EVENTS[0]].notna().values

    val_idx = np.nonzero(is_val)[0]
    val_tok = store.frames(val_idx)                                   # (n_val, P, d) fp16
    val_flat = torch.from_numpy(val_tok.reshape(-1, store.dim))
    print(f'{len(val_idx)} held-out frames ({val_flat.shape[0]:,} tokens) loaded, {time.time() - t0:.0f}s', flush=True)
    val_lab = lab.iloc[val_idx].reset_index(drop=True)
    val_ann = annotated[val_idx]

    rng = np.random.default_rng(0)
    subsets = {'random': np.sort(rng.choice(len(val_idx), args.n_random, replace=False))}
    for e in EVENTS:
        subsets[e] = np.nonzero(val_ann & (val_lab[e].fillna(0).values > 0))[0]
    subsets['any_event'] = np.nonzero(val_ann & (val_lab[list(EVENTS)].fillna(0).values > 0).any(1))[0]
    subsets['annotated_no_event'] = np.nonzero(val_ann & ~(val_lab[list(EVENTS)].fillna(0).values > 0).any(1))[0]
    print('subset sizes (frames): ' + ', '.join(f'{k} {len(v)}' for k, v in subsets.items()), flush=True)

    # annotated train-pool frames, for choosing the latent (labels never touch the SAE)
    tr_idx = np.nonzero(~is_val & annotated)[0]
    tr_lab = lab.iloc[tr_idx].reset_index(drop=True)
    tr_codes = {n: np.empty((len(tr_idx), 1024), np.float16) for n in names}
    for a in range(0, len(tr_idx), 8192):
        tok = store.frames(tr_idx[a:a + 8192])
        for n, (sae, norm, _) in saes.items():
            tr_codes[n][a:a + len(tok)] = max_codes(sae, norm, tok)
    print(f'{len(tr_idx)} annotated train-pool frames encoded, {time.time() - t0:.0f}s', flush=True)

    res = {'tokens_dir': args.tokens_dir, 'val_pools': val_pools, 'n_val_frames': int(len(val_idx)),
           'n_val_annotated_frames': int(val_ann.sum()), 'n_train_annotated_frames': int(len(tr_idx)),
           'subset_frames': {k: int(len(v)) for k, v in subsets.items()},
           'positives_val': {e: int((val_lab[e][val_ann] > 0).sum()) for e in EVENTS},
           'positives_train': {e: int((tr_lab[e] > 0).sum()) for e in EVENTS}, 'saes': {}}
    for n, (sae, norm, _) in saes.items():
        r = {}
        ev = evaluate_sae(sae, norm, val_flat, mode='threshold')
        var_val = ev['total_var_per_token']
        r['val'] = {m: {k: v[k] for k in ('fve', 'l0_per_token', 'dead_frac', 'n_dead')} for m, v in ev['prefixes'].items()}
        r['rare'] = {}
        for s, fi in subsets.items():
            if len(fi) == 0:
                continue
            es = evaluate_sae(sae, norm, torch.from_numpy(val_tok[fi].reshape(-1, store.dim)), mode='threshold')
            r['rare'][s] = {m: {'fve': v['fve'], 'fve_vs_val_var': 1 - v['mse_per_token'] / var_val,
                                'l0_per_token': v['l0_per_token']} for m, v in es['prefixes'].items()}
        codes = max_codes(sae, norm, val_tok)[val_ann]
        r['auroc'] = {}
        for e in EVENTS:
            au_val = auroc_cols(codes, val_lab[e][val_ann].values > 0)
            au_tr = auroc_cols(tr_codes[n], tr_lab[e].values > 0)
            r['auroc'][e] = {}
            for m in sae.prefixes:
                j = int(np.argmax(au_tr[:m]))
                jo = int(np.argmax(au_val[:m]))
                r['auroc'][e][str(m)] = {'selected_latent': j, 'selected_train_auroc': float(au_tr[j]),
                                         'selected_heldout_auroc': float(au_val[j]),
                                         'oracle_latent': jo, 'oracle_heldout_auroc': float(au_val[jo])}
        res['saes'][n] = r
        print(f'{n} done, {time.time() - t0:.0f}s', flush=True)

    res['stability'] = {}
    for pr in args.pairs.split(','):
        a, b = pr.split(':')
        sa, sb = load_sae(root / 'sae' / a / 'sae.pt')[0], load_sae(root / 'sae' / b / 'sae.pt')[0]
        res['stability'][pr] = decoder_stability(sa.W_dec.data, sb.W_dec.data, sa.prefixes)

    out = root / 'diagnostics' / f'sae_compare_{args.name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))

    P = ['128', '256', '512', '1024']
    print('\nVAL (held-out 1 fps frames): FVE / L0 / dead% per prefix')
    for n, r in res['saes'].items():
        print(f'  {n:34s} ' + '  '.join(f"{m}: {r['val'][m]['fve']:.3f}/{r['val'][m]['l0_per_token']:.1f}/"
                                        f"{100 * r['val'][m]['dead_frac']:.1f}" for m in P))
    print('\nRARE MOMENT: FVE_vs_val_var at prefix 128 / 1024')
    for n, r in res['saes'].items():
        print(f'  {n:34s} ' + '  '.join(f"{s}: {v['128']['fve_vs_val_var']:.3f}/{v['1024']['fve_vs_val_var']:.3f}"
                                        for s, v in r['rare'].items()))
    print('\nAUROC (selected on train pools -> held-out) [oracle] per prefix')
    for e in EVENTS:
        for n, r in res['saes'].items():
            print(f'  {e} {n:34s} ' + '  '.join(f"{m}: {r['auroc'][e][m]['selected_heldout_auroc']:.3f} "
                                               f"[{r['auroc'][e][m]['oracle_heldout_auroc']:.3f}]" for m in P))
    print('\nSTABILITY median max-cos (same prefix) / frac>0.9')
    for pr, st in res['stability'].items():
        print(f'  {pr}  ' + '  '.join(f"{m}: {st[m]['median_maxcos_same_prefix']:.3f}/"
                                      f"{st[m]['frac_gt_0.9_same_prefix']:.2f}" for m in P))
    print(f'-> {out}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
