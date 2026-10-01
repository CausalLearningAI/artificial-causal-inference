"""
Raw-token quality check of the SAE input encoders (no SAE): DINOv2-base (448) vs DINOv3 ViT-B/16 (512), same
frames and the same foreground patches (the mask always comes from DINOv2, src/eci/foreground.py FgEncoder).

Frames: labelled frames only, sampled per video with the positives over-represented (AUC does not depend on
the class balance): mice = dataset/mice/v1/annotations.csv rows with Y_nn / Y_np / Y_nt all present (144 videos),
label = social contact (any of the three); ants = dataset/ants/eci/annotations.csv (v2 + v3), label = grooming
(Y_B2F or Y_Y2F). Per video up to --n-pos positive and --n-neg negative frames (seeded).

Per frame and encoder: CLS token, mean of the foreground patch tokens, mean of all 1024 patch tokens.
Linear probe: standardize + logistic regression (C = 0.1 and 1), 5-fold cross-validation grouped by video,
AUC of the pooled out-of-fold predictions.

Position share of the token variance: on up to --n-pos-tokens tokens per frame (foreground patches, and
uniformly random patches of the whole frame), the fraction of the total token variance (summed over dims) that
is explained by the patch position (1024 groups: between-position sum of squares / total sum of squares).
The chance level of that ratio (1023 / n tokens for unstructured data) is reported too.

Output: <out-dir>/<domain>/probe.json (+ features.npz with the pooled features, labels and videos).

Usage:
    python scripts/eci/encoder_probe.py --domain mice
    python scripts/eci/encoder_probe.py --domain ants --n-pos 15 --n-neg 15
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, obs_rows  # noqa: E402

LABELS = {'mice': ['Y_nn', 'Y_np', 'Y_nt'], 'ants': ['Y_B2F', 'Y_Y2F']}


def sample_rows(ann_path, domain, n_pos, n_neg, seed):
    cols = LABELS[domain]
    a = pd.read_csv(ann_path, usecols=['observation_id'] + cols)
    ok = a[cols].notna().all(1).values
    y = (a[cols].fillna(0).sum(1).values > 0) & ok
    rng = np.random.default_rng(seed)
    rows = []
    for _, idx in a.index.to_series()[ok].groupby(a['observation_id'][ok]):
        idx = idx.values
        pos, neg = idx[y[idx]], idx[~y[idx]]
        rows.append(rng.choice(pos, min(n_pos, len(pos)), replace=False))
        rows.append(rng.choice(neg, min(n_neg, len(neg)), replace=False))
    rows = np.sort(np.concatenate(rows))
    return rows, y[rows].astype(np.int8), a['observation_id'].values[rows]


@torch.no_grad()
def forward(model, pix, device):
    """-> CLS (B, d) float32, patch tokens (B, 1024, d) float16 (as stored for the SAEs)."""
    n_prefix = 1 + (getattr(model.config, 'num_register_tokens', 0) or 0)
    with torch.inference_mode():
        hs = model(pixel_values=pix.to(device, non_blocking=True)).last_hidden_state.float()
    assert hs.shape[1] == n_prefix + 1024 and torch.isfinite(hs).all(), hs.shape
    return hs[:, 0], hs[:, n_prefix:].half()


def probe(X, y, groups, C, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        pred[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def position_share(tok, pos):
    """tok (n, d) float32, pos (n,) -> between-position SS / total SS (summed over dims), chance 1023 / n."""
    tok = tok.astype(np.float64)
    mu = tok.mean(0)
    total = float(((tok - mu) ** 2).sum())
    cnt = np.bincount(pos, minlength=1024).astype(np.float64)
    sums = np.zeros((1024, tok.shape[1]))
    np.add.at(sums, pos, tok)
    used = cnt > 0
    means = sums[used] / cnt[used, None]
    between = float((cnt[used, None] * (means - mu) ** 2).sum())
    return {'share': between / total, 'chance': float((used.sum() - 1) / len(pos)), 'n_tokens': int(len(pos)),
            'n_positions': int(used.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', default='mice', choices=DOMAINS)
    p.add_argument('--encoder2', default='dinov3_base')
    p.add_argument('--n-pos', type=int, default=40)
    p.add_argument('--n-neg', type=int, default=80)
    p.add_argument('--n-pos-tokens', type=int, default=32, help='tokens per frame for the position share')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--out-dir', default=str(REPO / 'results/vision/eci_encoders'))
    args = p.parse_args()
    dom = get_domain(args.domain)
    out = Path(args.out_dir) / args.domain
    out.mkdir(parents=True, exist_ok=True)
    ann = dom.ann_path
    rows, y, vid = sample_rows(ann, args.domain, args.n_pos, args.n_neg, args.seed)
    print(f'{args.domain}: {len(rows)} frames, {int(y.sum())} positive, {len(np.unique(vid))} videos', flush=True)

    device = torch.device('cuda')
    enc = FgEncoder(args.encoder2, device)
    bgs = FgBackgrounds(dom.eci_dir / 'fg448/background', obs_rows(ann), RULES[dom.fg_rule], device)
    paths = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
    loader = torch.utils.data.DataLoader(enc.dataset([str(REPO / 'dataset' / pth) for pth in paths[rows]], rows),
                                         batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    names = ('dinov2_base', args.encoder2)
    feats = {f'{e}_{k}': [] for e in names for k in ('cls', 'fg', 'all')}
    ptok = {(e, k): [] for e in names for k in ('fg', 'all')}
    ppos = {k: [] for k in ('fg', 'all')}
    n_fg_all = []
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for b, (pix, grey, r, pix2) in enumerate(loader):
        r = r.numpy()
        cls1, tok1 = forward(enc.model, pix, device)
        mask, _ = bgs.mask(tok1, grey.to(device), r)
        cls2, tok2 = forward(enc.model2, pix2, device)
        m = mask.float()[..., None]
        n = m.sum(1).clamp_min(1)
        for e, cls, tok in ((names[0], cls1, tok1), (names[1], cls2, tok2)):
            feats[f'{e}_cls'].append(cls.cpu().numpy())
            feats[f'{e}_fg'].append(((tok.float() * m).sum(1) / n).cpu().numpy())
            feats[f'{e}_all'].append(tok.float().mean(1).cpu().numpy())
        mk = mask.cpu().numpy()
        n_fg_all.append(mk.sum(1))
        for i in range(len(r)):  # the same patches for both encoders
            fgp = np.nonzero(mk[i])[0]
            sel = {'fg': rng.choice(fgp, min(args.n_pos_tokens, len(fgp)), replace=False),
                   'all': rng.choice(1024, args.n_pos_tokens, replace=False)}
            for k, s in sel.items():
                ppos[k].append(s)
                st = torch.from_numpy(s).to(device)
                ptok[(names[0], k)].append(tok1[i, st].cpu().numpy())
                ptok[(names[1], k)].append(tok2[i, st].cpu().numpy())
        if b % 50 == 0:
            print(f'  batch {b}  {(b + 1) * args.batch_size}/{len(rows)}  {time.time() - t0:.0f}s', flush=True)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    n_fg = np.concatenate(n_fg_all)
    np.savez(out / 'features.npz', rows=rows, y=y, vid=vid, n_fg=n_fg, **feats)

    res = {'domain': args.domain, 'n_frames': int(len(rows)), 'n_pos': int(y.sum()),
           'n_videos': int(len(np.unique(vid))), 'frames_without_fg': int((n_fg == 0).sum()),
           'n_fg_median': float(np.median(n_fg)), 'probe_auc': {}, 'position_share': {}}
    groups = pd.factorize(vid)[0]
    for k, X in feats.items():
        res['probe_auc'][k] = {f'C={C}': probe(X, y, groups, C) for C in (0.1, 1.0)}
        print(k, res['probe_auc'][k], flush=True)
    for (e, k), v in ptok.items():
        res['position_share'][f'{e}_{k}'] = position_share(np.concatenate(v).astype(np.float32), np.concatenate(ppos[k]))
        print(e, k, res['position_share'][f'{e}_{k}'], flush=True)
    res['elapsed_s'] = round(time.time() - t0, 1)
    (out / 'probe.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
