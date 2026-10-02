"""
Where on screen do a foreground SAE's latents fire, relative to the odor corner? (ECI, mice v1; used to compare the
unaligned SAE fg448 with the odor-aligned SAE fg448al, src/eci/foreground.py align_rot90.)

For each SAE, a seeded sample of its own training tokens (every --stride-th block of 8,192 tokens (about 55 frames) of every shard of
the SAE's token store; the tokens are the SAE's own input, so patch positions are in the SAE's own frame coordinates:
raw camera frame for fg448, odor corner at the top right for fg448al) is encoded (threshold inference), and per latent:
  near_share        share of the latent's summed activation on tokens inside the video's near-odor zone (the zone of
                    src/eci/spatial_encode.py zone_maps: patch centres within 0.5 x arena side of the arena corner on
                    the odor side; turned with the frames for an aligned SAE). Coordinate-free: the same quantity for
                    both SAEs. near_enrichment = near_share / the share of all sampled foreground tokens in the zone.
  quadrant_share    share of the latent's arena map mass (summed activation per patch position, all videos pooled, the
                    explorer's arena map) in each image quadrant TL / TR / BL / BR of the SAE's own frame.
  map_corr_TR_BL    Pearson correlation over the 1024 positions of the latent's arena map from the videos whose odor
                    corner is TR and from those whose corner is BL (both in the SAE's own frame). A latent tied to the
                    odor corner gives a high value only when the frames are aligned; one tied to the camera frame gives
                    a high value only when they are not.
Selected neurons: the primary NES selections (codes_max, per-video mean / bout rate, t, Bonferroni, full window,
prefixes 128 and 1024) of <nes root>/<sae>/summary.csv and maxpool_bouts/summary.csv.

Output: <out-dir>/arena_<sae>.npz (per corner group: summed activation (1024 positions, n_latents), token counts,
frames) and <out-dir>/arena_compare.json (per SAE: all-latent summaries + per selected neuron metrics).

Usage:
    python scripts/eci/align_arena.py --sae matryoshka_btk_1024_k16_fg448_s0 --sae matryoshka_btk_1024_k16_fg448al_s0
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
from src.eci.domain import get_domain  # noqa: E402
from src.eci.foreground import GRID, ODOR_ROT90, FgTokenStore, obs_rows  # noqa: E402
from src.eci.sae import load_sae  # noqa: E402
from src.eci.spatial_encode import zone_maps  # noqa: E402

GROUPS = ('TR', 'BL', 'BR')
PRIMARY = {'mean_activation': ('', 'p{}_max_mean_t_bonferroni_full'),
           'bouts': ('maxpool_bouts', 'p{}_bout_rate_q0.95_g0_t_bonferroni_full')}


def selections(nes_dir, prefixes=(128, 1024)):
    out = []
    for kind, (sub, fmt) in PRIMARY.items():
        f = nes_dir / sub / 'summary.csv' if sub else nes_dir / 'summary.csv'
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s = s[s['setting'].isin([fmt.format(p) for p in prefixes]) & s['neuron'].notna()]
        for r in s.itertuples():
            out.append({'kind': kind, 'analysis_id': r.analysis_id, 'prefix': int(r.prefix), 'round': int(r.round),
                        'neuron': int(r.neuron), 'tau': float(r.tau), 'p': float(r.p)})
    return out


@torch.no_grad()
def accumulate(sae_dir, dom, corner_csv, stride, chunk=8192, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sae, norm, ck = load_sae(sae_dir / 'sae.pt', device)
    align = ck.get('align', 'none')
    m = json.loads((sae_dir / 'metrics.json').read_text())
    tdir = Path(m['args']['tokens_dir'])
    store = FgTokenStore(tdir if tdir.is_absolute() else REPO / tdir)
    if any(i.get('align', 'none') != align for i in store.info):
        raise SystemExit(f'{tdir}: token store align differs from the SAE ({align})')
    if int(ck.get('motion_delta', 0) or 0) or ck.get('bg_sub', False):
        raise SystemExit('static raw-token SAEs only')
    ranges = obs_rows(dom.ann_path)
    ids = sorted(ranges, key=lambda o: ranges[o][0])
    starts = np.array([ranges[o][0] for o in ids])
    corner = pd.read_csv(corner_csv).set_index('observation_id')['odor_corner']
    vid_group = np.array([GROUPS.index(corner[o]) for o in ids])
    zm, _ = zone_maps('mice', ids, corner_csv)  # near odor = 0, raw frame coordinates
    near = zm == 0
    if align == 'odor':
        for k, o in enumerate(ids):
            near[k] = np.rot90(near[k].reshape(GRID, GRID), ODOR_ROT90[corner[o]]).ravel()
    elif align != 'none':
        raise SystemExit(f'unknown align {align}')
    near_t = torch.from_numpy(near).to(device)
    L, P = sae.n_latents, GRID * GRID
    S = torch.zeros(len(GROUPS), P, L, dtype=torch.float64, device=device)
    occ = torch.zeros(len(GROUPS), P, dtype=torch.float64, device=device)
    z_near = torch.zeros(L, dtype=torch.float64, device=device)
    n_near, frames = 0, [set() for _ in GROUPS]
    t0, n_tok = time.time(), 0
    for s in range(len(store.dirs)):
        tok, row, pos = store.tokens(s), store.row(s), store.pos(s).astype(np.int64)
        for b, a in enumerate(range(0, len(row), chunk)):
            if b % stride:
                continue
            r = row[a:a + chunk]
            v = np.searchsorted(starts, r, side='right') - 1
            g = torch.from_numpy(vid_group[v]).to(device)
            p = torch.from_numpy(pos[a:a + chunk]).to(device)
            z = sae.encode(norm(torch.from_numpy(np.asarray(tok[a:a + chunk])).to(device)), mode='threshold').double()
            gp = g * P + p
            S.view(-1, L).index_add_(0, gp, z)
            occ.view(-1).index_add_(0, gp, torch.ones_like(gp, dtype=torch.float64))
            nz = near_t[torch.from_numpy(v).to(device), p]
            z_near += z[nz].sum(0)
            n_near += int(nz.sum())
            for gi in range(len(GROUPS)):
                frames[gi].update(np.unique(r[vid_group[v] == gi]).tolist())
            n_tok += len(r)
        print(f'  {sae_dir.name}: shard {s + 1}/{len(store.dirs)}, {n_tok:,} tokens, {time.time() - t0:.0f}s', flush=True)
    return {'S': S.cpu().numpy(), 'occ': occ.cpu().numpy(), 'z_near': z_near.cpu().numpy(), 'n_near': n_near,
            'n_tok': n_tok, 'frames': np.array([len(f) for f in frames]), 'align': align}


def metrics(acc):
    S, frames = acc['S'], acc['frames']
    M = S.sum(0)  # (P, L) summed activation per position
    z_all = M.sum(0)
    base = acc['n_near'] / acc['n_tok']
    near_share = np.where(z_all > 0, acc['z_near'] / np.maximum(z_all, 1e-12), np.nan)
    q = M.reshape(GRID, GRID, -1)
    h = GRID // 2
    quad = np.stack([q[:h, :h].sum((0, 1)), q[:h, h:].sum((0, 1)), q[h:, :h].sum((0, 1)), q[h:, h:].sum((0, 1))])
    quad = quad / np.maximum(quad.sum(0), 1e-12)
    mt, mb = S[0] / frames[0], S[1] / frames[1]

    def corr_cols(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-30)
    occ_corr = float(corr_cols(acc['occ'][0][:, None] / frames[0], acc['occ'][1][:, None] / frames[1])[0])
    return {'near_share': near_share, 'near_enrichment': near_share / base, 'quad': quad,
            'map_corr_TR_BL': corr_cols(mt, mb), 'base_near_share': base, 'occupancy_corr_TR_BL': occ_corr,
            'weight': z_all / z_all.sum()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sae', action='append', required=True)
    p.add_argument('--stride', type=int, default=8, help='use every stride-th 8,192-token block of each shard')
    p.add_argument('--out-dir', default=str(REPO / 'results/vision/eci_align/mice'))
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    dom = get_domain('mice')
    corner_csv = dom.eci_dir / 'odor_corner.csv'
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fj = out / 'arena_compare.json'
    res = json.loads(fj.read_text()) if fj.exists() else {}
    for sae in args.sae:
        f = out / f'arena_{sae}.npz'
        if f.exists() and not args.overwrite:
            z = np.load(f)
            acc = {k: z[k] for k in z.files}
            acc['n_near'], acc['n_tok'], acc['align'] = int(acc['n_near']), int(acc['n_tok']), str(acc['align'])
        else:
            acc = accumulate(dom.eci_dir / 'sae' / sae, dom, corner_csv, args.stride)
            np.savez(f, **acc)
        mt = metrics(acc)
        w = mt['weight']
        r = {'align': acc['align'], 'n_tokens': int(acc['n_tok']), 'frames_per_group': dict(zip(GROUPS, acc['frames'].tolist())),
             'base_near_share': float(mt['base_near_share']), 'occupancy_corr_TR_BL': mt['occupancy_corr_TR_BL'],
             'all_latents': {'map_corr_TR_BL_weighted': float(np.nansum(w * mt['map_corr_TR_BL'])),
                             'map_corr_TR_BL_median': float(np.nanmedian(mt['map_corr_TR_BL'])),
                             'n_near_enrichment_gt2': int((mt['near_enrichment'] > 2).sum()),
                             'n_near_enrichment_gt1.5': int((mt['near_enrichment'] > 1.5).sum())}}
        sel = selections(dom.nes_root / sae)
        for s in sel:
            j = s['neuron']
            s.update({'near_share': float(mt['near_share'][j]), 'near_enrichment': float(mt['near_enrichment'][j]),
                      'quadrant_share': dict(zip(('TL', 'TR', 'BL', 'BR'), np.round(mt['quad'][:, j], 3).tolist())),
                      'map_corr_TR_BL': float(mt['map_corr_TR_BL'][j])})
        r['selected'] = sel
        res[sae] = r
        print(json.dumps({k: v for k, v in r.items() if k != 'selected'}, indent=1), flush=True)
        fj.write_text(json.dumps(res, indent=1))
    print(f'-> {fj}')


if __name__ == '__main__':
    main()
