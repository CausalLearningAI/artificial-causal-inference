"""
ECI v2.0 full token store (eci_tokens/v2_full/<dom>/<shard>/: tok.f16 (n, 1024) fp16 = every token of the N2 mask
dilated by one token (the v1-style ring of the CTX variant; contains N2), tok_frame.npy, tok_pos.npy, flag_n2.npy,
flag_tight.npy (per-token bool; the dilated flag is all True), frames.parquet (series, obs, k, role, n_kept = N2 tokens,
n_store)) - identity check and manifest. Motion blocks computed later must mark a neighbour as MISSING (not zero) when the
patch is not stored in that frame.
    check --dom D [--shard TAG]  codes of the saved tokens (PASS B deployed SAE) == the stored PASS B frame codes
    manifest                     -> eci_tokens/v2_full/manifest.json (videos, frames, tokens, shard sizes)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ROOT = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full')
B = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_0B')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def shards(dom):
    return sorted(p for p in (ROOT / dom).iterdir() if p.is_dir() and not p.name.startswith('_') and not p.name.endswith('.tmp'))


def cmd_check(a):
    import torch
    from src.eci.v2 import sae as SA
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = json.loads((REPO / f'results/vision/eci_v2_0B/{a.dom}/best.json').read_text())
    sae, norm = SA.load(Path(best['dir']) / 'sae_theta.pt', dev)
    sp = [p for p in shards(a.dom) if not a.shard or p.name == a.shard][0]
    F = pd.read_parquet(sp / 'frames.parquet')
    tf = np.load(sp / 'tok_frame.npy')
    pos = np.load(sp / 'tok_pos.npy')
    tok = np.memmap(sp / 'tok.f16', np.float16, 'r', shape=(len(tf), 1024))
    st = np.r_[0, np.cumsum(np.bincount(tf, minlength=len(F)))]
    n2 = np.load(sp / 'flag_n2.npy')
    assert (np.diff(st) == F.n_store.values).all()
    assert (np.bincount(tf[n2], minlength=len(F)) == F.n_kept.values).all()  # N2-flagged subset = the N2 token set
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(F), 40, replace=False))
    res = {'shard': sp.name, 'frames': 0, 'max_abs_diff': 0.0, 'tokens': 0, 'exact_frames': 0}
    cache = {}
    for f in pick:
        r = F.iloc[f]
        key = (r.series, r.obs)
        if key not in cache:
            cache.clear()
            cache[key] = np.load(B / a.dom / 'videos' / r.series / f'{r.obs}.npz')
        z = cache[key]
        x = torch.from_numpy(np.asarray(tok[st[f]:st[f + 1]])[n2[st[f]:st[f + 1]]]).to(dev)
        with torch.no_grad():
            c = sae.encode(norm(x.float()), mode='threshold').float()
        cm = (c.max(0).values if len(c) else torch.zeros(sae.n_latents, device=dev)).half().float().cpu().numpy()
        ref = z['codes_max'][int(r.k)].astype(np.float32)
        d = float(np.abs(cm - ref).max())
        nk_ref = int(z['n_kept'][int(r.k)])
        assert nk_ref == len(x), (key, r.k, nk_ref, len(x))
        res['frames'] += 1
        res['tokens'] += int(len(x))
        res['max_abs_diff'] = max(res['max_abs_diff'], d)
        res['exact_frames'] += int(d == 0)
    (ROOT / a.dom / '_meta').mkdir(parents=True, exist_ok=True)
    (ROOT / a.dom / '_meta' / f'identity_{sp.name}.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


def cmd_manifest(a):
    out = {'spec': 'configs/eci_v2/representation.json (v2.0: DINOv3-L/16 block 20 rev ea8dc28, native, canonical, 5 fps, '
                   'N2 thin-part mask; ALL kept tokens fp16)', 'domains': {}}
    for dom in ('mice', 'ants', 'frogs'):
        if not (ROOT / dom).exists():
            continue
        D = {'shards': {}, 'videos': 0, 'frames': 0, 'tokens': 0, 'bytes': 0}
        for sp in shards(dom):
            F = pd.read_parquet(sp / 'frames.parquet')
            nb = sum(f.stat().st_size for f in sp.iterdir())
            D['shards'][sp.name] = {'videos': int(F.groupby(['series', 'obs']).ngroups), 'frames': int(len(F)),
                                    'tokens': int(F.n_store.sum()), 'bytes': int(nb),
                                    'videos_list': sorted(f'{s}/{o}' for s, o in F.groupby(['series', 'obs']).groups)}
            for k in ('videos', 'frames', 'tokens', 'bytes'):
                D[k] += D['shards'][sp.name][k]
        out['domains'][dom] = D
        log(f'{dom}: {D["videos"]} videos, {D["frames"]:,} frames, {D["tokens"]:,} tokens, {D["bytes"] / 1e12:.3f} TB')
    (ROOT / 'manifest.json').write_text(json.dumps(out, indent=1))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('check', 'manifest'))
    p.add_argument('--dom', default='ants')
    p.add_argument('--shard', default='')
    a = p.parse_args()
    {'check': cmd_check, 'manifest': cmd_manifest}[a.step](a)


if __name__ == '__main__':
    main()
