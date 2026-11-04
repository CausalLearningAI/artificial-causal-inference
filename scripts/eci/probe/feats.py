"""
ECI linear-probe ceiling: per-frame pooled features of the raw representations and of the SAE codes, on the labelled
frames (results/vision/eci_probe/LOG.md has the design and every frame count).

Representations (frame features = pooling over the frame's masked tokens: mean, and max; mean||max is built later):
  d2    raw DINOv2-base last-layer tokens, v1's fg448al mask (768)       mice: stored 1 fps token shards (stride 5, per-video offset)
                                                                          ants: re-encoded (GPU) on the eval 1 fps grid
  v1c   v1 SAE s0 codes (1024; threshold mode) of those same tokens
  d3    raw DINOv3-L/16 block-20 tokens x_t on mask v5 M_t (1024)         v2.1_m5 store, 5 fps
  V     the V input x_t + (x_{t+1} - x_{t-1}) / 2 (1024)                  (v22_full.feats_v, same edge rule)
  V_s<s> v2 SAE arm V codes of the V input (2048), seeds 1001-1003

Steps
  frames                         (CPU) -> BEE/<dom>/frames.parquet (mice: 864,000 labelled frames; ants: 127,200)
  mice3 --task i --ntask n       (CPU) d3 / V / V_s* of mice videos chunk i -> BEE/mice/parts/t<i>_<name>.npy
  mice1 --task i                 (CPU) d2 / v1c of the mice frames held in v1 token shard i (0-15)
  ants3 --task i --ntask n       (CPU) d3 / V / V_s* of ants videos chunk i (eval 1 fps frames only)
  ants1                          (GPU) d2 / v1c of the ants eval frames (eval.py v1 token pipeline)
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for p in ('', 'scripts/eci/v2', 'scripts/eci/v2_eval', 'scripts/eci/v2_prob'):
    sys.path.insert(0, str(REPO / p))
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/probe')
OUT = REPO / 'results/vision/eci_probe'
V1TOK = Path('/mnt/beegfs/locatgrp/shared/rcadei/nfs_offload/dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1/shards')
V22F = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/full/frames.parquet')
ANTS_F = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_eval/ants/frames.parquet')
V_SEEDS = (1001, 1002, 1003)


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def local_dir():
    d = Path(os.environ.get('LOCAL_DIR', f'/localhome/{os.environ.get("USER")}/{os.environ.get("SLURM_JOB_ID", "x")}_probe'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_parts(arrs, rows, od, tag):
    """write {name: array} + rows parquet to local disk, then copy each to od once."""
    od.mkdir(parents=True, exist_ok=True)
    loc = local_dir()
    for n_, x in arrs.items():
        np.save(loc / 'tmp.npy', x)
        shutil.copyfile(loc / 'tmp.npy', od / f'{tag}_{n_}.npy')
    (loc / 'tmp.npy').unlink(missing_ok=True)
    rows.to_parquet(od / f'{tag}_rows.parquet')


# ---------------------------------------------------------------------------------------------- frames
def cmd_frames(a):
    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'])
    ann['row'] = np.arange(len(ann))
    W = pd.read_parquet(V22F, columns=['obs', 'k', 'row'])
    order = pd.unique(W.obs.values)
    A = ann[ann.observation_id.isin(order) & ann.Y_nn.notna()].rename(columns={'observation_id': 'obs', 'frame_idx': 'k'})
    A['vo'] = A.obs.map({o: i for i, o in enumerate(order)})
    A = A.sort_values(['vo', 'k']).reset_index(drop=True).drop(columns='vo')
    isH = A.obs.str.endswith('_H').values
    A['part'] = np.where(isH & (A.k.values < 4500), 'H_first15', 'atlas')
    assert A[A.part == 'atlas'].row.isin(W.row).all() and (A.part == 'atlas').sum() == len(W) == 648000
    for c in ('Y_nn', 'Y_np', 'Y_nt'):
        assert A[c].notna().all(), c
    import v22_core as C
    A['unit'] = A.obs.str.rsplit('_', n=2).str[0]
    A['y_nose_nose'] = (A.Y_nn > 0) | (A.Y_np > 0)
    A['y_nn'], A['y_np'], A['y_nt'] = A.Y_nn > 0, A.Y_np > 0, A.Y_nt > 0
    F = C.add_annot(A)
    # the v1 DINOv2 token store grid: stride 5 with a per-video offset -> the frames the store actually holds
    srows = np.concatenate([np.load(V1TOK / f'shard_{i:02d}' / 'frames.npz')['rows'] for i in range(16)])
    F['grid1'] = np.isin(F.row.values, srows)
    assert F.grid1.sum() * 5 == len(F) and (F.groupby('obs').grid1.sum() * 5 == F.groupby('obs').size()).all()
    g = F.groupby('obs', sort=False).k.agg(['min', 'max', 'size'])
    assert (g['max'] - g['min'] + 1 == g['size']).all()
    (BEE / 'mice').mkdir(parents=True, exist_ok=True)
    keep = ['obs', 'k', 'row', 'unit', 'annot', 'sg', 'nonsg', 'part', 'grid1', 'y_nose_nose', 'y_nn', 'y_np', 'y_nt']
    F[keep].to_parquet(BEE / 'mice' / 'frames.parquet')
    # ants: the eval.py frame table + focal labels (single = Y2F xor B2F, double = Y2F and B2F)
    G = pd.read_parquet(ANTS_F)
    y, b = G.Y_Y2F.values > 0, G.Y_B2F.values > 0
    G['y_groom_any'], G['y_Y2F'], G['y_B2F'], G['y_single'], G['y_double'] = y | b, y, b, y ^ b, y & b
    G[['obs', 'k', 'experiment', 'series', 'src', 'row', 'frame_path', 'y_groom_any', 'y_Y2F', 'y_B2F', 'y_single',
       'y_double']].to_parquet((BEE / 'ants').mkdir(parents=True, exist_ok=True) or BEE / 'ants' / 'frames.parquet')
    res = {}
    for nm, D in (('mice_all', F), ('mice_grid1', F[F.grid1])):
        for part in ('all', 'atlas', 'H_first15'):
            E = D if part == 'all' else D[D.part == part]
            for grp, m in (('all', np.ones(len(E), bool)), ('sg', E.sg.values), ('nonsg', E.nonsg.values),
                           ('unknown', E.annot.values == 'NA')):
                e = E[m]
                res[f'{nm}|{part}|{grp}'] = {'frames': len(e), 'videos': int(e.obs.nunique()), 'pools': int(e.unit.nunique()),
                                             **{b: int(e[f'y_{b}'].sum()) for b in ('nose_nose', 'nn', 'np', 'nt')}}
    for ex, e in [('all', G)] + list(G.groupby('experiment')):
        res[f'ants|{ex}'] = {'frames': len(e), 'videos': int(e.obs.nunique()),
                             **{b: int(e[f'y_{b}'].sum()) for b in ('groom_any', 'Y2F', 'B2F', 'single', 'double')}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'frames.json').write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


# ---------------------------------------------------------------------------------------------- v2 store features
def pool(f, lf, n, dev, init):
    """per-frame mean and max of rows f (frame index lf in 0..n-1) -> (mean, max, count)."""
    import torch
    cnt = torch.bincount(lf, minlength=n).float()
    s = torch.zeros((n, f.shape[1]), device=dev).index_add_(0, lf, f)
    m = torch.full((n, f.shape[1]), init, device=dev).index_reduce_(0, lf, f, 'amax', include_self=True)
    m[cnt == 0] = 0
    return s / cnt.clamp(min=1)[:, None], m, cnt


def v2_video(dom, ser, obs, keep_k, saes, dev, CH=400):
    """features of the frames keep_k (sorted) of one video -> dict name -> (len(keep_k), d) float16, ntok."""
    import torch
    import v21_train as T
    import v22_full as VF
    V_ = T.Video(dom, ser, obs)
    kpos = pd.Series(np.arange(len(V_.k)), index=V_.k)
    pos = kpos.reindex(keep_k).values
    assert not np.isnan(pos).any(), (obs, 'frames missing in store')
    pos = pos.astype(np.int64)
    out = {nm: np.zeros((len(pos), d), np.float16) for nm, d in
           [('d3_mean', 1024), ('d3_max', 1024), ('V_mean', 1024), ('V_max', 1024)] +
           [(f'{k}_{r}', 2048) for k in saes for r in ('mean', 'max')]}
    ntok = np.zeros(len(pos), np.int32)
    lo = 0
    while lo < len(pos):
        hi = lo
        while hi < len(pos) and pos[hi] - pos[lo] < CH:
            hi += 1
        b0, b1 = int(pos[lo]), int(pos[hi - 1]) + 1
        W = V_.window(b0, b1)
        f, fi = VF.feats_v(W, dev)  # V input of the M_t tokens of frames b0..b1-1
        c = np.flatnonzero(((W['flag'] & 1) > 0) & (W['fi'] >= b0) & (W['fi'] < b1))
        assert (W['fi'][c] == fi).all()
        x = torch.from_numpy(W['tok'][c]).to(dev).float()
        lf = torch.from_numpy(fi - b0).to(dev)
        n = b1 - b0
        sel = pos[lo:hi] - b0
        res = {}
        res['d3_mean'], res['d3_max'], cnt = pool(x, lf, n, dev, -np.inf)
        res['V_mean'], res['V_max'], _ = pool(f, lf, n, dev, -np.inf)
        for k, (sae, norm) in saes.items():
            with torch.no_grad():
                cc = sae.encode(norm(f), mode='threshold').float()
            res[f'{k}_mean'], res[f'{k}_max'], _ = pool(cc, lf, n, dev, 0.0)
            del cc
        for nm, v in res.items():
            out[nm][lo:hi] = v[sel].half().cpu().numpy()
        ntok[lo:hi] = cnt[sel].cpu().numpy().astype(np.int32)
        lo = hi
    return out, ntok


def run_v2(dom, F, vids, tag, a):
    import torch
    from src.eci.v2 import sae as SA
    import v22_encode as EN
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    saes = {f'V_s{s}': SA.load(EN.sae_path(dom, f'V_s{s}'), dev) for s in V_SEEDS}
    store = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tokens/v2_full') / dom / 'v2.1_m5' / 'videos'
    parts, rows, t0, nt = [], [], time.time(), 0
    for vi, obs in enumerate(vids):
        g = F[F.obs == obs]
        ser = next(s for s in (['v1'] if dom == 'mice' else ['v2', 'v3', 'v4']) if (store / s / obs).exists())
        o, ntok = v2_video(dom, ser, obs, g.k.values, saes, dev)
        parts.append(o)
        rows.append(pd.DataFrame({'obs': obs, 'k': g.k.values, 'ntok_v5': ntok}))
        nt += int(ntok.sum())
        log(f'{tag}: video {vi + 1}/{len(vids)} {obs}: {len(g)} frames, {nt:,} tokens, {time.time() - t0:.0f}s')
    arrs = {nm: np.concatenate([p[nm] for p in parts]) for nm in parts[0]}
    save_parts(arrs, pd.concat(rows, ignore_index=True), BEE / dom / 'parts', tag)
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs' / f'{dom}_{tag}.json').write_text(json.dumps({'videos': len(vids), 'frames': int(sum(len(r) for r in rows)),
                                                                'tokens': nt, 'seconds': time.time() - t0}))


def chunks(vids, i, n):
    return list(np.array_split(np.array(vids, dtype=object), n)[i])


def cmd_mice3(a):
    F = pd.read_parquet(BEE / 'mice' / 'frames.parquet', columns=['obs', 'k'])
    run_v2('mice', F, chunks(list(pd.unique(F.obs)), a.task, a.ntask), f't{a.task}', a)


def cmd_ants3(a):
    F = pd.read_parquet(BEE / 'ants' / 'frames.parquet', columns=['obs', 'k'])
    run_v2('ants', F, chunks(list(pd.unique(F.obs)), a.task, a.ntask), f't{a.task}', a)


# ---------------------------------------------------------------------------------------------- v1 (DINOv2) mice
def v1_feats(z, lf, n, saes, dev):
    import torch
    res = {}
    res['d2_mean'], res['d2_max'], cnt = pool(z, lf, n, dev, -np.inf)
    for k, (sae, norm) in saes.items():
        with torch.no_grad():
            cc = sae.encode(norm(z), mode='threshold').float()
        res[f'{k}_mean'], res[f'{k}_max'], _ = pool(cc, lf, n, dev, 0.0)
    return res, cnt


def cmd_mice1(a):
    import torch
    from src.eci.sae import load_sae
    import eval as EV
    dev = torch.device('cpu')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    saes = {'v1c': load_sae(REPO / EV.V1SAE['mice'], dev)[:2]}
    F = pd.read_parquet(BEE / 'mice' / 'frames.parquet', columns=['obs', 'k', 'row', 'grid1'])
    want = pd.Series(np.arange(len(F)), index=F.row.values)
    sd = V1TOK / f'shard_{a.task:02d}'
    fz = np.load(sd / 'frames.npz')
    srows, nfg = fz['rows'].astype(np.int64), fz['n_fg'].astype(np.int64)
    off = np.r_[0, np.cumsum(nfg)]
    meta = json.loads((sd / 'shard.json').read_text())
    tok = np.memmap(sd / 'tokens.f16', np.float16, 'r', shape=(int(meta['n_tokens']), 768))
    trow = np.memmap(sd / 'row.i32', np.int32, 'r')
    assert off[-1] == meta['n_tokens'] == len(trow)
    hit = np.flatnonzero(np.isin(srows, F.row.values))
    log(f'shard {a.task}: {len(srows)} frames, {len(hit)} labelled frames of the 144 annotated videos')
    acc = {}
    t0 = time.time()
    # contiguous runs of hit frames -> one read each
    br = np.flatnonzero(np.diff(hit) != 1) + 1
    for run in np.split(hit, br):
        if not len(run):
            continue
        for s0 in range(0, len(run), 2000):
            r = run[s0:s0 + 2000]
            t_a, t_b = int(off[r[0]]), int(off[r[-1] + 1])
            z = torch.from_numpy(np.asarray(tok[t_a:t_b])).float()
            tr = np.asarray(trow[t_a:t_b])
            lf_np = np.repeat(np.arange(len(r)), nfg[r])
            assert (tr == srows[r][lf_np]).all(), 'token rows != frame rows'
            res, cnt = v1_feats(z, torch.from_numpy(lf_np), len(r), saes, dev)
            for nm, v in res.items():
                acc.setdefault(nm, []).append(v.half().numpy())
            acc.setdefault('_rows', []).append(srows[r])
            acc.setdefault('_cnt', []).append(cnt.numpy().astype(np.int32))
    rws = np.concatenate(acc.pop('_rows'))
    cnt = np.concatenate(acc.pop('_cnt'))
    arrs = {nm: np.concatenate(v) for nm, v in acc.items()}
    fi = want.reindex(rws).values.astype(np.int64)
    R = pd.DataFrame({'obs': F.obs.values[fi], 'k': F.k.values[fi], 'row': rws, 'ntok_fg448al': cnt})
    assert F.grid1.values[fi].all()
    save_parts(arrs, R, BEE / 'mice' / 'parts1', f's{a.task}')
    log(f'shard {a.task} done: {len(R)} frames, {time.time() - t0:.0f}s')


# ---------------------------------------------------------------------------------------------- v1 (DINOv2) ants
def cmd_ants1(a):
    """eval.py's v1 token pipeline (DINOv2-base 448, fp32 forward, fp16 tokens, ants fg rule) on the eval frames of
    video chunk task / ntask; runs on CPU (no GPU slot within 2 h) or GPU."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from src.eci.foreground import RULES, FgBackgrounds, FgEncoder, encode_batch, obs_rows
    from src.eci.sae import load_sae
    import eval as EV
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    torch.set_num_threads(ncpu)
    F0 = pd.read_parquet(BEE / 'ants' / 'frames.parquet')
    vids = chunks(list(pd.unique(F0.obs)), a.task, a.ntask)
    F = F0[F0.obs.isin(vids)]
    saes = {'v1c': load_sae(REPO / EV.V1SAE['ants'], dev)[:2]}
    enc = FgEncoder('dinov2_base', dev)
    X, cnt_all = {}, np.zeros(len(F0), np.int32)
    loc = local_dir() / 'frames'
    t0 = time.time()
    for src, g in F.groupby('src'):
        ann = REPO / EV.ANN[src]
        rows = g.row.values.astype(np.int64)
        srcp = [REPO / 'dataset' / p for p in g.frame_path.values]
        dst = [loc / p for p in g.frame_path.values]
        for d in {p.parent for p in dst}:
            d.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda sd: shutil.copyfile(sd[0], sd[1]), zip(srcp, dst)))
        log(f'{src}: staged {len(dst):,} JPGs ({time.time() - t0:.0f}s)')
        fb = FgBackgrounds(REPO / 'dataset/ants/eci30/fg448/background', obs_rows(ann), RULES['ants'], dev, align='none', cache=8)
        ds = enc.dataset([str(p) for p in dst], rows, None)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, prefetch_factor=4)
        gi, o = g.index.values, 0
        for pix, grey, rr in dl:
            with torch.no_grad():
                tok = encode_batch(enc.model, pix, dev)
                mk, _ = fb.mask(tok, grey.to(dev), rr.numpy())
                fi, pi = torch.nonzero(mk, as_tuple=True)
                res, cnt = v1_feats(tok[fi, pi].float(), fi, len(tok), saes, dev)
            for nm, v in res.items():
                X.setdefault(nm, np.zeros((len(F0), v.shape[1]), np.float16))[gi[o:o + len(tok)]] = v.half().cpu().numpy()
            cnt_all[gi[o:o + len(tok)]] = cnt.cpu().numpy()
            assert (rr.numpy() == rows[o:o + len(tok)]).all()
            o += len(tok)
            if (o // len(tok)) % 20 == 0:
                log(f'{src}: {o}/{len(g)} frames, {o / (time.time() - t0):.2f} fps')
        assert o == len(g)
        shutil.rmtree(loc, ignore_errors=True)
        log(f'{src}: encoded {len(g):,} frames ({time.time() - t0:.0f}s)')
    ii = F.index.values
    save_parts({k: v[ii] for k, v in X.items()}, F[['obs', 'k']].assign(ntok_fg=cnt_all[ii], tab=ii),
               BEE / 'ants' / 'parts1', f't{a.task}')
    log(f'ants1 task {a.task}: {len(F)} frames, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntask', type=int, default=1)
    a = ap.parse_args()
    globals()[f'cmd_{a.step}'](a)
