"""
Ants v4 (healthy colony) + v4_infested with the ONLINE atlas ants model (v36 'antsfg'): DINOv2-base at 448 (whole
512 px dataset JPG resized, 32 x 32 patches, fp32 forward rounded to fp16), the 'ants' foreground mask (per-video
background of each 10-min window), Matryoshka BatchTopK SAE matryoshka_btk_1024_k16_antsfg_s0 (no temporal input,
global-threshold inference), per-frame codes_max / codes_mean / n_fg over the mask patches. No retraining.

The 30-min ants layout dataset/ants/eci30 (scripts/eci/ants30_chain.sh, 2026-10-15) already encoded every v4 /
v4_infested frame (t = 2 and 6, 81 videos x 9000 five-fps frames, the dataset JPGs) with exactly this SAE and
encoder (scripts/eci/fg_encode_extend.py; its verify.json passed). This script checks that claim independently, then
builds a v4-only layout from those rows (the row space of the new model's layout dataset/ants/eci_r4v4) so the NES /
DINES runners see only the v4 videos, exactly as the R4 ('new') v4 page does.

Frame convention: the online pipeline encodes the 5 fps dataset JPGs (ants/v4*/frames/full/<obs>/frame_XXXXXX.jpg),
the same JPGs the labels are joined to; the R4 2-source-frame offset (results/vision/eci_rollout_r4/v4_offset.json)
concerns R4's decoding of the source videos and does not apply here.

Steps
    verify  (GPU) re-encode from JPEG, through src/eci/fg_encode.py encode_rows (the standard path), 2 ants v3 videos
            of the online 10-min layout (dataset/ants/eci: 3_1_1 t=2, 3_1_7 t=6; 3000 frames each) and 1 v4 + 1
            v4_infested video of eci30 (9000 frames each); compare with the stored codes: per-frame cosine, active-set
            (firing) identity, n_fg -> results/vision/eci_online_v4/verify.json
    layout  (CPU) v4-only layout on BeeGFS (LAYOUT below): annotations.csv / experiment.csv = the eci30 rows of
            v4 + v4_infested (checked equal to dataset/ants/eci_r4v4 on observation_id / frame_idx / frame_path / T
            and the labels), codes/<sae>/ = the eci30 rows copied (checked byte-equal), sae/ and fg448/ links;
            dataset/ants/eci_c1v4 and dataset/ants/eci_c1v4_noday -> LAYOUT (one layout, two NES result roots:
            with and without recording-day conditioning) -> results/vision/eci_online_v4/layout.json
    check   (CPU) after NES: per-video summaries and activation NES of the day-unmatched run vs eci30's own v4 rows /
            results (same codes, same videos: must be identical) -> results/vision/eci_online_v4/check.json
    manifest (login safe) results/vision/eci_online_v4/manifest.json

Usage: scripts/eci/online_v4.sh (Slurm), STEP=verify | layout | check; manifest: python scripts/eci/online_v4.py manifest
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SAE = 'matryoshka_btk_1024_k16_antsfg_s0'
OLD = REPO / 'dataset/ants/eci'      # online 10-min layout (v2, v3): the published v36 codes
E30 = REPO / 'dataset/ants/eci30'    # 30-min layout, holds v4 / v4_infested
R4V4 = REPO / 'dataset/ants/eci_r4v4'  # the new model's v4 layout (row space reference, read only)
LAYOUT = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/online_v4/layout')
LINKS = ('eci_c1v4', 'eci_c1v4_noday')
OUT = REPO / 'results/vision/eci_online_v4'
V3_VIDEOS = ('3_1_1', '3_1_7')
OUTPUTS = ('codes_max', 'codes_mean', 'n_fg')
KEYS = ['observation_id', 'frame_idx', 'frame_path', 'experiment', 'T']
LABELS = ['Y_B2F', 'Y_BOL', 'Y_FOL', 'Y_Y2F', 'Y_YOL']
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return os.environ.get('ONLINE_V4_HEAD', '')


def stage(rels, ds_loc, workers=16):
    t0 = time.time()
    for p in sorted({str(Path(r).parent) for r in rels}):
        (ds_loc / p).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: shutil.copyfile(REPO / 'dataset' / r, ds_loc / r), rels, chunksize=64))
    log(f'  staged {len(rels):,} JPGs in {time.time() - t0:.0f}s')


def compare(new, ref):
    """new / ref dicts of codes_max, codes_mean (N, m) fp16 and n_fg (N,)."""
    out = {}
    for n in ('codes_max', 'codes_mean'):
        a, b = np.asarray(new[n], np.float32), np.asarray(ref[n], np.float32)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        both0 = (na == 0) & (nb == 0)
        cos = np.where(both0, 1.0, (a * b).sum(1) / np.maximum(na * nb, 1e-12))
        act_a, act_b = a > 0, b > 0
        same = (act_a == act_b).all(1)
        out[n] = {'n_frames': int(len(a)), 'cos_min': float(cos.min()), 'cos_p01': float(np.percentile(cos, 1)),
                  'cos_median': float(np.median(cos)), 'frac_frames_cos_ge_0.9999': float((cos >= 0.9999).mean()),
                  'max_abs_diff': float(np.abs(a - b).max()), 'byte_equal_frac': float((a == b).all(1).mean()),
                  'frames_identical_active_set_frac': float(same.mean()),
                  'active_entries_mismatch_frac': float((act_a != act_b).mean()),
                  'active_per_frame_mean_new_ref': [float(act_a.sum(1).mean()), float(act_b.sum(1).mean())],
                  'mismatch_entries': int((act_a != act_b).sum()),
                  'mismatch_entries_max_abs_value': float(np.abs(np.where(act_a != act_b, a + b, 0)).max())}
    fa, fb = np.asarray(new['n_fg']).astype(int), np.asarray(ref['n_fg']).astype(int)
    out['n_fg'] = {'exact_frac': float((fa == fb).mean()), 'max_abs_diff': int(np.abs(fa - fb).max()),
                   'mean_new_ref': [float(fa.mean()), float(fb.mean())]}
    return out


def step_verify(args):
    import torch
    from src.eci.fg_encode import encode_rows
    from src.eci.foreground import obs_rows
    loc = Path(os.environ['LOCAL_DIR'])
    sae_pt = OLD / 'sae' / SAE / 'sae.pt'
    e30 = pd.read_csv(E30 / 'experiment.csv')
    v4_videos = (e30[e30.experiment == 'v4'].observation_id.iloc[0], e30[e30.experiment == 'v4_infested'].observation_id.iloc[0])
    jobs = [(OLD, v) for v in V3_VIDEOS] + [(E30, v) for v in v4_videos]
    res = {'sae': SAE, 'sae_checkpoint': str(sae_pt), 'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename,
           'head': git_head(), 'videos': {}}
    t_all = time.time()
    for lay, obs in jobs:
        ann = lay / 'annotations.csv'
        lo, hi = obs_rows(ann)[obs]
        fp = pd.read_csv(ann, usecols=['frame_path'])['frame_path'].values
        rows = np.arange(lo, hi)
        ds = loc / 'dataset'
        stage([fp[r] for r in rows], ds)
        t0 = time.time()
        [new] = encode_rows(rows, fp, [str(sae_pt)], str(lay / 'fg448' / 'background'), str(ann), str(ds),
                            batch_size=128, num_workers=min(12, len(os.sched_getaffinity(0))))
        dt = time.time() - t0
        cd = lay / 'codes' / SAE
        ref = {n: np.asarray(np.load(cd / f'{n}.npy', mmap_mode='r')[lo:hi]) for n in OUTPUTS}
        r = compare(new, ref)
        r.update(layout=str(lay.relative_to(REPO)), rows=[lo, hi], frames=int(hi - lo), encode_s=round(dt, 1),
                 fps=round((hi - lo) / dt, 1), first_frame=fp[lo], last_frame=fp[hi - 1])
        res['videos'][obs] = r
        log(f'{obs} ({lay.name}): {hi - lo} frames, {dt:.0f}s; max cos_min {r["codes_max"]["cos_min"]:.6f}, identical '
            f'active set {r["codes_max"]["frames_identical_active_set_frac"]:.4f}, n_fg exact {r["n_fg"]["exact_frac"]:.4f}')
        shutil.rmtree(ds / 'ants')
    res['frames_encoded'] = int(sum(v['frames'] for v in res['videos'].values()))
    res['elapsed_s'] = round(time.time() - t_all, 1)
    # pass rule = the stored layouts' own verify bounds (scripts/eci/fg_encode_extend.py): cos_min > 0.99, n_fg diff <= 5
    res['passed'] = bool(all(v['codes_max']['cos_min'] > 0.99 and v['codes_mean']['cos_min'] > 0.99
                             and v['n_fg']['max_abs_diff'] <= 5 for v in res['videos'].values()))
    OUT.mkdir(parents=True, exist_ok=True)
    vf = OUT / f'verify{os.environ.get("VERIFY_TAG", "")}.json'
    vf.write_text(json.dumps(res, indent=1))
    log(f'verify passed={res["passed"]} -> {vf}')
    if not res['passed']:
        raise SystemExit('verification FAILED')


def sha(a, lo, hi, chunk=50000):
    h = hashlib.sha256()
    for i in range(lo, hi, chunk):
        h.update(np.ascontiguousarray(a[i:min(i + chunk, hi)]).tobytes())
    return h.hexdigest()


def step_layout(args):
    from src.eci.foreground import obs_rows
    ann30 = pd.read_csv(E30 / 'annotations.csv')
    keep = ann30['experiment'].isin(['v4', 'v4_infested']).values
    idx = np.nonzero(keep)[0]
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    if not (len(idx) == hi - lo):
        raise SystemExit('v4 rows of eci30 are not contiguous')
    ann = ann30.iloc[lo:hi].reset_index(drop=True)
    r4 = pd.read_csv(R4V4 / 'annotations.csv')
    if len(r4) != len(ann):
        raise SystemExit(f'row count {len(ann)} != eci_r4v4 {len(r4)}')
    for c in KEYS:
        if not (r4[c].astype(str).values == ann[c].astype(str).values).all():
            raise SystemExit(f'column {c} differs from eci_r4v4')
    for c in LABELS:
        a, b = ann[c].values.astype(float), r4[c].values.astype(float)
        if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
            raise SystemExit(f'label {c} differs from eci_r4v4')
    e30 = pd.read_csv(E30 / 'experiment.csv')
    ex = e30[e30.experiment.isin(['v4', 'v4_infested'])].reset_index(drop=True)
    exr4 = pd.read_csv(R4V4 / 'experiment.csv')
    if sorted(ex.observation_id) != sorted(exr4.observation_id):
        raise SystemExit('experiment.csv videos differ from eci_r4v4')
    tmp = Path(str(LAYOUT) + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    cd = tmp / 'codes' / SAE
    cd.mkdir(parents=True)
    ann.to_csv(tmp / 'annotations.csv', index=False)
    ex.to_csv(tmp / 'experiment.csv', index=False)
    (tmp / 'sae').mkdir()
    os.symlink(OLD / 'sae' / SAE, tmp / 'sae' / SAE)
    os.symlink(E30 / 'fg448', tmp / 'fg448')
    src = E30 / 'codes' / SAE
    loc = Path(os.environ['LOCAL_DIR'])
    hashes = {}
    for n in OUTPUTS:
        a = np.load(src / f'{n}.npy', mmap_mode='r')
        t0 = time.time()
        part = np.ascontiguousarray(a[lo:hi])  # one sequential read of the slice
        np.save(loc / f'{n}.npy', part)
        shutil.copyfile(loc / f'{n}.npy', cd / f'{n}.npy')
        (loc / f'{n}.npy').unlink()
        b = np.load(cd / f'{n}.npy', mmap_mode='r')
        hashes[n] = {'src_slice': sha(a, lo, hi), 'copy': sha(b, 0, hi - lo), 'shape': list(b.shape), 'dtype': str(b.dtype)}
        if hashes[n]['src_slice'] != hashes[n]['copy']:
            raise SystemExit(f'{n}: copy differs from the eci30 slice')
        log(f'  {n} {b.shape} copied and byte-checked in {time.time() - t0:.0f}s')
    cfg = json.loads((src / 'config.json').read_text())
    cfg.update(row_order='dataset/ants/eci_c1v4/annotations.csv (= eci_c1v4_noday, same dir)', n_rows=int(hi - lo),
               source={'codes': str(src), 'rows': [lo, hi], 'rule': 'the v4 + v4_infested rows of the 30-min layout '
                       'eci30, copied byte for byte (sha256 in results/vision/eci_online_v4/layout.json)'})
    cfg.pop('reuse', None)
    cfg.pop('shard_ranges', None)
    (cd / 'config.json').write_text(json.dumps(cfg, indent=1))
    shutil.copyfile(src / 'verify.json', cd / 'verify_eci30.json')
    (cd / 'DONE').touch()
    if LAYOUT.exists():
        raise SystemExit(f'{LAYOUT} exists: remove it first')
    tmp.rename(LAYOUT)
    for name in LINKS:
        p = REPO / 'dataset/ants' / name
        if p.is_symlink() or p.exists():
            if p.resolve() != LAYOUT.resolve():
                raise SystemExit(f'{p} exists and points elsewhere')
        else:
            os.symlink(LAYOUT, p)
    ranges = obs_rows(LAYOUT / 'annotations.csv')
    n_lab = ann.groupby('observation_id')['Y_Y2F'].apply(lambda s: int(s.notna().sum()))
    info = {'layout': str(LAYOUT), 'links': [f'dataset/ants/{n}' for n in LINKS], 'sae': SAE, 'eci30_rows': [lo, hi],
            'n_rows': int(hi - lo), 'n_videos': len(ranges),
            'videos_per_arm': {f'{k[0]} t={k[1]}': int(v) for k, v in ex.groupby(['experiment', 'T']).size().items()},
            'videos_by_day': {f'{k[0]} t={k[1]} day {k[2]}': int(v)
                              for k, v in ex.groupby(['experiment', 'T', 'recording_date']).size().items()},
            'frames_per_video': sorted({b - a for a, b in ranges.values()}),
            'labelled_videos': int((n_lab > 0).sum()), 'labelled_frames': int(n_lab.sum()),
            'same_rows_as_eci_r4v4': True, 'hashes': hashes, 'head': git_head()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'layout.json').write_text(json.dumps(info, indent=1))
    log(f'layout {LAYOUT}: {hi - lo} rows, {len(ranges)} videos -> {OUT / "layout.json"}')


def nes_root(lay):
    return REPO / f'results/vision/ants/{lay}/nes/{SAE}'


def step_check(args):
    """Day-unmatched v4-only run vs eci30's own NES (same codes, same videos): per-video means and the activation
    searches of the v4 analyses must agree exactly (bout / latency thresholds are pooled over the layout's frames,
    so those differ by design and are not compared)."""
    res = {}
    a = np.load(nes_root('eci_c1v4_noday') / '_cache/video_summaries_max.npz', allow_pickle=True)
    b = np.load(REPO / f'results/vision/ants/eci30/nes/{SAE}/_cache/video_summaries_max.npz', allow_pickle=True)
    ib = {o: i for i, o in enumerate(b['observation_id'])}
    sel = np.array([ib[o] for o in a['observation_id']])
    res['videos_compared'] = int(len(sel))
    res['summary_max_abs_diff'] = {k: float(np.abs(a[k] - b[k][sel]).max())
                                   for k in a.files if k.endswith(('__mean', '__rate')) and k in b.files}
    s_new = pd.read_csv(nes_root('eci_c1v4_noday') / 'summary.csv')
    s_old = pd.read_csv(REPO / f'results/vision/ants/eci30/nes/{SAE}/summary.csv')
    ids = sorted(set(s_new.analysis_id) & set(s_old.analysis_id))
    cols = ['analysis_id', 'prefix', 'pooling', 'outcome_type', 'test', 'correction', 'window', 'round', 'neuron', 'n_units']
    m = s_new[s_new.analysis_id.isin(ids)][cols + ['tau', 'p']].merge(
        s_old[s_old.analysis_id.isin(ids)][cols + ['tau', 'p']], on=cols, how='outer', suffixes=('', '_eci30'),
        indicator=True)
    res['activation_rows_compared'] = int(len(m))
    res['activation_rows_only_one_side'] = int((m['_merge'] != 'both').sum())
    both = m[m['_merge'] == 'both']
    res['activation_max_abs_tau_diff'] = float(np.nanmax(np.abs(both.tau - both.tau_eci30))) if len(both) else None
    res['analyses'] = ids
    res['passed'] = bool(res['activation_rows_only_one_side'] == 0 and (res['activation_max_abs_tau_diff'] or 0) < 1e-6
                         and max(res['summary_max_abs_diff'].values()) < 1e-9)
    (OUT / 'check.json').write_text(json.dumps(res, indent=1, default=str))
    log(json.dumps(res, default=str))


def step_manifest(args):
    def ls(p):
        return sorted(str(x.relative_to(REPO)) for x in p.iterdir()) if p.exists() else []
    code = {}
    for lay in LINKS:
        for f in (OUT / 'nes_code').glob(f'{lay}*.json'):
            code[f.stem] = json.loads(f.read_text())
    man = {'what': 'ants v4 (healthy) + v4_infested, t=2 and 6, 30 min, with the ONLINE atlas ants model (v36 antsfg)',
           'model': {'sae': SAE, 'sae_checkpoint': str(OLD / 'sae' / SAE / 'sae.pt'), 'encoder': 'facebook/dinov2-base, '
                     '448 px whole frame, last_hidden_state fp16', 'mask': "'ants' foreground rule", 'temporal': 'none',
                     'firing': 'SAE global threshold (per-frame codes_max over the mask patches); bouts: pooled q 0.95'},
           'verify': 'results/vision/eci_online_v4/verify.json', 'layout': 'results/vision/eci_online_v4/layout.json',
           'check': 'results/vision/eci_online_v4/check.json',
           'codes': {'dir': f'{LAYOUT}/codes/{SAE}', 'links': [f'dataset/ants/{n}/codes/{SAE}' for n in LINKS],
                     'source': f'dataset/ants/eci30/codes/{SAE} (v4 rows, byte copy)'},
           'nes': {lay: {'root': str(nes_root(lay).relative_to(REPO)), 'day': lay == 'eci_c1v4',
                         'label': 'recording-day conditioned' if lay == 'eci_c1v4' else 'day-unmatched',
                         'summaries': ls(nes_root(lay) / '_cache')} for lay in LINKS},
           'dines': {lay: f'results/vision/ants/{lay}/methods/ants_{lay}/{SAE}' for lay in LINKS},
           'nes_code': code, 'head': git_head(), 'written': time.strftime('%Y-%m-%dT%H:%M:%S')}
    for f in ('verify', 'layout', 'check'):
        p = OUT / f'{f}.json'
        if p.exists():
            man[f + '_result'] = {k: v for k, v in json.loads(p.read_text()).items() if k in ('passed', 'frames_encoded', 'n_rows', 'n_videos', 'activation_max_abs_tau_diff')}
    for lay in LINKS:
        c = REPO / man['dines'][lay] / 'code.json'
        if c.exists():
            man['dines_code_' + lay] = json.loads(c.read_text()).get('head')
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


def step_hits(args):
    """Primary hits (pooling max, t test, Bonferroni, whole window, bout / latency threshold q 0.95, merge gap 0, bout rule 0, latency untransformed) per layout variant x method x outcome x analysis x
    prefix: number of reported neurons and their ids in selection order -> hits.csv / hits.json."""
    rows = []
    for lay in LINKS:
        base = nes_root(lay)
        dines = REPO / f'results/vision/ants/{lay}/methods/ants_{lay}/{SAE}/dines'
        src = {('NES', 'activation'): base / 'pairs/summary.csv', ('NES', 'bout_rate'): base / 'pairs/maxpool_bouts/summary.csv',
               ('NES', 'latency'): base / 'pairs/maxpool_latency/summary.csv', ('DINES', 'activation'): dines / 'summary.csv',
               ('DINES', 'bout_rate'): dines / 'maxpool_bouts/summary.csv',
               ('DINES', 'latency'): dines / 'maxpool_latency/summary.csv'}
        for (meth, outc), f in src.items():
            if not f.exists():
                rows.append({'layout': lay, 'method': meth, 'outcome': outc, 'missing': str(f)})
                continue
            t = pd.read_csv(f)
            win = 'common' if outc == 'latency' else 'full'
            otype = {'activation': 'mean', 'bout_rate': 'bout_rate', 'latency': 'latency'}[outc]
            t = t[(t.pooling == 'max') & (t.test == 't') & (t.correction == 'bonferroni') & (t.window == win)
                  & (t.outcome_type == otype)]
            for c, v in (('threshold_q', 0.95), ('merge_gap', 0), ('bout_rule', 0), ('transform', 'none')):
                if c in t.columns:  # the primary bout / latency setting (build_explorer DEFAULT_OUTCOMES)
                    t = t[t[c].astype(str) == v] if isinstance(v, str) else t[np.isclose(t[c].astype(float), v)]
            for (aid, pre), g in t.groupby(['analysis_id', 'prefix'], sort=False):
                g = g[g['round'] >= 1].sort_values('round')
                r = {'layout': lay, 'day': 'conditioned' if lay == 'eci_c1v4' else 'day-unmatched', 'method': meth,
                     'outcome': outc, 'analysis_id': aid, 'prefix': int(pre), 'n_units': int(t[t.analysis_id == aid].n_units.iloc[0]),
                     'n_hits': len(g), 'neurons': ' '.join(str(int(n)) for n in g['neuron']),
                     'taus': ' '.join(f'{x:+.3g}' for x in g.get('tau', []))}
                if meth == 'DINES' and 'status' in g:
                    r['n_kept'] = int((g.status == 'kept').sum())
                    r['n_undetermined'] = int((g.status == 'undetermined').sum())
                rows.append(r)
    h = pd.DataFrame(rows)
    h.to_csv(OUT / 'hits.csv', index=False)
    print(h.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('verify', 'layout', 'check', 'manifest', 'hits'))
    A = ap.parse_args()
    {'verify': step_verify, 'layout': step_layout, 'check': step_check, 'manifest': step_manifest, 'hits': step_hits}[A.step](A)
