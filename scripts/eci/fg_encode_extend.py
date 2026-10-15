"""
Codes of a new ECI layout that REUSES the codes of an old one: a frame of the new annotations.csv that the old
annotations.csv also has (same observation_id, frame_idx and frame_path) takes its old codes rows byte for byte; only
the other frames are encoded. Built for the ants 30-min layout (ANTS_ECI=eci30, scripts/eci/ants_prepare.py): its v2
rows and frames 0-2999 of its v3 videos are in the 10-min layout (dataset/ants/eci), which the published atlas SAEs
(antsfg, antsfull) already encoded; frames 3000-8999 of v3 and every v4 / v4_infested frame are new.

Why reuse is exact: the frames < 3000 are never rewritten by the 30-min extension (scripts/05_deploy/extend_30min.py),
the SAE and encoder are unchanged, and the ants mask of a frame depends only on its video's background of that
10-min window (src/eci/foreground.py bg_windows; the domain's bg_window = 3000): the first window's background IS the
old 10-min file (linked, not recomputed), the later windows get their own (scripts/eci/fg_background.py, same recipe).
Encoding is DINOv2-base at 448, fp32 forward rounded to fp16 (as the stored codes), then per SAE its own mask rule
(ants / all) and max / mean pooling (src/eci/fg_encode.py fg_sae_pool): one forward pass serves every SAE (static,
unaligned, DINOv2 SAEs only).

Steps (Slurm: scripts/eci/fg_encode_extend.sh; the new layout is the domain's, the old one --old-eci):
    setup    (CPU) row map new -> old (<new eci>/reuse/<old eci name>.npz: old_row per new row, -1 = encode; .json
             summary per experiment); links <new eci>/sae/<sae> -> the old SAE dir; the new background dir
             <new eci>/fg448/background with windows.json and a link to the old background of every window whose
             rows are exactly an old video (all of v2 and the first 10 min of v3)
    encode   (GPU, --shard i of --n-shards) the rows to encode, JPGs staged to $LOCAL_DIR first, every --sae in one
             pass -> <new codes>/<sae>/shards/shard_XX/{codes_max, codes_mean, n_fg, rows}.npy + DONE
    merge    (GPU for the check) per --sae: codes_max / codes_mean / n_fg of all new rows = old rows copied by runs +
             the shards; config.json (the old config, new row order / backgrounds / reuse); DONE; <sae>_mean sibling
             (links, as the 10-min layout); then verify.json:
               reuse_exact   every reused row byte-equal to its old row (all rows, all three arrays)
               recompute     n_check random reused rows and n_check random new rows re-encoded from JPEG through the
                             standard path (src/eci/fg_encode.py encode_rows, one SAE at a time): cos_min, n_fg diff
             pass = reuse exact and, on both row sets, cos_min > 0.99 and n_fg max diff <= 5 (the bounds of
             scripts/eci/fg_encode_all.py --verify; cross-GPU mask jitter is why they are not 0); shards deleted on pass
    plan     (login safe) rows reused / to encode per experiment, GPU-hours at --fps frames/s per GPU, disk per SAE

Usage:
    ANTS_ECI=eci30 python scripts/eci/fg_encode_extend.py setup --sae <antsfg> --sae <antsfull>
    ANTS_ECI=eci30 python scripts/eci/fg_encode_extend.py encode --sae <antsfg> --sae <antsfull> --shard 3 --n-shards 24
    ANTS_ECI=eci30 python scripts/eci/fg_encode_extend.py merge --sae <antsfg> --sae <antsfull> --n-shards 24
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.eci.domain import DOMAINS, get_domain  # noqa: E402
from src.eci.encode import shard_ranges  # noqa: E402
from src.eci.foreground import (BG_WINDOWS_FILE, bg_window_of, bg_windows, obs_rows)  # noqa: E402

OUTPUTS = ('codes_max', 'codes_mean', 'n_fg')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731


def reuse_path(dom, old_eci):
    return dom.eci_dir / 'reuse' / f'{Path(old_eci).name}.npz'


def row_map(new_ann, old_ann):
    """(n_new,) int64 old row of every new row (-1 = not in the old layout); key = (observation_id, frame_idx,
    frame_path), so a frame that moved or was renamed is never reused."""
    cols = ['observation_id', 'frame_idx', 'frame_path']
    a = pd.read_csv(new_ann, usecols=cols, dtype={'observation_id': str})
    b = pd.read_csv(old_ann, usecols=cols, dtype={'observation_id': str})
    b['old_row'] = np.arange(len(b), dtype=np.int64)
    m = a.merge(b, on=cols, how='left', validate='1:1')
    return m['old_row'].fillna(-1).astype(np.int64).values, a


def runs(old_row):
    """Maximal runs of reused rows contiguous in both layouts -> (new_lo, old_lo, length) int64 arrays."""
    r = np.flatnonzero(old_row >= 0)
    if len(r) == 0:
        return np.zeros((0, 3), np.int64)
    brk = np.flatnonzero((np.diff(r) != 1) | (np.diff(old_row[r]) != 1)) + 1
    starts, ends = np.r_[0, brk], np.r_[brk, len(r)]
    return np.stack([r[starts], old_row[r[starts]], ends - starts], 1)


def step_setup(dom, args):
    old_eci = Path(args.old_eci)
    old_row, a = row_map(dom.ann_path, old_eci / 'annotations.csv')
    rp = reuse_path(dom, old_eci)
    rp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(rp, old_row=old_row)
    exp = a['frame_path'].str.split('/').str[1].values
    summ = {'old_eci': str(old_eci), 'new_eci': str(dom.eci_dir), 'n_rows': int(len(old_row)),
            'n_reused': int((old_row >= 0).sum()), 'n_encode': int((old_row < 0).sum()), 'n_runs': int(len(runs(old_row))),
            'per_experiment': {e: {'rows': int((exp == e).sum()), 'reused': int(((exp == e) & (old_row >= 0)).sum())}
                               for e in pd.unique(exp)}}
    rp.with_suffix('.json').write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))
    for s in args.sae:  # the SAE stays where it was trained (neuron ids unchanged)
        dst, src = dom.eci_dir / 'sae' / s, old_eci / 'sae' / s
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)
        log(f'sae {dst} -> {dst.resolve()}')
    # backgrounds: a new window that is exactly an old video's rows (in order) keeps that video's background file
    bg_new, bg_old = dom.eci_dir / 'fg448/background', old_eci / 'fg448/background'
    bg_new.mkdir(parents=True, exist_ok=True)
    if dom.bg_window is not None:
        if bg_window_of(bg_new) not in (None, dom.bg_window):
            raise SystemExit(f'{bg_new}: windows {bg_window_of(bg_new)}, the domain asks {dom.bg_window}')
        (bg_new / BG_WINDOWS_FILE).write_text(json.dumps({'frames': dom.bg_window}))
    old_ranges = obs_rows(old_eci / 'annotations.csv')
    old_win = bg_windows(old_ranges, bg_window_of(bg_old))  # old videos with one background: that file = the video
    n_link = n_new = 0
    for wid, (lo, hi) in bg_windows(obs_rows(dom.ann_path), dom.bg_window).items():
        o = wid.split('__w')[0]
        olo, ohi = old_ranges.get(o, (-1, -1))
        same = (wid == o and old_win.get(o) == (olo, ohi) and ohi - olo == hi - lo and old_row[lo] == olo
                and np.array_equal(old_row[lo:hi], np.arange(olo, ohi)))
        if same and (bg_old / f'{o}.npz').exists():
            if not (bg_new / f'{wid}.npz').exists():
                (bg_new / f'{wid}.npz').symlink_to((bg_old / f'{o}.npz').resolve())
            n_link += 1
        else:
            n_new += 1
    log(f'backgrounds {bg_new}: {n_link} windows linked to {bg_old}, {n_new} to compute (scripts/eci/fg_background.py)')


def load_saes(dom, saes, device):
    """[(name, sae, norm, FgBackgrounds or None)] of static, unaligned DINOv2 SAEs (one forward pass serves all)."""
    from src.eci.foreground import RULES, FgBackgrounds
    from src.eci.sae import load_sae
    ranges = obs_rows(dom.ann_path)
    out = []
    for s in saes:
        sae, norm, ck = load_sae(dom.eci_dir / 'sae' / s / 'sae.pt', device)
        bad = {k: ck.get(k) for k in ('motion_delta', 'bg_sub', 'align', 'encoder')
               if ck.get(k) not in (None, 0, False, 'none', 'dinov2_base')}
        if bad:
            raise SystemExit(f'{s}: only static unaligned DINOv2 SAEs here, checkpoint has {bad}')
        rule = RULES[ck.get('fg_rule', 'fg448')]
        bgs = None if rule.get('all', False) else FgBackgrounds(dom.eci_dir / 'fg448/background', ranges, rule, device)
        out.append((s, sae, norm, bgs))
    return out


def stage(rels, src_root, dst_root, workers=16):
    """Copy dataset-relative paths src_root/r -> dst_root/r."""
    t0 = time.time()
    for p in sorted({str(Path(r).parent) for r in rels}):
        (dst_root / p).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: shutil.copyfile(src_root / r, dst_root / r), rels, chunksize=64))
    log(f'staged {len(rels):,} JPGs in {time.time() - t0:.0f}s')


def step_encode(dom, args):
    import torch
    from src.eci.fg_encode import fg_sae_pool
    from src.eci.foreground import FrameDatasetFG, encode_batch, load_encoder_fg
    old_row = np.load(reuse_path(dom, args.old_eci))['old_row']
    todo = np.flatnonzero(old_row < 0)
    lo, hi = shard_ranges(len(todo), args.n_shards)[args.shard]
    rows = todo[lo:hi]
    dirs = {s: dom.codes_root / s / 'shards' / f'shard_{args.shard:02d}' for s in args.sae}
    if all((d / 'DONE').exists() for d in dirs.values()):
        log(f'[SKIP] shard {args.shard} done')
        return
    log(f'shard {args.shard}/{args.n_shards}: {len(rows):,} rows to encode (of {len(todo):,})')
    fp = pd.read_csv(dom.ann_path, usecols=['frame_path'])['frame_path'].values[rows]
    ds = REPO / 'dataset'
    if os.environ.get('LOCAL_DIR'):
        ds = Path(os.environ['LOCAL_DIR']) / 'dataset'
        stage(list(fp), REPO / 'dataset', ds)
    device = torch.device('cuda')
    saes = load_saes(dom, args.sae, device)
    _, processor, model = load_encoder_fg(device=device)
    loader = torch.utils.data.DataLoader(FrameDatasetFG([str(ds / p) for p in fp], processor, rows),
                                         batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                         pin_memory=True, prefetch_factor=4)
    tmp = {s: Path(str(d) + '.tmp') for s, d in dirs.items()}
    mm = {}
    for (s, sae, _, _) in saes:
        if tmp[s].exists():
            shutil.rmtree(tmp[s])
        tmp[s].mkdir(parents=True)
        m = sae.n_latents
        mm[s] = {n: np.lib.format.open_memmap(tmp[s] / f'{n}.npy', 'w+', np.float16, (len(rows), m))
                 for n in ('codes_max', 'codes_mean')}
        mm[s]['n_fg'] = np.lib.format.open_memmap(tmp[s] / 'n_fg.npy', 'w+', np.int16, (len(rows),))
        np.save(tmp[s] / 'rows.npy', rows)
    t0, cur = time.time(), 0
    for b, (pix, grey, r) in enumerate(loader):
        r = r.numpy()
        assert (r == rows[cur:cur + len(r)]).all()
        tok = encode_batch(model, pix, device)
        grey = grey.to(device, non_blocking=True)
        for (s, sae, norm, bgs) in saes:
            mask = (torch.ones(tok.shape[:2], dtype=torch.bool, device=device) if bgs is None
                    else bgs.mask(tok, grey, r)[0])
            mx, mean = fg_sae_pool(sae, norm, tok, mask)
            mm[s]['codes_max'][cur:cur + len(r)] = mx.half().cpu().numpy()
            mm[s]['codes_mean'][cur:cur + len(r)] = mean.half().cpu().numpy()
            mm[s]['n_fg'][cur:cur + len(r)] = mask.sum(1).short().cpu().numpy()
        cur += len(r)
        if b % 100 == 0:
            log(f'  batch {b:5d}  {cur:7d}/{len(rows)}  {cur / (time.time() - t0):6.1f} frames/s')
    assert cur == len(rows)
    el = time.time() - t0
    for s, d in dirs.items():
        for a in mm[s].values():
            a.flush()
        (tmp[s] / 'shard.json').write_text(json.dumps({'shard': args.shard, 'n_rows': int(len(rows)),
                                                       'elapsed_s': round(el, 1), 'saes_in_pass': args.sae}))
        if d.exists():
            shutil.rmtree(d)
        tmp[s].rename(d)
        (d / 'DONE').touch()
    log(f'done {len(rows):,} rows x {len(saes)} SAEs in {el:.0f}s ({len(rows) / el:.1f} frames/s)')


def step_merge(dom, args):
    import torch
    from src.eci.fg_encode import encode_rows
    from src.eci.somp_encode import _cos_fg
    old_eci = Path(args.old_eci)
    old_row = np.load(reuse_path(dom, old_eci))['old_row']
    R = runs(old_row)
    todo = np.flatnonzero(old_row < 0)
    n = len(old_row)
    fp_all = pd.read_csv(dom.ann_path, usecols=['frame_path'])['frame_path'].values
    for s in args.sae:
        out, old = dom.codes_root / s, old_eci / 'codes' / s
        if not (old / 'DONE').exists():
            raise SystemExit(f'{old}/DONE missing')
        if not (out / 'DONE').exists():
            sh = [out / 'shards' / f'shard_{i:02d}' for i in range(args.n_shards)]
            if [d for d in sh if not (d / 'DONE').exists()]:
                raise SystemExit(f'{s}: unfinished shards {[d.name for d in sh if not (d / "DONE").exists()]}')
            for name in OUTPUTS:
                src = np.load(old / f'{name}.npy', mmap_mode='r')
                dst = np.lib.format.open_memmap(out / f'{name}.npy.tmp', 'w+', src.dtype, (n,) + src.shape[1:])
                for nlo, olo, k in R:
                    for a in range(0, k, 65536):
                        b = min(a + 65536, k)
                        dst[nlo + a:nlo + b] = src[olo + a:olo + b]
                cur = 0
                for d in sh:
                    rows = np.load(d / 'rows.npy')
                    assert (rows == todo[cur:cur + len(rows)]).all(), d
                    x = np.load(d / f'{name}.npy', mmap_mode='r')
                    for a in range(0, len(rows), 65536):
                        dst[rows[a:a + 65536]] = x[a:a + 65536]
                    cur += len(rows)
                assert cur == len(todo)
                dst.flush()
                del dst
                (out / f'{name}.npy.tmp').rename(out / f'{name}.npy')
                log(f'{s}: merged {name}')
            cfg = json.loads((old / 'config.json').read_text())
            cfg.update(row_order=str(dom.ann_path.relative_to(REPO)), n_rows=int(n),
                       backgrounds=str(dom.eci_dir / 'fg448/background'), n_shards=args.n_shards,
                       reuse={'from': str(old), 'map': str(reuse_path(dom, old_eci)), 'n_reused': int(n - len(todo)),
                              'n_encoded': int(len(todo)), 'rule': 'rows with the same (observation_id, frame_idx, '
                              'frame_path) copied byte for byte; the others encoded (scripts/eci/fg_encode_extend.py)'})
            cfg.pop('shard_ranges', None)
            (out / 'config.json').write_text(json.dumps(cfg, indent=1))
            (out / 'DONE').touch()
        sib = dom.codes_root / f'{s}_mean'  # mean-pooling sibling result set (links, as in the old layout)
        if (old_eci / 'codes' / f'{s}_mean').exists() and not (sib / 'DONE').exists():
            sib.mkdir(parents=True, exist_ok=True)
            for f in ('codes_mean.npy', 'n_fg.npy'):
                if not (sib / f).exists():
                    (sib / f).symlink_to((out / f).resolve())
            (sib / 'config.json').write_text(json.dumps({
                'aggregation': "mean over the frame's patches (foreground patches for foreground SAEs)",
                'source': str(out.resolve()), 'files': 'symlinks to the source codes_mean.npy [and n_fg.npy]'}, indent=1))
            (sib / 'DONE').touch()

        # ---- verify
        res = {'sae': s, 'n_rows': int(n), 'n_reused': int(n - len(todo)), 'n_encoded': int(len(todo))}
        exact = {}
        for name in OUTPUTS:
            a, b = np.load(out / f'{name}.npy', mmap_mode='r'), np.load(old / f'{name}.npy', mmap_mode='r')
            ok = True
            for nlo, olo, k in R:
                for x in range(0, k, 65536):
                    y = min(x + 65536, k)
                    ok &= bool(np.array_equal(np.asarray(a[nlo + x:nlo + y]).view(np.uint8),
                                              np.asarray(b[olo + x:olo + y]).view(np.uint8)))
            exact[name] = ok
        res['reuse_exact'] = exact
        nf = np.load(out / 'n_fg.npy', mmap_mode='r')
        res['codes_rows_all_zero_new'] = int((np.asarray(np.load(out / 'codes_max.npy', mmap_mode='r')[
            todo[::max(1, len(todo) // 20000)]]) == 0).all(1).sum())
        rng = np.random.default_rng(0)
        reused = np.flatnonzero(old_row >= 0)
        sets = {'reused': np.sort(rng.choice(reused, min(args.n_check, len(reused)), replace=False)),
                'encoded': np.sort(rng.choice(todo, min(args.n_check, len(todo)), replace=False))}
        Z = {k: np.load(out / f'{k}.npy', mmap_mode='r') for k in ('codes_max', 'codes_mean')}
        res['recompute'] = {}
        for k, rows in sets.items():
            if len(rows) == 0:
                continue
            [ref] = encode_rows(rows, fp_all, [dom.eci_dir / 'sae' / s / 'sae.pt'], dom.eci_dir / 'fg448/background',
                                dom.ann_path, REPO / 'dataset', batch_size=16, num_workers=8)
            c = {}
            for key in ('codes_max', 'codes_mean'):
                x = torch.from_numpy(np.asarray(Z[key][rows], dtype=np.float32))
                y = torch.from_numpy(ref[key].astype(np.float32))
                c[key] = {'cos_min': float(_cos_fg(x, y).min()), 'max_abs_diff': float((x - y).abs().max())}
            c['n_fg_exact_frac'] = float((nf[rows] == ref['n_fg']).mean())
            c['n_fg_max_abs_diff'] = int(np.abs(nf[rows].astype(int) - ref['n_fg']).max())
            c['n_frames'] = int(len(rows))
            res['recompute'][k] = c
        res['passed'] = bool(all(exact.values()) and all(
            v['codes_max']['cos_min'] > 0.99 and v['codes_mean']['cos_min'] > 0.99 and v['n_fg_max_abs_diff'] <= 5
            for v in res['recompute'].values()))
        (out / 'verify.json').write_text(json.dumps(res, indent=1))
        log(json.dumps(res, indent=1))
        if not res['passed']:
            raise SystemExit(f'{s}: VERIFY FAILED')
        if (out / 'shards').exists():
            shutil.rmtree(out / 'shards')
            log(f'{s}: shards deleted')


def step_plan(dom, args):
    """Login-safe estimate from experiment.csv files (no annotations.csv read)."""
    old = pd.read_csv(Path(args.old_eci) / 'experiment.csv', dtype={'observation_id': str})
    sys.path.insert(0, str(REPO / 'scripts/eci'))
    from ants_prepare import LAYOUTS, load_experiment
    old_ids = set(old['observation_id'])
    rows, reused = {}, {}
    for name, (keep, nf, _) in LAYOUTS[dom.ECI].items():
        p = dom.raw_experiments[name]
        if not p.exists():
            print(f'{name}: {p} missing, skipped')
            continue
        e = load_experiment(name, p, keep)
        rows[name] = len(e) * nf
        reused[name] = int(e['observation_id'].isin(old_ids).sum()) * min(nf, 3000)
    n_enc = sum(rows.values()) - sum(reused.values())
    gb = sum(rows.values()) * 1024 * 2 * 2 / 1e9
    print(json.dumps({'rows': rows, 'reused': reused, 'n_rows': sum(rows.values()), 'n_encode': n_enc,
                      'gpu_hours_one_pass': round(n_enc / args.fps / 3600, 1),
                      'codes_gb_per_sae': round(gb, 2)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('setup', 'encode', 'merge', 'plan'))
    ap.add_argument('--domain', default='ants', choices=DOMAINS)
    ap.add_argument('--old-eci', default=str(REPO / 'dataset/ants/eci'), help='the layout whose codes are reused')
    ap.add_argument('--sae', action='append', default=[])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=24)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=14)
    ap.add_argument('--n-check', type=int, default=48, help='merge: rows recomputed per row set (reused / encoded)')
    ap.add_argument('--fps', type=float, default=25.0, help='plan: frames/s per GPU')
    args = ap.parse_args()
    dom = get_domain(args.domain)
    if Path(args.old_eci).resolve() == dom.eci_dir.resolve():
        raise SystemExit(f'--old-eci is the domain layout itself ({dom.eci_dir}); set ANTS_ECI to the new layout')
    if args.step in ('setup', 'encode', 'merge') and not args.sae:
        raise SystemExit('give --sae (repeatable)')
    {'setup': step_setup, 'encode': step_encode, 'merge': step_merge, 'plan': step_plan}[args.step](dom, args)


if __name__ == '__main__':
    main()
