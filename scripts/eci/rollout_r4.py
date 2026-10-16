"""
R4 ROLLOUT: the new ECI recipe (decided on the public benchmarks MABe22 / CalMS21) on the lab domains.

Recipe per domain
    frames   the 5 fps grid of the existing pipeline (annotations.csv rows), decoded from the NATIVE source video
             (source frame = start_frame + STEP k + OFFSET: mice 6k+2 @30 fps, ants v2 6k+2, ants v3 6k+1, frogs 12k+5
             @60 fps), with the standardisation's chain at side S ([hflip,]scale=S:S:force_original_aspect_ratio=
             decrease,pad=S:S:(ow-iw)/2:(oh-ih)/2, ffmpeg 6.1.1, rgb24 through a pipe; no JPEG anywhere).
    encoder  DINOv3-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m), blocks 1-20 of 24 (the model is truncated after
             block 20, then its final LayerNorm = the benchmarks' 'block 20 after the final LayerNorm'), CLS + 4
             registers dropped, bf16 autocast + SDPA, tokens rounded to fp16. S sized to the animal (~10 patches per
             body length, never above native, multiple of 16): mice 1024 (64 x 64), ants v2 768 (48 x 48), ants v3
             832 (52 x 52), frogs 800 (50 x 50).
    mask     the CURRENT per-domain rule (src/eci/foreground.py RULES: mice 'fg448', ants 'ants', frogs 'frogs') with the
             existing per-video backgrounds, computed exactly as C1 on the 32 x 32 grid (DINOv2-base at 448 on the 512 px
             frame + the grey 512 px frame), on C1's own input: the frame's 512 px dataset JPG (staged to local disk per
             job). The decoded native frame resized to 512 was measured first (maskdiag: same rule, same backgrounds)
             and inflates the ants masks by up to 42% (the backgrounds / thresholds are JPG quantities); on the JPGs the
             32-grid count matches the C1 n_fg (mean |diff| 0.3-1.1 cells, frogs exact). Projected to the G x G grid by patch centre (idx = floor((g + 0.5) 32 / G), MABe22 map_mask;
             at G = 64 this is exactly the 2 x 2 children of a 448 cell, CalMS21 to_grid).
    context  x = [z_t, z_t - z_{t-10}] (10 frames = 2 s at 5 fps, same patch; t-10 clipped to the video's first frame)
             -> variant 'd2' (2048-d); variant 'd0' = z_t only (1024-d). Both from the same encoder pass.
    SAE      Matryoshka BatchTopK, 512 latents, prefixes 128/256/512, k = 16 (the benchmark train_one: TokenNorm per
             1024-d block, b_dec = geometric median, Adam 5e-4, 500 warm-up, cosine, 6000 steps x 4096 tokens, AuxK),
             3 seeds per variant; theta* (src/eci/sae_calibrate.py) on the held-out CALIBRATION videos; the seed with the
             highest FVE (= lowest normalised reconstruction error) on the held-out EVALUATION videos is kept. No labels.
    codes    per 5 fps frame and neuron: max over the frame's kept tokens under theta* (codes_max), mean over them
             (codes_mean), the argmax token (G-grid position, -1 = silent), n_fg (kept G-grid tokens), n_fg32 (kept
             32-grid cells, comparable with the C1 codes' n_fg).

Steps (scripts/eci/rollout_r4.sh runs each in Slurm; every heavy input / output is staged on $LOCAL_DIR)
    plan      (CPU) per domain: videos, row maps, source paths, ffprobe checks, held-out split, prepass bursts,
              encode shards; the ants layouts eci_r4v2 / eci_r4v3 (annotations / experiment subsets on BeeGFS,
              symlinked as dataset/ants/eci_r4v*) -> OUT/<dom>/plan.json
    check     (GPU) frame alignment (decoded frame vs dataset JPG at k-1, k, k+1), seek == sequential decode,
              truncated DINOv3 == the benchmark hook encoder, throughput, mask n_fg32 vs C1 n_fg -> OUT/<dom>/check.json
    prepass   (GPU) the sampled bursts only: kept tokens z_t and z_{t-10} of ~12k frames -> BEE/<dom>/prepass/,
              mask sheet OUT/<dom>/mask_sheet.jpg, OUT/<dom>/prepass.json
    train     (GPU) 2 variants x 3 seeds, theta*, held-out stats, best seed -> OUT/<dom>/sae/<name>/
    encode    (GPU, one task per shard of videos) streaming full pass -> BEE/<dom>/videos/<obs>.npz
    merge     (CPU) per variant codes dir BEE/<dom>/codes/<name>/ (codes_max / codes_mean / argmax / n_fg / n_fg32 .npy,
              config.json, DONE), symlinked into the domain's eci codes root; SAE dir symlinked into its sae root
    summaries (CPU) per video and neuron: time spent, bout rate, latency (fires iff > 0 under theta*)
    align     (CPU) behaviour alignment test check (descriptive): r4 d2 / d0 vs C1 and the deployed atlas SAE
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
OUT = Path(os.environ.get('R4_OUT') or REPO / 'results/vision/eci_rollout_r4')  # R4_OUT / R4_BEE: self-test roots
BEE = Path(os.environ.get('R4_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/r4')
FFMPEG = Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg'
FFPROBE = Path.home() / '.conda/envs/ffmpeg611/bin/ffprobe'
D3L = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
LAYER, N_PRE, D = 20, 5, 1024
WIDTH, K, PREFIXES = 512, 16, (128, 256, 512)
LAG = 10  # 5 fps frames = 2 s
VARIANTS = ('d2', 'd0')
SEEDS = (0, 1, 2)
FRAMES_TARGET = 12_000  # prepass frames per domain (train + held-out)
BURST = (0, 5, 10, 15, 20, 25)  # 5 fps offsets decoded per burst; train frames = offsets 10..25, prev = offset - 10
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731
# DINOv3 autocast dtype: bf16 (the validated recipe; every step that writes tokens / codes asserts it). R4_AMP=fp16 only for
# the 'check' step on GPUs without bf16 (debug_gpu 2080 Ti): checks of alignment / seek / mask / truncation, not codes.
AMP = os.environ.get('R4_AMP', 'bf16')
# fp16 for whole domains run on debug_gpu (no bf16 there; decided by the orchestrator 2026-10-15: ants_v2): every token of
# such a domain (prepass, theta*, full pass) is fp16; the encode step refuses a precision that differs from its prepass.
FP16_DOMAINS = {'ants_v2', 'ants_v2p'}


def check_amp(dom):
    if AMP != 'bf16' and not os.environ.get('R4_TEST') and dom not in FP16_DOMAINS:
        raise SystemExit(f'{dom}: tokens / codes are written only with the bf16 recipe (R4_AMP={AMP})')

# v4 / v4_infested source offset 2 (results/vision/eci_rollout_r4/v4_offset.json: 3 videos x 3 frames per series, offsets
# 0-5; offset 2 is closest to the dataset JPG on every frame with clear motion; static frames are ambiguous within 1-2
# source frames, some old-file frames k < 3000 favour 0)
OFF_V4, OFF_V4I = 2, 2
DOMS = {
    'mice': dict(ann='dataset/mice/v1/annotations.csv', exp='data/mice/v1/experiment.csv', fps=30, step=6, offset=2,
                 S=1024, rule='fg448', bg='dataset/mice/v1/eci/fg448/background', eci='dataset/mice/v1/eci',
                 layout=None, domain='mice', ants_eci=None, native=2064, body=303,
                 ref_nfg='dataset/mice/v1/eci/codes/matryoshka_btk_4096_k16_fg448m5_s0/n_fg.npy',
                 c1='dataset/mice/v1/eci/codes/matryoshka_btk_4096_k16_fg448m5_s0',
                 deployed='dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0',
                 p_store=0.45, chunk=300, par=6, threads=4, bs=16, n_shards=14),
    'ants_v2': dict(ann='dataset/ants/eci30/annotations.csv', exp='dataset/ants/eci30/experiment.csv', fps=30, step=6,
                    offset=2, S=768, rule='ants', bg='dataset/ants/eci30/fg448/background', eci='dataset/ants/eci_r4v2',
                    layout='eci_r4v2', select=('v2', None), srcdir='data/ants/v2/observations/source', domain='ants',
                    ants_eci='eci_r4v2', native=770, body=64,
                    ref_nfg='dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/n_fg.npy',
                    c1='dataset/ants/eci/codes/matryoshka_btk_4096_k16_antsfgm5_s0',
                    deployed='dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0', c1_ann='dataset/ants/eci/annotations.csv',
                    p_store=1.0, chunk=600, par=4, threads=4, bs=32, n_shards=1),
    'ants_v3': dict(ann='dataset/ants/eci30/annotations.csv', exp='dataset/ants/eci30/experiment.csv', fps=30, step=6,
                    offset=1, S=832, rule='ants', bg='dataset/ants/eci30/fg448/background', eci='dataset/ants/eci_r4v3',
                    layout='eci_r4v3', select=('v3', (2, 6, 8)), srcdir='data/ants/v3/observations/source',
                    domain='ants', ants_eci='eci_r4v3', native=824, body=76,
                    ref_nfg='dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/n_fg.npy',
                    c1='dataset/ants/eci/codes/matryoshka_btk_4096_k16_antsfgm5_s0',
                    deployed='dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0', c1_ann='dataset/ants/eci/annotations.csv',
                    p_store=1.0, chunk=600, par=4, threads=4, bs=32, n_shards=4),
    # ants v4: healthy (v4) + infested (v4_infested) Exp4 colonies, t = 2 and 6, 30 min; per-series source offset
    # (measured with the dataset JPGs, results/vision/eci_rollout_r4/v4_offset.json); body length not measured (v4 is not
    # in results/vision/animal_scale): S = native 812 rounded up to 816 as the other ants series
    'ants_v4': dict(ann='dataset/ants/eci30/annotations.csv', exp='dataset/ants/eci30/experiment.csv', fps=30, step=6,
                    offset={'v4': OFF_V4, 'v4_infested': OFF_V4I}, S=816, rule='ants',
                    bg='dataset/ants/eci30/fg448/background', eci='dataset/ants/eci_r4v4', layout='eci_r4v4',
                    select=(('v4', 'v4_infested'), (2, 6)),
                    srcdir={'v4': 'data/ants/v4/observations/source', 'v4_infested': 'data/ants/v4_infested/observations/source'},
                    domain='ants', ants_eci='eci_r4v4', native=812, body=None,
                    ref_nfg='dataset/ants/eci30/codes/matryoshka_btk_1024_k16_antsfg_s0/n_fg.npy', c1=None, deployed=None,
                    p_store=1.0, chunk=600, par=4, threads=4, bs=32, n_shards=10),
    'frogs': dict(ann='dataset/frogs/eci/annotations.csv', exp='data/frogs/v1/experiment.csv', fps=60, step=12,
                  offset=5, S=800, rule='frogs', bg='dataset/frogs/eci/fg448/background', eci='dataset/frogs/eci',
                  layout=None, srcdir='data/frogs/v1/observations/source', domain='frogs', ants_eci=None, native=800,
                  body=73, ref_nfg='dataset/frogs/eci/codes/matryoshka_btk_4096_k16_frogsfgm5_s0/n_fg.npy',
                  c1=None, deployed=None, p_store=1.0, chunk=600, par=4, threads=4, bs=32, n_shards=3),
}


def bee(dom):
    """BeeGFS root of a domain: r4/<dom>; the pooled ants SAE and its series: r4/ants_pooled[/<series>]."""
    c = DOMS.get(dom, {})
    if dom == 'ants_pooled':
        return BEE / 'ants_pooled'
    return BEE / 'ants_pooled' / c['series'] if c.get('pooled') else BEE / dom


def sae_root(dom):
    return OUT / 'ants_pooled' / 'sae' if DOMS.get(dom, {}).get('pooled') else OUT / dom / 'sae'


def prepass_json(dom):
    """The prepass record whose token precision a domain's codes must match (pooled series: their own series)."""
    return OUT / (DOMS[dom].get('base') or dom) / 'prepass.json'


def voff(c, v):
    """Source-frame offset of a plan video (per video since ants v4; older plans: the domain constant)."""
    return int(v['offset']) if 'offset' in v else int(c['offset'])


# r4 pooled (user's design correction 2026-10-16): ONE ants SAE per variant for all ant experiments, trained on equal
# prepass frame counts per experiment (v2, v3, v4, v4_infested); the series keep their own input sizes and masks.
# v3 is re-encoded at source offset 2 (results/vision/eci_rollout_r4/lagcheck.json: 24 / 24 frames best at 2; the
# per-series v3 codes used 1, from the old 10-min files).
DOMS['ants_pooled'] = dict(DOMS['ants_v3'], S=None, layout=None, eci=None, domain='ants', ants_eci=None, native=None,
                           body=None, n_shards=0)
for _b, _s, _off in (('ants_v2', 'v2', 2), ('ants_v3', 'v3', 2), ('ants_v4', 'v4', None)):
    DOMS[f'{_b}p'] = dict(DOMS[_b], pooled=True, base=_b, series=_s, layout=f'eci_r4{_s}p', eci=f'dataset/ants/eci_r4{_s}p',
                          ants_eci=f'eci_r4{_s}p', **({'offset': _off} if _off is not None else {}))


def sae_name(v, seed):
    return f'matryoshka_btk_{WIDTH}_k{K}_r4{v}_s{seed}'


def local_dir():
    d = os.environ.get('LOCAL_DIR')
    if not d:
        raise SystemExit('LOCAL_DIR is not set (the job staging dir /localhome/$USER/$SLURM_JOB_ID)')
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def obs_ranges_of(ann_path, extra_cols=()):
    a = pd.read_csv(ann_path, usecols=['observation_id', 'frame_idx', *extra_cols])
    o = a['observation_id'].values
    ch = np.r_[0, np.flatnonzero(o[1:] != o[:-1]) + 1, len(o)]
    ids = o[ch[:-1]]
    assert len(set(ids)) == len(ids), 'observations not contiguous'
    return a, {i: (int(s), int(e)) for i, s, e in zip(ids, ch[:-1], ch[1:])}


def load_plan(dom):
    return json.loads((OUT / dom / 'plan.json').read_text())


def atomic_copy(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    if Path(src).is_dir():
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        if dst.exists():
            shutil.rmtree(dst)
    else:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------------------------- plan
def ffprobe(path):
    r = subprocess.run([str(FFPROBE), '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration', '-of', 'json',
                        str(path)], capture_output=True, text=True, timeout=120)
    j = json.loads(r.stdout)
    s = j['streams'][0]
    return {'codec': s.get('codec_name'), 'w': int(s['width']), 'h': int(s['height']), 'rate': s.get('r_frame_rate'),
            'nb_frames': int(s['nb_frames']) if s.get('nb_frames', 'N/A') not in ('N/A', None) else None,
            'duration': float(j['format']['duration'])}


def cmd_plan(args):
    for dom in args.domains:
        c = DOMS[dom]
        t0 = time.time()
        A, R = obs_ranges_of(REPO / c['ann'], ('source_frame',) if dom == 'frogs' else ())
        for o, (lo, hi) in R.items():
            fi = A['frame_idx'].values[lo:hi]
            if not (fi == np.arange(hi - lo)).all():
                raise SystemExit(f'{dom} {o}: frame_idx is not 0..n-1')
        if dom == 'frogs':  # stored mapping must equal 12 k + 5
            sf = A['source_frame'].values
            k = A['frame_idx'].values
            if not (sf == c['step'] * k + c['offset']).all():
                raise SystemExit('frogs: source_frame != 12 k + 5')
        E = pd.read_csv(REPO / c['exp'])
        if dom.startswith('ants'):
            ex, ts = c['select']
            exs = ex if isinstance(ex, tuple) else (ex,)
            sel = E[E['experiment'].isin(exs) & (E['T'].isin(ts) if ts else True)]
            vids = [o for o in R if o in set(sel['observation_id'])]
        else:
            vids = list(R)
        Ei = E.set_index('observation_id')
        videos, lay = [], 0
        for o in vids:
            lo, hi = R[o]
            if dom == 'mice':
                path = REPO / 'data/mice/source' / Ei.loc[o, 'observation_file']
                hflip = 0
                assert int(Ei.loc[o, 'start_frame']) == 0
            elif dom == 'frogs':
                path = REPO / c['srcdir'] / Ei.loc[o, 'observation_file']
                hflip = int(Ei.loc[o, 'hflip'])
                assert int(Ei.loc[o, 'start_frame']) == 0
            else:
                ex_o = str(Ei.loc[o, 'experiment'])
                sd = c['srcdir'][ex_o] if isinstance(c['srcdir'], dict) else c['srcdir']
                path = REPO / sd / f'{o}.mkv'
                hflip = 0
            off = c['offset'][str(Ei.loc[o, 'experiment'])] if isinstance(c['offset'], dict) else c['offset']
            if off < 0:
                raise SystemExit(f'{dom}: source offset not measured')
            videos.append({'obs': o, 'src_lo': lo, 'n': hi - lo, 'lay_lo': lay, 'path': str(path), 'hflip': hflip,
                           'offset': int(off)})
            lay += hi - lo
        # ffprobe every source (headers only)
        with ThreadPoolExecutor(16) as ex_:
            probes = list(ex_.map(lambda v: ffprobe(v['path']), videos))
        bad = []
        for v, p in zip(videos, probes):
            v['probe'] = p
            need = voff(c, v) + c['step'] * (v['n'] - 1) + 1
            have = p['nb_frames'] if p['nb_frames'] else int(round(p['duration'] * c['fps']))
            v['src_frames'] = have
            if have < need or max(p['w'], p['h']) not in range(c['native'] - 8, c['native'] + 9):
                bad.append((v['obs'], have, need, p['w'], p['h']))
        if bad:
            raise SystemExit(f'{dom}: {len(bad)} videos too short / wrong size, e.g. {bad[:5]}')
        # held-out split (by video, seeded): every ~6th video, alternating calibration (A) / evaluation (B)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(videos))
        n_held = max(6, int(round(len(videos) / 6)))
        n_held += n_held % 2
        for j, i in enumerate(order):
            videos[i]['role'] = 'calib' if j < n_held // 2 else ('eval' if j < n_held else 'train')
        # prepass bursts: per video n_b bursts evenly spread with a seeded jitter; burst start a >= 0, a + 25 < n
        n_b = int(np.ceil(FRAMES_TARGET / (4 * len(videos))))
        for i, v in enumerate(videos):
            g = np.random.default_rng([1, i])
            span = v['n'] - 26
            u = g.random(n_b)
            a = np.floor((np.arange(n_b) + u) * span / n_b).astype(int)
            assert a.min() >= 0 and a.max() + 25 < v['n'] and (np.diff(a) >= 0).all()
            v['bursts'] = a.tolist()
        # encode shards: consecutive videos, balanced by frame count
        n = np.array([v['n'] for v in videos])
        cuts = np.searchsorted(np.cumsum(n), np.linspace(0, n.sum(), c['n_shards'] + 1)[1:-1])
        shards = np.split(np.arange(len(videos)), cuts + 1)
        shards = [s.tolist() for s in shards if len(s)]
        G = c['S'] // 16
        plan = {'domain': dom, 'cfg': c, 'G': G, 'n_videos': len(videos), 'n_frames': int(n.sum()),
                'patches_per_body': c['body'] * c['S'] / c['native'] / 16 if c['body'] else None, 'bursts_per_video': n_b,
                'n_prepass_frames': 4 * n_b * len(videos), 'roles': {r: sum(v['role'] == r for v in videos)
                                                                    for r in ('train', 'calib', 'eval')},
                'shards': shards, 'shard_frames': [int(n[s].sum()) for s in shards], 'videos': videos}
        # ants layouts: subsets of the eci30 layout on BeeGFS, symlinked as dataset/ants/eci_r4v*
        if c['layout']:
            L = bee(dom) / 'layout'
            L.mkdir(parents=True, exist_ok=True)
            Afull = pd.read_csv(REPO / c['ann'])
            idx = np.concatenate([np.arange(v['src_lo'], v['src_lo'] + v['n']) for v in videos])
            Afull.iloc[idx].to_csv(L / 'annotations.csv', index=False)
            E[E['observation_id'].isin(vids)].set_index('observation_id').loc[vids].reset_index().to_csv(
                L / 'experiment.csv', index=False)
            (L / 'codes').mkdir(exist_ok=True)
            (L / 'sae').mkdir(exist_ok=True)
            if not (L / 'fg448').exists():
                os.symlink(REPO / 'dataset/ants/eci30/fg448', L / 'fg448')
            link = REPO / 'dataset/ants' / c['layout']
            if not link.exists():
                os.symlink(L, link)
            plan['layout_dir'] = str(L)
        (OUT / dom).mkdir(parents=True, exist_ok=True)
        (OUT / dom / 'plan.json').write_text(json.dumps(plan, indent=1, default=str))
        log(f'{dom}: {len(videos)} videos, {n.sum():,} frames, G {G}, patches/body {plan["patches_per_body"]}, '
            f'{n_b} bursts/video ({plan["n_prepass_frames"]:,} prepass frames), roles {plan["roles"]}, '
            f'shards {plan["shard_frames"]}, codecs {sorted({p["codec"] for p in probes})}, '
            f'sizes {sorted({(p["w"], p["h"]) for p in probes})} ({time.time() - t0:.0f}s)')


def cmd_reshard(args):
    """Re-cut a domain's encode shards to about --shard-frames frames each (short jobs for backfill); mice: the
    behaviour-annotated videos (Y_nn present) first, so the test-check alignment can start early. Writes plan.json."""
    dom = args.domain
    c = DOMS[dom]
    P = load_plan(dom)
    order = list(range(len(P['videos'])))
    if dom == 'mice':
        a = pd.read_csv(REPO / c['ann'], usecols=['observation_id', 'Y_nn'])
        lab = set(a.loc[a.Y_nn.notna(), 'observation_id'])
        for v in P['videos']:
            v['annotated'] = v['obs'] in lab
        order = [i for i in order if P['videos'][i]['annotated']] + [i for i in order if not P['videos'][i]['annotated']]
    shards, cur, nf = [], [], 0
    for i in order:
        if cur and (nf + P['videos'][i]['n'] > args.shard_frames or
                    (dom == 'mice' and P['videos'][i]['annotated'] != P['videos'][cur[0]]['annotated'])):
            shards.append(cur)
            cur, nf = [], 0
        cur.append(i)
        nf += P['videos'][i]['n']
    shards.append(cur)
    P['shards'] = shards
    P['shard_frames'] = [int(sum(P['videos'][i]['n'] for i in s)) for s in shards]
    if dom == 'mice':
        P['shards_annotated'] = [j for j, s in enumerate(shards) if P['videos'][s[0]]['annotated']]
    (OUT / dom / 'plan.json').write_text(json.dumps(P, indent=1, default=str))
    log(f'{dom}: {len(shards)} shards, frames {P["shard_frames"]}' +
        (f', annotated shards {P["shards_annotated"]}' if dom == 'mice' else ''))


# ---------------------------------------------------------------------------------------------- decode
def decode(path, s0, n, step, S, hflip, fps, threads=4):
    """Source frames s0 + step i (i < n) at side S, (n, S, S, 3) uint8. Input seek to (s0 - 0.5) / fps (accurate seek:
    the first frame kept is the first with pts >= that time = source frame s0), then select every step-th frame."""
    vf = f"select='not(mod(n\\,{step}))'" + (',hflip' if hflip else '') + \
        f',scale={S}:{S}:force_original_aspect_ratio=decrease,pad={S}:{S}:(ow-iw)/2:(oh-ih)/2'
    cmd = [str(FFMPEG), '-nostdin', '-v', 'error', '-threads', str(threads)]
    if s0 > 0:
        cmd += ['-ss', f'{(s0 - 0.5) / fps:.6f}']
    cmd += ['-i', str(path), '-an', '-sn', '-vf', vf, '-fps_mode', 'passthrough', '-frames:v', str(n),
            '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    p = subprocess.run(cmd, capture_output=True)
    need = n * S * S * 3
    if p.returncode != 0 or len(p.stdout) != need:
        raise RuntimeError(f'decode {path} s0 {s0} n {n}: rc {p.returncode}, {len(p.stdout)} of {need} bytes; '
                           f'{p.stderr.decode()[-400:]}')
    return np.frombuffer(p.stdout, np.uint8).reshape(n, S, S, 3)


def load_jpgs(paths):
    """512 px dataset JPGs (the C1 mask input) -> (n, 512, 512, 3) uint8."""
    from PIL import Image
    out = np.empty((len(paths), 512, 512, 3), np.uint8)
    for i, f in enumerate(paths):
        with Image.open(f) as im:
            a = np.asarray(im.convert('RGB'))
        if a.shape != (512, 512, 3):
            raise ValueError(f'{f}: shape {a.shape}')
        out[i] = a
    return out


def stage_jpgs(rels, dst, workers=32):
    """Copy dataset-relative JPG paths from REPO/dataset to dst (same layout), one bulk copy (as rollout_c1)."""
    t0 = time.time()
    for d in sorted({str(Path(r).parent) for r in rels}):
        (dst / d).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda r: shutil.copyfile(REPO / 'dataset' / r, dst / r), rels, chunksize=64))
    log(f'  staged {len(rels):,} JPGs in {time.time() - t0:.0f}s')


class Prefetch:
    """Ordered decode jobs [(key, kwargs)] run `par` at a time in threads; get() yields (key, frames) in order."""

    def __init__(self, jobs, par):
        self.jobs, self.ex = list(jobs), ThreadPoolExecutor(par)
        self.par = par
        self.futs = {}
        self.next = 0
        self.dec_s = 0.0
        self.lock = threading.Lock()
        for _ in range(par + 1):
            self._submit()

    def _run(self, kw):
        t0 = time.time()
        x = decode(**{k_: v_ for k_, v_ in kw.items() if k_ != 'jpgs'})
        J = load_jpgs(kw['jpgs']) if kw.get('jpgs') is not None else None
        with self.lock:
            self.dec_s += time.time() - t0
        return x, J

    def _submit(self):
        if self.next < len(self.jobs):
            k, kw = self.jobs[self.next]
            self.futs[self.next] = (k, self.ex.submit(self._run, kw))
            self.next += 1

    def __iter__(self):
        for i in range(len(self.jobs)):
            k, f = self.futs.pop(i)
            x = f.result()
            self._submit()
            yield k, x
        self.ex.shutdown()


# ---------------------------------------------------------------------------------------------- encoders + mask
class Pipe:
    """DINOv2-base 448 mask tokens (+ the current rule with the existing backgrounds) and DINOv3-L block-20 tokens."""

    def __init__(self, dom, dev, src_ranges):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from src.eci.foreground import RULES, FgBackgrounds
        self.torch, self.dev, self.c = torch, dev, DOMS[dom]
        self.S, self.G = self.c['S'], self.c['S'] // 16
        self.proc2 = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True,
                                                        size={'height': 448, 'width': 448}, do_center_crop=False)
        self.m2 = AutoModel.from_pretrained('facebook/dinov2-base', attn_implementation='sdpa').to(dev).eval()
        m3 = AutoModel.from_pretrained(D3L, attn_implementation='sdpa').to(dev).eval()
        assert len(m3.layer) == 24
        m3.layer = torch.nn.ModuleList(list(m3.layer)[:LAYER])
        self.m3 = m3.requires_grad_(False)
        self.m2.requires_grad_(False)
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]
        self.bgs = FgBackgrounds(REPO / self.c['bg'], src_ranges, RULES[self.c['rule']], dev)
        g = np.floor((np.arange(self.G) + 0.5) * 32 / self.G).astype(np.int64)
        self.proj = torch.from_numpy((g[:, None] * 32 + g[None, :]).ravel()).to(dev)  # G*G -> 32-grid cell
        self.gpu_s = 0.0

    def frame512(self, x):
        """x (B, 3, S, S) float 0..255 -> (B, 3, 512, 512) uint8 (bicubic, antialias)."""
        F = self.torch.nn.functional
        if x.shape[-1] == 512:
            return x.round().clamp(0, 255).to(self.torch.uint8)
        return F.interpolate(x, size=(512, 512), mode='bicubic', antialias=True,
                             align_corners=False).round().clamp(0, 255).to(self.torch.uint8)

    @staticmethod
    def grey(x512):
        """PIL convert('L'): (R 19595 + G 38470 + B 7471 + 32768) >> 16."""
        t = x512.int()
        return ((t[:, 0] * 19595 + t[:, 1] * 38470 + t[:, 2] * 7471 + 32768) >> 16).to(x512.dtype)

    def run(self, frames, src_rows, jpg=None):
        """frames (B, S, S, 3) uint8 numpy, src_rows (B,) source-layout rows, jpg (B, 512, 512, 3) uint8 = the dataset
        JPGs of these rows (C1's mask input; None = the decoded frame resized to 512) -> z (B, G*G, 1024) fp16, mask
        (B, G*G) bool, n32 (B,) kept 32-grid cells, x512 (B, 3, 512, 512) uint8 (the mask's input frame)."""
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            x = torch.from_numpy(np.array(frames)).to(self.dev).permute(0, 3, 1, 2).float()
            if jpg is not None:
                assert len(jpg) == len(frames)
                x512 = torch.from_numpy(jpg).to(self.dev).permute(0, 3, 1, 2).contiguous()
            else:
                x512 = self.frame512(x)
            grey = self.grey(x512)
            pix = self.proc2(images=list(x512), return_tensors='pt', device=self.dev)['pixel_values'].to(self.dev)
            with torch.autocast('cuda', dtype=torch.float16):
                h2 = self.m2(pixel_values=pix).last_hidden_state
            tok2 = h2[:, 1:].float().half()
            assert tok2.shape[1] == 1024 and torch.isfinite(tok2).all()
            m32, _ = self.bgs.mask(tok2, grey, np.asarray(src_rows))
            xin = (x / 255 - self.mean) / self.std
            with torch.autocast('cuda', dtype=torch.bfloat16 if AMP == 'bf16' else torch.float16):
                h3 = self.m3(pixel_values=xin).last_hidden_state
            z = h3[:, N_PRE:].float()
            assert z.shape[1] == self.G * self.G and torch.isfinite(z).all(), z.shape
            z = z.half()
            m = m32[:, self.proj]
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        return z, m, m32.sum(1), x512


def prev_index(k, k0, B):
    """Index of frame max(k - LAG, 0) in [ring (LAG slots, slot = frame % LAG), batch (k0..k0+B-1)]: -> ('b', i) / ('r', s)
    / ('z', 0) (the video's frame 0 kept separately)."""
    j = max(k - LAG, 0)
    if j >= k0:
        return 'b', j - k0
    if j == 0:
        return 'z', 0
    return 'r', j % LAG


class VideoCtx:
    """Rolling per-video buffer of the last LAG frames' full token grids -> prev tokens of a batch."""

    def __init__(self, torch, G, dev):
        self.torch = torch
        self.ring = torch.zeros((LAG, G * G, D), dtype=torch.float16, device=dev)
        self.z0 = None

    def prev(self, z, k0):
        B = z.shape[0]
        P = self.torch.empty_like(z)
        if k0 == 0:
            self.z0 = z[0].clone()
        for i in range(B):
            w, j = prev_index(k0 + i, k0, B)
            P[i] = z[j] if w == 'b' else (self.z0 if w == 'z' else self.ring[j])
        for i in range(max(0, B - LAG), B):
            self.ring[(k0 + i) % LAG] = z[i]
        return P


def sae_codes(torch, saes, z, P, m):
    """Kept tokens of a batch -> per variant (codes_max (B, W) fp16, codes_mean (B, W) fp16, argmax (B, W) int16 = the
    G-grid position of the max token (lowest position on ties, -1 = silent), token L0 sum, n tokens)."""
    B = z.shape[0]
    bi, pi = m.nonzero(as_tuple=True)
    zt = z[bi, pi]
    out = {}
    nk = torch.bincount(bi, minlength=B).float()
    big = 1 << 30
    for v, (sae, norm) in saes.items():
        x = torch.cat([zt, (zt.float() - P[bi, pi].float()).half()], 1) if v == 'd2' else zt
        zz = sae.encode(norm(x.float()), mode='threshold') if len(x) else torch.zeros((0, WIDTH), device=z.device)
        cm = torch.zeros((B, WIDTH), dtype=torch.float32, device=z.device)
        cm.index_reduce_(0, bi, zz, 'amax', include_self=True)
        cs = torch.zeros_like(cm).index_add_(0, bi, zz)
        hit = (zz == cm[bi]) & (zz > 0)
        cand = torch.where(hit, pi[:, None].expand_as(zz), torch.full_like(zz, big, dtype=torch.int64))
        am = torch.full((B, WIDTH), big, dtype=torch.int64, device=z.device)
        am.scatter_reduce_(0, bi[:, None].expand_as(cand), cand, 'amin', include_self=True)
        am[am == big] = -1
        out[v] = (cm.half(), (cs / nk.clamp(min=1)[:, None]).half(), am.to(torch.int16), float((zz > 0).sum()), len(x))
    return out


def load_saes(dom, dev):
    from src.eci.sae import load_sae
    best = json.loads((sae_root(dom) / 'best.json').read_text())
    saes = {}
    for v in VARIANTS:
        sae, norm, ck = load_sae(sae_root(dom) / best[v]['name'] / 'sae.pt', dev)
        assert abs(float(sae.threshold) - best[v]['theta_star']) < 1e-6, 'theta* not baked into the checkpoint'
        saes[v] = (sae, norm)
    return saes, best


# ---------------------------------------------------------------------------------------------- check
def jpg_grey(dom, rows, A):
    from PIL import Image
    out = []
    for r in rows:
        with Image.open(REPO / 'dataset' / A['frame_path'].values[r]) as im:
            out.append(np.asarray(im.convert('L'), dtype=np.float32))
    return np.stack(out)


def cmd_check(args):
    """Alignment / seek / encoder / throughput / mask checks on 2 videos per domain."""
    import torch
    dev = torch.device('cuda')
    for dom in args.domains:
        c = DOMS[dom]
        P = load_plan(dom)
        A, R = obs_ranges_of(REPO / c['ann'], ('frame_path',))
        res = {'domain': dom}
        vids = [P['videos'][i] for i in (0, len(P['videos']) // 2)]
        pipe = Pipe(dom, dev, R)
        # 1. frame alignment: decoded frame k (resized to 512, grey) vs dataset JPG rows k-1, k, k+1
        al = []
        for v in vids:
            ks = [5, v['n'] // 3, v['n'] // 2 + 7, v['n'] - 20]
            if dom == 'ants_v3':
                ks += [2995, 3003, 6001]
            for k in ks:
                fr = decode(v['path'], voff(c, v) + c['step'] * k, 1, c['step'], c['S'], v['hflip'], c['fps'])
                x = torch.from_numpy(np.ascontiguousarray(fr)).to(dev).permute(0, 3, 1, 2).float()
                g = pipe.grey(pipe.frame512(x)).float().cpu().numpy()[0]
                rows = [v['src_lo'] + k + d for d in (-1, 0, 1)]
                J = jpg_grey(dom, rows, A)
                mad = [float(np.abs(g - j).mean()) for j in J]
                al.append({'obs': v['obs'], 'k': k, 'mad_km1_k_kp1': mad, 'ok': bool(mad[1] < min(mad[0], mad[2]))})
        res['alignment'] = al
        log(f'{dom} alignment: ' + json.dumps(al))
        # 2. seek decode == sequential decode
        v = vids[0]
        seq = decode(v['path'], voff(c, v), 40, c['step'], c['S'], v['hflip'], c['fps'])
        sk = decode(v['path'], voff(c, v) + c['step'] * 23, 10, c['step'], c['S'], v['hflip'], c['fps'])
        res['seek_equal'] = bool((seq[23:33] == sk).all())
        res['seek_max_abs_diff'] = int(np.abs(seq[23:33].astype(int) - sk.astype(int)).max())
        log(f'{dom} seek == sequential: {res["seek_equal"]} (max abs diff {res["seek_max_abs_diff"]})')
        # 3. truncated DINOv3 vs the benchmark hook encoder (fp32 reference, bf16 both)
        if args.encoder_check and dom == args.domains[0]:
            sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
            import eci_enc_encode as EE
            ref = EE.Encoder('dinov3l', c['S'], [LAYER], dev, bf16=False)
            refb = EE.Encoder('dinov3l', c['S'], [LAYER], dev, bf16=True)
            x = torch.from_numpy(np.ascontiguousarray(seq[:4])).to(dev).permute(0, 3, 1, 2).float()
            xin = (x / 255 - pipe.mean) / pipe.std
            with torch.no_grad():
                r32 = ref.forward(xin)[LAYER]
                rb = refb.forward(xin)[LAYER]
                z, _, _, _ = pipe.run(seq[:4], np.arange(v['src_lo'], v['src_lo'] + 4))
            cos = torch.nn.functional.cosine_similarity(z.float(), r32, dim=-1)
            cosb = torch.nn.functional.cosine_similarity(z.float(), rb, dim=-1)
            res['encoder_cos_vs_fp32_hook'] = {'mean': float(cos.mean()), 'min': float(cos.min()),
                                               'p01': float(cos.flatten().quantile(0.01))}
            res['encoder_cos_vs_bf16_hook'] = {'mean': float(cosb.mean()), 'min': float(cosb.min())}
            log(f'{dom} encoder check: {res["encoder_cos_vs_fp32_hook"]} | vs bf16 hook {res["encoder_cos_vs_bf16_hook"]}')
            del ref, refb
            torch.cuda.empty_cache()
        # 4. throughput + mask vs the C1 n_fg over 2 x 300 frames (streaming path, decode prefetch)
        ref = np.load(REPO / c['ref_nfg'], mmap_mode='r')
        thr = []
        for v in vids:
            n = min(300, v['n'])
            jobs = [((v['obs'], k0), dict(path=v['path'], s0=voff(c, v) + c['step'] * k0, n=min(c['chunk'] // 2, n - k0),
                                          step=c['step'], S=c['S'], hflip=v['hflip'], fps=c['fps'], threads=c['threads']))
                    for k0 in range(0, n, c['chunk'] // 2)]
            t0 = time.time()
            pipe.gpu_s = 0.0
            n32, nG = [], []
            pf = Prefetch(jobs, c['par'])
            for (o, k0), (fr, _) in pf:
                for b in range(0, len(fr), c['bs']):
                    rows = v['src_lo'] + k0 + b + np.arange(len(fr[b:b + c['bs']]))
                    z, m, c32, _ = pipe.run(fr[b:b + c['bs']], rows)
                    n32.append(c32.cpu().numpy())
                    nG.append(m.sum(1).cpu().numpy())
            wall = time.time() - t0
            n32 = np.concatenate(n32)
            nG = np.concatenate(nG)
            rf = np.asarray(ref[v['src_lo']:v['src_lo'] + n]).astype(int)
            thr.append({'obs': v['obs'], 'frames': n, 'wall_s': wall, 'fps_wall': n / wall, 'gpu_s': pipe.gpu_s,
                        'fps_gpu': n / pipe.gpu_s, 'decode_s_sum': pf.dec_s,
                        'n32_mean': float(n32.mean()), 'ref_nfg_mean': float(rf.mean()),
                        'n32_exact_frac': float((n32 == rf).mean()), 'n32_absdiff_mean': float(np.abs(n32 - rf).mean()),
                        'n32_corr': float(np.corrcoef(n32, rf)[0, 1]) if rf.std() > 0 else None,
                        'nG_mean': float(nG.mean()), 'frames_0_kept': int((nG == 0).sum())})
            log(f'{dom} stream {json.dumps(thr[-1])}')
        res['stream'] = thr
        (OUT / dom).mkdir(parents=True, exist_ok=True)
        (OUT / dom / 'check.json').write_text(json.dumps(res, indent=1))
        del pipe
        torch.cuda.empty_cache()


def cmd_maskdiag(args):
    """Why does the 32-grid mask differ from the C1 n_fg? Same 300 frames of 2 videos, four inputs to the CURRENT rule:
    frame = dataset JPG (C1's input) or native-decoded (R4), DINOv2 forward fp32 (C1) or fp16 autocast (R4)."""
    import torch
    from PIL import Image
    dev = torch.device('cuda')
    for dom in args.domains:
        c = DOMS[dom]
        P = load_plan(dom)
        A, R = obs_ranges_of(REPO / c['ann'], ('frame_path',))
        pipe = Pipe(dom, dev, R)
        ref = np.load(REPO / c['ref_nfg'], mmap_mode='r')
        res = {}
        for v in [P['videos'][i] for i in (0, len(P['videos']) // 2)]:
            n = 200
            ks = np.arange(0, n)
            fr = decode(v['path'], voff(c, v), n, c['step'], c['S'], v['hflip'], c['fps'])
            jp = np.stack([np.asarray(Image.open(REPO / 'dataset' / A.frame_path.values[v['src_lo'] + k]).convert('RGB'))
                           for k in ks])
            rows = v['src_lo'] + ks
            rf = np.asarray(ref[rows]).astype(int)
            out = {}
            for src in ('jpg', 'native'):
                for prec in ('fp32', 'fp16'):
                    cnt = []
                    for b in range(0, n, 20):
                        with torch.no_grad():
                            if src == 'jpg':
                                x512 = torch.from_numpy(jp[b:b + 20]).to(dev).permute(0, 3, 1, 2).contiguous()
                            else:
                                x = torch.from_numpy(np.ascontiguousarray(fr[b:b + 20])).to(dev).permute(0, 3, 1, 2).float()
                                x512 = pipe.frame512(x)
                            grey = pipe.grey(x512)
                            pix = pipe.proc2(images=list(x512), return_tensors='pt', device=dev)['pixel_values'].to(dev)
                            with torch.autocast('cuda', dtype=torch.float16, enabled=prec == 'fp16'):
                                h2 = pipe.m2(pixel_values=pix).last_hidden_state
                            tok2 = h2[:, 1:].float().half()
                            m32, _ = pipe.bgs.mask(tok2, grey, rows[b:b + 20])
                        cnt.append(m32.sum(1).cpu().numpy())
                    cnt = np.concatenate(cnt).astype(int)
                    out[f'{src}_{prec}'] = {'mean': float(cnt.mean()), 'ref_mean': float(rf.mean()),
                                            'exact_frac': float((cnt == rf).mean()),
                                            'absdiff_mean': float(np.abs(cnt - rf).mean())}
            res[v['obs']] = out
            log(f'{dom} {v["obs"]}: ' + json.dumps(out))
        (OUT / dom).mkdir(parents=True, exist_ok=True)
        (OUT / dom / 'maskdiag.json').write_text(json.dumps(res, indent=1))
        del pipe
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- prepass
def mask_sheet(items, G, path):
    """items: [(x512 (3, 512, 512) uint8 numpy, mask (G*G,) bool, title)] -> one JPEG (red tint = kept tokens)."""
    from PIL import Image, ImageDraw
    tiles = []
    for x, m, t in items:
        im = x.transpose(1, 2, 0).astype(np.float32)
        mm = np.kron(m.reshape(G, G), np.ones((1, 1))).astype(bool)
        up = np.array(Image.fromarray(mm.astype(np.uint8) * 255).resize((512, 512), Image.NEAREST)) > 0
        im[up] = 0.55 * im[up] + 0.45 * np.array([255, 0, 0], np.float32)
        tile = Image.fromarray(im.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(tile).text((6, 6), t, fill=(255, 255, 0))
        tiles.append(tile)
    nc = 4
    nr = int(np.ceil(len(tiles) / nc))
    sheet = Image.new('RGB', (nc * 512, nr * 512))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % nc) * 512, (i // nc) * 512))
    sheet.save(path, quality=85)


def cmd_prepass(args):
    import torch
    dom = args.domain
    check_amp(dom)
    c = DOMS[dom]
    P = load_plan(dom)
    dev = torch.device('cuda')
    A, R = obs_ranges_of(REPO / c['ann'], ('frame_path',))
    loc = local_dir() / 'prepass'
    loc.mkdir(parents=True, exist_ok=True)
    jdir = local_dir() / 'jpg'
    fp = A['frame_path'].values
    jobs = []
    for vi, v in enumerate(P['videos']):
        for a in v['bursts']:
            rr = [v['src_lo'] + a + o for o in BURST]
            jobs.append(((vi, a), dict(path=v['path'], s0=voff(c, v) + c['step'] * a, n=len(BURST),
                                       step=c['step'] * 5, S=c['S'], hflip=v['hflip'], fps=c['fps'], threads=2,
                                       jpgs=[jdir / fp[r] for r in rr])))
    if args.limit:  # self-test: bursts spread over the videos
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    st = ThreadPoolExecutor(1).submit(stage_jpgs, sorted({str(f.relative_to(jdir)) for _, kw in jobs for f in kw['jpgs']}),
                                      jdir)
    pipe = Pipe(dom, dev, R)
    G = pipe.G
    st.result()
    ftok = open(loc / 'tok.f16', 'wb')
    fprev = open(loc / 'prev.f16', 'wb')
    tok_frame, tok_pos, frames = [], [], []
    sheet, align = [], []
    t0 = time.time()
    n_tok = 0
    batch = []
    par = max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) // 2)

    def flush(batch):
        nonlocal n_tok
        fr = np.concatenate([b[1][0] for b in batch])
        jp = np.concatenate([b[1][1] for b in batch])
        rows = np.concatenate([[P['videos'][vi]['src_lo'] + a + o for o in BURST] for (vi, a), _ in batch])
        z, m, n32, x512 = pipe.run(fr, rows, jp)
        for bi, ((vi, a), _) in enumerate(batch):
            v = P['videos'][vi]
            g = np.random.default_rng([2, vi, a])
            for j in range(2, len(BURST)):
                i, ip = bi * len(BURST) + j, bi * len(BURST) + j - 2
                pos = torch.nonzero(m[i]).flatten()
                keep = pos if (v['role'] != 'train' or c['p_store'] >= 1) else \
                    pos[torch.from_numpy(g.random(len(pos)) < c['p_store']).to(dev)]
                zt, zp = z[i, keep], z[ip, keep]
                ftok.write(zt.cpu().numpy().tobytes())
                fprev.write(zp.cpu().numpy().tobytes())
                fid = len(frames)
                tok_frame.append(np.full(len(keep), fid, np.int32))
                tok_pos.append(keep.cpu().numpy().astype(np.int16))
                k = a + BURST[j]
                frames.append({'video': vi, 'obs': v['obs'], 'k': k, 'src_row': v['src_lo'] + k,
                               'lay_row': v['lay_lo'] + k, 'role': v['role'], 'n_kept': int(len(pos)),
                               'n_stored': int(len(keep)), 'n32': int(n32[i])})
                n_tok += len(keep)
                if v['role'] == 'eval' and len(sheet) < 12 and (not sheet or sheet[-1][2].split()[0] != v['obs']) \
                        and j == 2:
                    sheet.append((x512[i].cpu().numpy(), m[i].cpu().numpy(), f'{v["obs"]} k{k} kept {len(pos)}'))
                if len(align) < 12 and j == 2 and bi == 0:
                    xd = torch.from_numpy(np.ascontiguousarray(fr[i:i + 1])).to(dev).permute(0, 3, 1, 2).float()
                    gq = pipe.grey(pipe.frame512(xd)).float().cpu().numpy()[0]  # the DECODED frame vs its JPGs
                    J = jpg_grey(dom, [v['src_lo'] + k + d for d in (-1, 0, 1)], A)
                    mad = [float(np.abs(gq - jj).mean()) for jj in J]
                    align.append({'obs': v['obs'], 'k': int(k), 'mad': mad, 'ok': bool(mad[1] <= min(mad[0], mad[2]))})

    pf = Prefetch(jobs, par)
    per = max(1, (24 if c['S'] <= 832 else 16) // len(BURST))
    for i, (key, fr) in enumerate(pf):
        batch.append((key, fr))
        if len(batch) == per:
            flush(batch)
            batch = []
        if i % 500 == 0:
            log(f'{dom} prepass: {i}/{len(jobs)} bursts, {n_tok:,} tokens, {time.time() - t0:.0f}s '
                f'(GPU {pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f})')
    if batch:
        flush(batch)
    ftok.close()
    fprev.close()
    F = pd.DataFrame(frames)
    F.to_parquet(loc / 'frames.parquet')
    np.save(loc / 'tok_frame.npy', np.concatenate(tok_frame))
    np.save(loc / 'tok_pos.npy', np.concatenate(tok_pos))
    ok = sum(a['ok'] for a in align)
    ref = np.load(REPO / c['ref_nfg'], mmap_mode='r')
    rf = np.asarray(ref[F['src_row'].values]).astype(int)
    n32 = F['n32'].values
    info = {'domain': dom, 'S': c['S'], 'G': G, 'n_bursts': len(jobs), 'n_frames': len(F), 'n_tokens': n_tok,
            'frames_by_role': F.role.value_counts().to_dict(), 'tokens_by_role': F.groupby('role').n_stored.sum().to_dict(),
            'kept_per_frame': {'mean': float(F.n_kept.mean()), 'median': float(F.n_kept.median()),
                               'p10': float(F.n_kept.quantile(0.1)), 'p90': float(F.n_kept.quantile(0.9)),
                               'frames_0_kept': int((F.n_kept == 0).sum())},
            'n32_vs_c1_nfg': {'exact_frac': float((n32 == rf).mean()), 'absdiff_mean': float(np.abs(n32 - rf).mean()),
                              'frac_absdiff_gt5': float((np.abs(n32 - rf) > 5).mean()),
                              'corr': float(np.corrcoef(n32, rf)[0, 1]), 'n32_mean': float(n32.mean()),
                              'ref_mean': float(rf.mean())},
            'alignment_vs_jpg': {'ok': ok, 'n': len(align), 'items': align},
            'wall_s': time.time() - t0, 'gpu_s': pipe.gpu_s, 'decode_thread_s': pf.dec_s,
            'gpu': torch.cuda.get_device_name(0), 'amp': AMP}
    log(json.dumps({k: v for k, v in info.items() if k != 'alignment_vs_jpg'}))
    log(f'alignment vs JPG: {ok}/{len(align)} frames closest to their own JPG ' + json.dumps([a['mad'] for a in align]))
    (OUT / dom).mkdir(parents=True, exist_ok=True)
    if sheet:
        mask_sheet(sheet, G, OUT / dom / 'mask_sheet.jpg')
    (OUT / dom / 'prepass.json').write_text(json.dumps(info, indent=1, default=float))
    if ok < len(align) / 2:
        raise SystemExit(f'ALIGNMENT FAILED: only {ok}/{len(align)} decoded frames closest to their JPG')
    if True:
        atomic_copy(loc, bee(dom) / 'prepass')
        log(f'-> {bee(dom) / "prepass"}')


# ---------------------------------------------------------------------------------------------- pool (ants)
def cmd_pool(args):
    """r4 pooled ants training set (CPU): from the per-series prepass pools (ants_v2: v2; ants_v3: v3 t=2/6/8; ants_v4:
    v4 + v4_infested t=2/6), per experiment and role the SAME number of frames, the smallest experiment's count
    (uniform seeded draw; roles stay those of the series plans, i.e. held-out frames come from held-out videos of every
    experiment). All stored tokens of a drawn frame are kept. -> BEE/ants_pooled/prepass/, OUT/ants_pooled/prepass.json"""
    E = pd.read_csv(REPO / 'dataset/ants/eci30/experiment.csv').set_index('observation_id')['experiment']
    parts = []
    for b in ('ants_v2', 'ants_v3', 'ants_v4'):
        F = pd.read_parquet(BEE / b / 'prepass' / 'frames.parquet')
        F['series'], F['experiment'], F['frame_in_series'] = b, E.loc[F.obs].values, np.arange(len(F))
        F['amp'] = json.loads((OUT / b / 'prepass.json').read_text()).get('amp', 'bf16')
        parts.append(F)
    A = pd.concat(parts, ignore_index=True)
    cnt = A.groupby(['experiment', 'role']).size().unstack()
    m = cnt.min(0)
    log('frames per experiment x role:\n' + cnt.to_string() + f'\ndrawn per experiment: {m.to_dict()}')
    g = np.random.default_rng(0)
    keep = np.concatenate([g.choice(np.flatnonzero((A.experiment == ex).values & (A.role == r).values), int(m[r]),
                                    replace=False) for ex in sorted(A.experiment.unique()) for r in m.index])
    keep.sort()
    S = A.iloc[keep].reset_index(drop=True)
    loc = local_dir() / 'pool'
    loc.mkdir(parents=True, exist_ok=True)
    ft, fp_ = open(loc / 'tok.f16', 'wb'), open(loc / 'prev.f16', 'wb')
    tok_frame, tok_pos, n_tok = [], [], 0
    for b in ('ants_v2', 'ants_v3', 'ants_v4'):
        pre = BEE / b / 'prepass'
        tf = np.load(pre / 'tok_frame.npy')
        tp = np.load(pre / 'tok_pos.npy')
        N = len(tf)
        tok = np.memmap(pre / 'tok.f16', np.float16, 'r', shape=(N, D))
        prv = np.memmap(pre / 'prev.f16', np.float16, 'r', shape=(N, D))
        sub = S[S.series == b]
        starts = np.r_[0, np.cumsum(np.bincount(tf, minlength=int(tf.max()) + 1))]
        for new_id, fi in zip(sub.index.values, sub.frame_in_series.values):
            a_, e_ = starts[fi], starts[fi + 1]
            assert (tf[a_:e_] == fi).all()
            ft.write(np.asarray(tok[a_:e_]).tobytes())
            fp_.write(np.asarray(prv[a_:e_]).tobytes())
            tok_frame.append(np.full(e_ - a_, new_id, np.int32))
            tok_pos.append(tp[a_:e_])
            n_tok += e_ - a_
        log(f'{b}: {len(sub)} frames gathered')
    ft.close()
    fp_.close()
    # token order must follow frame order for the readers: sort the per-series blocks by frame id
    tfr = np.concatenate(tok_frame)
    order = np.argsort(tfr, kind='stable')
    if not (order == np.arange(len(order))).all():
        tokm = np.memmap(loc / 'tok.f16', np.float16, 'r', shape=(n_tok, D))
        prvm = np.memmap(loc / 'prev.f16', np.float16, 'r', shape=(n_tok, D))
        for nm_, src in (('tok2.f16', tokm), ('prev2.f16', prvm)):
            with open(loc / nm_, 'wb') as fo:
                for a_ in range(0, n_tok, 500_000):
                    fo.write(np.asarray(src[order[a_:a_ + 500_000]]).tobytes())
        del tokm, prvm
        os.replace(loc / 'tok2.f16', loc / 'tok.f16')
        os.replace(loc / 'prev2.f16', loc / 'prev.f16')
    np.save(loc / 'tok_frame.npy', tfr[order])
    np.save(loc / 'tok_pos.npy', np.concatenate(tok_pos)[order])
    S['n_stored'] = np.bincount(tfr, minlength=len(S))
    S.drop(columns=['frame_in_series']).to_parquet(loc / 'frames.parquet')
    info = {'domain': 'ants_pooled', 'n_frames': int(len(S)), 'n_tokens': int(n_tok),
            'frames_by_experiment_role': {f'{ex}|{r}': int(v) for (ex, r), v in S.groupby(['experiment', 'role']).size().items()},
            'tokens_by_experiment': {k_: int(v) for k_, v in S.groupby('experiment').n_stored.sum().items()},
            'available_frames_by_experiment_role': {f'{ex}|{r}': int(v) for (ex, r), v in cnt.stack().items()},
            'amp': 'mixed: ' + ', '.join(f'{b} {a}' for b, a in S.groupby('series').amp.first().items()),
            'sources': [str(BEE / b / 'prepass') for b in ('ants_v2', 'ants_v3', 'ants_v4')],
            'note': 'v2 has treatments 1 and 2 only (all 44 valid videos used); v3 t = 2, 6, 8; v4 / v4_infested t = 2, 6. '
                    'v3 prepass tokens were decoded at source offset 1 (33 ms early; irrelevant for a training sample).'}
    (OUT / 'ants_pooled').mkdir(parents=True, exist_ok=True)
    (OUT / 'ants_pooled' / 'prepass.json').write_text(json.dumps(info, indent=1))
    atomic_copy(loc, BEE / 'ants_pooled' / 'prepass')
    log(json.dumps(info))


# ---------------------------------------------------------------------------------------------- train
def train_one_r4(arm, cfg, seed, X, steps, out, dev, n_init=500_000, bs=4096, lr=5e-4, warmup=500):
    """scripts/benchmark/mabe22/eci_train.py train_one (token mode), unchanged except the init sample size n_init
    (TokenNorm fit + geometric-median b_dec; 500k there, smaller on the 11 GB debug GPUs) and a chunked init gather."""
    import torch
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    N, din = X.shape
    ii = torch.from_numpy(np.sort(g.choice(N, min(n_init, N), replace=False))).to(dev)
    init = X[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [D] * (din // D), seed=seed)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=WIDTH, prefixes=PREFIXES, k=K, seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    torch.cuda.empty_cache()
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm, p = torch.randperm(N, generator=gen, device=dev), 0
    history, t0, ntok = [], time.time(), 0
    for step in range(steps):
        if p + bs > N:
            perm, p = torch.randperm(N, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        ntok += len(b)
        logs, gn = _train_step(sae, opt, norm(X[b]), step, steps, lr, warmup, 1.0)
        if step % 1000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1), batch_tokens=int(len(b)))
            history.append(logs)
            log(f"    {arm} s{seed} step {step}/{steps} fve {logs['fve']:.4f} l0 {logs['l0']:.2f} dead {logs['n_dead']} "
                f"thr {logs['threshold']:.3f} {logs['elapsed_s']}s")
    sae.eval()
    dead = int((sae.tokens_since_fired >= sae.dead_tokens).sum())
    extra = {'config': 'r4_rollout', 'arm': arm, 'cfg': cfg, 'seed': seed, 'n_init': int(min(n_init, N))}
    save_checkpoint(out / 'sae.pt', sae, norm, extra=extra)
    info = {**extra, 'n_pool_tokens': N, 'steps': steps, 'token_presentations': ntok, 'train_time_s': time.time() - t0,
            'dead_at_end_train': dead, 'threshold_ema': float(sae.threshold), 'gpu': torch.cuda.get_device_name(0),
            'history': history}
    (out / 'train.json').write_text(json.dumps(info, indent=1, default=float))
    log(f'    {arm} s{seed}: trained in {time.time() - t0:.0f}s, final fve {history[-1]["fve"]:.4f}, dead {dead}')


def cmd_train(args):
    import torch
    sys.path.insert(0, str(REPO / 'scripts/benchmark/mabe22'))
    import eci_train as ET
    from src.eci.sae import MatryoshkaBatchTopKSAE, load_sae, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    dom = args.domain
    dev = torch.device('cuda')
    src = bee(dom) / 'prepass'
    loc = local_dir() / 'prepass'
    t0 = time.time()
    atomic_copy(src, loc)
    log(f'staged prepass in {time.time() - t0:.0f}s')
    F = pd.read_parquet(loc / 'frames.parquet')
    tf = np.load(loc / 'tok_frame.npy')
    N = len(tf)
    tok = np.memmap(loc / 'tok.f16', np.float16, 'r', shape=(N, D))
    prv = np.memmap(loc / 'prev.f16', np.float16, 'r', shape=(N, D))
    role = F.role.values[tf]
    lens_all = F.n_stored.values
    log(f'{dom}: {N:,} tokens of {len(F):,} frames; by role ' + json.dumps(pd.Series(role).value_counts().to_dict()))
    ET.D, ET.WIDTH = D, WIDTH

    def build(cfg_, din, seed):
        return MatryoshkaBatchTopKSAE(d_in=din, n_latents=WIDTH, prefixes=PREFIXES, k=K, seed=seed)
    ET.build = build
    out_root = sae_root(dom)
    res = {'domain': dom, 'n_tokens': {r: int((role == r).sum()) for r in ('train', 'calib', 'eval')},
           'n_frames': {r: int((F.role == r).sum()) for r in ('train', 'calib', 'eval')}, 'variants': {}}
    best = {}
    for v in VARIANTS:
        def X_of(r, cap=0):
            sel = np.flatnonzero(role == r)
            if cap and len(sel) > cap:  # GPU memory (11 GB debug GPUs): uniform seeded subsample of the pool
                sel = np.sort(np.random.default_rng(0).choice(sel, cap, replace=False))
                log(f'{dom} {v}: {r} pool capped to {cap:,} of {int((role == r).sum()):,} tokens (seed 0)')
            X = torch.empty((len(sel), D * (2 if v == 'd2' else 1)), dtype=torch.float16, device=dev)
            for a in range(0, len(sel), 200_000):
                s_ = sel[a:a + 200_000]
                t = torch.from_numpy(np.ascontiguousarray(tok[s_])).to(dev)
                X[a:a + len(s_), :D] = t
                if v == 'd2':
                    p_ = torch.from_numpy(np.ascontiguousarray(prv[s_])).to(dev)
                    X[a:a + len(s_), D:] = (t.float() - p_.float()).half()
                    del p_
                del t
            torch.cuda.empty_cache()
            return X
        cfg = dict(sp='btk', k=K, matry=True, prefixes=list(PREFIXES), variant=v, lag=LAG if v == 'd2' else None,
                   encoder=D3L, layer=LAYER, S=DOMS[dom]['S'], rule=DOMS[dom]['rule'], domain=dom)
        Xtr = X_of('train', args.cap)
        ltr = F.n_stored.values[F.role.values == 'train']
        if args.cap:
            res['train_pool_cap'] = args.cap
        torch.backends.cuda.matmul.allow_tf32 = True
        for seed in SEEDS:
            od = out_root / f'{v}_s{seed}'
            if (od / 'sae.pt').exists() and not args.force:
                log(f'[SKIP] {od}')
                continue
            train_one_r4(f'{dom}_{v}', cfg, seed, Xtr, args.steps, od, dev, n_init=args.n_init)
            torch.cuda.empty_cache()
        del Xtr
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        Xca, Xev = X_of('calib'), X_of('eval')
        R = {}
        for seed in SEEDS:
            od = out_root / f'{v}_s{seed}'
            sae, norm, ck = load_sae(od / 'sae.pt', dev)
            ema = float(sae.threshold)
            th, info = calibrate_threshold(sae, Xca, norm=norm)
            set_threshold(sae, th)
            st = sparsity_stats(sae, Xev, th, norm=norm)
            fire = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
            sse = {p: 0.0 for p in PREFIXES}
            s1 = torch.zeros(Xev.shape[1], dtype=torch.float64, device=dev)
            s2 = 0.0
            with torch.no_grad():
                for a in range(0, len(Xev), 65536):
                    x = norm(Xev[a:a + 65536].float())
                    zz = sae.encode(x, mode='threshold')
                    fire += (zz > 0).sum(0).double()
                    s1 += x.double().sum(0)
                    s2 += float(x.double().pow(2).sum())
                    for p in PREFIXES:
                        sse[p] += float((sae.decode(zz, p) - x).double().pow(2).sum())
            n = len(Xev)
            tss = s2 - float(s1.pow(2).sum()) / n
            fr = (fire / n).cpu().numpy()
            R[seed] = {'theta_star': th, 'theta_ema': ema, 'calib': {k_: v_ for k_, v_ in info.items()
                                                                      if not isinstance(v_, (list, np.ndarray))},
                       'eval_l0_per_token': st['l0'], 'eval_fve': st['fve'],
                       'eval_fve_by_prefix': {str(p): 1 - sse[p] / tss for p in PREFIXES},
                       'eval_mse_norm': sse[WIDTH] / n, 'eval_dead_frac': float((fr == 0).mean()),
                       'eval_dead_by_prefix': {str(p): float((fr[:p] == 0).mean()) for p in PREFIXES},
                       'eval_fire_rate_median': float(np.median(fr)), 'eval_fire_rate_max': float(fr.max())}
            # theta* baked into a copy of the checkpoint (state_dict 'threshold')
            ck['state_dict']['threshold'] = torch.tensor(float(th))
            ck['theta_star'] = float(th)
            ck['theta_info'] = R[seed]
            torch.save(ck, od / 'sae_theta.pt')
            log(f'{dom} {v} s{seed}: theta* {th:.4f} (EMA {ema:.4f}) | eval L0 {st["l0"]:.2f} FVE {st["fve"]:.4f} '
                f'dead {R[seed]["eval_dead_frac"]:.3f} FVE by prefix {json.dumps(R[seed]["eval_fve_by_prefix"])}')
        b = max(SEEDS, key=lambda s: R[s]['eval_fve'])
        name = sae_name(v, b)
        nd = out_root / name
        nd.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_root / f'{v}_s{b}' / 'sae_theta.pt', nd / 'sae.pt')
        shutil.copyfile(out_root / f'{v}_s{b}' / 'train.json', nd / 'train.json')
        metrics = {'args': {'domain': dom, 'tokens_dir': str(bee(dom) / 'prepass'), 'n_latents': WIDTH,
                            'prefixes': ','.join(map(str, PREFIXES)), 'k': K, 'seed': b, 'variant': v, 'lag': LAG,
                            'encoder': D3L, 'layer': LAYER, 'S': DOMS[dom]['S'], 'fg_rule': DOMS[dom]['rule']},
                   'selection': 'seed with the highest FVE on the held-out evaluation videos (no labels)',
                   'amp': json.loads(prepass_json(dom).read_text()).get('amp', 'bf16'),
                   'train_pool_cap': args.cap or None,
                   'seeds': R, 'chosen_seed': b}
        (nd / 'metrics.json').write_text(json.dumps(metrics, indent=1, default=float))
        best[v] = {'name': name, 'seed': b, 'theta_star': R[b]['theta_star']}
        res['variants'][v] = {'seeds': R, 'chosen': b, 'name': name}
        del Xca, Xev
        torch.cuda.empty_cache()
    (out_root / 'best.json').write_text(json.dumps(best, indent=1))
    (OUT / dom / 'train.json').write_text(json.dumps(res, indent=1, default=float))
    log(f'{dom}: best {json.dumps(best)} ({time.time() - t0:.0f}s)')


# ---------------------------------------------------------------------------------------------- encode
def cmd_encode(args):
    import torch
    dom = args.domain
    check_amp(dom)
    c = DOMS[dom]
    P = load_plan(dom)
    pre_amp = json.loads(prepass_json(dom).read_text()).get('amp', 'bf16')
    if pre_amp != AMP:
        raise SystemExit(f'{dom}: prepass tokens are {pre_amp}, this encode would be {AMP}')
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    _, R = obs_ranges_of(REPO / c['ann'])
    outdir = Path(args.out_dir) if args.out_dir else bee(dom) / 'videos'
    outdir.mkdir(parents=True, exist_ok=True)
    vids = P['shards'][args.shard] if args.videos is None else [int(x) for x in args.videos.split(',')]
    todo = [vi for vi in vids if not (outdir / f'{P["videos"][vi]["obs"]}.npz').exists()]
    log(f'{dom} shard {args.shard}: {len(vids)} videos, {len(todo)} to do -> {outdir}')
    if not todo:
        return
    loc = local_dir() / 'enc'
    loc.mkdir(parents=True, exist_ok=True)
    jdir = local_dir() / 'jpg'
    fp = pd.read_csv(REPO / c['ann'], usecols=['frame_path'])['frame_path'].values
    jobs, rels = [], []
    for vi in todo:
        v = P['videos'][vi]
        n = v['n'] if not args.max_frames else min(v['n'], args.max_frames)
        rels += list(fp[v['src_lo']:v['src_lo'] + n])
        for k0 in range(0, n, c['chunk']):
            m_ = min(c['chunk'], n - k0)
            jobs.append(((vi, k0, n), dict(path=v['path'], s0=voff(c, v) + c['step'] * k0, n=m_,
                                           step=c['step'], S=c['S'], hflip=v['hflip'], fps=c['fps'],
                                           threads=c['threads'],
                                           jpgs=[jdir / f for f in fp[v['src_lo'] + k0:v['src_lo'] + k0 + m_]])))
    st = ThreadPoolExecutor(1).submit(stage_jpgs, rels, jdir)
    pipe = Pipe(dom, dev, R)
    saes, best = load_saes(dom, dev)
    G = pipe.G
    st.result()
    pf = Prefetch(jobs, c['par'])
    T0 = time.time()
    cur, acc, ctx, tv = None, None, None, None
    tot_frames = 0

    def finish(vi, acc, tv):
        v = P['videos'][vi]
        arr = {k_: np.concatenate(x) for k_, x in acc.items() if k_ not in ('l0', 'ntok')}
        n = len(arr['n_fg'])
        f = loc / f'{v["obs"]}.npz'
        np.savez(f, **arr, l0=np.array([acc['l0'][v_] for v_ in VARIANTS]),
                 ntok=np.array([acc['ntok'][v_] for v_ in VARIANTS]), obs=v['obs'], n=n)
        atomic_copy(f, outdir / f.name)
        f.unlink()
        dt = time.time() - tv
        fl0 = {v_: float((arr[f'max_{v_}'] > 0).sum(1).mean()) for v_ in VARIANTS}
        tl0 = {v_: acc['l0'][v_] / max(acc['ntok'][v_], 1) for v_ in VARIANTS}
        log(f'{dom} {v["obs"]} ({v["role"]}): {n} frames in {dt:.0f}s ({n / dt:.1f} fps) kept/frame '
            f'{arr["n_fg"].mean():.0f} | token L0 {json.dumps({k_: round(x, 2) for k_, x in tl0.items()})} | frame L0 '
            f'{json.dumps({k_: round(x, 1) for k_, x in fl0.items()})} | 0-kept frames {(arr["n_fg"] == 0).sum()}')

    for (vi, k0, n), (fr, jp) in pf:
        v = P['videos'][vi]
        if vi != cur:
            if cur is not None:
                finish(cur, acc, tv)
            cur, tv = vi, time.time()
            ctx = VideoCtx(torch, G, dev)
            acc = {'n_fg': [], 'n_fg32': [], 'l0': {v_: 0.0 for v_ in VARIANTS}, 'ntok': {v_: 0 for v_ in VARIANTS},
                   **{f'{p}_{v_}': [] for p in ('max', 'mean', 'arg') for v_ in VARIANTS}}
        for b in range(0, len(fr), c['bs']):
            kb = k0 + b
            x = fr[b:b + c['bs']]
            rows = v['src_lo'] + kb + np.arange(len(x))
            z, m, n32, _ = pipe.run(x, rows, jp[b:b + c['bs']])
            Pv = ctx.prev(z, kb)
            with torch.no_grad():
                out = sae_codes(torch, saes, z, Pv, m)
            acc['n_fg'].append(m.sum(1).to(torch.int16).cpu().numpy())
            acc['n_fg32'].append(n32.to(torch.int16).cpu().numpy())
            for v_, (cm, cmean, am, l0, nt) in out.items():
                acc[f'max_{v_}'].append(cm.cpu().numpy())
                acc[f'mean_{v_}'].append(cmean.cpu().numpy())
                acc[f'arg_{v_}'].append(am.cpu().numpy())
                acc['l0'][v_] += l0
                acc['ntok'][v_] += nt
        tot_frames += len(fr)
    if cur is not None:
        finish(cur, acc, tv)
    wall = time.time() - T0
    log(f'{dom} shard {args.shard} DONE: {tot_frames:,} frames in {wall:.0f}s ({tot_frames / wall:.1f} fps wall), '
        f'GPU busy (pipe) {pipe.gpu_s:.0f}s, decode thread-s {pf.dec_s:.0f}, {torch.cuda.get_device_name(0)}')


# ---------------------------------------------------------------------------------------------- merge
def cmd_merge(args):
    dom = args.domain
    c = DOMS[dom]
    P = load_plan(dom)
    vdir = bee(dom) / 'videos'
    best = json.loads((sae_root(dom) / 'best.json').read_text())
    N = P['n_frames']
    missing = [v['obs'] for v in P['videos'] if not (vdir / f'{v["obs"]}.npz').exists()]
    if missing:
        raise SystemExit(f'{dom}: {len(missing)} videos not encoded, e.g. {missing[:5]}')
    eci = REPO / c['eci']
    stats = {}
    for v_ in VARIANTS:
        name = best[v_]['name']
        cd = bee(dom) / ('layout/codes' if c['layout'] else 'codes') / name
        tmp = cd.with_name(name + '.tmp')
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        mx = np.lib.format.open_memmap(tmp / 'codes_max.npy', 'w+', np.float16, (N, WIDTH))
        mn = np.lib.format.open_memmap(tmp / 'codes_mean.npy', 'w+', np.float16, (N, WIDTH))
        am = np.lib.format.open_memmap(tmp / 'argmax.npy', 'w+', np.int16, (N, WIDTH))
        nf = np.zeros(N, np.int16)
        n32 = np.zeros(N, np.int16)
        l0, ntok = 0.0, 0
        fire = np.zeros(WIDTH, np.int64)
        for v in P['videos']:
            z = np.load(vdir / f'{v["obs"]}.npz')
            assert int(z['n']) == v['n'], (v['obs'], int(z['n']), v['n'])
            s = slice(v['lay_lo'], v['lay_lo'] + v['n'])
            mx[s], mn[s], am[s] = z[f'max_{v_}'], z[f'mean_{v_}'], z[f'arg_{v_}']
            nf[s], n32[s] = z['n_fg'], z['n_fg32']
            j = VARIANTS.index(v_)
            l0 += float(z['l0'][j])
            ntok += int(z['ntok'][j])
            fire += (z[f'max_{v_}'] > 0).sum(0)
        nonfinite = int(sum((~np.isfinite(mx[a:a + 200_000].astype(np.float32))).any(1).sum()
                            for a in range(0, N, 200_000)))
        mx.flush(), mn.flush(), am.flush()
        del mx, mn, am
        np.save(tmp / 'n_fg.npy', nf)
        np.save(tmp / 'n_fg32.npy', n32)
        cfg = {'sae': name, 'domain': dom, 'n_rows': N, 'width': WIDTH, 'layout_ann': str(REPO / c['eci'] / 'annotations.csv')
               if c['layout'] else str(REPO / c['ann']), 'backgrounds': str(REPO / c['bg']), 'align': 'none',
               'fg_rule': c['rule'], 'encoder': D3L, 'layer': LAYER, 'S': c['S'], 'G': c['S'] // 16, 'variant': v_,
               'lag': LAG if v_ == 'd2' else None, 'theta_star': best[v_]['theta_star'],
               'files': {'codes_max': 'max over kept tokens under theta*', 'codes_mean': 'mean over kept tokens',
                         'argmax': 'G-grid position of the max token (-1 = silent)', 'n_fg': 'kept G-grid tokens',
                         'n_fg32': 'kept 32-grid cells (comparable with C1 n_fg)'},
               'recipe': 'scripts/eci/rollout_r4.py',
               'amp': json.loads(prepass_json(dom).read_text()).get('amp', 'bf16')}
        (tmp / 'config.json').write_text(json.dumps(cfg, indent=1))
        st = {'n_frames': N, 'rows_nonfinite': nonfinite, 'token_l0_mean': l0 / max(ntok, 1), 'n_tokens': ntok,
              'kept_per_frame_mean': float(nf.mean()), 'frames_0_kept_frac': float((nf == 0).mean()),
              'neuron_frame_fire_rate': {'dead_frac': float((fire == 0).mean()),
                                         'median': float(np.median(fire / N)), 'max': float((fire / N).max())}}
        (tmp / 'verify.json').write_text(json.dumps(st, indent=1))
        if nonfinite:
            raise SystemExit(f'{dom} {v_}: {nonfinite} non-finite rows')
        (tmp / 'DONE').touch()
        if cd.exists():
            shutil.rmtree(cd)
        tmp.rename(cd)
        # links: codes root and sae root of the domain
        if not c['layout']:
            link = eci / 'codes' / name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(cd, link)
        sl = eci / 'sae' / name
        if not (sl.is_symlink() or sl.exists()):
            os.symlink(sae_root(dom) / name, sl)
        stats[v_] = st
        log(f'{dom} {v_} merged -> {cd}: {json.dumps(st)}')
    (OUT / dom / 'merge.json').write_text(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------------------------------- summaries
def codes_dir_of(dom, name):
    c = DOMS[dom]
    return bee(dom) / ('layout/codes' if c['layout'] else 'codes') / name


def exact_theta(vals, n_units, k=K):
    """sae_calibrate's exact rule on a flat array of values: the (n_units k + 1)-th largest value, so that the mean count
    per unit of values > theta is k (fewer only on a tie at the cut)."""
    v = np.sort(np.asarray(vals, np.float32).ravel())[::-1]
    j = int(n_units * k)
    th = float(v[j])
    return th, {'n_units': int(n_units), 'n_values': int(len(v)), 'tie_at_cut': int((v == th).sum()),
                'mean_count_above': float((v > th).sum() / n_units)}


def rule_stats(fire, n_frames, top=16):
    """fire: (W,) firing frame counts of each neuron over n_frames -> participation numbers."""
    r = fire / max(n_frames, 1)
    o = np.sort(fire)[::-1]
    return {'fires_per_frame': float(fire.sum() / max(n_frames, 1)), 'neurons_ever_firing': int((fire > 0).sum()),
            'neurons_firing_ge_1pct': int((r >= 0.01).sum()), 'top16_share': float(o[:top].sum() / max(o.sum(), 1)),
            'median_neuron_rate': float(np.median(r)), 'max_neuron_rate': float(r.max())}


def write_rule_codes(cd, name, suffix, mx, thr_j, desc, c):
    """Thresholded codes set <name>_<suffix> for the unchanged NES runners: codes_max = value where value > thr_j (per
    neuron, (W,)), else 0; n_fg copied; symlinked into the domain's codes / sae roots."""
    fd = cd.with_name(f'{name}_{suffix}')
    tmp = fd.with_name(fd.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    Nr = mx.shape[0]
    y = np.lib.format.open_memmap(tmp / 'codes_max.npy', 'w+', np.float16, (Nr, WIDTH))
    for a_ in range(0, Nr, 200_000):
        blk = np.asarray(mx[a_:a_ + 200_000])
        y[a_:a_ + 200_000] = np.where(blk.astype(np.float32) > thr_j[None], blk, 0)
    y.flush()
    del y
    shutil.copyfile(cd / 'n_fg.npy', tmp / 'n_fg.npy')
    cfg = json.loads((cd / 'config.json').read_text())
    cfg.update({'sae': fd.name, 'parent_codes': str(cd), 'rule': desc,
                'files': {'codes_max': f'codes_max of {name} where it fires under the rule, else 0',
                          'n_fg': 'copied from the parent'}})
    (tmp / 'config.json').write_text(json.dumps(cfg, indent=1))
    np.save(tmp / 'threshold_per_neuron.npy', thr_j.astype(np.float32))
    (tmp / 'DONE').touch()
    if fd.exists():
        shutil.rmtree(fd)
    tmp.rename(fd)
    links = [(REPO / c['eci'] / 'sae' / fd.name, sae_root(c['dom']) / name)]
    if not c['layout']:
        links.append((REPO / c['eci'] / 'codes' / fd.name, fd))
    for l_, t_ in links:
        if not (l_.is_symlink() or l_.exists()):
            os.symlink(t_, l_)
    return fd


def cmd_ftheta(args):
    """Frame-level firing rules per domain and variant (label-free, one global parameter each), on the SAE's held-out
    prepass frames (CALIBRATION videos calibrate, EVALUATION videos check):
      fnorm  (PRIMARY) scale-normalised frame theta: s_j = median of neuron j's non-zero token activations under theta*
             on the calibration tokens (never-active neurons: the median of all s_j); v_j = codes_max_j / s_j; theta_n =
             the (N_frames k + 1)-th largest v over the calibration frames (k = 16); fires iff v_j > theta_n.
      ftheta global frame theta: theta_f = the (N_frames k + 1)-th largest codes_max over the calibration frames; fires
             iff codes_max_j > theta_f.
      gt0    fires iff codes_max_j > 0 (theta* only).
      top16  (reported only) the 16 largest v_j of each frame fire.
    Writes the thresholded codes sets <sae>_fnorm and <sae>_ftheta for the unchanged NES runners -> OUT/<dom>/ftheta.json.
    CPU."""
    import torch
    from src.eci.sae import load_sae
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    dom = args.domain
    c = {**DOMS[dom], 'dom': dom}
    best = json.loads((sae_root(dom) / 'best.json').read_text())
    out = {'domain': dom, 'k': K, 'primary_rule': 'fnorm', 'variants': {}}
    if c.get('pooled'):  # a series of the pooled ants SAE: thresholds calibrated once on the pooled prepass
        FT = json.loads((OUT / 'ants_pooled' / 'ftheta.json').read_text())
        out['calibrated_on'] = 'ants_pooled (held-out calibration frames of every ants experiment)'
        for v_ in VARIANTS:
            sj = np.load(OUT / 'ants_pooled' / f'scale_{v_}.npy').astype(np.float64)
            R = {k_: v for k_, v in FT['variants'][v_].items() if k_ not in ('fullpass', 'codes')}
            out['variants'][v_] = apply_rules(dom, c, best[v_]['name'], sj, R)
        (OUT / dom / 'ftheta.json').write_text(json.dumps(out, indent=1))
        return
    pre = bee(dom) / 'prepass'
    F = pd.read_parquet(pre / 'frames.parquet')
    tf = np.load(pre / 'tok_frame.npy')
    N = len(tf)
    tok = np.memmap(pre / 'tok.f16', np.float16, 'r', shape=(N, D))
    prv = np.memmap(pre / 'prev.f16', np.float16, 'r', shape=(N, D))
    held = np.flatnonzero(F.role.isin(['calib', 'eval']).values)
    sel = np.flatnonzero(np.isin(tf, held))
    fr_of = tf[sel]
    is_cal_tok = F.role.values[fr_of] == 'calib'
    out['frames'] = {r: int((F.role == r).sum()) for r in ('calib', 'eval')}
    cal = F.role.values == 'calib'
    ev = F.role.values == 'eval'
    for v_ in VARIANTS:
        sae, norm, _ = load_sae(sae_root(dom) / best[v_]['name'] / 'sae.pt', 'cpu')
        fvt = torch.zeros((len(F), WIDTH), dtype=torch.float32)
        nz_j, nz_v = [], []
        with torch.no_grad():
            for a_ in range(0, len(sel), 100_000):
                s_ = sel[a_:a_ + 100_000]
                t = torch.from_numpy(np.ascontiguousarray(tok[s_])).float()
                x = torch.cat([t, (t - torch.from_numpy(np.ascontiguousarray(prv[s_])).float()).half().float()], 1) \
                    if v_ == 'd2' else t
                z = sae.encode(norm(x), mode='threshold')
                fvt.index_reduce_(0, torch.from_numpy(fr_of[a_:a_ + 100_000].astype(np.int64)), z, 'amax',
                                  include_self=True)
                zc = z[torch.from_numpy(is_cal_tok[a_:a_ + 100_000])]
                ii, jj = torch.nonzero(zc, as_tuple=True)
                nz_j.append(jj.numpy())
                nz_v.append(zc[ii, jj].numpy())
        fv = fvt.numpy()
        nz_j, nz_v = np.concatenate(nz_j), np.concatenate(nz_v)
        o = np.argsort(nz_j, kind='stable')
        nz_j, nz_v = nz_j[o], nz_v[o]
        cuts = np.searchsorted(nz_j, np.arange(WIDTH + 1))
        sj = np.array([np.median(nz_v[cuts[j]:cuts[j + 1]]) if cuts[j + 1] > cuts[j] else np.nan for j in range(WIDTH)],
                      np.float64)
        n_never = int(np.isnan(sj).sum())
        sj[np.isnan(sj)] = np.nanmedian(sj)
        th_f, info_f = exact_theta(fv[cal], cal.sum())
        vn = fv / sj[None]
        th_n, info_n = exact_theta(vn[cal], cal.sum())
        thr = rule_thresholds(sj, th_n, th_f)
        R = {'theta_star': best[v_]['theta_star'], 'theta_f': th_f, 'theta_n': th_n, 'calib_ftheta': info_f,
             'calib_fnorm': info_n, 'scale': {'n_never_active_calib': n_never, 'median': float(np.median(sj)),
                                               'p10': float(np.quantile(sj, 0.1)), 'p90': float(np.quantile(sj, 0.9)),
                                               'n_calib_tokens': int(is_cal_tok.sum())},
             'fires_per_frame_heldout': {r: {'calib': float((fv[cal] > thr[r][None]).sum(1).mean()),
                                             'eval': float((fv[ev] > thr[r][None]).sum(1).mean())} for r in thr}}
        if 'experiment' in F:  # pooled calibration: per experiment
            R['fires_per_frame_heldout_by_experiment'] = {
                ex: {r: {'calib': float((fv[cal & (F.experiment.values == ex)] > thr[r][None]).sum(1).mean()),
                         'eval': float((fv[ev & (F.experiment.values == ex)] > thr[r][None]).sum(1).mean())}
                     for r in thr} for ex in sorted(F.experiment.unique())}
        np.save(OUT / dom / f'scale_{v_}.npy', sj.astype(np.float32))
        if dom == 'ants_pooled':  # calibration only; the series apply it
            out['variants'][v_] = R
            log(f'{dom} {v_}: theta_n {th_n:.4f} theta_f {th_f:.4f} | ' + json.dumps(R['fires_per_frame_heldout']) +
                ' | ' + json.dumps(R.get('fires_per_frame_heldout_by_experiment', {})))
            continue
        name = best[v_]['name']
        mx = np.load(codes_dir_of(dom, name) / 'codes_max.npy', mmap_mode='r')
        hrow = F.lay_row.values[held]
        oo = np.argsort(hrow)
        full = np.empty((len(held), WIDTH), np.float32)
        full[oo] = np.asarray(mx[hrow[oo]]).astype(np.float32)
        d_ = np.abs(full - fv[held])
        R['prepass_vs_fullpass'] = {'n_frames': int(len(held)), 'max_abs_diff': float(d_.max()),
                                    'mean_abs_diff': float(d_.mean()),
                                    'fire_agreement_fnorm': float(((full > thr['fnorm'][None]) ==
                                                                   (fv[held] > thr['fnorm'][None])).mean())}
        out['variants'][v_] = apply_rules(dom, c, name, sj, R)
    (OUT / dom / 'ftheta.json').write_text(json.dumps(out, indent=1))


def rule_thresholds(sj, th_n, th_f):
    return {'fnorm': (th_n * np.asarray(sj)).astype(np.float32), 'ftheta': np.full(WIDTH, th_f, np.float32),
            'gt0': np.zeros(WIDTH, np.float32)}


def apply_rules(dom, c, name, sj, R):
    """Full-pass participation under every rule (+ the per-frame top-16 rule on v) and the thresholded codes sets."""
    th_n, th_f = R['theta_n'], R['theta_f']
    thr = rule_thresholds(sj, th_n, th_f)
    cd = codes_dir_of(dom, name)
    mx = np.load(cd / 'codes_max.npy', mmap_mode='r')
    Nr = mx.shape[0]
    fire = {r: np.zeros(WIDTH, np.int64) for r in (*thr, 'top16')}
    for a_ in range(0, Nr, 200_000):
        blk = np.asarray(mx[a_:a_ + 200_000]).astype(np.float32)
        for r in thr:
            fire[r] += (blk > thr[r][None]).sum(0)
        vb = blk / np.asarray(sj, np.float32)[None]
        top = np.argpartition(-vb, K, axis=1)[:, :K]
        ok = np.take_along_axis(vb, top, 1) > 0
        np.add.at(fire['top16'], top[ok], 1)
    R['fullpass'] = {r: rule_stats(f, Nr) for r, f in fire.items()}
    R['codes'] = {r: str(write_rule_codes(cd, name, r, mx, thr[r],
                                          {'fnorm': f'fires iff codes_max_j / s_j > theta_n = {th_n:.6g}',
                                           'ftheta': f'fires iff codes_max_j > theta_f = {th_f:.6g}'}[r], c))
                  for r in ('fnorm', 'ftheta')}
    if c.get('pooled'):
        np.save(OUT / dom / f'scale_{"d2" if "r4d2" in name else "d0"}.npy', np.asarray(sj, np.float32))
    log(f'{dom} {name}: theta_n {th_n:.4f} theta_f {th_f:.4f} | full pass {json.dumps(R["fullpass"])} | '
        f'prepass vs full pass {json.dumps(R.get("prepass_vs_fullpass"))}')
    return R


def video_rules(x, fps=5.0):
    """(n, W) bool firing -> time spent, bouts, bout rate / min, latency s, censored."""
    n = len(x)
    starts = x[0].astype(np.int64) + (x[1:] & ~x[:-1]).sum(0)
    first = np.where(x.any(0), x.argmax(0), n)
    return x.mean(0), starts, starts / (n / fps / 60), first / fps, ~x.any(0)


def cmd_summaries(args):
    """Per video and neuron, under the three firing rules side by side (OUT/<dom>/ftheta.json): 'fnorm' (PRIMARY:
    codes_max_j / s_j > theta_n), 'ftheta' (codes_max_j > theta_f), 'gt0' (codes_max_j > 0, theta* only)."""
    dom = args.domain
    P = load_plan(dom)
    best = json.loads((sae_root(dom) / 'best.json').read_text())
    FT = json.loads((OUT / dom / 'ftheta.json').read_text())
    RULES_ = ('fnorm', 'ftheta', 'gt0')
    Q = ('time_spent', 'n_bouts', 'bout_rate_per_min', 'latency_s', 'latency_censored')
    S = {}
    for v_ in VARIANTS:
        name = best[v_]['name']
        R = FT['variants'][v_]
        sj = np.load(OUT / dom / f'scale_{v_}.npy').astype(np.float32)
        thr = {'fnorm': (R['theta_n'] * sj).astype(np.float32), 'ftheta': np.full(WIDTH, R['theta_f'], np.float32),
               'gt0': np.zeros(WIDTH, np.float32)}
        mx = np.load(codes_dir_of(dom, name) / 'codes_max.npy', mmap_mode='r')
        nv = len(P['videos'])
        arr = {f'{q}_{r}': np.zeros((nv, WIDTH), np.float32) for q in Q for r in RULES_}
        act = np.zeros((nv, WIDTH), np.float32)
        for i, v in enumerate(P['videos']):
            m = np.asarray(mx[v['lay_lo']:v['lay_lo'] + v['n']]).astype(np.float32)
            act[i] = m.mean(0)
            for r in RULES_:
                for q, val in zip(Q, video_rules(m > thr[r][None])):
                    arr[f'{q}_{r}'][i] = val
        f = OUT / dom / f'summaries_{v_}.npz'
        np.savez_compressed(f, obs=np.array([v['obs'] for v in P['videos']]),
                            n_frames=np.array([v['n'] for v in P['videos']]), mean_activation=act, sae=name,
                            theta_n=R['theta_n'], scale=sj, theta_f=R['theta_f'], primary_rule='fnorm', **arr,
                            rule='fnorm (PRIMARY): fires iff codes_max_j / s_j > theta_n (s_j = median non-zero token '
                                 'activation of neuron j on the calibration tokens); ftheta: codes_max_j > theta_f; gt0: '
                                 'codes_max_j > 0 (theta* baked into the SAE); bout = maximal run of firing frames; '
                                 'latency = first firing frame / 5 fps, censored at the video length')
        S[v_] = {r: {'median_time_spent': float(np.median(arr[f'time_spent_{r}'].mean(0))),
                     'mean_time_spent': float(arr[f'time_spent_{r}'].mean()),
                     'median_bout_rate_per_min': float(np.median(arr[f'bout_rate_per_min_{r}'].mean(0))),
                     'never_firing_video_neuron_frac': float(arr[f'latency_censored_{r}'].mean())} for r in RULES_}
        log(f'{dom} {v_}: {f} | ' + json.dumps(S[v_]))
    (OUT / dom / 'summaries.json').write_text(json.dumps(S, indent=1))


def rprec_halves(codes, y, rows, dev, chunk=256):
    """Per neuron on the frames `rows`: R-precision P = share of positives among the top N frames (N = number of
    positives; ties at the cut shared pro rata = expected value under random tie-breaking), and the normalised
    R~ = (P - pi) / (1 - pi), pi = base rate. Unit weights. -> (R~ (m,) numpy, pi)."""
    import torch
    X = codes[rows]
    yy = y[rows].bool()
    n, m = X.shape
    Np = int(yy.sum())
    pi = Np / max(n, 1)
    out = np.full(m, np.nan)
    if Np == 0 or Np == n:
        return out, pi
    yf = yy.float()
    for a_ in range(0, m, chunk):
        x = X[:, a_:a_ + chunk].float()
        cut = torch.topk(x, Np, dim=0).values[-1]  # the N-th largest value per neuron
        above = x > cut[None]
        tied = x == cut[None]
        n_above = above.sum(0).float()
        n_tied = tied.sum(0).float()
        pos_above = (above.float() * yf[:, None]).sum(0)
        pos_tied = (tied.float() * yf[:, None]).sum(0)
        P = (pos_above + pos_tied * (Np - n_above) / n_tied.clamp(min=1)) / Np
        out[a_:a_ + chunk] = ((P - pi) / (1 - pi)).cpu().numpy()
    return out, pi


def rprec_cf(codes, y, valid, half, dev):
    """Cross-fitted normalised R-precision: the neuron with the highest R~ on one half of the units is scored on the
    other half, both directions averaged (as sae_levers.score_model's cf_* metrics)."""
    import torch
    R, dirs = {}, []
    for h in (0, 1):
        rows = torch.from_numpy(np.flatnonzero(valid & (half == h))).to(codes.device)
        R[h] = rprec_halves(codes, y, rows, dev)
    for s_, t_ in ((0, 1), (1, 0)):
        sc = np.where(np.isfinite(R[s_][0]), R[s_][0], -np.inf)
        j = int(np.argmax(sc))
        dirs.append({'select_half': s_, 'neuron': j, 'select_rprec_norm': float(sc[j]),
                     'test_rprec_norm': float(R[t_][0][j]), 'test_base_rate': float(R[t_][1])})
    return {'cf_rprec_norm': float(np.mean([d['test_rprec_norm'] for d in dirs])), 'cf_rprec_dirs': dirs,
            'cf_rprec_neurons': [d['neuron'] for d in dirs]}


# ---------------------------------------------------------------------------------------------- align
def cmd_align(args):
    """Behaviour alignment test check (descriptive; never changes the recipe): best-neuron cross-fitted AUROC / AP /
    honest top-1% (sae_levers.score_model) of r4 d2 / d0, C1 and the deployed atlas SAE on the SAME labelled frames
    (every 5th labelled frame, seeded offset per video as rollout_c1.labelled_fps1)."""
    import torch
    from fg_extract_train import fps1_rows
    from sae_levers import score_model
    dom = args.domain
    c = DOMS[dom]
    P = load_plan(dom)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = json.loads((sae_root(dom) / 'best.json').read_text())
    lay_ann = REPO / c['eci'] / 'annotations.csv' if c['layout'] else REPO / c['ann']
    if dom == 'mice':
        a = pd.read_csv(lay_ann, usecols=['observation_id', 'frame_idx', 'Y_nn', 'Y_np', 'Y_nt'])
        ok = a[['Y_nn', 'Y_np', 'Y_nt']].notna().all(1).values
        lab = {'nose_nose': (a.Y_nn.values > 0) | (a.Y_np.values > 0), 'nose_tail': a.Y_nt.values > 0}
    else:
        a = pd.read_csv(lay_ann, usecols=['observation_id', 'frame_idx', 'Y_Y2F', 'Y_B2F', 'Y_YOL'])
        ok = a[['Y_Y2F', 'Y_B2F']].notna().all(1).values
        lab = {'groom_any': (a.Y_Y2F.values > 0) | (a.Y_B2F.values > 0), 'groom_yellow': a.Y_Y2F.values > 0,
               'groom_blue': a.Y_B2F.values > 0}
    ranges = {v['obs']: (v['lay_lo'], v['lay_lo'] + v['n']) for v in P['videos']}
    rows = np.sort(np.concatenate(list(fps1_rows(ranges, 5, 0).values())))
    rows = rows[ok[rows]]
    vdir = bee(dom) / 'videos'
    if args.partial:  # only videos already encoded (per-video files, before the merge)
        have = [v for v in P['videos'] if (vdir / f'{v["obs"]}.npz').exists()]
        keep = np.zeros(len(a), bool)
        for v in have:
            keep[v['lay_lo']:v['lay_lo'] + v['n']] = True
        rows = rows[keep[rows]]
        log(f'{dom} partial: {len(have)} of {len(P["videos"])} videos encoded, {len(rows):,} labelled frames')
    if not len(rows):
        (OUT / dom / 'align.json').write_text(json.dumps({'domain': dom, 'note': 'no labelled frames'}))
        log(f'{dom}: no labelled frames, alignment skipped')
        return
    labels = list(lab)
    yol_valid = None
    if dom != 'mice' and a.Y_YOL.notna().values[rows].sum() >= 100:  # on-the-lid (yellow), where labelled
        lab['onlid_yellow'] = np.nan_to_num(a.Y_YOL.values) > 0
        yol_valid = a.Y_YOL.notna().values[rows]
        labels.append('onlid_yellow')
    obs = a.observation_id.values[rows]
    unit = pd.Series(obs).str.rsplit('_', n=2).str[0].values if dom == 'mice' else obs
    units = sorted(set(unit))
    half = pd.Series(unit).map({u: i % 2 for i, u in enumerate(units)}).values
    Yt = torch.from_numpy(np.stack([lab[b][rows] for b in labels], 1)).to(dev)
    valid = {b: np.ones(len(rows), bool) for b in labels}
    if yol_valid is not None:
        valid['onlid_yellow'] = yol_valid
    sets = {f'r4_{v_}': (bee(dom) / ('layout/codes' if c['layout'] else 'codes') / best[v_]['name'], rows)
            for v_ in VARIANTS}
    if c.get('c1'):  # old codes; ants: rows of the 10-min layout by (observation, frame_idx)
        if c['layout']:
            A1 = pd.read_csv(REPO / c['c1_ann'], usecols=['observation_id', 'frame_idx'])
            key = pd.Series(np.arange(len(A1)), index=pd.MultiIndex.from_arrays([A1.observation_id, A1.frame_idx]))
            r1 = key.reindex(pd.MultiIndex.from_arrays([obs, a.frame_idx.values[rows]])).values
            assert not np.isnan(r1).any(), 'labelled frames missing from the 10-min layout'
            r1 = r1.astype(np.int64)
        else:
            r1 = rows
        sets['C1'] = (REPO / c['c1'], r1)
        sets['deployed'] = (REPO / c['deployed'], r1)
    Rz = {'domain': dom, 'n_frames': int(len(rows)), 'n_videos': int(len(set(obs))), 'n_units': len(units),
          'base_rate': {b: float(lab[b][rows][valid[b]].mean()) for b in labels},
          'n_valid': {b: int(valid[b].sum()) for b in labels},
          'note': 'IN-SAMPLE for every SAE (no labels used in training; the SAEs saw tokens of these videos except the '
                  'r4 held-out videos). Neuron selection cross-fitted over two halves of the units (mice pools, ants '
                  'videos).', 'models': {}}
    for nm, (cd, rr) in sets.items():
        t0 = time.time()
        if args.partial and nm.startswith('r4_'):
            vv = nm[3:]
            X = np.zeros((len(rr), WIDTH), np.float16)
            nf = np.zeros(len(rr), np.float64)
            for v in P['videos']:
                j = np.flatnonzero((rr >= v['lay_lo']) & (rr < v['lay_lo'] + v['n']))
                if len(j):
                    z = np.load(vdir / f'{v["obs"]}.npz')
                    X[j] = z[f'max_{vv}'][rr[j] - v['lay_lo']]
                    nf[j] = z['n_fg'][rr[j] - v['lay_lo']]
            cm = X
        else:
            cm = np.load(Path(cd) / 'codes_max.npy', mmap_mode='r')
            order = np.argsort(rr)
            X = np.empty((len(rr), cm.shape[1]), np.float16)
            X[order] = cm[rr[order]]
            nf = np.load(Path(cd) / 'n_fg.npy', mmap_mode='r')[rr].astype(np.float64)
        codes = torch.from_numpy(X).to(dev)
        rank = np.argsort(np.argsort(nf + np.random.default_rng(0).random(len(nf)) * 1e-3))
        decile = np.minimum(rank * 10 // len(nf), 9)
        res, _ = score_model(codes, Yt, labels, valid, half, decile, dev)
        Rz['models'][nm] = {'codes': str(cd), 'width': int(cm.shape[1]),
                            **{b: {k_: v_ for k_, v_ in r.items() if k_ != 'dirs'} for b, r in res.items()}}
        for li, b in enumerate(labels):  # PRIMARY (user, 2026-10-16): cross-fitted normalised R-precision
            rp = rprec_cf(codes, Yt[:, li], valid[b], half, dev)
            Rz['models'][nm][b].update(rp)
            br = float(Yt[:, li].cpu().numpy()[valid[b]].mean())
            Rz['models'][nm][b]['cf_ap_lift'] = Rz['models'][nm][b]['cf_ap'] / br if br > 0 else None
        log(f'{dom} {nm} ({cm.shape[1]} neurons, {time.time() - t0:.0f}s): ' + json.dumps(
            {b: {k_: (round(Rz['models'][nm][b][k_], 4) if Rz['models'][nm][b][k_] is not None else None) for k_ in ('cf_rprec_norm', 'cf_auroc', 'cf_ap', 'cf_ap_lift',
                                                                 'cf_top1')} for b in labels}))
        del codes
    Rz['partial'] = bool(args.partial)
    (OUT / dom / ('align_partial.json' if args.partial else 'align.json')).write_text(json.dumps(Rz, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- bout spot check
def cmd_spotcheck(args):
    """One annotated bout per domain (mice nose-nose Y_nn, ants grooming Y_Y2F; frogs: no behaviour labels, a random
    frame): start / middle / end frame decoded from the native video (resized to 512) next to its dataset JPG, with the
    grey mean |diff| against the JPGs at k-1, k, k+1 -> OUT/<dom>/bout_spotcheck.{jpg,json}. CPU only."""
    from PIL import Image, ImageDraw
    dom = args.domain
    c = DOMS[dom]
    P = load_plan(dom)
    col = {'mice': 'Y_nn', 'ants_v2': 'Y_Y2F', 'ants_v3': 'Y_Y2F'}.get(dom)
    A = pd.read_csv(REPO / c['ann'], usecols=['observation_id', 'frame_idx', 'frame_path'] + ([col] if col else []))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(P['videos']))
    pick = None
    for i in order:
        v = P['videos'][i]
        y = A[col].values[v['src_lo']:v['src_lo'] + v['n']] if col else None
        if col is None:
            pick = (v, [v['n'] // 2])
            break
        if y is None or np.isnan(y).all():
            continue
        on = np.nan_to_num(y) > 0
        st = np.flatnonzero(on[1:] & ~on[:-1]) + 1
        for s0 in st:
            e = s0
            while e < len(on) and on[e]:
                e += 1
            if e - s0 >= 5:
                pick = (v, [int(s0), int((s0 + e - 1) // 2), int(e - 1)])
                break
        if pick:
            break
    v, ks = pick
    rows, info = [], []
    for k in ks:
        fr = decode(v['path'], voff(c, v) + c['step'] * k, 1, c['step'], c['S'], v['hflip'], c['fps'], threads=2)[0]
        nat = np.asarray(Image.fromarray(fr).resize((512, 512), Image.BICUBIC))
        g = np.asarray(Image.fromarray(nat).convert('L'), np.float32)
        J = [np.asarray(Image.open(REPO / 'dataset' / A.frame_path.values[v['src_lo'] + k + d]).convert('RGB'))
             for d in (-1, 0, 1)]
        mad = [float(np.abs(g - np.asarray(Image.fromarray(j).convert('L'), np.float32)).mean()) for j in J]
        lab = (float(A[col].values[v['src_lo'] + k]) if col else None)
        info.append({'obs': v['obs'], 'k': k, 'label': col, 'label_value': lab, 'mad_km1_k_kp1': mad,
                     'closest_is_k': bool(mad[1] <= min(mad[0], mad[2]))})
        tile = Image.new('RGB', (1024, 512))
        tile.paste(Image.fromarray(nat), (0, 0))
        tile.paste(Image.fromarray(J[1]), (512, 0))
        ImageDraw.Draw(tile).text((6, 6), f'{v["obs"]} k{k} {col}={lab} native (left) vs JPG (right) '
                                          f'MAD k-1/k/k+1 {mad[0]:.2f}/{mad[1]:.2f}/{mad[2]:.2f}', fill=(255, 255, 0))
        rows.append(tile)
    sheet = Image.new('RGB', (1024, 512 * len(rows)))
    for i, t in enumerate(rows):
        sheet.paste(t, (0, 512 * i))
    sheet.save(OUT / dom / 'bout_spotcheck.jpg', quality=85)
    (OUT / dom / 'bout_spotcheck.json').write_text(json.dumps(info, indent=1))
    log(f'{dom} bout spot check: ' + json.dumps(info))


# ---------------------------------------------------------------------------------------------- report
def gpu_usage():
    """Per job log: GPU name, wall from the first to the last [gpu] sample, mean utilisation -> {log name: dict}."""
    out = {}
    for f in sorted((OUT / 'logs').glob('r4_*.out')):
        ut, ts, name = [], [], None
        for line in f.read_text(errors='replace').splitlines():
            if line.startswith('[gpu] '):
                p = [x.strip() for x in line[6:].split(',')]
                try:
                    ts.append(pd.Timestamp(p[0].replace('/', '-')))
                    name = p[1]
                    ut.append(float(p[2].split()[0]))
                except Exception:
                    pass
        if ut:
            out[f.name] = {'gpu': name, 'samples': len(ut), 'gpu_h': (ts[-1] - ts[0]).total_seconds() / 3600 + 1 / 60,
                           'util_mean': float(np.mean(ut)), 'util_median': float(np.median(ut))}
    return out


def nes_top(res_dir, n=3):
    """Primary NES selections (rollout_c1.PRIMARY settings) of a result dir -> {outcome: {analysis: [(prefix, round,
    neuron, direction, tau, p)]}}."""
    from rollout_c1 import PRIMARY
    out = {}
    for oc, (sub, flt) in PRIMARY.items():
        f = res_dir / sub / 'summary.csv'
        if not f.exists():
            continue
        t = pd.read_csv(f)
        for k_, v_ in flt.items():
            if k_ in t:
                t = t[np.isclose(t[k_], v_)] if isinstance(v_, (int, float)) else t[t[k_] == v_]
        t = t[t['round'] > 0]
        out[oc] = {aid: [(int(r['prefix']), int(r['round']), int(r['neuron']), r['direction'], float(r['tau']),
                          float(r['p'])) for _, r in g.sort_values(['prefix', 'round']).iterrows()][:3 * n]
                   for aid, g in t.groupby('analysis_id')}
    return out


def cmd_report(args):
    R = {'gpu_jobs': gpu_usage(), 'domains': {}}
    for dom in DOMS:
        if not (OUT / dom / 'plan.json').exists():
            continue
        P = load_plan(dom)
        d = {'S': DOMS[dom]['S'], 'G': P['G'], 'patches_per_body': P['patches_per_body'], 'n_videos': P['n_videos'],
             'n_frames': P['n_frames']}
        for k_ in ('check', 'prepass', 'train', 'merge', 'align', 'align_partial', 'ftheta', 'summaries', 'maskdiag',
                   'bout_spotcheck'):
            f = OUT / dom / f'{k_}.json'
            if f.exists():
                d[k_] = json.loads(f.read_text())
        bj = sae_root(dom) / 'best.json'
        if bj.exists():
            best = json.loads(bj.read_text())
            dm = DOMS[dom]
            root = REPO / ('results/vision/' + {'mice': 'mice/eci', 'frogs': 'frogs/eci'}.get(dom, f'ants/{dm["ants_eci"]}')
                           + '/nes')
            d['nes'] = {}
            for v_ in VARIANTS:
                for rule, suf in (('fnorm', '_fnorm'), ('ftheta', '_ftheta'), ('gt0', '')):
                    rd = root / (best[v_]['name'] + suf)
                    if not (rd / 'summary.csv').exists():
                        continue
                    e = {'dir': str(rd), 'core': nes_top(rd)}
                    if dom.startswith('ants'):
                        e['pairs'] = nes_top(rd / 'pairs')
                        e['pairs_noday'] = nes_top(rd / 'pairs_noday')
                    d['nes'][f'{v_}_{rule}'] = e
                    key = {'mice': 'mice', 'frogs': 'frogs'}.get(dom, f'ants_{dm["ants_eci"]}')
                    nd = OUT / 'ness' / key / (best[v_]['name'] + suf) / 'dines'
                    if (nd / 'summary.csv').exists():  # NESS (= DINES) on the same outcomes
                        d.setdefault('ness', {})[f'{v_}_{rule}'] = {'dir': str(nd), 'core': nes_top(nd)}
        vdir = bee(dom) / 'videos'
        d['videos_encoded'] = sum((vdir / f'{v["obs"]}.npz').exists() for v in P['videos']) if vdir.exists() else 0
        R['domains'][dom] = d
    # seeds: chosen seed and every seed's held-out stats (OUT/<dom>/train.json)
    for dom in list(R['domains']) + ['ants_pooled']:
        f = OUT / dom / 'train.json'
        if f.exists():
            t = json.loads(f.read_text())
            R.setdefault('seeds', {})[dom] = {v_: {'chosen': r['chosen'], 'per_seed': {
                s_: {k_: x[k_] for k_ in ('eval_fve', 'eval_l0_per_token', 'eval_dead_frac', 'theta_star')}
                for s_, x in r['seeds'].items()}} for v_, r in t['variants'].items()}
    # r4 pooled ants manifest (atlas builder entry point)
    if (OUT / 'ants_pooled' / 'sae' / 'best.json').exists():
        best = json.loads((OUT / 'ants_pooled' / 'sae' / 'best.json').read_text())
        M = {'name': 'r4 pooled', 'description': 'ONE ants SAE per variant (d2 = [z, z - z_{t-2 s}], d0 = z) for v2, v3 '
                                                  '(t = 2, 6, 8), v4 and v4_infested (t = 2, 6); per-series input sizes',
             'sae': {v_: {'name': best[v_]['name'], 'dir': str(OUT / 'ants_pooled' / 'sae' / best[v_]['name']),
                          'theta_star': best[v_]['theta_star']} for v_ in VARIANTS},
             'prepass': str(BEE / 'ants_pooled' / 'prepass'), 'primary_rule': 'fnorm', 'series': {}}
        for dom in ('ants_v2p', 'ants_v3p', 'ants_v4p'):
            c = DOMS[dom]
            e = {'layout': str(REPO / c['eci']), 'ANTS_ECI': c['ants_eci'], 'codes_root': str(bee(dom) / 'layout' / 'codes'),
                 'S': c['S'], 'source_offset': c['offset'], 'variants': {}}
            for v_ in VARIANTS:
                nm = best[v_]['name']
                e['variants'][v_] = {
                    'codes': {r: str(bee(dom) / 'layout' / 'codes' / (nm + suf)) for r, suf in
                              (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
                    'codes_exist': {r: (bee(dom) / 'layout' / 'codes' / (nm + suf) / 'DONE').exists() for r, suf in
                                    (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
                    'nes': {r: str(REPO / 'results/vision/ants' / c['ants_eci'] / 'nes' / (nm + suf)) for r, suf in
                            (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
                    'ness': {r: str(OUT / 'ness' / f'ants_{c["ants_eci"]}' / (nm + suf)) for r, suf in
                             (('gt0', ''), ('fnorm', '_fnorm'), ('ftheta', '_ftheta'))},
                    'summaries': str(OUT / dom / f'summaries_{v_}.npz')}
            e['align'] = str(OUT / dom / 'align.json')
            M['series'][c['series']] = e
        (OUT / 'ants_pooled').mkdir(parents=True, exist_ok=True)
        (OUT / 'ants_pooled' / 'manifest.json').write_text(json.dumps(M, indent=1))
    (OUT / 'report.json').write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({dom: {k_: v_ for k_, v_ in d.items() if k_ in ('S', 'G', 'patches_per_body', 'n_frames',
                                                                     'videos_encoded')}
                      for dom, d in R['domains'].items()}, indent=1))
    tot = sum(j['gpu_h'] for j in R['gpu_jobs'].values())
    print(f'GPU hours logged: {tot:.1f} over {len(R["gpu_jobs"])} jobs, mean util '
          f'{np.mean([j["util_mean"] for j in R["gpu_jobs"].values()]) if R["gpu_jobs"] else float("nan"):.1f}%')


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('plan', 'reshard', 'check', 'maskdiag', 'pool', 'prepass', 'train', 'encode', 'merge', 'summaries', 'align', 'report', 'spotcheck', 'ftheta'))
    p.add_argument('--domain', default='mice', choices=list(DOMS))
    p.add_argument('--domains', nargs='+', default=list(DOMS))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--videos', default=None, help='comma-separated plan video indices (instead of --shard)')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--out-dir', default='')
    p.add_argument('--limit', type=int, default=0, help='prepass: first N bursts only (test; nothing copied)')
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--cap', type=int, default=0, help='train: max train-pool tokens (0 = all)')
    p.add_argument('--n-init', type=int, default=500_000, help='train: TokenNorm / b_dec init sample')
    p.add_argument('--shard-frames', type=int, default=60_000)
    p.add_argument('--force', action='store_true')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--encoder-check', action='store_true')
    args = p.parse_args()
    {'plan': cmd_plan, 'reshard': cmd_reshard, 'check': cmd_check, 'maskdiag': cmd_maskdiag, 'pool': cmd_pool, 'prepass': cmd_prepass, 'train': cmd_train, 'encode': cmd_encode,
     'merge': cmd_merge, 'summaries': cmd_summaries, 'align': cmd_align, 'report': cmd_report, 'spotcheck': cmd_spotcheck, 'ftheta': cmd_ftheta}[args.step](args)


if __name__ == '__main__':
    main()
