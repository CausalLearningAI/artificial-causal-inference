"""
R4 "background neurons" diagnosis (2026-10-16). Read-only on every existing file; writes results/vision/eci_r4_bgdiag/.

Animal pixels ("dark" truth): the lab mask's own pixel cue on the 512 px dataset JPG, (grey < 60) & (pix_bg - grey > 40)
for mice / ants (FG_RULE dark_abs / dark_rel), (pix_bg - grey > 30) inside the dish ROI for frogs (FG_RULE_FROGS).
Per G-grid token: dark fraction of its patch (box average of the 512 px map). Token class: body = dark fraction >= 0.25;
edge = not body but a dark pixel within the 3 x 3 token neighbourhood; off = no dark pixel within the 3 x 3 neighbourhood.

H1  page clips: real per-patch heat files (patch/*.npz) -> peak per top clip (build_explorer clip_peaks) vs the page pk;
    class of the peak token on the clip frame. Same for the online ants model (32 grid) as reference.
H2  prepass frames (kept-token positions stored by rollout_r4 prepass): kept fraction, class shares of kept tokens at the
    G grid and of kept cells at the 32 grid, kept area / dark area; overlay PNGs.
H3  per neuron, on the sampled prepass frames: class of its argmax token (parent argmax.npy) where it fires under fnorm;
    between-video share (eta^2) of its per-frame firing over the full codes; online eta^2 for comparison.
H4  per neuron scale s_j (threshold_per_neuron / theta_n) vs off-share; NES picks under fnorm / ftheta / plain.
H6  pooled ants: per-neuron firing rate per experiment (v2 / v3 / v4 / v4_infested).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter

REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('DIAG_OUT') or REPO / 'results/vision/eci_r4_bgdiag')
SITE = REPO / 'results/vision/eci_atlas_v37/site/assets/data'
RES = REPO / 'results/vision/eci_atlas_v37/res'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/r4')
LOCAL = Path(os.environ.get('LOCAL_DIR', '/tmp/diag_r4bg'))
N_SAMPLE = int(os.environ.get('N_SAMPLE', 1500))
rng = np.random.default_rng(0)
S2 = 'matryoshka_btk_512_k16_r4d2_s2'
DOMS = {
    'mice': dict(res='mice_d2', sae='matryoshka_btk_512_k16_r4d2_s0', prefix='r4miced2', kind='mice',
                 eci=REPO / 'dataset/mice/v1/eci', pre=BEE / 'mice/prepass', nes=REPO / 'results/vision/mice/eci/nes',
                 online=dict(codes=REPO / 'dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0',
                             ann=REPO / 'dataset/mice/v1/annotations.csv')),
    'frogs': dict(res='frogs_d2', sae=S2, prefix='r4frogsd2', kind='frogs', eci=REPO / 'dataset/frogs/eci',
                  pre=BEE / 'frogs/prepass', nes=REPO / 'results/vision/frogs/eci/nes',
                  online=dict(codes=REPO / 'dataset/frogs/eci/codes/matryoshka_btk_1024_k16_frogsfg_s0',
                              ann=REPO / 'dataset/frogs/eci/annotations.csv')),
}
for s in ('v2', 'v3', 'v4'):
    DOMS[f'ants_{s}p'] = dict(res=f'ants_{s}p_d2', sae=S2, prefix=f'r4{s}pd2', kind='ants',
                              eci=REPO / f'dataset/ants/eci_r4{s}p', pre=BEE / 'ants_pooled/prepass', series=f'ants_{s}',
                              nes=REPO / f'results/vision/ants/eci_r4{s}p/nes')
ONLINE_ANTS = dict(work=REPO / 'results/vision/ants/eci/nes/matryoshka_btk_1024_k16_antsfg_s0/_cache/explorer',
                   prefix='antsfg', bg=REPO / 'dataset/ants/eci/fg448/background',
                   codes=REPO / 'dataset/ants/eci/codes/matryoshka_btk_1024_k16_antsfg_s0',
                   ann=REPO / 'dataset/ants/eci/annotations.csv')
log = lambda *a: print(*a, flush=True)  # noqa: E731


# ------------------------------------------------------------------ helpers
def jpath(rel):
    p = LOCAL / 'ds' / rel
    return p if p.exists() else REPO / 'dataset' / rel


_bg = {}


def background(bgdir, obs):
    k = (str(bgdir), obs)
    if k not in _bg:
        f = Path(bgdir) / f'{obs}.npz'
        if not f.exists():
            f = sorted(Path(bgdir).glob(f'{obs}__*.npz'))[0]
        z = np.load(f)
        _bg[k] = (z['pix_bg'].astype(np.float32), z['roi'] if 'roi' in z.files else None)
    return _bg[k]


def dark_map(rel, bgdir, obs, kind):
    g = np.asarray(Image.open(jpath(rel)).convert('L'), np.float32)
    pb, roi = background(bgdir, obs)
    if kind == 'frogs':
        d = (pb - g) > 30
        if roi is not None:
            d &= roi
    else:
        d = (g < 60) & ((pb - g) > 40)
    return d


def token_frac(d, G):
    return np.asarray(Image.fromarray(d.astype(np.float32)).resize((G, G), Image.BOX), np.float32)


def classes(frac):
    """(G, G) dark fraction -> (G, G) int8: 0 body, 1 edge, 2 off."""
    body = frac >= 0.25
    near = maximum_filter(frac, size=3) > 0
    return np.where(body, 0, np.where(near, 1, 2)).astype(np.int8)


def stage(rels):
    rels = sorted(set(r for r in rels if not (LOCAL / 'ds' / r).exists()))
    if not rels or not os.environ.get('LOCAL_DIR'):
        return
    (LOCAL / 'ds').mkdir(parents=True, exist_ok=True)
    lst = LOCAL / f'stage_{len(rels)}_{rng.integers(1e9)}.txt'
    lst.write_text('\n'.join(rels) + '\n')
    subprocess.run(f'tar -C {(REPO / "dataset").resolve()} -cf - -T {lst} | tar -C {LOCAL / "ds"} -xf -', shell=True,
                   check=True)
    log(f'staged {len(rels)} JPGs')


def discovered_from_site(prefix):
    """{neuron: [outcome|analysis, ...]} from the page data files of one model (NES + DINES rows)."""
    out = {}
    for f in sorted(SITE.glob(f'{prefix}*.json')):
        if '_max_' in f.name:
            continue
        d = json.loads(f.read_text())
        for k, v in d.get('results', {}).items():
            for r in v.get('rows', []) if isinstance(v, dict) else []:
                if isinstance(r, dict) and r.get('neuron') is not None:
                    out.setdefault(int(r['neuron']), []).append(k)
    return out


def shown_neurons(prefix):
    out = {}
    for f in sorted(SITE.glob(f'{prefix}_max_*.json')):
        out.update(json.loads(f.read_text())['neurons'])
    return out


def nes_picks(d):
    """NES picks per rule ({fnorm, ftheta, plain}: sorted neuron list) from the R4 NES summary.csv files."""
    out = {}
    for rule, suf in (('fnorm', '_fnorm'), ('ftheta', '_ftheta'), ('plain', '')):
        base = d['nes'] / (d['sae'] + suf)
        s = set()
        for sub in ('.', 'maxpool_bouts', 'maxpool_latency'):
            f = base / sub / 'summary.csv'
            if f.exists():
                t = pd.read_csv(f, usecols=lambda c: c in ('neuron', 'analysis_id'))
                if 'neuron' in t:
                    s |= set(t['neuron'].dropna().astype(int))
        out[rule] = sorted(s)
    return out


def peak(m):
    acc = gaussian_filter(np.maximum(m, 0).sum(0), 1.0)
    if not acc.max() > 0:
        return None, None
    iy, ix = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return [round((ix + 0.5) / acc.shape[1], 3), round((iy + 0.5) / acc.shape[0], 3)], (iy, ix)


# ------------------------------------------------------------------ H1: page clips
def h1(name, work, prefix, bgdir_of, kind, disc, sheet_dir, kr=8):
    picks = json.loads((work / 'picks.json').read_text())
    fp = picks['frame_path']
    shown = shown_neurons(prefix)
    by = picks['by_key']['max']
    order = sorted((j for j in shown if j in by), key=lambda j: (int(j) not in disc, int(j)))[:40]
    need = []
    for j in order:
        for c in by[j]['clips']['all']['frame']['top'][:kr]:
            need.append(fp[str(c['rows'][0])])
    stage(need)
    res, pk_err, cls_all, sheets = [], [], [], []
    for j in order:
        pf = sorted((work / 'patch').glob(f'*{int(j):04d}.npz'))
        if not pf:
            continue
        pz = np.load(pf[0])
        sig = str(pz['sig'])
        m = pz['all__frame__top'].astype(np.float32)
        clips = by[j]['clips']['all']['frame']['top'][:kr]
        page_pk = shown[j]['clips']['all']['frame'].get('pk', [])
        G = m.shape[-1]
        cl, mass = [], []
        for k, c in enumerate(clips[:len(m)]):
            p, ij = peak(m[k:k + 1])
            if k < len(page_pk) and p is not None and page_pk[k] is not None:
                pk_err.append(max(abs(p[0] - page_pk[k][0]), abs(p[1] - page_pk[k][1])))
            if ij is None:
                continue
            rel = fp[str(c['rows'][0])]
            d = dark_map(rel, bgdir_of(c['obs']), c['obs'], kind)
            cc = classes(token_frac(d, G))
            cl.append(int(cc[ij]))
            h = np.maximum(m[k], 0)
            mass.append([float(h[cc == q].sum() / max(h.sum(), 1e-9)) for q in range(3)])
            if int(j) in disc and len(sheets) < 24:
                sheets.append((rel, c['obs'], d, m[k], ij, int(j), int(cc[ij])))
        cls_all += cl
        res.append({'neuron': int(j), 'discovered': int(j) in disc, 'heat': sig.split('-')[0], 'grid': G,
                    'peak_class_counts': np.bincount(cl, minlength=3).tolist(),
                    'heat_mass_body_edge_off': np.mean(mass, 0).round(3).tolist() if mass else None})
    cls = np.bincount(cls_all, minlength=3)
    if sheets:
        tiles = []
        for rel, obs, d, hm, (iy, ix), j, q in sheets:
            im = Image.open(jpath(rel)).convert('RGB')
            a = np.asarray(im).astype(np.float32)
            H = np.asarray(Image.fromarray(hm).resize((512, 512), Image.BILINEAR))
            w = np.clip(H / max(H.max(), 1e-9), 0, 1)[..., None]
            a = a * (1 - 0.5 * w) + np.array([255, 60, 0]) * 0.5 * w
            a[d] = a[d] * 0.4 + np.array([0, 255, 255]) * 0.6
            im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(im)
            G = hm.shape[-1]
            x, y = (ix + 0.5) * 512 / G, (iy + 0.5) * 512 / G
            dr.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 0), width=3)
            dr.rectangle([0, 0, 512, 22], fill=(0, 0, 0))
            dr.text((4, 4), f'n{j} {obs} peak={["body", "edge", "off"][q]}', fill=(255, 255, 255))
            tiles.append(im.resize((256, 256), Image.LANCZOS))
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        sh = Image.new('RGB', (cols * 256, rows * 256))
        for i, t in enumerate(tiles):
            sh.paste(t, ((i % cols) * 256, (i // cols) * 256))
        sh.save(sheet_dir / f'clips_{name}.jpg', quality=85)
    return {'neurons_checked': len(res), 'discovered_checked': sum(r['discovered'] for r in res),
            'pk_recomputed_vs_page_max_abs_err_frac': float(max(pk_err)) if pk_err else None,
            'pk_pairs': len(pk_err),
            'peak_token_share_body_edge_off_all': (cls / max(cls.sum(), 1)).round(3).tolist(), 'n_clips': int(cls.sum()),
            'peak_token_share_body_edge_off_discovered':
                (lambda b: (b / max(b.sum(), 1)).round(3).tolist())(
                    np.sum([r['peak_class_counts'] for r in res if r['discovered']] or [[0, 0, 0]], 0)),
            'per_neuron': res}


# ------------------------------------------------------------------ video ids / eta^2
def video_index(ann):
    o = pd.read_csv(ann, usecols=['observation_id'])['observation_id'].astype(str).values
    starts = np.r_[0, np.nonzero(o[1:] != o[:-1])[0] + 1]
    assert len(starts) == len(set(o)), 'videos are not contiguous'
    return o, starts


def per_video_fire(codes_path, starts, n):
    C = np.load(codes_path, mmap_mode='r')
    assert len(C) == n, (codes_path, len(C), n)
    W = C.shape[1]
    ends = np.r_[starts[1:], n]
    S = np.zeros((len(starts), W), np.int64)
    for v, (a, b) in enumerate(zip(starts, ends)):
        S[v] = (np.asarray(C[a:b]) > 0).sum(0)
    return S, ends - starts


def eta2(S, nv):
    N = nv.sum()
    p = S.sum(0) / N
    pv = S / nv[:, None]
    between = (nv[:, None] * (pv - p) ** 2).sum(0) / N
    tot = p * (1 - p)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(tot > 0, between / tot, np.nan), p


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {'median': round(float(np.median(x)), 3), 'q25': round(float(np.quantile(x, .25)), 3),
            'q75': round(float(np.quantile(x, .75)), 3), 'n': int(len(x))} if len(x) else None


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 5:
        return None
    from scipy.stats import spearmanr
    rho, pv = spearmanr(a[k], b[k])
    return {'rho': round(float(rho), 3), 'p': float(pv), 'n': int(k.sum())}


# ------------------------------------------------------------------ per domain
def run_dom(name):
    d = DOMS[name]
    codes = d['eci'] / 'codes'
    cfg = json.loads((codes / (d['sae'] + '_fnorm') / 'config.json').read_text())
    G, bgdir, kind = int(cfg['G']), Path(cfg['backgrounds']), d['kind']
    parent = Path(cfg['parent_codes'])
    ann = Path(cfg['layout_ann'])
    out = {'G': G, 'S': cfg['S'], 'rule': cfg['rule']}
    disc = discovered_from_site(d['prefix'])
    picks_rule = nes_picks(d)
    out['discovered_on_page'] = {str(k): sorted(set(v)) for k, v in sorted(disc.items())}
    out['nes_picks_by_rule'] = picks_rule
    log(f'== {name}: G {G}, {len(disc)} discovered neurons on the page')
    # H1
    work = RES / d['res'] / (d['sae'] + '_fnorm') / '_cache' / 'explorer'
    out['H1'] = h1(name, work, d['prefix'], lambda o: bgdir, kind, disc, OUT / 'overlays')
    log(f'{name} H1', {k: v for k, v in out['H1'].items() if k != 'per_neuron'})
    # prepass sample
    F = pd.read_parquet(d['pre'] / 'frames.parquet')
    tf = np.load(d['pre'] / 'tok_frame.npy', mmap_mode='r')
    tp = np.load(d['pre'] / 'tok_pos.npy', mmap_mode='r')
    fidx = np.arange(len(F))
    if 'series' in d:
        fidx = fidx[F['series'].values == d['series']]
    sel = np.sort(rng.choice(fidx, min(N_SAMPLE, len(fidx)), replace=False))
    A = pd.read_csv(ann, usecols=['observation_id', 'frame_path'])
    nfg = np.load(parent / 'n_fg.npy', mmap_mode='r')
    lr = F['lay_row'].values[sel]
    out['prepass_lay_row_check_nfg_eq_nkept'] = float(np.mean(np.asarray(nfg[lr]) == F['n_kept'].values[sel]))
    stage(A['frame_path'].values[lr].tolist())
    tf_a = np.asarray(tf)
    order = np.argsort(tf_a, kind='stable')
    bounds = np.searchsorted(tf_a[order], np.r_[sel, sel + 1].reshape(2, -1))
    arg = np.load(parent / 'argmax.npy', mmap_mode='r')
    cf = np.load(codes / (d['sae'] + '_fnorm') / 'codes_max.npy', mmap_mode='r')
    ARG, CF = np.asarray(arg[lr]).astype(np.int64), np.asarray(cf[lr]).astype(np.float32)
    g32 = np.floor((np.arange(G) + 0.5) * 32 / G).astype(np.int64)
    rows_h2, ncls = [], np.zeros((512, 3), np.int64)
    over = []
    for i, f in enumerate(sel):
        rel, obs = A['frame_path'].values[lr[i]], str(A['observation_id'].values[lr[i]])
        dm = dark_map(rel, bgdir, obs, kind)
        frac = token_frac(dm, G)
        cc = classes(frac).ravel()
        pos = np.asarray(tp[order[bounds[0, i]:bounds[1, i]]]).astype(np.int64)
        kc = np.bincount(cc[pos], minlength=3)
        frac32 = token_frac(dm, 32)
        c32 = classes(frac32).ravel()
        cells = np.unique(g32[pos // G] * 32 + g32[pos % G])
        k32 = np.bincount(c32[cells], minlength=3)
        n_kept = int(F['n_kept'].values[f])
        rows_h2.append([n_kept / G ** 2, *(kc / max(kc.sum(), 1)), *(k32 / max(k32.sum(), 1)),
                        n_kept * (512 / G) ** 2 / max(dm.sum(), 1), dm.mean()])
        fire = np.nonzero((CF[i] > 0) & (ARG[i] >= 0))[0]
        np.add.at(ncls, (fire, cc[ARG[i, fire]]), 1)
        if len(over) < 6 and i % max(1, len(sel) // 6) == 0:
            over.append((rel, obs, dm, pos, n_kept))
    R = np.array(rows_h2)
    cols = ['kept_frac_of_grid', 'kept_tok_body', 'kept_tok_edge', 'kept_tok_off', 'kept32_body', 'kept32_edge',
            'kept32_off', 'kept_area_over_dark_area', 'dark_frac_of_frame']
    out['H2'] = {'n_frames': len(sel), 'stored_tokens_subset': bool((F['n_stored'] < F['n_kept']).any()),
                 'median': dict(zip(cols, np.median(R, 0).round(3).tolist())),
                 'mean': dict(zip(cols, R.mean(0).round(3).tolist()))}
    log(f'{name} H2', out['H2'])
    # overlays
    tiles = []
    for rel, obs, dm, pos, n_kept in over:
        im = np.asarray(Image.open(jpath(rel)).convert('RGB')).astype(np.float32)
        mk = np.zeros(G * G, bool)
        mk[pos] = True
        M = np.asarray(Image.fromarray(mk.reshape(G, G).astype(np.uint8) * 255).resize((512, 512), Image.NEAREST)) > 0
        a = im.copy()
        a[M] = a[M] * 0.55 + np.array([255, 0, 0]) * 0.45
        a[dm] = a[dm] * 0.3 + np.array([0, 255, 255]) * 0.7
        t = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(t)
        dr.rectangle([0, 0, 512, 22], fill=(0, 0, 0))
        dr.text((4, 4), f'{obs} kept {n_kept}/{G * G} red=kept token cyan=animal px', fill=(255, 255, 255))
        tiles.append(t.resize((384, 384), Image.LANCZOS))
    if tiles:
        sh = Image.new('RGB', (3 * 384, 2 * 384))
        for i, t in enumerate(tiles):
            sh.paste(t, ((i % 3) * 384, (i // 3) * 384))
        sh.save(OUT / 'overlays' / f'mask_{name}.jpg', quality=85)
    # H3a: argmax class per neuron
    nf = ncls.sum(1)
    off = np.where(nf >= 20, ncls[:, 2] / np.maximum(nf, 1), np.nan)
    body = np.where(nf >= 20, ncls[:, 0] / np.maximum(nf, 1), np.nan)
    dn = sorted(disc)
    o_vid, starts = video_index(ann)
    S, nv = per_video_fire(codes / (d['sae'] + '_fnorm') / 'codes_max.npy', starts, len(o_vid))
    e2, p_fn = eta2(S, nv)
    out['H3'] = {'argmax_off_share_all': q(off), 'argmax_body_share_all': q(body),
                 'argmax_off_share_discovered': q(off[dn]) if dn else None,
                 'argmax_body_share_discovered': q(body[dn]) if dn else None,
                 'n_neurons_off_share_gt_0.5': int(np.nansum(off > 0.5)), 'n_neurons_scored': int(np.isfinite(off).sum()),
                 'eta2_all': q(e2), 'eta2_discovered': q(e2[dn]) if dn else None,
                 'spearman_eta2_vs_off_share': spearman(e2, off),
                 'discovered': {str(j): {'off': None if not np.isfinite(off[j]) else round(float(off[j]), 3),
                                         'body': None if not np.isfinite(body[j]) else round(float(body[j]), 3),
                                         'n_fire_sampled': int(nf[j]), 'eta2': round(float(e2[j]), 3),
                                         'rate_fnorm': round(float(p_fn[j]), 4)} for j in dn}}
    # H4: scale, rules
    tpn = np.load(codes / (d['sae'] + '_fnorm') / 'threshold_per_neuron.npy').astype(np.float64)
    th = float(re.search(r'theta_n = ([0-9.]+)', cfg['rule']).group(1))
    s_j = tpn / th
    Sp, _ = per_video_fire(parent / 'codes_max.npy', starts, len(o_vid))
    St, _ = per_video_fire(codes / (d['sae'] + '_ftheta') / 'codes_max.npy', starts, len(o_vid))
    _, p_pl = eta2(Sp, nv)
    _, p_ft = eta2(St, nv)
    alive = p_fn > 0
    sets = {r: v for r, v in picks_rule.items()}
    out['H4'] = {'s_j': q(s_j), 'spearman_s_j_vs_off_share': spearman(s_j, off),
                 'spearman_s_j_vs_eta2': spearman(s_j, e2),
                 'spearman_s_j_vs_fnorm_rate_over_plain_rate': spearman(s_j, p_fn / np.maximum(p_pl, 1e-12)),
                 'fires_per_frame_mean': {'fnorm': round(float(p_fn.sum()), 2), 'ftheta': round(float(p_ft.sum()), 2),
                                          'plain_gt0': round(float(p_pl.sum()), 2)},
                 'n_alive_fnorm': int(alive.sum()),
                 's_j_rank_pct_of_picks': {r: q([float((s_j < s_j[j]).mean()) for j in v]) if v else None
                                           for r, v in sets.items()},
                 'off_share_of_picks': {r: q(off[v]) if v else None for r, v in sets.items()},
                 'eta2_of_picks': {r: q(e2[v]) if v else None for r, v in sets.items()},
                 'pick_overlap_fnorm_ftheta': len(set(sets['fnorm']) & set(sets['ftheta'])),
                 'pick_overlap_fnorm_plain': len(set(sets['fnorm']) & set(sets['plain']))}
    out['_per_neuron'] = {'off': off.tolist(), 'body': body.tolist(), 'eta2': e2.tolist(), 's_j': s_j.tolist(),
                          'rate_fnorm': p_fn.tolist(), 'rate_plain': p_pl.tolist(), 'rate_ftheta': p_ft.tolist()}
    exp_rate = None
    if 'series' in d:  # H6 per experiment rate (fnorm)
        e = pd.read_csv(d['eci'] / 'experiment.csv', dtype={'observation_id': str})
        vids = o_vid[starts]
        ex = e.set_index('observation_id').loc[vids, 'experiment'].values
        exp_rate = {x: (S[ex == x].sum(0) / nv[ex == x].sum()).tolist() for x in sorted(set(ex))}
        exp_rate['_frames'] = {x: int(nv[ex == x].sum()) for x in sorted(set(ex))}
        if name == 'ants_v4p':  # H5: infested vs healthy within recording day F
            ee = e.set_index('observation_id').loc[vids]
            rv = S / nv[:, None]
            h5 = {}
            for j in sorted(set(dn)):
                r = {}
                for day in sorted(ee['recording_date'].unique()):
                    for x in sorted(set(ex)):
                        k = (ee['recording_date'].values == day) & (ex == x)
                        if k.any():
                            r[f'{x}|{day}'] = [round(float(rv[k, j].mean()), 4), int(k.sum())]
                h5[str(j)] = r
            out['H5_v4_rate_by_arm_day'] = h5
    return out, exp_rate


def online_eta(name, d):
    if 'online' not in d:
        return None
    o, starts = video_index(d['online']['ann'])
    S, nv = per_video_fire(d['online']['codes'] / 'codes_max.npy', starts, len(o))
    e2, p = eta2(S, nv)
    pre = {'mice': 'fg448al', 'frogs': 'frogsfg'}[name]
    disc = sorted(set(discovered_from_site(pre)) | set(discovered_from_site(pre + 'v37')) |
                  set(discovered_from_site(pre + 'dines')))
    disc = [j for j in disc if j < len(e2)]
    return {'eta2_all_alive': q(e2[p > 0]), 'eta2_discovered': q(e2[disc]) if disc else None, 'n_disc': len(disc),
            'fires_per_frame_mean': round(float(p.sum()), 2)}


def main():
    (OUT / 'overlays').mkdir(parents=True, exist_ok=True)
    doms = [x for x in (sys.argv[1:] or list(DOMS)) if x in DOMS]
    rep = {}
    rates = {}
    for name in doms:
        try:
            rep[name], er = run_dom(name)
            if er:
                rates[name] = er
            rep[name]['online_eta2'] = online_eta(name, DOMS[name])
        except Exception as ex:  # fail loud in the report, continue with the other domains
            import traceback
            traceback.print_exc()
            rep[name] = {'error': repr(ex)}
        (OUT / f'diag_{name}.json').write_text(json.dumps(rep[name], indent=1, default=float))
    # online ants reference (H1 + eta2)
    if 'online_ants' in sys.argv[1:] or not sys.argv[1:]:
        try:
            oa = ONLINE_ANTS
            disc = discovered_from_site('antsfg')
            disc.update(discovered_from_site('antsfgdines'))
            r = {'H1': h1('online_ants', oa['work'], 'antsfg', lambda o: oa['bg'], 'ants', disc, OUT / 'overlays', kr=16)}
            o, starts = video_index(oa['ann'])
            S, nv = per_video_fire(oa['codes'] / 'codes_max.npy', starts, len(o))
            e2, p = eta2(S, nv)
            dn = [j for j in sorted(disc) if j < len(e2)]
            r['eta2_all_alive'] = q(e2[p > 0])
            r['eta2_discovered'] = q(e2[dn])
            r['fires_per_frame_mean'] = round(float(p.sum()), 2)
            rep['online_ants'] = r
        except Exception as ex:
            import traceback
            traceback.print_exc()
            rep['online_ants'] = {'error': repr(ex)}
        (OUT / 'diag_online_ants.json').write_text(json.dumps(rep['online_ants'], indent=1, default=float))
    # H6 pooled ants
    if rates:
        R = {}
        for name, er in rates.items():
            for x, v in er.items():
                if x != '_frames':
                    R[x] = np.array(v)
        X = sorted(R)
        M = np.stack([R[x] for x in X])  # (experiments, 512)
        alive = M.max(0) > 1e-4
        share = M / np.maximum(M.sum(0), 1e-12)
        top = share.max(0)
        h6 = {'experiments': X, 'n_alive': int(alive.sum()),
              'n_share_ge_0.9_one_experiment': int(((top >= 0.9) & alive).sum()),
              'n_share_ge_0.75_one_experiment': int(((top >= 0.75) & alive).sum()),
              'uniform_share': round(1 / len(X), 3),
              'by_experiment_ge_0.9': {x: int(((share.argmax(0) == i) & (top >= 0.9) & alive).sum())
                                       for i, x in enumerate(X)},
              'mean_fires_per_frame': {x: round(float(R[x].sum()), 2) for x in X}}
        disc_all = {}
        for name in rates:
            for k in rep[name].get('discovered_on_page', {}):
                disc_all.setdefault(int(k), []).append(name)
        h6['discovered'] = {str(j): {'series': v, 'share': dict(zip(X, share[:, j].round(3).tolist()))}
                            for j, v in sorted(disc_all.items())}
        h6['discovered_share_ge_0.75'] = int(sum(top[j] >= 0.75 for j in disc_all))
        h6['n_discovered'] = len(disc_all)
        rep['H6_ants_pooled'] = h6
        (OUT / 'diag_H6.json').write_text(json.dumps(h6, indent=1))
    slim = {k: ({kk: vv for kk, vv in v.items() if kk != '_per_neuron'} if isinstance(v, dict) else v)
            for k, v in rep.items()}
    (OUT / f'diagnosis{"_" + "_".join(sys.argv[1:]) if sys.argv[1:] else ""}.json').write_text(
        json.dumps(slim, indent=1, default=float))
    log('done')


if __name__ == '__main__':
    main()
