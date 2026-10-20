"""
CalMS21 resolution test, two extra DINOv3-L/16 block-20 token grids (rules PRE-REGISTERED in calms21_res.sh).

Thin wrapper over scripts/benchmark/calms21/ (eci_encode.py, eci_train.py, eci_res_score.py), which are NOT edited:
the modules are loaded by file path under private names (mabe22 has files called eci_encode.py / eci_train.py too, and
calms21 eci_train.main does `import eci_train as ET` = the MABe22 machinery module), then patched in memory:
    eci_encode.CONFIGS  + d3l_768_L20 ('dinov3l', 768, 20; 48 x 48 grid), d3l_1280_L20 ('dinov3l', 1280, 20; 80 x 80)
    eci_encode.to_grid  any G: a G-grid patch is kept iff its 32-grid parent is, parent = floor((g + 0.5) * 32 / G)
                        per axis (patch-centre rule, = mabe22 eci_enc_encode.map_mask for G > 32); self-test: identical
                        to the old kron result at G = 64 and the identity at G = 32
    eci_train.DIM       + both configs, 1024 (DINOv3-L hidden size, as d3l_512_L20 / d3l_1024_L20)
    eci_train.load_shard  the original hard-codes the per-frame token count as n32 x (4 if G == 64 else 1); for the new
                        grids it is recomputed per frame from the shard's pos32 cells with to_grid (and checked against
                        the token and _pos.npy row counts); every other config goes to the original function
Modes:
    selftest            to_grid checks (login safe, tiny)
    encode <args>       eci_encode.main with <args>
    train <args>        eci_train.main with <args>
    score               summary -> results/vision/eci_resolution/calms21/{res.json, res.md} (login safe: reads JSONs and
                        the small pos32 / frames files)
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
C21 = REPO / 'scripts/benchmark/calms21'
NEW = {'d3l_768_L20': ('dinov3l', 768, 20), 'd3l_1280_L20': ('dinov3l', 1280, 20)}
GRID = {'d3l_512_L20': 32, 'd3l_768_L20': 48, 'd3l_1024_L20': 64, 'd3l_1280_L20': 80}
RES = REPO / 'results/vision/eci_resolution/calms21'


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, C21 / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def to_grid(pos32, G):
    """32-grid kept cells -> kept cells of a G x G grid (patch-centre parent rule), int16, row-major."""
    if G == 32:
        return pos32.astype(np.int16)
    m = np.zeros((32, 32), bool)
    m.flat[pos32.astype(np.int64)] = True
    idx = np.floor((np.arange(G) + 0.5) * 32 / G).astype(np.int64)
    return np.flatnonzero(m[np.ix_(idx, idx)]).astype(np.int16)


def selftest(EN=None):
    EN = EN or load('c21_eci_encode', 'eci_encode.py')
    old = EN.to_grid  # the unpatched function (kron at 64, identity at 32)
    g = np.random.default_rng(0)
    tok = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/calms21')
    cases = [np.sort(g.choice(1024, n, replace=False)).astype(np.int16) for n in (0, 1, 5, 75, 300, 1024)]
    for f in ('eval_0of6', 'train_3of6'):
        import pandas as pd
        n32 = pd.read_parquet(tok / f'{f}_frames.parquet').n32.values
        p = np.load(tok / f'{f}_pos32.npy')
        st = np.r_[0, np.cumsum(n32)]
        cases += [p[st[j]:st[j + 1]] for j in range(0, len(n32), 7)]
    for p in cases:
        assert np.array_equal(to_grid(p, 32), old(p, 32)) and np.array_equal(to_grid(p, 32), p.astype(np.int16))
        assert np.array_equal(to_grid(p, 64), old(p, 64)), 'G = 64 differs from the kron rule'
        for G in (48, 80):
            q = to_grid(p, G)
            par = (np.floor((q // G + 0.5) * 32 / G).astype(int) * 32 + np.floor((q % G + 0.5) * 32 / G).astype(int))
            assert set(par.tolist()) <= set(p.tolist()) and (len(p) == 0) == (len(q) == 0)
            assert set(par.tolist()) == set(p.tolist()), 'every kept 32-cell must have >= 1 child at G >= 32'
    full = np.arange(1024)
    assert len(to_grid(full, 48)) == 48 ** 2 and len(to_grid(full, 80)) == 80 ** 2
    print(f'selftest OK: {len(cases)} masks; G 32 identity, G 64 == kron, G 48 / 80 parent sets exact')


def patch_encode():
    EN = load('c21_eci_encode', 'eci_encode.py')
    selftest(EN)
    EN.CONFIGS.update(NEW)
    EN.to_grid = to_grid
    return EN


def patch_train():
    CT = load('c21_eci_train', 'eci_train.py')
    assert CT.DIM['d3l_512_L20'] == CT.DIM['d3l_1024_L20'] == 1024
    for c in NEW:
        CT.DIM[c] = 1024
        assert c not in CT.G64
    orig = CT.load_shard

    def load_shard(base, cfg, lag, stage, tok='mask', set_='train'):
        if cfg not in NEW:
            return orig(base, cfg, lag, stage, tok, set_)
        import shutil
        import pandas as pd
        assert tok == 'mask' and lag == 60, (tok, lag)
        G = NEW[cfg][1] // 16
        Fm = pd.read_parquet(CT.TOK / f'{base}_frames.parquet')
        p32 = np.load(CT.TOK / f'{base}_pos32.npy')
        st = np.r_[0, np.cumsum(Fm.n32.values)]
        lens = np.array([len(to_grid(p32[st[j]:st[j + 1]], G)) for j in range(len(Fm))], np.int64)
        arr = {}
        for k in ('t', 'p60'):
            src = CT.TOK / f'{base}_{cfg}_{k}.npy'
            loc = stage / src.name
            shutil.copyfile(src, loc)
            arr[k] = np.load(loc)
            loc.unlink()
        assert lens.sum() == len(arr['t']) == len(arr['p60']) == len(np.load(CT.TOK / f'{base}_{cfg}_pos.npy')), \
            (base, cfg, lens.sum(), arr['t'].shape)
        D = arr['t'].shape[1]
        X = np.empty((len(arr['t']), 2 * D), np.float16)
        for c in range(0, len(X), 400_000):
            ta = arr['t'][c:c + 400_000]
            X[c:c + len(ta), :D] = ta
            X[c:c + len(ta), D:] = (ta.astype(np.float32) - arr['p60'][c:c + 400_000].astype(np.float32)).astype(np.float16)
        return X, lens, Fm
    CT.load_shard = load_shard
    return CT


# --------------------------------------------------------------------------------------------------------- scoring
ARMS = ['d3l_512_L20', 'd3l_768_L20', 'd3l_1024_L20', 'd3l_1280_L20']
NAME = {'d3l_512_L20': 'D3L-32', 'd3l_768_L20': 'D3L-48', 'd3l_1024_L20': 'D3L-64', 'd3l_1280_L20': 'D3L-80'}
E1 = [('d3l_768_L20', 'd3l_512_L20'), ('d3l_1024_L20', 'd3l_768_L20'), ('d3l_1280_L20', 'd3l_1024_L20')]


def sacct_gpu_h(prefix):
    """-> {job name: GPU-hours (elapsed x 1 GPU)} for COMPLETED jobs named prefix*."""
    out = subprocess.run(['sacct', '-n', '-P', '-X', '-S', '2026-10-19', '--name', ','.join(prefix), '-o',
                          'JobID,JobName,State,ElapsedRaw,NodeList'], capture_output=True, text=True).stdout
    rows = [r.split('|') for r in out.strip().splitlines() if r]
    return [{'job': r[0], 'name': r[1], 'state': r[2], 'gpu_h': int(r[3]) / 3600, 'node': r[4]} for r in rows]


def score():
    import pandas as pd
    S_ = load('c21_eci_res_score', 'eci_res_score.py')
    from calms_common import OUT, PLAN, TOK
    labs = json.loads((PLAN / 'eval_plan.json').read_text())['scored_labels']
    BL = json.loads((OUT / 'bodylen' / 'bodylen.json').read_text())['body_len_native_px_median']
    P32 = []
    for f in sorted(TOK.glob('*of6_frames.parquet')):
        n32 = pd.read_parquet(f).n32.values
        p = np.load(TOK / f.name.replace('_frames.parquet', '_pos32.npy'))
        st = np.r_[0, np.cumsum(n32)]
        P32 += [p[st[j]:st[j + 1]] for j in range(len(n32))]
    rate = {}
    for f in sorted((OUT / 'encode_json').glob('*.json')):
        if 'smoke' in f.name or f.name.startswith('p2_'):
            continue
        for c, v in json.loads(f.read_text())['per_config'].items():
            r = rate.setdefault(c, [0.0, 0])
            r[0] += v['gpu_s']
            r[1] += v['inputs']
    jobs = sacct_gpu_h(['res_calms_enc_train', 'res_calms_enc_eval', 'res_calms_tr_d3l_768_L20',
                        'res_calms_tr_d3l_1280_L20'])
    out = {'labels': labs, 'body_len_native_px': BL, 'arms': {}, 'e1': {}, 'jobs': jobs}
    for a in ARMS:
        R = S_.load(a)
        ps = S_.per_seed(R, labs)
        v = np.array(list(ps.values()))
        G = GRID[a]
        S = G * 16
        out['arms'][a] = {
            'name': NAME[a], 'grid': G, 'input_px': S, 'patches_per_body': BL * S / 1024 / 16,
            'upsampled_beyond_native': S > 1024, 'seeds': list(ps), 'per_seed_primary_auroc': list(ps.values()),
            'mean': float(v.mean()) if len(v) else None,
            'se': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            'sd': float(v.std(ddof=1)) if len(v) > 1 else None,
            'cf_auroc': {b: S_.mean_q(R, b, 'cf_auroc') for b in labs} if R else {},
            'cf_ap': {b: S_.mean_q(R, b, 'cf_ap') for b in labs} if R else {},
            'encoder_tokens_per_frame': 2 * (G * G + 5),
            'sae_tokens_per_frame': float(np.mean([len(to_grid(p, G)) for p in P32])),
            'frames_per_gpu_s': rate[a][1] / rate[a][0] / 2 if a in rate else None,
            'encode_gpu_s_summed': rate[a][0] if a in rate else None,
            'source': 'new (this test)' if a in NEW else 'existing per-seed eval JSONs (eci_res.sh)'}
    for a, b in E1:
        A, B = out['arms'][a]['per_seed_primary_auroc'], out['arms'][b]['per_seed_primary_auroc']
        if len(A) > 1 and len(B) > 1:
            d, lo, hi, se, df = S_.welch(A, B)
            out['e1'][f'{NAME[a]} vs {NAME[b]}'] = {'delta': d, 'ci': [lo, hi], 'df': df, 'verdict': S_.verdict(d, lo, hi)}
    # e2: smallest grid whose primary AUROC does NOT 'LOSE' versus the best-mean grid
    ok = [a for a in ARMS if len(out['arms'][a]['per_seed_primary_auroc']) > 1]
    if len(ok) == len(ARMS):
        best = max(ok, key=lambda a: out['arms'][a]['mean'])
        cmp = {}
        for a in ARMS:
            if a == best:
                cmp[a] = {'verdict': 'BEST'}
                continue
            d, lo, hi, se, df = S_.welch(out['arms'][a]['per_seed_primary_auroc'], out['arms'][best]['per_seed_primary_auroc'])
            cmp[a] = {'delta': d, 'ci': [lo, hi], 'df': df, 'verdict': S_.verdict(d, lo, hi)}
        choice = next(a for a in ARMS if cmp[a]['verdict'] != 'LOSES')
        out['e2'] = {'best_mean': best, 'vs_best': cmp, 'choice': choice, 'choice_grid': GRID[choice]}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'res.json').write_text(json.dumps(out, indent=1, default=float))
    L = ['# CalMS21: DINOv3-L block 20 token grid 32 / 48 / 64 / 80 (pre-registered, scripts/eci/resolution/calms21_res.sh)',
         '', f'primary AUROC = mean over {", ".join(labs)} of the cross-fitted best-neuron AUROC (AUROC-picked neuron), '
         'split seeds 0 and 1 averaged; one value per SAE seed (0-5). Same frames, same mask cells (pass A, 32 grid), '
         'same recipe (BatchTopK 4096, k 16, 6000 steps, Delta 2 s). D3L-32 and D3L-64 are the existing arms of '
         'eci_res.sh (their per-seed eval JSONs, not rerun). D3L-80 (1280 px) is UPSAMPLED beyond the native 1024 '
         f'canvas: it tests token density, not image detail. Body length {BL:.1f} native px.', '',
         '| arm | grid | input px | patches / body | per-seed primary AUROC | mean ± SE (sd) primary AUROC | cf AUROC '
         + ' / '.join(labs) + ' | encoder tokens / frame (t and t-60) | SAE tokens / frame (after mask) | '
         'encode frames / GPU-s |', '|---|---|---|---|---|---|---|---|---|---|']
    for a in ARMS:
        r = out['arms'][a]
        m = f"{r['mean']:.4f} ± {r['se']:.4f} (sd {r['sd']:.4f})" if r['se'] is not None else 'missing'
        fr = f"{r['frames_per_gpu_s']:.1f}" if r['frames_per_gpu_s'] else 'n/a'
        L.append(f"| {r['name']} ({a}) | {r['grid']} | {r['input_px']}{' (upsampled)' if r['upsampled_beyond_native'] else ''} "
                 f"| {r['patches_per_body']:.1f} | {', '.join(f'{x:.3f}' for x in r['per_seed_primary_auroc'])} | {m} | "
                 + ' / '.join(f"{r['cf_auroc'][b]:.3f}" for b in r['cf_auroc']) + f" | {r['encoder_tokens_per_frame']:,} "
                 f"| {r['sae_tokens_per_frame']:.0f} | {fr} |")
    L += ['', '## (e1) adjacent grids, WIN = delta primary AUROC >= +0.010 and Welch 95% CI excludes 0 (LOSES symmetric)',
          '', '| contrast | delta primary AUROC [95% CI] | df | verdict |', '|---|---|---|---|']
    for k, c in out['e1'].items():
        L.append(f"| {k} | {c['delta']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | {c['df']:.1f} | {c['verdict']} |")
    if 'e2' in out:
        e = out['e2']
        L += ['', f"## (e2) grid choice: best-mean grid = {NAME[e['best_mean']]}; choice = smallest grid not LOSING to it "
              f"= **{NAME[e['choice']]} ({e['choice_grid']} x {e['choice_grid']})**", '',
              '| arm | delta primary AUROC vs best [95% CI] | verdict |', '|---|---|---|']
        for a, c in e['vs_best'].items():
            L.append(f"| {NAME[a]} | " + (f"{c['delta']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]" if 'delta' in c
                                          else '-') + f" | {c['verdict']} |")
    L += ['', '## jobs (GPU-hours = elapsed x 1 GPU, sacct)', '', '| job | name | state | GPU-h | node |',
          '|---|---|---|---|---|']
    L += [f"| {j['job']} | {j['name']} | {j['state']} | {j['gpu_h']:.2f} | {j['node']} |" for j in jobs]
    L.append(f"\ntotal GPU-h (all listed jobs): {sum(j['gpu_h'] for j in jobs):.2f}")
    (RES / 'res.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == 'selftest':
        selftest()
    elif mode == 'encode':
        EN = patch_encode()
        sys.argv = ['eci_encode.py'] + rest
        EN.main()
    elif mode == 'train':
        CT = patch_train()
        sys.argv = ['eci_train.py'] + rest
        CT.main()
    elif mode == 'score':
        score()
    else:
        raise SystemExit(f'unknown mode {mode}')


if __name__ == '__main__':
    main()
