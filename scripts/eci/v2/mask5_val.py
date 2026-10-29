"""
Mask v5: cheap per-domain patch mask without SAM at inference (pre-registration: scripts/eci/v2/mask5_prereg.md).
Route (b) = logistic probe on DINOv3-L block-20 tokens + 3x3 smoothing + core rule (+ optional N2 thin-token OR).

Steps (wrapper scripts/eci/v2/mask4_chain.sh style; PY = this file):
  build --dom D   (GPU) tokens + N2 for 5-frame windows around the gold / sheet frames, and training frames with offline
                  SAM pseudo-labels (B_l_full) -> BEE/<dom>/*.npz
  fit             (CPU) probe per domain, dev tuning, test scoring, store numbers -> OUT/table.md, OUT/results.json
  sheet           (CPU) -> OUT/{mice,ants,frogs}.png
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci/v2'))
OUT = REPO / 'results/vision/eci_v2_0B/mask_approval_v3'
OUT4 = REPO / 'results/vision/eci_v2_0B/mask_approval_v2'
BEE = Path('/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5')
log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)  # noqa: E731
SER = {'mice': ['v1', 'v2'], 'ants': ['v2', 'v3', 'v4'], 'frogs': ['juv', 'tad']}
N_TRAIN = {'mice': 60, 'ants': 40, 'frogs': 60}  # per series
FRAMES = {'mice': 3.89e6, 'ants': 1.82e6, 'frogs': 2.76e6}


def app_tokens(app, patch=16):
    import torch
    import torch.nn.functional as F
    return (F.avg_pool2d(app.float()[:, None], patch).flatten(1) * patch * patch >= 16)


def cmd_build(a):
    import torch
    import mask3_val as MV
    from src.eci.v2 import mask4 as M4
    from src.eci.v2.encode import Encoder
    dev = torch.device('cuda')
    enc = Encoder(dev)
    if not torch.cuda.is_bf16_supported():  # RTX 2080 Ti: the store's bf16 autocast is not available -> fp32 tokens
        import src.eci.v2.encode as EN

        def call32(self, x):
            xin = (x.float() / 255 - self.mean) / self.std
            with torch.no_grad():
                h = self.m(pixel_values=xin).last_hidden_state
            z = h[:, EN.N_PRE:].float()
            assert torch.isfinite(z).all()
            return z.half()
        EN.Encoder.__call__ = call32
        log('encoder: fp32 (bf16 unsupported on this GPU)')
    sam, proc = M4.load_sam('large', dev)
    split = json.loads((OUT / 'split.json').read_text())
    test_keys = set(split[a.dom]['test'])
    for ser in SER[a.dom]:
        (BEE / a.dom).mkdir(parents=True, exist_ok=True)
        o, _ = MV.series_pair(a.dom, ser, dev, rule='m3b')
        meta = json.loads((OUT4 / 'cap' / f'{a.dom}_{ser}.json').read_text())
        test_vids = {k.split(f'{a.dom}_{ser}_', 1)[1].rsplit('_', 1)[0] for k in test_keys if k.startswith(f'{a.dom}_{ser}_')}

        def run(v, ks):
            fr = o.run_job(**o.job(v, ks[0], len(ks), threads=4))
            o.cap.clear()
            o.frames(v, fr, ks)
            cap = {key: torch.cat([c[key] for c in o.cap]) for key in ('x', 'core0', 'pix', 'app', 'keep')}
            torch.cuda.synchronize()
            t0 = time.time()
            z = torch.cat([enc(cap['x'][i:i + 1]) for i in range(0, len(ks))])
            torch.cuda.synchronize()
            return cap, z, (time.time() - t0) / len(ks)
        tenc = []
        # ---- 5-frame windows around gold + sheet frames
        for m in meta:
            v = o.V[m['obs']]
            k = m['k']
            ks = list(range(max(0, k - 2), min(v['n'] - 1, k + 2) + 1))
            cap, z, te = run(v, ks)
            tenc.append(te)
            it = ks.index(k)
            c4 = M4V_load_cap(a.dom, ser, m['obs'], k)
            same = float((cap['x'][it].permute(1, 2, 0).cpu().numpy().astype(np.uint8) == c4['x'][-1]).mean())
            np.savez(BEE / a.dom / f'win_{ser}_{m["obs"]}_{k}.npz', z=z.cpu().numpy(), it=it, ks=np.array(ks),
                     n2=cap['keep'].cpu().numpy(), n2app=app_tokens(cap['app']).cpu().numpy(),
                     x=cap['x'][it].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                     n2pix=np.packbits(cap['pix'][it].cpu().numpy(), -1), h=o.h, G=o.G, role=m['role'], same_x=same)
            log(f'win {ser} {m["obs"]} k{k} role {m["role"]} x identical to mask4 capture: {same:.4f}')
        # ---- training frames (videos without test frames), SAM B_l_full pseudo-labels
        gold_frames = {(m['obs'], m['k']) for m in meta}
        pool = [f for f in MV.sample_frames(o) if f[0] not in test_vids and f not in gold_frames]
        g = np.random.default_rng(5)
        pick = [pool[i] for i in g.permutation(len(pool))[:N_TRAIN[a.dom]]]
        for j, (obs, k) in enumerate(pick):
            v = o.V[obs]
            try:
                cap, z, te = run(v, [k])
            except RuntimeError as e:
                log(f'DECODE FAIL {obs} {k}: {str(e)[:100]}')
                continue
            tenc.append(te)
            pix, app = cap['pix'][0].cpu().numpy(), cap['app'][0].cpu().numpy()
            core = cap['core0'][0].cpu().numpy() | (pix & ~app)
            objs = M4.objects(pix, core, o.h)
            ch = M4.sam_full(sam, proc, cap['x'][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), objs, dev)
            u, _ = M4.assemble(ch, objs, pix, core, o.h, rule='B', app=app)
            lab = M4.tokens(u, o.h, dev=dev)
            np.savez(BEE / a.dom / f'train_{ser}_{obs}_{k}.npz', z=z[0].cpu().numpy(), lab=lab,
                     n2=cap['keep'][0].cpu().numpy(), n2app=app_tokens(cap['app'])[0].cpu().numpy(), G=o.G)
            if j % 10 == 0:
                log(f'train {ser} {j + 1}/{len(pick)} lab {int(lab.sum())} n2 {int(cap["keep"][0].sum())}')
        log(f'{a.dom} {ser}: encode s/frame median {np.median(tenc):.4f} on {torch.cuda.get_device_name(0)}')
        (BEE / a.dom / f'enc_time_{ser}.json').write_text(json.dumps({'s_per_frame_median': float(np.median(tenc)),
                                                                        'device': torch.cuda.get_device_name(0)}))


def M4V_load_cap(dom, ser, obs, k):
    import mask4_val as M4V
    return M4V.load_cap(dom, ser, obs, k)


# ------------------------------------------------------------------------------------------------ fit / score
def postproc(p, G, t, t_core, n2app=None, smooth=True):
    """p (G*G,) prob -> keep (G*G,) bool."""
    import cv2
    P = p.reshape(G, G).astype(np.float32)
    ps = cv2.blur(P, (3, 3), borderType=cv2.BORDER_REPLICATE) if smooth else P
    keep = ps >= t
    if n2app is not None:
        A = n2app.reshape(G, G)
        kd = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # N2 thin tokens attached to the mask (through other N2 thin tokens, 8-connected)
        n, lab = cv2.connectedComponents((A | keep).astype(np.uint8), connectivity=8)
        good = np.unique(lab[keep & (lab > 0)])
        keep = keep | (A & np.isin(lab, good) & (kd | A))
    n, lab = cv2.connectedComponents(keep.astype(np.uint8), connectivity=8)
    if n > 1:
        core = np.unique(lab[(P >= t_core) & (lab > 0)])
        keep = np.isin(lab, core[core > 0])
    return keep.ravel()


def score(keep, gold_pix, thin, G, h):
    import cv2
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask4 as M4
    S = gold_pix.shape[0]
    gtok = M4.tokens(gold_pix, h)
    gfrac = F.avg_pool2d(torch.from_numpy(gold_pix).float()[None, None], S // G).flatten().numpy()
    bg = keep & (gfrac < 0.05)
    n, lab = cv2.connectedComponents(keep.reshape(G, G).astype(np.uint8), connectivity=8)
    iso = sum(1 for c in range(1, n) if not (gfrac.reshape(G, G)[lab == c] >= 0.05).any())
    st = {'tp': int((keep & gtok).sum()), 'kept': int(keep.sum()), 'gold': int(gtok.sum()), 'bg': int(bg.sum()),
          'iso': iso, 'thin_n': len(thin), 'thin_hit': 0}
    if len(thin):
        ti = np.clip(np.asarray(thin).round().astype(int), 0, S - 1)
        cell = (ti[:, 1] // (S // G)) * G + ti[:, 0] // (S // G)
        st['thin_hit'] = int(keep[cell].sum())
    return st


def summ(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    rec, pre = s['tp'] / max(s['gold'], 1), s['tp'] / max(s['kept'], 1)
    return {'token_f1': 2 * rec * pre / max(rec + pre, 1e-9), 'token_recall': rec, 'token_precision': pre,
            'thin_recall': s['thin_hit'] / max(s['thin_n'], 1), 'bg_tok_per_frame': s['bg'] / len(rows),
            'iso_bg_comp_per_frame': s['iso'] / len(rows), 'tok_per_frame': s['kept'] / len(rows), 'n_frames': len(rows),
            'n_thin': s['thin_n']}


def fit_probe(Z, Y, wd=1e-3, steps=200):
    import torch
    Z = torch.from_numpy(Z).float()
    Y = torch.from_numpy(Y).float()
    mu, sd = Z.mean(0), Z.std(0) + 1e-4
    X = (Z - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pw = (1 - Y).sum() / Y.sum().clamp(min=1)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        lo = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, Y, pos_weight=pw.sqrt())
        loss = lo + wd * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {'mu': mu.numpy(), 'sd': sd.numpy(), 'w': w.detach().numpy(), 'b': float(b.detach())}


def probe_p(P, z):
    x = (z.astype(np.float32) - P['mu']) / P['sd']
    return 1 / (1 + np.exp(-(x @ P['w'] + P['b'])))


def cmd_fit(a):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    split = json.loads((OUT / 'split.json').read_text())
    res = {}
    L = ['# Mask v5 (no SAM at inference): test-split metrics (pre-registered in scripts/eci/v2/mask5_prereg.md)', '']
    for dom in a.doms.split(','):
        D = BEE / dom
        # ---- train
        Zs, Ys = [], []
        g = np.random.default_rng(0)
        for f in sorted(D.glob('train_*.npz')):
            z_ = np.load(f)
            y = z_['lab'].astype(bool)
            neg = np.flatnonzero(~y)
            sel = np.concatenate([np.flatnonzero(y), g.choice(neg, min(len(neg), max(4 * int(y.sum()), 200)), replace=False)])
            Zs.append(z_['z'][sel])
            Ys.append(y[sel])
        Z, Y = np.concatenate(Zs), np.concatenate(Ys).astype(np.float32)
        t0 = time.time()
        P = fit_probe(Z, Y)
        log(f'{dom}: probe on {len(Zs)} frames, {len(Y)} tokens ({int(Y.sum())} pos), {time.time() - t0:.0f}s')
        np.savez(D / 'probe.npz', **{k: np.asarray(v) for k, v in P.items()})
        # ---- frames
        fr = {}
        for f in sorted(D.glob('win_*.npz')):
            z_ = np.load(f)
            ser, rest = f.stem[4:].split('_', 1)
            obs, k = rest.rsplit('_', 1)
            key = f'{dom}_{ser}_{obs}_{k}'
            it = int(z_['it'])
            G, h = int(z_['G']), float(z_['h'])
            t1 = time.time()
            pw = np.stack([probe_p(P, z_['z'][i]) for i in range(len(z_['z']))])
            tp = (time.time() - t1) / len(pw)
            fr[key] = {'p': pw, 'it': it, 'G': G, 'h': h, 'n2': z_['n2'].astype(bool), 'n2app': z_['n2app'].astype(bool),
                       'role': str(z_['role']), 'tprobe': tp, 'ser': ser, 'obs': obs, 'k': int(k), 'same_x': float(z_['same_x'])}
        gold = {}
        for key in fr:
            gf = OUT4 / 'gold' / f'{key}.npz'
            if gf.exists():
                gz = np.load(gf)
                S = fr[key]['G'] * 16
                gold[key] = (np.unpackbits(gz['pix'], -1)[:, :S].astype(bool), gz['thin'])

        def eval_set(keys, fn):
            return summ([score(fn(key), gold[key][0], gold[key][1], fr[key]['G'], fr[key]['h']) for key in keys])
        n2fn = lambda key: fr[key]['n2'][fr[key]['it']]  # noqa: E731

        def newfn(cfg):
            return lambda key: postproc(fr[key]['p'][fr[key]['it']], fr[key]['G'], cfg['t'], cfg['t_core'],
                                        fr[key]['n2app'][fr[key]['it']] if cfg['or_n2'] else None)
        devk = [k_ for k_ in split[dom]['dev'] if k_ in fr]
        testk = [k_ for k_ in split[dom]['test'] if k_ in fr]
        n2dev = eval_set(devk, n2fn)
        grid = []
        for t in (0.2, 0.3, 0.4, 0.5, 0.6):
            for tc in (0.5, 0.8, 0.9, 0.95):
                for orn in (False, True):
                    cfg = {'t': t, 't_core': tc, 'or_n2': orn}
                    m = eval_set(devk, newfn(cfg))
                    grid.append((cfg, m))
        ok = [(c, m) for c, m in grid if m['thin_recall'] >= n2dev['thin_recall'] and m['bg_tok_per_frame'] <= n2dev['bg_tok_per_frame']]
        cfg, mdev = max(ok or grid, key=lambda cm: cm[1]['token_f1'])
        # ---- test (once)
        import mask4_val as M4V
        arm = {}
        for key in testk:
            q = fr[key]
            am, _ = M4V.arm_masks(dom, q['ser'], {'obs': q['obs'], 'k': q['k']})
            arm[key] = am['B_s_crop'][1]
        out = {'N2': eval_set(testk, n2fn), 'new': eval_set(testk, newfn(cfg)), 'B_s_crop': eval_set(testk, lambda key: arm[key])}
        o_ = out
        passed = (o_['new']['token_f1'] > o_['N2']['token_f1'] and o_['new']['thin_recall'] > o_['N2']['thin_recall']
                  and o_['new']['bg_tok_per_frame'] <= o_['N2']['bg_tok_per_frame'])
        # ---- store numbers on all windows (gold + sheet frames)
        rat, ins, ut = [], [], []
        import cv2
        for key, q in fr.items():
            ks = [postproc(q['p'][i], q['G'], cfg['t'], cfg['t_core'], q['n2app'][i] if cfg['or_n2'] else None) for i in range(len(q['p']))]
            U = np.any(ks, 0)
            kt = ks[q['it']]
            G = q['G']
            Sd = np.any([cv2.dilate(n.reshape(G, G).astype(np.uint8), np.ones((3, 3), np.uint8)).ravel() > 0 for n in q['n2']], 0)
            if kt.sum():
                rat.append(U.sum() / kt.sum())
                ins.append((kt & Sd).sum() / kt.sum())
            ut.append(U.sum())
        store = {'union5_over_t': float(np.mean(rat)), 'new_inside_S_N2': float(np.mean(ins)),
                 'union_tok_per_frame': float(np.mean(ut)),
                 'store_TB': float(np.mean(ut) * 1024 * 2 * FRAMES[dom] / 1e12),
                 'probe_s_per_frame': float(np.median([q['tprobe'] for q in fr.values()])),
                 'x_identical_min': float(min(q['same_x'] for q in fr.values()))}
        res[dom] = {'cfg': cfg, 'dev': {'new': mdev, 'N2': n2dev}, 'test': out, 'pass': bool(passed), 'store': store,
                    'n_train_frames': len(Zs), 'n_train_tokens': int(len(Y)), 'test_keys': testk}
        log(json.dumps({dom: {k: v for k, v in res[dom].items() if k != 'test_keys'}}, default=float))
        np.save(D / 'cfg.npy', cfg)
        cols = ['token_f1', 'token_recall', 'token_precision', 'thin_recall', 'bg_tok_per_frame', 'iso_bg_comp_per_frame', 'tok_per_frame']
        L += [f'## {dom}: route (b) probe, cfg {cfg}; PASS = {passed}', '',
              '| arm | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
        for arm_, m in out.items():
            L.append(f'| {arm_} | ' + ' | '.join(f'{m[c]:.3f}' for c in cols) + ' |')
        L += ['', f'test: {out["new"]["n_frames"]} frames, {out["new"]["n_thin"]} thin points; store: {store}', '']
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / f'table{a.tag}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step')
    ap.add_argument('--dom', default='mice')
    ap.add_argument('--doms', default='mice,ants,frogs')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    globals()['cmd_' + a.step](a)


if __name__ == '__main__':
    main()
