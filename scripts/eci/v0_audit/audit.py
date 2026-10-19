"""
Audit of the v0 replication (original ECI paper, arXiv 2510.14073 v2, ants v2 / ISTAnt): why does it not
reproduce the paper's grooming neuron #394 (F1 0.398) and position neuron #550 (F1 0.568)?

Re-uses the cached DINOv2-S/14 patch tokens of scripts/eci/v0/ants_v2_v0.py (44 videos x 1200 frames x
256 patches x 384, on BeeGFS) and its SAE trainer (overcomplete TopKSAE 384 -> 4608, k 20, Adam 5e-4,
MSE, clip 1.0, 10 epochs, no shuffle).

PRE-REGISTERED variants (fixed before any result of this script was seen):
  SAE training (both on single patch tokens, as the original s3-train_sae.py collate does):
    b128 : 128 frames per batch (paper Table 2).               seeds 0-4
    b20  : 20 frames per batch (s3-train_sae.py default).      seeds 0-2
  Frame encoding (what the SAE is applied to; per frame, then averaged over the video's frames):
    mean : mean of the 256 patch tokens (paper App. D.2 text, current s4-sae_encoding.py default)
    tok0 : patch token 0 = the top-left patch. The old s4-sae_encoding.py (commits <= 5f8ffad,
           2025-09-10..16) defaulted to --pooler CLS = batch[:, 0, :], but overcomplete's DinoV2 returns
           patch tokens only, so "CLS" was really the top-left patch. The ISTAnt notebook reads codes from
           SAE/TOPK/DINOV2/conorome/codes, the OLD output layout without the pooler folder, so the paper's
           codes were most likely written by that old script.
    patchavg (diagnostic, not a paper recipe): SAE on every patch token (its training distribution),
           codes averaged over the 256 patches.
  Search: single-pass Welch t-test per code on video means, raw p < 0.05 (the paper's NES, no Bonferroni);
          Bonferroni count reported next to it; original recursive NES (t-test) as a cross-check.
  Grooming: label = yellow OR blue grooms the focal ant (the notebook's grooming_2focal); frame rule
          code > 0 (prf_rule_sign0_multi). Reported per variant: best-F1 code over all codes (the paper's
          selection, in-sample), its AUROC, R~ (normalised R-precision, 0 = chance, 1 = perfect), the
          always-on F1 baseline, and a CROSS-FITTED version (select best-F1 code on one half of the
          videos, score on the other half, 2 folds, stratified by arm).
  Position: per NES hit, eta^2 of its video means by position (1-9) with a 5000-permutation p, and its
          F1 for the paper's 550 label "position 1 and batch a-d"; the best code for that label overall.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/eci/v0'))
from ants_v2_v0 import train_sae, FVE, N_FRAMES, N_CODES, DIM  # noqa: E402
from nes_original import neural_effect_search  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode(sae, tokens, mode, device, bs=128):
    n = len(tokens)
    out = np.empty((n, N_CODES), dtype=np.float32)
    fve = FVE()
    with torch.inference_mode():
        for s in range(0, n, bs):
            f = torch.from_numpy(tokens[s:s + bs]).to(device)
            if mode == 'mean':
                x = f.mean(1)
                _, z, xh = sae(x)
                fve.update(x, xh)
            elif mode == 'tok0':
                x = f[:, 0]
                _, z, xh = sae(x)
                fve.update(x, xh)
            elif mode == 'patchavg':
                x = f.reshape(-1, DIM)
                _, z, xh = sae(x)
                fve.update(x, xh)
                z = z.reshape(len(f), -1, N_CODES).mean(1)
            out[s:s + len(f)] = z.float().cpu().numpy()
    return out, fve.value()


def f1_all(act, y):
    pos = y == 1
    tp = act[pos].sum(0).astype(np.float64)
    fp = act[~pos].sum(0).astype(np.float64)
    fn = pos.sum() - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0)
        return np.where(p + r > 0, 2 * p * r / (p + r), 0)


def r_tilde(score, y):
    R = int(y.sum())
    top = np.argsort(-score, kind='stable')[:R]
    prec = y[top].mean()
    prev = y.mean()
    return float((prec - prev) / (1 - prev))


def eta2(v, g):
    v = np.asarray(v, float)
    sst = ((v - v.mean()) ** 2).sum()
    if sst == 0:
        return float('nan')
    return float(sum((g == k).sum() * (v[g == k].mean() - v.mean()) ** 2 for k in np.unique(g)) / sst)


def perm_p(v, g, rng, n=5000):
    obs = eta2(v, g)
    null = np.array([eta2(v, rng.permutation(g)) for _ in range(n)])
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n))


def analyse(codes, videos, y, pos1_label, rng):
    nv = len(videos)
    T = (videos.treatment.to_numpy() == 2).astype(int)
    position = videos.position.to_numpy()
    vid = np.repeat(np.arange(nv), N_FRAMES)
    vc = codes.reshape(nv, N_FRAMES, N_CODES).astype(np.float64).mean(1)
    act = codes > 0
    alive = np.where(act.any(0))[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        t, p = ttest_ind(vc[T == 1], vc[T == 0], axis=0, equal_var=False)
    valid = np.isfinite(p)
    m = int(valid.sum())
    hits = [int(j) for j in np.where(valid & (p < 0.05))[0]]
    hits.sort(key=lambda j: p[j])
    prev = y.mean()
    res = dict(n_alive=int(len(alive)), n_testable=m, n_hits_raw=len(hits), expected_fp=0.05 * m,
               n_hits_bonf=int((valid & (p < 0.05 / max(m, 1))).sum()), min_p=float(np.nanmin(p)) if m else None,
               always_on_f1=float(2 * prev / (1 + prev)), prevalence=float(prev))
    # original recursive NES (t-test), cross-check
    df = pd.DataFrame(vc, columns=[f'N_{j + 1}' for j in range(N_CODES)])
    df.insert(0, 'T', T)
    rec = neural_effect_search(df, method='t-test', alpha=0.05, k=2000, neuron_prefix='N_', fallback_start_neur_index=4)
    res['recursive_nes_hits'] = [int(s) for s in rec['neurons']]
    # grooming, in-sample (paper's selection)
    f1 = f1_all(act, y)
    f1pos = f1_all(act, pos1_label)
    auc = np.full(N_CODES, np.nan)
    for j in alive:
        auc[j] = roc_auc_score(y, codes[:, j])
    jb = int(np.argmax(f1))
    order = np.argsort(-f1)
    rank = {int(j): int(np.where(order == j)[0][0]) + 1 for j in hits}
    res['best_f1_code'] = dict(neuron=f'N_{jb + 1}', f1=float(f1[jb]), auroc=float(auc[jb]), r_tilde=r_tilde(codes[:, jb], y),
                               p=float(p[jb]) if np.isfinite(p[jb]) else None, is_nes_hit=jb in hits)
    ja = int(np.nanargmax(auc))
    res['best_auroc_code'] = dict(neuron=f'N_{ja + 1}', auroc=float(auc[ja]), f1=float(f1[ja]), r_tilde=r_tilde(codes[:, ja], y),
                                  p=float(p[ja]) if np.isfinite(p[ja]) else None, is_nes_hit=ja in hits)
    jp = int(np.argmax(f1pos))
    res['best_pos1_batch_ad_code'] = dict(neuron=f'N_{jp + 1}', f1=float(f1pos[jp]), is_nes_hit=jp in hits,
                                          always_on_f1=float(2 * pos1_label.mean() / (1 + pos1_label.mean())))
    # grooming, cross-fitted by video (2 folds, stratified by arm)
    cf = []
    for rep in range(5):
        r2 = np.random.default_rng(100 + rep)
        fold = np.zeros(nv, int)
        for arm in (0, 1):
            idx = np.where(T == arm)[0]
            fold[r2.permutation(idx)[: len(idx) // 2]] = 1
        for k in (0, 1):
            tr, te = np.isin(vid, np.where(fold == k)[0]), np.isin(vid, np.where(fold != k)[0])
            js = int(np.argmax(f1_all(act[tr], y[tr])))
            yt = y[te]
            cf.append(dict(f1=float(f1_all(act[te][:, [js]], yt)[0]),
                           auroc=float(roc_auc_score(yt, codes[te, js])) if codes[te, js].std() > 0 else 0.5,
                           r_tilde=r_tilde(codes[te, js], yt), always_on=float(2 * yt.mean() / (1 + yt.mean()))))
    res['crossfit_best_f1'] = {k: float(np.mean([c[k] for c in cf])) for k in cf[0]}
    # per hit
    rows = []
    for j in hits:
        e, pp = perm_p(vc[:, j], position, rng)
        rows.append(dict(neuron=f'N_{j + 1}', p=float(p[j]), welch_t=float(t[j]), late=float(vc[T == 1, j].mean() - vc[T == 0, j].mean()),
                         frac_active=float(act[:, j].mean()), groom_f1=float(f1[j]), groom_f1_rank=rank[j],
                         groom_auroc=float(auc[j]), groom_r_tilde=r_tilde(codes[:, j], y),
                         pos1_batch_ad_f1=float(f1pos[j]), eta2_position=e, eta2_position_perm_p=pp,
                         eta2_frames_by_video=eta2(codes[:, j], vid)))
    res['hits'] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--token-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--plan', default='b128:0,1,2,3,4;b20:0,1,2')
    ap.add_argument('--sae-path', default=None, help='use this trained SAE instead of training (key: cached)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    device = args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    exp = pd.read_csv(REPO / 'data/ants/v2/experiment.csv')
    videos = exp[exp.valid == 1].reset_index(drop=True)
    assert len(videos) == 44
    fr = pd.read_csv(Path(args.token_dir) / 'frames.csv')
    assert (fr.observation_id.to_numpy() == np.repeat(videos.observation_id, N_FRAMES).to_numpy()).all()
    y = ((fr.groom_yellow + fr.groom_blue) > 0).to_numpy().astype(np.int8)
    pos1 = np.repeat(((videos.position == 1) & (videos.batch != 'e')).to_numpy(), N_FRAMES).astype(np.int8)
    log(f'grooming prevalence {y.mean():.4f}; pos1&batch a-d prevalence {pos1.mean():.4f}')
    tokens = np.concatenate([np.load(Path(args.token_dir) / f'tokens_batch_{b}.npy') for b in 'abcde'])
    assert tokens.shape == (44 * N_FRAMES, 256, DIM), tokens.shape
    log(f'tokens loaded {tokens.shape}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    jobs = [('cached', None, None)] if args.sae_path else [
        (f'{part.split(":")[0]}_s{sd}', int(part.split(':')[0][1:]), int(sd))
        for part in args.plan.split(';') for sd in part.split(':')[1].split(',')]
    for key, bf, seed in jobs:
            if (out / f'audit_{key}.json').exists():
                continue
            allres = {}
            t0 = time.time()
            if args.sae_path:
                sae = torch.load(args.sae_path, map_location=device, weights_only=False)
                sae.eval()
                hist = [dict(note=f'loaded {args.sae_path}')]
            else:
                sae, hist = train_sae(tokens, device, 10, bf, seed)
            log(f'{key}: SAE ready in {time.time() - t0:.0f}s, last epoch {hist[-1]}')
            allres[key] = dict(train_last_epoch=hist[-1], device=device)
            for mode in ['mean', 'tok0', 'patchavg']:
                codes, fve = encode(sae, tokens, mode, device)
                r = analyse(codes, videos, y, pos1, rng)
                r['fve_on_input'] = fve
                allres[key][mode] = r
                log(f"{key}/{mode}: FVE {fve:.3f} alive {r['n_alive']} hits {r['n_hits_raw']} (exp {r['expected_fp']:.2f}, "
                    f"bonf {r['n_hits_bonf']}) best-F1 {r['best_f1_code']} cf {r['crossfit_best_f1']} pos {r['best_pos1_batch_ad_code']}")
                for h in r['hits']:
                    log(f'    hit {h}')
            (out / f'audit_{key}.json').write_text(json.dumps(allres, indent=1))
    log('DONE')


if __name__ == '__main__':
    main()
