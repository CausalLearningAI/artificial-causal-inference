"""
Crop prototype step 3: gates of the single-mouse and pair/contact crop SAEs vs the fg448 SAE, on the
same 108k held-out annotated 5 fps frames (behaviour labels used for EVALUATION ONLY).

Frame codes: crop SAE codes (threshold inference) max-pooled over the frame's crops of that kind
(frames without such a crop -> all zeros); fg448: codes_max over foreground patches
(dataset/mice/v1/eci/fg448/eval_codes).
Per representation, seed and event (Y_nn, Y_np, Y_nt > 0):
    raw           best single-latent AUROC
    size_fg448    stratified AUROC within quintiles of the fg448 foreground size n_fg (as eval_sae_gates.py)
    dist          stratified AUROC within quintiles of the min inter-blob distance
    mono          best - second-best latent AUROC and # latents within 0.02 of the best (raw and size_fg448)
Baselines: fg size alone, and 'min inter-blob distance' alone (score = -distance; 0 when a merged
blob exists, 999 when < 2 blobs and none merged); both raw and (distance) within fg-size quintiles.
Stability: decoder_stability (s0 vs s1) and, for each event's best size-controlled s0 latent, its
max decoder cosine to the s1 dictionary.
Grids: top 16 crops (pair SAE s0 and single SAE s0) of each event's best size-controlled latent,
at most one crop per video per 10 s, with the frame's annotations printed (nn/np/nt).
Pass: pair beats fg448 (full dictionary, s0) on size-controlled AUROC for >= 2 of 3 events AND
pair raw AUROC beats the distance baseline on those same >= 2 events.

Output: dataset/mice/v1/eci/crops/gates.json, top16_<kind>_<event>.jpg
Usage: python scripts/eci/crops_gates.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/eci'))
from eval_sae_fg import EVENTS, auroc_cols  # noqa: E402
from eval_sae_gates import quintiles, stratified_auroc  # noqa: E402
from src.eci.crops import sample_crops  # noqa: E402
from src.eci.sae import decoder_stability, load_sae  # noqa: E402

CROPS = REPO / 'dataset/mice/v1/eci/crops'
FG = REPO / 'dataset/mice/v1/eci/fg448/eval_codes'
FG_SAE = 'matryoshka_btk_1024_k16_fg448'
KINDS = {'single': (0,), 'pair': (1, 2)}
SAE = 'matryoshka_btk_512_k8'


def load_eval():
    parts = [np.load(f) for f in sorted((CROPS / 'eval').glob('task_*.npz')) if '.tmp' not in f.name]
    cat = lambda k: np.concatenate([z[k] for z in parts])
    return {k: cat(k) for k in ('rows', 'geom', 'crop_row', 'crop_spec', 'cls')}


@torch.no_grad()
def crop_codes(sae_path, X, dev):
    sae, norm, _ = load_sae(sae_path, dev)
    out = []
    for a in range(0, len(X), 65536):
        out.append(sae.encode(norm(torch.from_numpy(X[a:a + 65536]).to(dev)), mode='threshold').cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, sae.n_latents), np.float32), sae.W_dec.detach().cpu()


def frame_max(codes, fidx, n_frames):
    out = np.zeros((n_frames, codes.shape[1]), np.float32)
    np.maximum.at(out, fidx, codes)
    return out


def metrics(X, y, strata_fg, strata_d):
    raw = auroc_cols(X, y)
    sc = stratified_auroc(X, y, strata_fg)
    sd = stratified_auroc(X, y, strata_d)
    def mono(a):
        o = np.sort(a)[::-1]
        return {'best': float(o[0]), 'second': float(o[1]), 'gap': float(o[0] - o[1]), 'n_within_0.02': int((a >= o[0] - 0.02).sum())}
    return {'raw': {'latent': int(np.argmax(raw)), 'auroc': float(raw.max())},
            'size_fg448': {'latent': int(np.argmax(sc)), 'auroc': float(sc.max()), 'raw_of_that': float(raw[np.argmax(sc)])},
            'dist': {'latent': int(np.argmax(sd)), 'auroc': float(sd.max())},
            'mono_raw': mono(raw), 'mono_sc': mono(sc)}, sc


def grid(tiles, path):
    W = Image.new('RGB', (8 * 160, 2 * 160), 'white')
    for k, im in enumerate(tiles[:16]):
        W.paste(im.resize((160, 160)), ((k % 8) * 160, (k // 8) * 160))
    W.save(path, quality=85)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann = pd.read_csv(REPO / 'dataset/mice/v1/annotations.csv', usecols=['observation_id', 'frame_path', *EVENTS])
    E = load_eval()
    rows = E['rows']
    o = np.argsort(rows); rows, geom = rows[o], E['geom'][o]
    fgp = [np.load(f) for f in sorted(FG.glob('task_*.npz')) if '.tmp' not in f.name]
    frows = np.concatenate([z['rows'] for z in fgp]); fo = np.argsort(frows)
    if not np.array_equal(frows[fo], rows):
        raise ValueError('crop eval rows differ from fg448 eval rows')
    nfg = np.concatenate([z['n_fg'] for z in fgp])[fo].astype(np.float64)
    fidx = np.searchsorted(rows, E['crop_row'])
    ctype = E['crop_spec'][:, 0].astype(int)
    lab = ann.iloc[rows]
    Y = {ev: lab[ev].values > 0 for ev in EVENTS}
    q_fg, q_d = quintiles(nfg), quintiles(geom[:, 4].astype(np.float64))
    n_frames = len(rows)
    res = {'n_frames': int(n_frames), 'n_pos': {ev: int(Y[ev].sum()) for ev in EVENTS},
           'crop_stats': {'n_crops_by_type': np.bincount(ctype, minlength=3).tolist(),
                          'blobs_per_frame_hist': {int(k): int(v) for k, v in zip(*np.unique(geom[:, 0], return_counts=True))},
                          'frac_frames_with_pair_crop': float(np.isin(np.arange(n_frames), fidx[ctype == 1]).mean()),
                          'frac_frames_with_merged_crop': float((geom[:, 2] > 0).mean()),
                          'frac_frames_with_contact_crop': float(np.isin(np.arange(n_frames), fidx[ctype >= 1]).mean()),
                          'frac_frames_with_single_crop': float((geom[:, 1] > 0).mean()),
                          'mean_single_per_frame': float(geom[:, 1].mean()),
                          'frac_min_dist_quintile_sizes': np.bincount(q_d, minlength=5).tolist()}}
    base = {}
    for ev in EVENTS:
        y = Y[ev]
        base[ev] = {'fg_size_raw': float(auroc_cols(nfg[:, None], y)[0]),
                    'min_dist_raw': float(auroc_cols(-geom[:, 4:5].astype(np.float64), y)[0]),
                    'min_dist_within_fg_size': float(stratified_auroc(-geom[:, 4:5].astype(np.float64), y, q_fg)[0]),
                    'fg_size_within_min_dist': float(stratified_auroc(nfg[:, None], y, q_d)[0])}
    res['baselines'] = base
    reps, best_sc, W = {}, {}, {}
    for s in ('s0', 's1'):
        Xf = np.concatenate([z[f'codes_max_{FG_SAE}_{s}'] for z in fgp])[fo].astype(np.float32)
        for m in (512, 1024):
            reps.setdefault(f'fg448_m{m}', {})[s] = {ev: metrics(Xf[:, :m], Y[ev], q_fg, q_d)[0] for ev in EVENTS}
        del Xf
        W[('fg448', s)] = load_sae(REPO / f'dataset/mice/v1/eci/sae/{FG_SAE}_{s}/sae.pt')[0].W_dec.detach()
    crop_act = {}
    for kind, types in KINDS.items():
        sel = np.isin(ctype, types)
        for s in ('s0', 's1'):
            codes, W[(kind, s)] = crop_codes(CROPS / 'sae' / f'{SAE}_{kind}_{s}' / 'sae.pt', E['cls'][sel], dev)
            X = frame_max(codes, fidx[sel], n_frames)
            r = {}
            for ev in EVENTS:
                r[ev], sc = metrics(X, Y[ev], q_fg, q_d)
                if s == 's0':
                    best_sc[(kind, ev)] = int(np.argmax(sc))
            reps.setdefault(kind, {})[s] = r
            if s == 's0':
                crop_act[kind] = (np.nonzero(sel)[0], codes)
            print(kind, s, {ev: (round(v['raw']['auroc'], 3), round(v['size_fg448']['auroc'], 3)) for ev, v in r.items()}, flush=True)
    res['reps'] = reps
    stab = {}
    for rep, pref in (('fg448', (512, 1024)), ('single', (64, 128, 256, 512)), ('pair', (64, 128, 256, 512))):
        stab[rep] = {'decoder': decoder_stability(W[(rep, 's0')], W[(rep, 's1')], pref)}
        A = W[(rep, 's0')] / W[(rep, 's0')].norm(dim=1, keepdim=True)
        B = W[(rep, 's1')] / W[(rep, 's1')].norm(dim=1, keepdim=True)
        C = (A @ B.t()).max(1).values
        key = 'fg448_m1024' if rep == 'fg448' else rep
        stab[rep]['best_latents'] = {ev: {'latent': reps[key]['s0'][ev]['size_fg448']['latent'],
                                          'max_cos_to_s1': float(C[reps[key]['s0'][ev]['size_fg448']['latent']])} for ev in EVENTS}
    res['stability'] = stab
    # gate
    wins_fg = [ev for ev in EVENTS if reps['pair']['s0'][ev]['size_fg448']['auroc'] > reps['fg448_m1024']['s0'][ev]['size_fg448']['auroc']]
    wins_d = [ev for ev in EVENTS if reps['pair']['s0'][ev]['raw']['auroc'] > base[ev]['min_dist_raw']]
    both = [ev for ev in wins_fg if ev in wins_d]
    res['gate'] = {'pair_beats_fg448_size_controlled': wins_fg, 'pair_beats_min_dist_raw': wins_d,
                   'both': both, 'passed': len(both) >= 2,
                   's1_check': {'pair_beats_fg448_size_controlled': [ev for ev in EVENTS if reps['pair']['s1'][ev]['size_fg448']['auroc']
                                                                     > reps['fg448_m1024']['s1'][ev]['size_fg448']['auroc']]}}
    (CROPS / 'gates.json').write_text(json.dumps(res, indent=1))
    # grids
    obs = ann.observation_id.values
    for kind in KINDS:
        cidx, codes = crop_act[kind]
        for ev in EVENTS:
            j = best_sc[(kind, ev)]
            order = np.argsort(-codes[:, j])
            tiles, seen = [], set()
            for c in order:
                if codes[c, j] <= 0 or len(tiles) >= 16:
                    break
                ci = cidx[c]
                r = int(E['crop_row'][ci]); key = (obs[r], r // 50)
                if key in seen:
                    continue
                seen.add(key)
                rgb = np.asarray(Image.open(REPO / 'dataset' / ann.frame_path.values[r]).convert('RGB'))
                x = sample_crops(torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None], torch.from_numpy(E['crop_spec'][ci:ci + 1]),
                                 torch.zeros(1, dtype=torch.long))[0]
                im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                t = ' '.join(e[2:] for e in EVENTS if ann[e].values[r] > 0) or '-'
                ImageDraw.Draw(im).text((4, 4), f'{t} t{int(E["crop_spec"][ci, 0])}', fill=(255, 0, 0))
                tiles.append(im)
            grid(tiles, CROPS / f'top16_{kind}_{ev}.jpg')
    # table
    print(json.dumps(res['crop_stats'], indent=1))
    print('baselines', json.dumps(base, indent=1))
    for rep, rr in reps.items():
        for s, r in rr.items():
            print(f'{rep:12s} {s}', ' | '.join(f'{ev}: raw {v["raw"]["auroc"]:.3f} sc {v["size_fg448"]["auroc"]:.3f} '
                                               f'dist {v["dist"]["auroc"]:.3f} gap {v["mono_sc"]["gap"]:.3f} n02 {v["mono_sc"]["n_within_0.02"]}'
                                               for ev, v in r.items()))
    print('stability', json.dumps({k: {'frac>0.9': v['decoder'][max(v['decoder'], key=int)]['frac_gt_0.9_same_prefix'],
                                       'best': v['best_latents']} for k, v in stab.items()}, indent=1))
    print('GATE', res['gate'])


if __name__ == '__main__':
    main()
