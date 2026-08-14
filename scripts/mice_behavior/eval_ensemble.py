"""Inference-side gains: D4 test-time augmentation, bout-aware temporal smoothing, ensembling.

Every result so far comes from a single checkpoint, argmax-free but also entirely un-post-
processed, and three sources of essentially free accuracy have never been collected:

1. TEST-TIME AUGMENTATION. The model is trained on all 8 D4 renderings of every sample, so
   its predictions should be D4-invariant and disagreement between renderings is pure
   variance. Averaging over ops removes it. Costs one encoder pass per op, nothing else.

2. BOUT-AWARE TEMPORAL SMOOTHING. Labels are contiguous bouts (median 2-3 frames), but
   predictions are emitted per frame with no temporal coupling at all. We already know the
   errors are partly jitter rather than misses: on the res448 checkpoint nn scores AP 0.4338
   frame-exact but 0.4608 with +-1 frame tolerance, i.e. predictions land NEAR positives.
   A centred moving average over each observation's prediction sequence converts that into
   frame-exact gains. Never crosses an observation boundary.

3. ENSEMBLING. Seven checkpoints exist across resolutions, context_k, optimizers and
   augmentations -- a genuinely diverse pool, which is exactly the condition under which
   averaging helps.

Avoiding val fitting
--------------------
Smoothing width, ensemble method and ensemble membership are all CHOSEN ON TRAINING
OBSERVATIONS (a held-out subset of the 120 the models were fitted on) and only then applied
to val, once. Selecting any of them on val would inflate exactly the number we are trying to
report -- with n=24 val observations that inflation would be substantial. Per-checkpoint val
baselines are printed alongside so the delta from post-processing is visible.

Usage:
    python scripts/mice_behavior/eval_ensemble.py --tta rot4
    sbatch scripts/mice_behavior/eval_ensemble.sh
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
import train_online_aug as toa
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.metrics import ap_report, rate_report, format_ap_report, format_rate_report
from src.mice_behavior.viz import plot_confusion_examples
from train_patchgrid_online import dummy_loader

EMB_DIM, PATCH_SIZE = toa.EMB_DIM, toa.PATCH_SIZE
TTA_OPS = {'none': [0], 'rot4': [0, 1, 2, 3], 'd4': list(range(8))}

# Renamed 2026-08-14 to the config-derived scheme (see rename_runs.py):
# res<input>_k<context>[s<stride>]_<encoder>_<augment>. The old 504_k2 entry is dropped -- that
# directory never held a checkpoint and only ever produced a FileNotFoundError skip.
DEFAULT_CKPTS = [
    'res448_k2_ft2_d4photo',        # best single model to date (macro AP 0.4889, r 0.542)
    'res448_k2_frozen_d4photo', 'res448_k2s2_frozen_d4photo', 'res448_k2_frozen_d4',
    'res504_k2_frozen_d4photo', 'res504_k2_frozen_d4',
    'res504_k1_frozen_d4', 'res504_k0_frozen_d4',
    'res224_k2_frozen_d4_decay20',
]


def load_spec(name):
    """Architecture is recovered from the state_dict's own shapes rather than trusted to the
    config, because cross_attn_dim/patch_pool_dim were argparse defaults and went unrecorded
    in every run before tonight."""
    d = gsf.FRAME_DIR / name
    cfg = json.load(open(d / 'config.json'))
    sd = torch.load(d / 'best_model.pt', map_location='cpu', weights_only=True)
    # a fine-tuned run's head is only meaningful with ITS encoder; pairing it with the
    # pretrained weights would silently produce garbage rather than fail
    enc_path = d / 'best_encoder.pt'
    spec = dict(
        name=name, dir=d, state=sd, encoder_path=(enc_path if enc_path.exists() else None),
        input_size=cfg.get('input_size', 224), context_k=cfg.get('context_k', 2),
        stride=cfg.get('stride', 1), n_heads=cfg['cfg']['n_heads'],
        hidden_dim=sd['head.0.weight'].shape[0],
        cross_attn_dim=sd['query'].shape[-1],
        patch_pool_dim=(sd['patch_pool.in_proj.weight'].shape[0]
                        if 'patch_pool.in_proj.weight' in sd else None),
        # both recovered the same way, for the same reason: the 2026-08-14 head ablation added
        # two axes, and a checkpoint from either arm loads as garbage (or not at all) if the
        # architecture is assumed rather than read off the tensors that are actually in it.
        patch_selfattn_dim=(sd['patch_selfattn.down.weight'].shape[0]
                            if 'patch_selfattn.down.weight' in sd else None),
        n_pool_queries=sd['patch_pool.query'].shape[1] if 'patch_pool.query' in sd else 1,
        use_motion=any(k.startswith('motion_pool') for k in sd),
        val_ap=cfg.get('ap_report', {}).get('macro/tol0', {}).get('ap'),
    )
    return spec


def build(spec, dev):
    m = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=spec['n_heads'], hidden_dim=spec['hidden_dim'],
        use_patch_grid=True, dropout=0.0, use_motion=spec['use_motion'],
        cross_attn_dim=spec['cross_attn_dim'], patch_pool_dim=spec['patch_pool_dim'],
        patch_selfattn_dim=spec['patch_selfattn_dim'], n_pool_queries=spec['n_pool_queries'])
    m.load_state_dict(spec['state'])
    return m.to(dev).eval()


def smooth(probs, sample_obs, width):
    """Centred moving average within each observation. width=1 is a no-op."""
    if width <= 1:
        return probs
    out = np.empty_like(probs)
    half = width // 2
    for o in np.unique(sample_obs):
        m = sample_obs == o
        seq = probs[m]
        pad = np.pad(seq, ((half, half), (0, 0)), mode='edge')
        csum = np.cumsum(np.vstack([np.zeros((1, seq.shape[1])), pad]), axis=0)
        out[m] = (csum[width:] - csum[:-width]) / width
    return out


def macro_ap(probs, labels):
    return float(np.mean([average_precision_score(labels[:, i], probs[:, i]) for i in (0, 1)]))


def rank_norm(p):
    """Per-column rank in [0,1] -- makes averaging insensitive to each model's calibration,
    which differs a lot here (calibration slopes ranged 0.06-0.12 across runs)."""
    out = np.empty_like(p)
    for i in range(p.shape[1]):
        order = p[:, i].argsort()
        r = np.empty(len(order))
        r[order] = np.arange(len(order))
        out[:, i] = r / max(len(order) - 1, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='*', default=None)
    p.add_argument('--tta', choices=['none', 'rot4', 'd4'], default='rot4')
    p.add_argument('--tta-top-k', type=int, default=3,
                    help='TTA costs one full encoder pass per op, so it is applied only to the '
                         'top-k checkpoints by train-subset AP rather than all of them')
    p.add_argument('--n-train-obs', type=int, default=12,
                    help='training observations used to CHOOSE smoothing width / ensemble method '
                         '/ membership. Nothing is selected on val.')
    p.add_argument('--widths', type=int, nargs='+', default=[1, 3, 5, 7, 9, 11, 15])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--read-workers', type=int, default=32)
    p.add_argument('--decode-workers', type=int, default=8)
    p.add_argument('--tag', type=str, default='ensemble')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT = gsf.FRAME_DIR / f'ensemble_{args.tag}'
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.checkpoints or DEFAULT_CKPTS
    specs = []
    for n in names:
        try:
            specs.append(load_spec(n))
        except Exception as e:
            print(f'  skip {n}: {e.__class__.__name__} {e}', flush=True)
    print(f'{len(specs)} checkpoints loaded', flush=True)
    for s in specs:
        print(f'  {s["name"]:<42} res={s["input_size"]} k={s["context_k"]} stride={s["stride"]} '
              f'ca={s["cross_attn_dim"]} pp={s["patch_pool_dim"]} val_ap={s["val_ap"]}', flush=True)

    pair_labels = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
    ann_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    o2p = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
    all_obs = pd.read_parquet(pair_labels)['observation_id'].unique().tolist()
    pools = sorted({o2p[o] for o in all_obs})
    val_pools = get_fixed_val_pools(pools)
    train_obs = [o for o in all_obs if o2p[o] not in val_pools]
    val_obs = [o for o in all_obs if o2p[o] in val_pools]
    sel_rng = np.random.default_rng(gsf.SEED)
    fit_obs = sorted(sel_rng.choice(train_obs, size=min(args.n_train_obs, len(train_obs)),
                                    replace=False).tolist())
    if args.smoke:
        fit_obs, val_obs = fit_obs[:2], val_obs[:2]
    print(f'\nfit-on (train) {len(fit_obs)} obs | report-on (val) {len(val_obs)} obs', flush=True)

    ann = pd.read_csv(ann_csv, usecols=['frame_path'])
    frame_paths = ann.frame_path.values
    jpeg_cache = {}

    def ensure_cached(idx, what):
        missing = np.array(sorted(set(int(g) for g in idx) - set(jpeg_cache)), dtype=np.int64)
        if not len(missing):
            return
        t0 = time.time()
        dl = DataLoader(toa._BytesReader(frame_paths[missing]), batch_size=None,
                        num_workers=args.read_workers, prefetch_factor=6, collate_fn=lambda x: x)
        for i, buf in dl:
            jpeg_cache[int(missing[i])] = buf
        print(f'  [{what}] read {len(missing):,} frames in {(time.time()-t0)/60:.1f} min', flush=True)

    metas, obs_of = {}, {}
    for split, obs in (('fit', fit_obs), ('val', val_obs)):
        for k, st in sorted({(s['context_k'], s['stride']) for s in specs}):
            m = FrameBatchData(str(ann_csv), str(pair_labels), obs, k, 1,
                               dummy_loader(1, 1), n_patches=1, stride=st)
            del m.flat
            metas[(split, k, st)] = m
        a2 = pd.read_csv(ann_csv, usecols=['observation_id', 'frame_idx'])
        a2 = a2[a2.observation_id.isin(set(obs))].reset_index()
        b = {o: int(g['index'].values[0]) for o, g in a2.groupby('observation_id', sort=False)}
        stt = np.array(sorted(b.values()))
        nm = np.array([kk for kk, _ in sorted(b.items(), key=lambda x: x[1])])
        any_meta = metas[(split, *sorted({(s['context_k'], s['stride']) for s in specs})[0])]
        obs_of[split] = nm[np.searchsorted(stt, any_meta.gi, side='right') - 1]

    base_encoder = AutoModel.from_pretrained(toa.MODEL_ID).to(dev).eval()
    base_encoder.requires_grad_(False)
    enc_cache = {None: base_encoder}

    def get_encoder(spec):
        """Fine-tuned checkpoints carry their own encoder weights; frozen ones share the base."""
        key = str(spec['encoder_path']) if spec['encoder_path'] else None
        if key not in enc_cache:
            e = AutoModel.from_pretrained(toa.MODEL_ID)
            # strict=False: since 2026-08-14 best_encoder.pt stores only the tensors that were
            # unfrozen during training, so the frozen remainder comes from the pretrained load
            # above rather than from a re-saved copy of it. Older full-encoder files still load
            # through the same call, as every one of their keys matches.
            e.load_state_dict(torch.load(spec['encoder_path'], map_location='cpu',
                                         weights_only=True), strict=False)
            enc_cache[key] = e.to(dev).eval().requires_grad_(False)
            print(f'  loaded fine-tuned encoder for {spec["name"]}', flush=True)
        return enc_cache[key]

    @torch.no_grad()
    def predict(spec, split, ops):
        """Mean probability over the given D4 ops. Sample ORDER is identical for every spec
        (one sample per annotated frame, ordered by observation then frame_idx), which is what
        makes predictions from different context_k/stride/resolution ensemble-able."""
        meta = metas[(split, spec['context_k'], spec['stride'])]
        order = np.arange(len(meta))
        n_patches = (spec['input_size'] // PATCH_SIZE) ** 2
        ensure_cached(np.unique((meta.gi[order][:, None] + meta.offsets_grid[None, :])
                                [~meta.pad_mask[order]]), f'{split}/{spec["name"][-18:]}')
        model = build(spec, dev)
        encoder = get_encoder(spec)
        acc = None
        for op in ops:
            batches = [order[i:i+args.batch_size] for i in range(0, len(order), args.batch_size)]
            dl = DataLoader(toa._SampleDataset(meta, batches, jpeg_cache, spec['input_size'],
                                               'd4' if op else 'none', 0, fixed_op=op),
                            batch_size=None, num_workers=args.decode_workers,
                            pin_memory=(dev.type == 'cuda'), prefetch_factor=4)
            P, L = [], []
            for imgs, offs, lbl, mask in dl:
                imgs = imgs.to(dev, non_blocking=True)
                B, T = imgs.shape[:2]
                with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                    tok = encoder(pixel_values=imgs.view(B*T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                    lg = model(tok.view(B, T, n_patches, EMB_DIM),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
                P.append(torch.sigmoid(lg).float().cpu())
                L.append(lbl)
            pr = torch.cat(P).numpy()
            acc = pr if acc is None else acc + pr
        del model
        torch.cuda.empty_cache() if dev.type == 'cuda' else None
        return acc / len(ops), torch.cat(L).numpy()

    # ---- stage 1: un-augmented predictions on both splits, cached to disk ----
    preds = {}
    labels = {}
    for split in ('fit', 'val'):
        for s in specs:
            f = OUT / f'pred_{split}_{s["name"]}_op0.npy'
            n_expect = len(metas[(split, s['context_k'], s['stride'])])
            # the cache key does not encode which observations were used, so a changed
            # --n-train-obs would otherwise reuse a mismatched array
            if f.exists() and np.load(f, mmap_mode='r').shape[0] == n_expect:
                preds[(split, s['name'])] = np.load(f)
            else:
                t0 = time.time()
                pr, lb = predict(s, split, [0])
                np.save(f, pr)
                preds[(split, s['name'])] = pr
                labels[split] = lb
                print(f'  {split}/{s["name"]}: {len(pr):,} samples in '
                      f'{(time.time()-t0)/60:.1f} min  macroAP={macro_ap(pr, lb):.4f}', flush=True)
            if split not in labels:
                labels[split] = metas[(split, s['context_k'], s['stride'])].labels

    # ---- stage 2: choose width / method / membership on the FIT split only ----
    fit_lb, val_lb = labels['fit'], labels['val']
    fit_o, val_o = obs_of['fit'], obs_of['val']
    ranked = sorted(specs, key=lambda s: -macro_ap(preds[('fit', s['name'])], fit_lb))
    print('\nfit-split single-model AP (this is what ordering/selection uses):', flush=True)
    for s in ranked:
        print(f'  {macro_ap(preds[("fit", s["name"])], fit_lb):.4f}  {s["name"]}', flush=True)

    best_w, best_wap = 1, -1
    for w in args.widths:
        v = macro_ap(smooth(preds[('fit', ranked[0]['name'])], fit_o, w), fit_lb)
        print(f'  smoothing width {w:>3}: fit AP {v:.4f}', flush=True)
        if v > best_wap:
            best_w, best_wap = w, v
    print(f'  -> chosen width {best_w}', flush=True)

    def combine(names_, split, method, w):
        ps = [smooth(preds[(split, n)], fit_o if split == 'fit' else val_o, w) for n in names_]
        ps = [rank_norm(x) for x in ps] if method == 'rank' else ps
        return np.mean(ps, axis=0)

    best = {'method': 'mean', 'members': [ranked[0]['name']], 'ap': -1}
    for method in ('mean', 'rank'):
        members = []
        for _ in range(len(ranked)):                      # greedy forward selection on fit
            cand = [(macro_ap(combine(members + [s['name']], 'fit', method, best_w), fit_lb),
                     s['name']) for s in ranked if s['name'] not in members]
            v, n = max(cand)
            if members and v <= cur:
                break
            members.append(n)
            cur = v
        print(f'  ensemble[{method}] fit AP {cur:.4f} with {len(members)}: {members}', flush=True)
        if method == 'rank' and len(members) == 1:
            # rank_norm is a strictly monotone per-column transform, so for a SINGLE member it
            # cannot change AP or ROC-AUC at all -- any apparent difference from 'mean' is pure
            # argsort tie-breaking (and frames are ordered by observation with contiguous bouts,
            # so index-order tie-breaks can even leak label structure). Meanwhile it replaces
            # probabilities with uniform ranks in [0,1], which pushes the mean prediction to ~0.5
            # against a ~1.5% true rate and destroys every rate metric: on the 2026-08-13 overnight
            # run it was selected by a meaningless 0.0016 fit-AP margin and took per-observation
            # Pearson r from 0.542 -> 0.030, MAE from 1.6pp -> 49.4pp, calib slope 0.30 -> 0.003.
            print('    ^ skipped: rank on 1 member is AP-invariant by construction and only '
                  'destroys calibration', flush=True)
            continue
        if cur > best['ap']:
            best = {'method': method, 'members': members, 'ap': cur}
    print(f'\n-> selected on fit: method={best["method"]}  width={best_w}  '
          f'members={best["members"]}', flush=True)

    # ---- stage 3: TTA for the selected members, re-tuned on fit, then applied to val ONCE ----
    ops = TTA_OPS[args.tta]
    if len(ops) > 1:
        for n in best['members'][:args.tta_top_k]:
            s = next(x for x in specs if x['name'] == n)
            for split in ('fit', 'val'):
                f = OUT / f'pred_{split}_{n}_tta{len(ops)}.npy'
                if f.exists():
                    preds[(split, n)] = np.load(f)
                    continue
                t0 = time.time()
                pr, _ = predict(s, split, ops)
                np.save(f, pr)
                base = macro_ap(preds[(split, n)], labels[split])
                preds[(split, n)] = pr
                print(f'  TTA{len(ops)} {split}/{n}: {(time.time()-t0)/60:.1f} min  '
                      f'AP {base:.4f} -> {macro_ap(pr, labels[split]):.4f}', flush=True)

    # ---- report on val ----
    print(f'\n{"="*76}\nVAL RESULTS (selection was made on train observations only)\n{"="*76}')
    print(f'{"variant":<46}{"macAP0":>9}{"macAP1":>9}{"aucNT":>8}{"aucNN":>8}')

    def report(tag, pv):
        a = ap_report(pv, val_lb, val_o, tolerances=(0, 1, 2))
        au = [roc_auc_score(val_lb[:, i], pv[:, i]) for i in (0, 1)]
        print(f'{tag:<46}{a["macro/tol0"]["ap"]:>9.4f}{a["macro/tol1"]["ap"]:>9.4f}'
              f'{au[0]:>8.4f}{au[1]:>8.4f}', flush=True)
        return a, au

    results = {}
    for s in specs:
        results[s['name']] = report(f'single: {s["name"][-38:]}', preds[('val', s['name'])])[0]
    ens = combine(best['members'], 'val', best['method'], best_w)
    print()
    results['ensemble_raw'] = report('ENSEMBLE (no smoothing)',
                                     combine(best['members'], 'val', best['method'], 1))[0]
    apr, auc = report(f'ENSEMBLE + smoothing(w={best_w}) + TTA{len(ops)}', ens)
    rr = rate_report(ens, val_lb, val_o)
    print()
    print(format_ap_report(apr, tolerances=(0, 1, 2)))
    print(f'\nROC-AUC  nt {auc[0]:.4f}  nn {auc[1]:.4f}')
    print(format_rate_report(rr))

    try:
        any_meta = metas[('val', specs[0]['context_k'], specs[0]['stride'])]
        plot_confusion_examples(ens, val_lb, val_o, any_meta.gi, frame_paths, OUT,
                                jpeg_cache=jpeg_cache,
                                title_prefix=f'ensemble(w={best_w},{best["method"]})  ')
    except Exception as e:
        print(f'  [viz] skipped ({e.__class__.__name__}: {e})', flush=True)

    json.dump({'args': vars(args), 'selected': best, 'width': best_w,
               'val_ap_report': apr, 'val_rate_report': rr,
               'val_roc_auc': {'nt': auc[0], 'nn': auc[1]},
               'singles': {k: v['macro/tol0']['ap'] for k, v in results.items()}},
              open(OUT / 'ensemble.json', 'w'), indent=2, default=str)
    print(f'\nSaved {OUT}/ensemble.json', flush=True)


if __name__ == '__main__':
    main()
