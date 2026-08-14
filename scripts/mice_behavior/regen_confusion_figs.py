"""Re-score a finished run on full val and (re)draw its qualitative error figures.

Why this exists
---------------
The confusion figures are drawn once, at the end of training, from probabilities that live
only in that process. When the figure code changes -- as it did on 2026-08-14, from a random
draw within each bucket to the most confident case per video -- every existing figure is
stale, and there is no way to redraw it without repeating a 6-hour training run. Worse, a run
launched BEFORE such a commit and finishing after it silently writes an old-style figure with
a new-style filename (job 63538641 started 12:26, the change landed 13:21), so the figure on
disk cannot be trusted to match the code in the tree.

The fix is to persist what the figures are made of. This script does one full-val forward pass
from the saved checkpoint and writes `val_probs.npz` into the run directory -- probabilities,
labels, global frame indices, observation ids. Every later figure change is then a sub-second
redraw (`--from-cache`) rather than another GPU pass, and the numbers behind a figure become
checkable after the fact instead of being lost with the process that made them.

The forward pass reproduces the run's own evaluation exactly: same fixed val pools, same
FrameBatchData construction, same no-augmentation rendering, same head config read back out of
the run's config.json. It is NOT a re-training and cannot change any reported metric; the AP
it prints is a check that the reconstruction is faithful, and it should match config.json's
`ap_report` to ~1e-3 (fp16 autocast is not bitwise deterministic across batch composition).

Usage:
    # one GPU pass per run, then all figures
    python scripts/mice_behavior/regen_confusion_figs.py --tag patchgrid256_dinov2_ft_b2

    # redraw only -- no GPU, no encoder, reads val_probs.npz
    python scripts/mice_behavior/regen_confusion_figs.py --tag ft_b4 --from-cache
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import grid_search_frame as gsf
from src.mice_behavior.batch_data import FrameBatchData
from src.mice_behavior.head_cfg import get_head_cfg
from src.mice_behavior.model import MouseFrameClassifier
from src.mice_behavior.pools import get_fixed_val_pools
from src.mice_behavior.viz import plot_confusion_examples, plot_error_strips
from train_patchgrid_online import dummy_loader
from train_online_aug import MODEL_ID, EMB_DIM, PATCH_SIZE, _BytesReader, _SampleDataset

DATASET_ROOT = Path('dataset')


def build_val_meta(cfg, ann_csv, pair_labels):
    """The run's own val split, rebuilt from its config. get_fixed_val_pools is deterministic,
    but config.json records the resulting pools, so verify rather than assume -- a run from
    before the split was frozen would otherwise be scored on the wrong videos in silence."""
    o2p = gsf.load_obs_to_pool_map(gsf.DATA_DIR)
    all_obs = pd.read_parquet(pair_labels)['observation_id'].unique().tolist()
    val_pools = get_fixed_val_pools(sorted({o2p[o] for o in all_obs}))
    recorded = cfg.get('val_pools')
    if recorded is not None and sorted(recorded) != sorted(val_pools):
        raise SystemExit(f'val split moved since this run: config.json has {sorted(recorded)}, '
                         f'get_fixed_val_pools now returns {sorted(val_pools)}. Refusing to '
                         f'score the run on videos it never validated on.')
    val_obs = [o for o in all_obs if o2p[o] in val_pools]
    vm = FrameBatchData(str(ann_csv), str(pair_labels), val_obs, cfg['context_k'], 1,
                        dummy_loader(1, 1), n_patches=1, stride=cfg.get('stride', 1))
    del vm.flat
    return vm


def load_jpeg_cache(path, needed, frame_paths, read_workers):
    """Memory-map the shared JPEG cache if it covers what we need, else read from NFS."""
    cache = {}
    if path:
        bin_p, idx_p = Path(f'{path}.bin'), Path(f'{path}.npz')
        if bin_p.exists() and idx_p.exists():
            m = np.load(idx_p)
            blob = np.memmap(bin_p, dtype=np.uint8, mode='r')
            offs, keys = m['offsets'], m['all_needed']
            cache = {int(k): blob[offs[i]:offs[i + 1]] for i, k in enumerate(keys)}
            print(f'JPEG cache memory-mapped from {bin_p} ({len(cache):,} frames)', flush=True)
    missing = np.array(sorted(set(int(g) for g in needed) - set(cache)), dtype=np.int64)
    if len(missing):
        t0 = time.time()
        dl = DataLoader(_BytesReader(frame_paths[missing]), batch_size=None,
                        num_workers=read_workers, prefetch_factor=6, collate_fn=lambda x: x)
        for i, buf in dl:
            cache[int(missing[i])] = buf
        print(f'read {len(missing):,} frames off NFS in {(time.time()-t0)/60:.1f} min', flush=True)
    return cache


def score(cfg, run_dir, vm, jpeg_cache, batch_size, decode_workers, dev):
    """Full-val forward pass from the saved checkpoint. Mirrors train_online_aug's evaluate()."""
    n_patches = (cfg['input_size'] // PATCH_SIZE) ** 2
    head = get_head_cfg()
    encoder = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    encoder.requires_grad_(False)

    enc_ckpt = run_dir / 'best_encoder.pt'
    if cfg.get('unfreeze_blocks', 0) > 0:
        if not enc_ckpt.exists():
            raise SystemExit(f'{run_dir.name} fine-tuned {cfg["unfreeze_blocks"]} blocks but has '
                             f'no best_encoder.pt -- scoring it with the stock encoder would '
                             f'silently evaluate a different model.')
        sd = torch.load(enc_ckpt, map_location=dev, weights_only=True)
        # strict=False: since 2026-08-14 best_encoder.pt holds ONLY the tensors that trained,
        # the frozen remainder coming from the hub. Older files carry the whole encoder and
        # match every key. Both are correct here; an empty overlay is not, so check.
        _, unexpected = encoder.load_state_dict(sd, strict=False)
        overlaid = len(sd) - len(unexpected)
        print(f'encoder: overlaid {overlaid}/{len(sd)} fine-tuned tensors from best_encoder.pt',
              flush=True)
        if overlaid == 0:
            raise SystemExit('best_encoder.pt shares no keys with the encoder -- wrong checkpoint.')

    model = MouseFrameClassifier(
        emb_dim=EMB_DIM, n_heads=head['n_heads'], hidden_dim=head['hidden_dim'],
        use_patch_grid=True, dropout=cfg.get('dropout', head['dropout']),
        use_motion=cfg.get('use_motion', False),
        cross_attn_dim=cfg.get('cross_attn_dim') or None,
        patch_pool_dim=cfg.get('patch_pool_dim') or None,
        patch_selfattn_dim=cfg.get('patch_selfattn_dim') or None,
        n_pool_queries=cfg.get('pool_queries', 1)).to(dev)
    model.load_state_dict(torch.load(run_dir / 'best_model.pt', map_location=dev,
                                     weights_only=True))
    model.eval()

    order = np.arange(len(vm))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    # augment='none' and seed 0: evaluation must be the single canonical rendering, exactly as
    # train_online_aug's evaluate() does it -- not a draw from the D4 orbit the model trained on.
    loader = DataLoader(_SampleDataset(vm, batches, jpeg_cache, cfg['input_size'], 'none', 0),
                        batch_size=None, num_workers=decode_workers,
                        pin_memory=(dev.type == 'cuda'), prefetch_factor=4)
    P, L, t0 = [], [], time.time()
    with torch.no_grad():
        for bi, (imgs, offs, lbl, mask) in enumerate(loader):
            imgs = imgs.to(dev, non_blocking=True)
            B, T = imgs.shape[:2]
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                tok = encoder(pixel_values=imgs.view(B * T, *imgs.shape[2:])).last_hidden_state[:, 1:]
                logits = model(tok.view(B, T, n_patches, EMB_DIM),
                               offsets=offs.to(dev), key_padding_mask=mask.to(dev))
            P.append(torch.sigmoid(logits).float().cpu()); L.append(lbl)
            if bi % 200 == 0:
                done = bi + 1
                print(f'  batch {done:,}/{len(batches):,} '
                      f'({(time.time()-t0)/60:.1f} min, eta '
                      f'{(time.time()-t0)/done*(len(batches)-done)/60:.1f} min)', flush=True)
    probs, labs = torch.cat(P).numpy(), torch.cat(L).numpy()
    # Explicit teardown: with several runs scored in one process, the previous run's encoder
    # (87M params), head and DataLoader workers would otherwise stay alive until the next
    # assignment rebinds them -- i.e. peak memory would hold TWO of everything at the moment
    # the next run loads. That is what OOM-killed job 63550097 at the start of its second run.
    del loader, model, encoder, P, L
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return probs, labs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', action='append', required=True,
                   help='run directory under results/vision/mice/frame (repeatable)')
    p.add_argument('--from-cache', action='store_true',
                   help='redraw from val_probs.npz; no GPU, no encoder, no forward pass')
    p.add_argument('--overwrite-probs', action='store_true',
                   help='re-run the forward pass even though val_probs.npz exists')
    p.add_argument('--n-rows', type=int, default=10, help='error strips per figure')
    p.add_argument('--context', type=int, default=3, help='context frames each side of the error')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--decode-workers', type=int, default=16)
    p.add_argument('--read-workers', type=int, default=32)
    p.add_argument('--jpeg-cache-file', type=str, default='dataset/mice/v1/jpegcache_k2')
    args = p.parse_args()

    pair_labels = gsf.build_pair_labels(gsf.DATA_DIR, gsf.DATASET_DIR, overwrite=False)
    ann_csv = gsf.DATASET_DIR / 'mice' / 'v1' / 'annotations.csv'
    ann = pd.read_csv(ann_csv, usecols=['frame_path', 'observation_id', 'frame_idx'])
    frame_paths = ann.frame_path.values
    obs_of_gi = ann.observation_id.values          # global frame index -> video
    fidx_of_gi = ann.frame_idx.values

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # One cache for every run in this invocation. build_val_meta refuses to proceed if a run's
    # val pools differ from the current fixed split, so all runs here read the SAME val frames
    # -- rebuilding the 444k-entry dict per run bought nothing and held two copies at a time.
    # Populated lazily so --from-cache never touches it.
    jpeg_cache = None
    for tag in args.tag:
        run_dir = gsf.FRAME_DIR / tag
        cfg_path = run_dir / 'config.json'
        if not cfg_path.exists():
            print(f'[skip] {tag}: no config.json', flush=True)
            continue
        cfg = json.load(open(cfg_path))
        print(f'\n{"="*70}\n{tag}  ({cfg["input_size"]}px k{cfg["context_k"]} '
              f'ft{cfg.get("unfreeze_blocks", 0)})\n{"="*70}', flush=True)

        vm = build_val_meta(cfg, ann_csv, pair_labels)
        npz = run_dir / 'val_probs.npz'
        if npz.exists() and not args.overwrite_probs:
            d = np.load(npz, allow_pickle=True)
            probs, labs, gi = d['probs'], d['labels'], d['gi']
            if len(gi) != len(vm) or not np.array_equal(gi, vm.gi):
                raise SystemExit(f'{npz} was written against a different val set '
                                 f'({len(gi):,} samples vs {len(vm):,} now) -- rerun with '
                                 f'--overwrite-probs.')
            print(f'probabilities loaded from {npz}', flush=True)
        elif args.from_cache:
            raise SystemExit(f'--from-cache but {npz} does not exist; run once on a GPU first.')
        else:
            if jpeg_cache is None:
                need = np.unique((vm.gi[:, None] + vm.offsets_grid[None, :])[~vm.pad_mask])
                jpeg_cache = load_jpeg_cache(args.jpeg_cache_file, need, frame_paths,
                                             args.read_workers)
            probs, labs = score(cfg, run_dir, vm, jpeg_cache, args.batch_size,
                                args.decode_workers, dev)
            gi = vm.gi
            np.savez_compressed(npz, probs=probs, labels=labs, gi=gi,
                                obs=obs_of_gi[gi].astype(str))
            print(f'saved {npz}', flush=True)

        ap = {n: float(average_precision_score(labs[:, i], probs[:, i]))
              for i, n in enumerate(('nt', 'nn'))}
        ref = cfg.get('ap_report', {})
        print(f'full-val AP  nt {ap["nt"]:.4f}  nn {ap["nn"]:.4f}  '
              f'macro {np.mean(list(ap.values())):.4f}'
              + (f'   (run recorded macro {ref["macro/tol0"]["ap"]:.4f})'
                 if 'macro/tol0' in ref else ''), flush=True)

        sample_obs = obs_of_gi[gi]
        # jpeg_cache is None on the --from-cache path; _load_frame then falls back to disk,
        # which is fine for the few hundred frames a figure actually renders.
        plot_confusion_examples(probs, labs, sample_obs, gi, frame_paths, run_dir,
                                jpeg_cache=jpeg_cache, dataset_root=str(DATASET_ROOT),
                                title_prefix=f'{tag}  ')
        plot_error_strips(probs, labs, sample_obs, gi, frame_paths, run_dir,
                          n_rows=args.n_rows, context=args.context, jpeg_cache=jpeg_cache,
                          dataset_root=str(DATASET_ROOT), title_prefix=f'{tag}  ',
                          frame_idx=fidx_of_gi[gi])


if __name__ == '__main__':
    main()
