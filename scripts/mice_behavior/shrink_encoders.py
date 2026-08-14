"""Strip re-saved pretrained weights out of fine-tuned encoder checkpoints.

A fine-tuning run used to write its whole encoder -- 346 MB of fp32 DINOv2-base -- even though
only the unfrozen blocks ever moved. At --unfreeze-blocks 2 that is 10 of 12 transformer blocks
written out byte-identical to facebook/dinov2-base, about 287 MB per run of pure duplication.
Measured across two runs on 2026-08-14, 50.9% of a ft2 and a ft6 encoder were identical to each
other; against the pretrained weights a ft2 encoder is ~83% redundant.

train_online_aug.py now saves only the trainable tensors, and every loader applies them over a
freshly pretrained encoder with strict=False. This script does the same to checkpoints already
on disk: it loads the pretrained reference, drops every tensor that still equals it exactly, and
rewrites the file. Runs launched before the trainer changed keep writing full encoders (Python
holds the module in memory), so re-run this after in-flight jobs land.

Exact equality is the test, not a tolerance -- a frozen tensor is bit-identical to the hub
weights because it was never touched by an optimizer step. Anything that differs at all is kept,
so a tensor that trained to a near-identical value is still preserved.

    python scripts/mice_behavior/shrink_encoders.py           # dry run, reports the saving
    python scripts/mice_behavior/shrink_encoders.py --apply   # rewrite in place
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModel

FRAME_DIR = Path('results/vision/mice/frame')
MODEL_ID = 'facebook/dinov2-base'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='rewrite the files (default: dry run)')
    p.add_argument('--pattern', default='*',
                    help="glob over run directory names. A training job rewrites best_encoder.pt "
                         "every time val AP improves, so restrict this to runs whose job has "
                         "ENDED -- rewriting one mid-write races the trainer.")
    args = p.parse_args()

    files = sorted(FRAME_DIR.glob(f'{args.pattern}/best_encoder.pt'))
    if not files:
        print('no best_encoder.pt files found'); return

    print(f'loading pretrained reference ({MODEL_ID})...', flush=True)
    ref = AutoModel.from_pretrained(MODEL_ID).state_dict()

    tot_before = tot_after = 0
    for f in files:
        sd = torch.load(f, map_location='cpu', weights_only=True)
        before = f.stat().st_size
        kept = {k: v for k, v in sd.items()
                if k not in ref or not torch.equal(v.cpu(), ref[k].cpu())}
        if len(kept) == len(sd):
            print(f'  {f.parent.name:<34} already minimal ({len(sd)} tensors, {before/2**20:.0f} MB)')
            tot_before += before; tot_after += before
            continue
        after = sum(v.numel() * v.element_size() for v in kept.values())
        print(f'  {f.parent.name:<34} {len(sd):3d} -> {len(kept):3d} tensors, '
              f'{before/2**20:6.1f} -> ~{after/2**20:5.1f} MB'
              f'  ({100*(1-after/before):.0f}% saved)')
        tot_before += before; tot_after += after
        if args.apply:
            torch.save(kept, f)

    print(f'\ntotal {tot_before/2**30:.2f} GiB -> ~{tot_after/2**30:.2f} GiB '
          f'({(tot_before-tot_after)/2**30:.2f} GiB recoverable)')
    if not args.apply:
        print('dry run -- pass --apply to rewrite')


if __name__ == '__main__':
    main()
