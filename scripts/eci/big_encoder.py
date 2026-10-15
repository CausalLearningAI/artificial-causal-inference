"""
BIG_ENCODER (idea B): does a bigger frozen DINOv2 make mouse nose-nose (nn) / nose-tail (nt) sniffing visible to an
UNSUPERVISED SAE neuron? DINOv2-large (ViT-L/14, d 1024, 24 layers) and DINOv2-giant (ViT-g/14, d 1536, 40 layers),
last-layer patch tokens, same 448 px input (the dinov2-base processor pixels), 32 x 32 grid, fg448 mask, 1 s motion
[x_t, x_t - x_{t-5}] (TokenNorm.fit_blocks([d, d])), Matryoshka BatchTopK 4096 / k 16, 6000 x 4096 steps, frame MAX
read-out, cross-fitted best neuron; the same 2 training pools x 3 SAE seeds and eval frames as idea A. The harness is
scripts/eci/invariant_tokens.py (encode / train with --model large|giant, kind id); the matched control is idea A's
'id' arm (DINOv2-base on the same pools). Supervised ceiling = mean_lin + mil_ctx on the eval t tokens.
PRE-REGISTERED RULE: as invariant_tokens (helps iff AP >= 1.5 x ref OR top-1% >= 0.25 on nn or nt, other-label AUROC
drop <= 0.02; ref = matched 'id', also vs the task-quoted c448 / btk16; dream top-1% >= 0.40).

Steps (scripts/eci/big_encoder.sh): encode / train run invariant_tokens.py; summary (login safe) is this file.
Outputs: results/vision/eci_big_encoder/ (gitignored).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import invariant_tokens as it  # noqa: E402


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        it.summarise(it.OUT_BIG, ['id', 'vitl', 'vitg'], [it.OUT_INV / 'eval'], [it.OUT_INV],
                     'big_encoder: DINOv2-large / giant vs DINOv2-base (matched id arm of invariant_tokens)')
    else:
        it.main()


if __name__ == '__main__':
    main()
