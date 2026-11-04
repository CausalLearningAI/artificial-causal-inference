# ECI v2.3 log: V SAE trained on all M_t tokens of all videos

Goal: retrain the v2 SAE (arm V input x + (x_{t+1} - x_{t-1}) / 2, mask v5) on ALL M_t tokens of ALL videos (~1 pass),
evaluate on every labelled mice frame (864,000 = atlas window + first 15 min of the 48 H videos), vs V (s1001-1006) and
v1 s0. Pre-registration: scripts/eci/v2/v23_prereg.md. Code: scripts/eci/v2/v23_train.py (count / featcheck / train),
v23_eval.py (ext frames, code identity); scoring reuses v22_full.py / v22_eval.py / v22_ants.py with V22_FRAMES=ext.
Big files: /mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_3/ (sae/, ext/).

## Timeline (2026-11-04)
- 12:20 count jobs 2001990 (mice) / 2001991 (ants): mice 648 videos, 3,888,000 frames, 748,150,181 M_t tokens (192.4 /
  frame), 1,048,111,383 store rows (1.40 x M), one pass = 182,653 steps. Ants (store) 232 videos, 104,754,034 M_t
  tokens, 25,574 steps. Read bench (mice, 16 threads): raw reads 12.4 GB/s, numpy V features 0.21M tokens/s (too slow
  for ~0.46M tokens/s per SAE) -> features moved to the GPU.
- 12:22 ext frame table (2002232): 864,000 frames = 648,000 atlas (ordered subset of the v2.2 full table, labels
  identical, asserted) + 216,000 first-15-min H frames (48 videos x 4,500). Counts in ext/frames.md.
- 12:30 COORDINATOR: ants DINOv3 tokens exist for only 232 of the 489 Exp2-4 videos (v2 44/45, v3 107/212, v4 40/113,
  v4_infested 41/119; v5, vA not encoded). Mice proceed; ants SAE training NOT started, waits for the coordinator's go.
  The ants path reads whatever video list the store manifest holds (v23_train.store_videos).
- featcheck (streamed numpy features vs v22_full.feats_v): exact (max abs diff 0) on 40 random windows per domain;
  re-run 2002267 with 10 explicit video-edge windows added.
- GPU queue: -p gpu estimated start 2026-11-20 -> SAE training on gpu100 (est. start ~15:09).
- CPU smoke test of the training loop (2002250, 16 windows per chunk, 30 steps): GPU-path features == numpy (exact, 16
  windows).

## Edits to shared v22 scripts (frame-set value, no fork; defaults unchanged)
- v22_core.py: V22_FRAMES=ext -> OUT22 = results/vision/eci_v2_3/ext, BEEF = eci_codes/v2_3/ext; cache_paths for ext.
- v22_full.py: BEEF / OUTF follow ext; cache --src deployed (v1 s0 deployed codes_max by annotations row).
- v22_encode.py: sae_path / output dirs for Vall_* keys -> eci_codes/v2_3, results/vision/eci_v2_3/codes.
- v22_ants.py: codes_path for Vall_* keys -> eci_codes/v2_3/ants/codes.
