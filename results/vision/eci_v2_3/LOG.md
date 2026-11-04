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
- 12:31 first mice training job 2002302 (gpu100, H100) started early. 3,000-step bench phase: 10.7 steps/s (one step =
  3 SAEs); full run: steps 0-2,000 at 91.9 steps/s (= 3.6 ms per SAE step), then steps 2,000-4,000 at 14.4 steps/s
  (ETA 207 min): the main thread (GPU feature building of each new chunk) was starved, most likely by the 14 loader
  threads copying memmap pages while holding the GIL. Loader wait itself was 0 s. Cancelled at 12:43, together with the
  first ants run 2003126 (same code).
- Fix: window rows read with os.preadv into a fresh buffer (releases the GIL) and the GPU features built inside the
  loader threads (torch ops release the GIL); the training thread only concatenates. Identity re-checked (2003169):
  preadv window == T.Video.window (asserted) and features == v22_full.feats_v, max abs diff 0 on 50 windows incl. 10
  video edges. Bench phase dropped (step time / read rate are logged every 2,000 steps). Resubmitted: mice 2003185,
  ants 2003186 (gpu100); gpu100 est. start reported 2026-11-05 03:25 (estimates have been pessimistic: 2002302 was
  estimated 14:09 and started 12:31). -p gpu estimate: 2026-11-20.
- 13:05 AMENDMENT 1 (user via coordinator; committed 0cc25bf before any ants SAE / score): ants training set = rule
  (valid v2; v3 treatments 2/6/8; v4 and v4_infested treatments 2/6) = the 232 store videos (asserted in
  v23_train.store_videos); ants alignment on v3 + v4 only (147 videos, 88,200 frames at 1 fps), halves A / B
  stratified by series x treatment (rng 20261105; ants/split.json: A 74, B 73 videos), select on A, measure on B
  (primary) and swapped. Code: v23_ants.py (tags: V s1001-1006, v1 s0-s10 job 2003151; Vall after training).
- ext V codes array 2002561: 52/72 ok, 20 tasks failed on eta380 (no /localhome/rcadei) -> v22_full.local_dir falls
  back to the node's /tmp; those tasks re-run as 2003103 (excluding eta380).
- v2.2 full report re-run after the v22 edits (V22_FRAMES=full, job 2002563): report.json and tables.md byte-identical
  (git diff empty) -> the edits do not change the v2.2 pipeline.
- v1 s0 ext cache (2002562): deployed codes_max on all 864,000 frames.

## Results (14:30-14:40)
- Training (gpu100, H100): mice 2003185 13:04-13:46, 182,653 steps = 1.00 pass over 748,150,181 tokens, 2,364 s train loop,
  77.3 steps/s (3 SAEs per step), store read 0.94 GB/s, loader wait 49 s; eval FVE s2001/2/3 0.8507 / 0.8507 / 0.8497
  (V s1001-1006: 0.8497-0.8509), L0 15.9, dead 5 / 16 / 8. Ants 2003186: 25,574 steps = 1.00 pass over 104,754,034
  tokens, 667 s, 38.4 steps/s (loader-bound: wait 373 s), FVE 0.8482 / 0.8491 / 0.8476. H100 time incl. the two
  cancelled runs: 14:15 + 3:51 + 41:43 + 13:21 = 1.22 H100-hours.
- Identity: ext codes of V s1001-1006 on the 648,000 atlas frames vs the v2.2 full-window codes: threshold flips only
  (max abs diff 2.23-2.24 = theta*, <= 5.1e-8 of entries > 0.01; ext/identity_codes.json); direct SAE forward pass vs
  assembled ext codes (300 frames): V s1001 0, V s1004 0.002, Vall s2001-2003 0 (ext/fwdcheck_*.json).
- Mice (ext/summary.md, design (a), rule auc): Vall - V nose-nose AUROC -0.036 [-0.069, -0.004], nose-tail AUROC
  +0.003 [-0.039, +0.048]; distinct picks 3/3. A1 NO, A2 NO (lower CI -0.039), A3 yes -> NOT adopted.
- Ants (ants/summary.md, A -> B): Vall - V grooming-to-focal (any) AUROC -0.001 [-0.014, +0.011], Pearson r per-video
  time -0.028 [-0.098, +0.026] -> A4 NO (r lower CI < -0.02).
- Sheets ext/sheets/sheet_mice_Vall_s2001_{auc,bal}.png: nose-nose #463 4/12 labelled, heads together; nose-tail #1540
  (auc) 0/12, mice bunched at the wall spout; #326 (bal) 0/12, heat along tails.
- v1 s1-s10: job 1959004 (now running) encodes only the 648,000 atlas frames (v2.2 frame table), not the ext set; v1 on
  ext = seed 0 only.
