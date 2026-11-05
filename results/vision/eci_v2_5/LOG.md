# ECI v2.5 log: native-resolution SAEs, temporal-context test, video edges excluded

Goal (user, overnight): (A) a SEPARATE mice nose-tail neuron, distinct from the nose-nose neuron, with recognisable
top frames; (B) show that temporal context helps, re-testing a summed motion term at native resolution after fixing the
v2.4 video-edge bug. Edge rule: frames without full temporal context (first / last L = 2 frames of every store video)
are excluded from training and from the scored frame set, for every model incl. V and v1.
Pre-registration: `scripts/eci/v2/v25_prereg.md` (commit fa83d20, before any training or score). Code:
`scripts/eci/v2/v25_train.py` (count / alpha, featcheck, train, rows, codes, merge), `v25_eval.py` (cache, tag, prep,
report, fwdcheck; reuses v22_core / v22_eval / v22_full / v23_ants / v24_top3 by patching module attributes only),
`v25_sheet.py` (figures), `v25_summary.py`. Big files: `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_5/`.

## Timeline (2026-11-05)
- 01:29-01:33 count / alpha / featcheck / rows (CPU jobs 2024596-2024606):
  mice 648 videos, 747,672,867 used M_t tokens (477,314 removed by the edge rule), alpha 1.12266, 1 pass = 182,537 steps;
  ants 232 videos, 104,700,084 used tokens (53,950 removed), alpha 1.18189, 2 passes = 51,123 steps.
  Featcheck bit-exact (50 windows incl. 10 at the edge boundary, all four arms); training windows cover exactly frames
  [2, T - 2) of every video. Scored frames: mice 864,000 -> 863,424 (576 edge frames = 4 x 144 videos); ants 440,412 span
  frames, 0 edge frames (spans are k = 2..2997) -> the ants scored set is unchanged.
- 01:33 pre-registration committed + pushed (fa83d20).
- 01:34 training: mice 2024620, ants 2024621 (gpu100, H100, both on gpu274), 12 SAEs per job (4 arms x 3 seeds) from one
  store read. GPU == numpy features bit-exact on the first 16 windows of both runs. Mice: 22 steps/s (12 SAEs per step),
  GPU util 83%, loader wait 2 s of the first 91 s, ETA 137 min.
- 01:35 codes path smoke-tested with a stand-in checkpoint (ants, 3 videos, 18 s); test files deleted.
- 01:43 scorer identity (V s1001-1006 re-tagged with the v2.5 code on the ORIGINAL ext frames, `identity/mice/report.json`):
  rule auc nose-nose AUROC 0.8053, nose-tail AUROC 0.7906, precision@top3/video 0.270 / 0.086 = v2.3 / v2.4 exactly.
