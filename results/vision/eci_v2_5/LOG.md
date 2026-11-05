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
- 01:50-01:58 sheet code tested on the v1 blocks (bug: v1 SAE path is a folder -> `sae.pt`; fixed). v1 s0 mice #611
  nose-nose 0/12 labelled, #414 nose-tail 2/12; ants #90 10/12; per-patch map max == stored frame code within 0.0073.
- 01:58 edge-mode caches for V s1001-1006 (P(on) recomputed on the edge-excluded sequences) and v1 s0; tags + prep.
- 02:21 ants training done (2024621: 47.3 min wall = 0.79 H100-hours; 51,123 steps = 2.00 passes; 21.2 steps/s with 12
  SAEs; GPU util ~80% while training; theta* / FVE calibration of 12 SAEs took ~6 min on the GPU). Eval FVE N0 0.875-0.876,
  Wh 0.845, W1 0.822, C1 0.789; dead latents at the end <= 3 / 2048; never firing on the eval sample <= 0.5%.
- 02:29-02:46 ants codes (49 CPU tasks), merge, caches, tags; forward check max |diff| 0 (12 models x 200 frames).
- 02:49 ants report: V A -> B AUROC 0.861 (= v2.3), v1 s0 precision@top3/video 0.788 (= metric_top3). Every motion arm is
  BELOW N0 on ants AUROC (paired): W1 -0.075 [-0.096, -0.053], Wh -0.024 [-0.040, -0.011], C1 -0.030 [-0.059, +0.000].
  Ants non-inferiority vs V: N0 passes (lower CI -0.010); W1 / Wh / C1 fail (-0.095 / -0.041 / -0.049). So no motion arm
  can pass (B and the deploy rule both need ants not worse) -> escalation ladder triggered.
- 02:55 AMENDMENT 1 (escalation 2, multi-lag concat M2 = [x, a1 Delta1, a2 Delta2], 3072-d) written; first alpha1 jobs ran
  out of memory (float64 copy of 2M x 3072), rewritten as running sums; alpha_1 mice 1.2486 / ants 1.2989, alpha_2
  reproduced; featcheck M2 bit-exact. Committed c4d3cf5 BEFORE M2 training.
- 03:02 M2 training: ants 2025276 (chunks of 256 windows = the main stream, paired with N0), mice 2025277 (chunks of 128
  windows for GPU memory -> not the same batches as N0). 3 SAEs each; ~40-45 steps/s.
- Escalation 3 code (wider SAE, arm suffix 'w', 4096 latents, prefixes + 4096) added, not used until its own amendment.
- 04:00 AMENDMENT 2 (escalation 3: wider SAE, 4096 latents, on Wh = Whw, plus the same-width static control N0w) committed
  e82567d before any wide training; ants 2025610 / mice 2025627. Submit-script bug caught before it ran (the codes step
  would have re-encoded the main arms instead of the wide keys): cancelled and resubmitted with explicit keys.
- 04:09 mice main training done (2024620: 2 h 35 min, 182,537 steps = 1.00 pass); eval FVE N0 0.874, Wh 0.845, W1 0.821,
  C1 0.786; dead latents <= 14 / 2048. 04:15 mice M2 done (1 h 13 min). Wide: ants 35 min, mice 1 h 46 min.
- ~06:00 the parent session restarted and stopped this agent; all queued Slurm chains (codes, merge, caches, tags,
  forward checks) had already completed on their own. Forward check max |diff| 0 for all 21 mice and 21 ants models.
- 08:44-08:49 (resumed) mice + ants reports (`mice/report.json`, `ants/report.json`), `v25_summary.py` -> summary.json,
  tables.md. Total v2.5 H100 time: 7.39 H100-hours (budget 8): main mice 2.58, main ants 0.79, M2 ants 0.44, M2 mice 1.22,
  wide ants 0.58, wide mice 1.77.
- RESULT: no (arm, rule) passes A, no motion arm passes B, no motion arm passes ants non-inferiority -> no deploy
  candidate. Closest: Whw (rule auc). Details in summary.md.
- 08:51-08:55 figures (`figures/sheet_mice_v1_N0w_Whw.png`, `figures/sheet_ants_v1_N0w_Whw.png`; blocks v1 s0 | N0w s2501
  rule dist | Whw best seed rule auc; 12 held-out frames, 1 per video, raw-code ranking; map max == stored code within 0.008).
  Looked at: mice nose-nose v1 #611 0/12 (bodies of clustered mice), N0w #1061 9/12 (heads together), Whw #1835 1/12 (gap
  between two nearby but separate mice); mice nose-tail v1 #414 2/12 (broad on bodies), N0w #2424 4/12 (head at another
  mouse's flank / rear in about half the frames), Whw #1935 1/12 (heat along tails: a tail detector, not nose-to-rear);
  ants v1 #90 10/12, N0w #1 12/12, Whw #73 9/12 (all clusters of ants around the focal; v2 maps far more localised).
  -> A's visual clause fails for Whw (`figures/visual.json`). summary.md not written (agent harness blocks report files);
  tables.md + summary.json hold the numbers.
