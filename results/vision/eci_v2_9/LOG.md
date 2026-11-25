# ECI v2.9 log: confirm rule sgap on fresh PRED-contact dictionaries, vary the sparsity k

- **Pre-registration:** `scripts/eci/v2/v29_prereg.md`, commit 7cbda9b (2026-11-25 ~07:30), before any v2.9 training or scoring.
- **Code:**
  - `v29_train.py` (the v2.7 PREDk training loop with a per-SAE k);
  - `v29_eval.py` (the v2.5 / v2.7 scoring, redirected);
  - `v29_sgap.py` (rule sgap and the per-dictionary metrics; reuses the v2.8 audit kernels);
  - `v29_summary.py` (applies the decisions literally);
  - wrappers `v29_*.sh`.
- **Tables:** `tables.md`. **Decisions:** `summary.json`. **Per dictionary:** `mice/dicts/*.json`, `ants/dicts/*.json`.
- **Big files:** `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9/` (SAE checkpoints `<dom>/sae/<key>/sae_theta.pt`, frame codes `<dom>/codes/<key>_max.npy`, caches `<dom>/cache/<key>_{raw,pon}.npy`).

## Timeline (2026-11-25)
- **07:22 Identity check** (job 3198265, gpu100, gpu275, 3 min).
  - Setup: v2.9 code, k = 16, seed 2701, mice, 5,001 steps.
  - Result: reproduces v2.7 PREDk_s2701 bit for bit. Loss 13147.3017578125 at step 0, loss 4065.628662109375 at step 5,000. L0, L1, aux, FVE 0.84846, dead latents 49 and batch contact-weight statistics all equal.
- **07:25 Training submitted:**
  - mice job 3198304 (gpu100 H100, gpu275): 12 SAEs, 120,000 steps = 0.657 pass, 3 h 33 min, 9.54 steps/s, median GPU utilisation 91%;
  - ants job 3198305 (same node): 25,561 steps = 1 pass, 57 min, 8.39 steps/s, median GPU utilisation 90.5%.
  - Total v2.9 GPU use: 4.56 H100-hours (budget 10).
  - Tokens are streamed from the beegfs token store, exactly as in v2.7. v2.7 does not stage tokens to /localhome: one 0.66-pass read of about 1.5 TB does not fit there.
- **Scoring:**
  - codes / merge / cache / tag / prep / forward check / sgap: arrays 3198306-3198331 (mice) and 3198332-3198357 (ants);
  - old models (v2.7 PREDk, V, v1 s0): jobs 3198369-3198370;
  - forward check (direct SAE pass vs the assembled frame codes, 200 frames): max abs difference 0.0 for all 24 new dictionaries;
  - official-rule picks (auc / dist / dist2) recomputed == the tags for every dictionary.
- **09:40 ants report** (job 3200278, 19 min; v1 s1-s10 dropped to avoid heavy NFS re-reads, v1 s0 kept).
- **~13:00** `v29_summary.py` (decisions); mice report job 3211027; POST HOC split combine; sheets 3211028-3211030; cards 3211031-3211033.

## Training health (eval FVE / L0 per token / dead latents at end of training / fraction never firing on 1.5M eval tokens)

| domain | arm (seeds) | eval FVE | L0 per token | dead latents | never firing on eval |
|---|---|---|---|---|---|
| mice | k = 16 (s2901-2906) | 0.8635-0.8641 | 15.93-15.97 | 34-63 | 3.8-4.7% |
| mice | k = 8 (s2901-2903) | 0.8173-0.8185 | 7.92 | 63-108 | 5.9-6.9% |
| mice | k = 32 (s2901-2903) | 0.8962 | 31.95-31.99 | 435-494 | 13.4-14.7% |
| ants | k = 16 (s2901-2906) | 0.8576-0.8584 | 16.10-16.15 | 2-4 | 0.7-1.3% |
| ants | k = 8 s2903 | 0.811 | 8.06 | 4 | 1.5% |
| ants | k = 8 s2901 | 0.800 | 8.07 | 2,333 | 65.4% |
| ants | k = 8 s2902 | 0.799 | 8.05 | 2,483 | 69.1% |
| ants | k = 32 (s2901-2903) | 0.8928-0.8932 | 32.23-32.30 | 27-33 | 4.8-6.4% |

Problems:
- ants k = 8 s2901 and s2902 collapsed (about 60% of latents dead).
- mice k = 32 has about 12% dead latents.

## Pre-registered results (applied literally, `summary.json`)
- **Q1 confirmation:** 1 of 6 fresh k = 16 dictionaries is accepted, PREDk16_s2901.
  - Old v2.7 PREDk scored with the same code: 2 of 3 (s2702, s2703).
  - s2904, s2905 and s2906 have no sgap pair: every nose-nose candidate with selection discrimination AUROC <= 0.4 fires on >= 10% of frames.
  - s2902 and s2903 fail the nose-nose criteria (AUROC 0.670 / 0.665, AP lower CI 0.030 / 0.030, precision@top3/video 0.106 / 0.093). s2903 also fails nose-tail (AP lower CI 0.075, precision@top3/video 0.286).
- **Q2 k (seeds 2901-2903):**

  | arm | accepted | median SG AP sum |
  |---|---|---|
  | k = 8 | 2/3 (s2901, s2902) | 0.271 |
  | k = 16 | 1/3 | 0.175 |
  | k = 32 | 0/3 (no sgap pair in any seed) | 0 |

  k = 8 has a strictly higher accepted fraction AND a higher median SG AP sum, so **k changes to 8**.
- **Q3 deliverable (mice):** pool = k = 8 s2901-2903; accepted s2901 and s2902. The maximum SG AP sum is **PREDk8_s2901** (SG AP sum 0.281): nose-nose #3422 (level 5), nose-tail #1027 (level 4). Held-out, 72 videos, 431,712 frames:

  | neuron | fires on | AUROC | AP [95% video CI] | precision@top3/video (chance 0.0157 nose-nose / 0.0103 nose-tail) | precision@top10/video | per-video time bias, s | per-video MAE, s |
  |---|---|---|---|---|---|---|---|
  | nose-nose #3422 | 3.89% of frames | 0.818 | 0.185 [0.148, 0.222] | 0.486 | 0.438 | +4.8 +- 1.7 | 10.3 +- 1.3 |
  | nose-tail #1027 | 2.54% | 0.847 | 0.218 [0.123, 0.304] | 0.402 | 0.343 | +11.5 +- 2.5 | 14.4 +- 2.2 |

  Top-3 overlap 0. For the fresh seeds, neurons were picked and dictionaries ranked on the SG pools only; the held-out pools entered only through the pass / fail gate.
- **Ants deliverable:** PREDk8_s2901 (same arm and seed); it meets the ants criteria. Grooming-to-focal (any):

  | direction | neuron | held-out AUROC [CI] | precision@top3/video [CI] |
  |---|---|---|---|
  | A -> B | #53 | 0.870 [0.848, 0.888] | 0.823 [0.737, 0.902] |
  | B -> A | #53 | 0.880 [0.857, 0.900] | 0.860 [0.787, 0.924] |

  **Caveat:** this is one of the two collapsed ants k = 8 dictionaries (2,333 dead latents).
- **Not chosen only because of the k rule:** PREDk16_s2901 has the highest SG AP sum of all scored dictionaries (0.401). Held-out: nose-nose #3461 AUROC 0.847, AP 0.271 [0.230, 0.308], precision@top3/video 0.588; nose-tail #1780 AUROC 0.845, AP 0.237 [0.137, 0.329], precision@top3/video 0.381. Its ants dictionary fails the ants criteria (A -> B AUROC 0.832).

## Errata to the pre-registration (reported by the coordinator, verified; no rule changed)
1. The SG selection side is 11 pools, 66 videos, **395,736** frames. The prereg's 431,712 is the size of the held-out side.
2. The 10% firing cap DOES change one audit pick: PREDk s2701 nose-nose #359 (fires on about 62% of frames) becomes #690. The s2702 and s2703 picks are unchanged. The prereg claimed the cap changes nothing; that held for s2702 / s2703 only.

## Additions after the pre-registration (POST HOC, not part of its decisions)
- **Sheets for v2.7 PREDk, judged by eye by another agent:**
  - s2703: nose-nose #3593 9/12, nose-tail #3009 11/12;
  - s2702: nose-nose 6/12;
  - s2701: nose-nose 2/12.
- **Random pool splits** (the v2.8 split check, 50 annotator-stratified splits; `v29_split.py`, `v29_split_combine.py`, `split/tables.md`).
  - Columns: same picks as official / chosen by select-side AP sum among the 15 PREDk dictionaries / meets the point thresholds.
  - Metric cells: mean +- SD of the evaluate-side values.

  | dictionary | same picks / chosen / meets thresholds | nose-nose AUROC | nose-nose AP | nose-nose precision@top3/video | nose-tail AUROC | nose-tail AP | nose-tail precision@top3/video |
  |---|---|---|---|---|---|---|---|
  | PREDk8_s2901 | 1.00 / 0.00 / 0.80 | 0.775 +- 0.011 | 0.169 +- 0.013 | 0.493 +- 0.030 | 0.805 +- 0.022 | 0.142 +- 0.048 | 0.371 +- 0.065 |
  | PREDk8_s2902 | 1.00 / 0.00 / 0.56 | 0.760 +- 0.011 | 0.138 +- 0.010 | 0.445 +- 0.032 | 0.795 +- 0.025 | 0.146 +- 0.051 | 0.342 +- 0.061 |
  | PREDk16_s2901 | 0.86 / 0.86 / 0.58 | 0.789 +- 0.046 | 0.237 +- 0.078 | 0.573 +- 0.164 | 0.812 +- 0.024 | 0.148 +- 0.054 | 0.351 +- 0.058 |
  | PREDk_s2702 | 0.62 / 0.04 / 0.78 | 0.773 +- 0.010 | 0.179 +- 0.016 | 0.543 +- 0.043 | 0.802 +- 0.024 | 0.151 +- 0.052 | 0.363 +- 0.058 |
  | PREDk_s2703 | 1.00 / 0.10 / 0.58 | 0.760 +- 0.011 | 0.217 +- 0.017 | 0.549 +- 0.039 | 0.795 +- 0.027 | 0.137 +- 0.048 | 0.356 +- 0.054 |
  | v1 s0 (no sgap pair; re-picked by best select AP) | - | 0.681 +- 0.006 | 0.050 +- 0.004 | 0.101 +- 0.019 | 0.761 +- 0.014 | 0.057 +- 0.011 | 0.159 +- 0.035 |

  Chance: AP 0.0208 (nose-nose) / 0.0101 (nose-tail). The top-3-overlap criterion is not computable from the split pickles and is omitted from "meets thresholds".
- **Leave one annotator out, PREDk8_s2901** (sgap re-picks #3422 / #1027 every time):

  | left out | nose-nose AP (chance) | nose-nose precision@top3/video | nose-tail AP (chance) | nose-tail precision@top3/video |
  |---|---|---|---|---|
  | SG | 0.161 (0.0259) | 0.510 | 0.120 (0.0128) | 0.360 |
  | SD | 0.194 (0.0152) | 0.511 | 0.069 (0.0053) | 0.231 |
  | MF | 0.182 (0.0163) | 0.537 | 0.060 (0.0029) | 0.333 |
  | MB | 0.183 (0.0159) | 0.417 | 0.364 (0.0166) | 0.625 |

  In all 4 the dictionary chosen by select-side AP sum is PREDk16_s2901, not PREDk8_s2901.
- **Sheets** (`v28_sheets.py --ver 29`, sgap picks; `results/vision/eci_v2_9/sheets`): PREDk8_s2901, PREDk16_s2901, PREDk16_s2902 (jobs 3211028-3211030). By-eye counts pending.
- **Alignment cards** (`v29_card.py` = v28_card with v2.9 paths; `results/vision/eci_v2_9/figs`), PREDk8_s2901 vs v1 s0:
  - nose-nose #3422 vs #611;
  - nose-tail #1027 vs #414;
  - ants grooming #53 vs #90.
  - Jobs 3211031-3211033.

## Follow-up (2026-11-25 afternoon; all POST HOC, not part of the pre-registered decisions)
- **Mice arm-level report:** job 3211027, written to `mice/report.json` (v2.5 report machinery; rules auc / dist / dist2; 23 s).
- **Sheets** (jobs 3211028-3211030, `sheets/mice/<key>/`):
  - Format: 12 top held-out frames by raw code, at most 2 per video; here all 12 tiles come from 12 distinct videos.
  - Tile identity (recomputed map max == stored code): 0.0 on every tile.
  - Self-check (200 random held-out frames x all latents): max abs difference <= 0.0078 (fp16).
  - By eye, strict definitions: nose-nose = two snouts touching or within a snout length; nose-tail = snout at the other's tail base / anogenital region.

  | dictionary, neuron | by eye | labels | what the other frames show |
  |---|---|---|---|
  | PREDk8_s2901 nose-nose #3422 | 9/12 | 7/12 | #2 and #7 are 3-4 mouse huddles with the max token at a head among bodies; #6 is snout to the other's flank |
  | PREDk8_s2901 nose-tail #1027 | 12/12 | 9/12 | all are head-to-rear trains, the 3 unlabelled included (#10 the least clear) |
  | PREDk16_s2901 nose-nose #3461 | 11/12 | 10/12 | #12 is snout to the other's shoulder / flank; #4 (unlabelled) looks like a true head-to-head |
  | PREDk16_s2901 nose-tail #1780 | 11/12 | 8/12 | #11 is ambiguous (vertical train, snout near the rear but partly occluded) |
  | PREDk16_s2902 nose-nose #3811 | 2/12 | 0/12 | mostly snout at another's flank, side-by-side resting and corner huddles; not a nose-nose neuron (it also failed the numeric criteria) |
  | PREDk16_s2902 nose-tail #2455 | 11/12 | 9/12 | #12 is snout to the other's side / rear; the frames overlap heavily with #1780's videos |

  These are single-observer judgements on 0.35-MP thumbnails; huddle frames are hard to call.
- **Alignment cards:** the first attempt (jobs 3211031-3211033) failed with "no token path". Cause: `v28_card.py` hard-codes `v28_sheets.init('27')`, so the v2.9 arms were not registered. It was not the PDF save, whose fix (`image.composite_image=False`) is inherited. Fix: `v29_card.py` forces `init('29')`, checked with a short srun (both v2.9 encoders load, k 8 / 16). Resubmitted as jobs 3211955-3211960:
  - mice nose-nose and nose-tail for PREDk8_s2901 (#3422 / #1027) and PREDk16_s2901 (#3461 / #1780), each vs v1 s0 #611 / #414;
  - ants grooming-to-focal for PREDk8_s2901 #53 and PREDk16_s2903 #49, each vs v1 s0 #90.
- **Per-video time on the held-out pools** (sum of P(on) x 0.2 s minus annotated time; mean +- SE over 72 videos):
  - Annotated time per video: nose-nose 18.8 +- 1.6 s, nose-tail 10.1 +- 1.5 s.
  - Constant-rate guess (SG prevalence x video length) MAE: nose-nose 13.9 +- 1.2 s, nose-tail 11.4 +- 1.0 s.

  | neuron | bias | MAE |
  |---|---|---|
  | PREDk8_s2901 nose-nose #3422 | +4.8 +- 1.7 s | 10.3 +- 1.3 s |
  | PREDk8_s2901 nose-tail #1027 | +11.5 +- 2.5 s | 14.4 +- 2.2 s |
  | PREDk16_s2901 nose-nose #3461 | -6.0 +- 1.2 s | 8.9 +- 0.9 s |
  | PREDk16_s2901 nose-tail #1780 | +21.4 +- 3.1 s | 22.4 +- 3.0 s |

  Both nose-tail neurons do worse than the constant-rate guess on per-video time.
- **Ants candidates side by side** (grooming-to-focal (any); held-out AUROC / precision@top3/video):

  | dictionary | eval FVE | dead latents | never fire on eval | A -> B | B -> A |
  |---|---|---|---|---|---|
  | PREDk16_s2903 (#49) | 0.858 | 2 | 1.1% | 0.884 / 0.869 | 0.889 / 0.778 |
  | PREDk8_s2901 (#53) | 0.800 | 2,333 | 65.4% | 0.870 / 0.823 | 0.880 / 0.860 |
  | v2.7 PREDk s2703 (#33) | 0.859 | 3 | 1.3% | 0.867 / 0.848 | 0.886 / 0.836 |
  | v1 s0 #90 (fixed; half B only, from the card job log) | - | - | - | 0.875 / 0.788 | - |
