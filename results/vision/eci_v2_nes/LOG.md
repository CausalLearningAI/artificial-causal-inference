# ECI v2-temporary NES log: the v1 atlas NES on the v2 representation

Goal: re-run the published v1 NES searches (do-pilot atlas, model "v1" = "online") with the SAME settings, changing only
the representation, and stage a "v2-temporary" atlas section (draft, not published).

## Dictionaries
- mice: v2.9 PREDk16_s2901 (4096 latents, Matryoshka levels 128 / 256 / 512 / 2048 / 4096); nose-nose #3461, nose-tail #1780.
- ants: v2.7 PREDk_s2703 (4096 latents, same levels); grooming-to-focal #33.

## Recon (2026-11-26)
- v1 NES result sets used by the atlas (scripts/eci/build_atlas_v37.py ONLINE):
  - mice: results/vision/mice/eci/nes/matryoshka_btk_1024_k16_fg448al_s0 (run 2026-10-07; habituation = its last 15 min;
    the published v36 mice page used the whole 30 min of habituation = the 'hwhole' rows of the same runs);
  - ants: results/vision/ants/eci30/nes/matryoshka_btk_1024_k16_antsfg_s0 (30-min layout, 232 videos; 2026-10-15);
    ants v4 on the page = results/vision/ants/eci_c1v4_noday/... (same SAE, v4-only layout, no day covariate).
- v1 settings (sanity.json): `run_nes.py --primary-pooling max` (codes_max primary, per-video mean activation, t-test,
  Bonferroni 0.05, full window), nuisance none, select tau, no recording-day covariate, core analysis set, prefixes
  128 / 256 / 1024; bout rate (`run_nes_bouts.py`, q 0.95, gap 0) for both domains; latency (`run_nes_latency.py`, q 0.95,
  common window) for mice only.
- Coverage of the stored v2 codes (NOT enough for NES):
  - mice v2.9 codes (`eci_codes/v2_9/mice/codes/PREDk16_s2901_max.npy`): 144 of 432 v1 videos (the 24 annotated pools),
    863,424 of 2,592,000 frames;
  - ants v2.7 codes (`eci_codes/v2_7/ants/codes/PREDk_s2703_max.npy`): v3 107 + v4 40 videos, first 10 min only,
    440,412 of 1,824,000 frames; no v2, no v4_infested.
  - The v2_full token store holds every video at full length (mice 648 videos, ants 232), so the missing codes are a
    CPU job: `scripts/eci/v2/v2nes_codes.py` (the v2.9 scoring code `v25_train.cmd_codes` on a full row table).

## Code
- `scripts/eci/v2/v2nes_codes.py` rows / codes / layout: per-frame max codes of one dictionary on every frame of the v1
  NES layout (mice dataset/mice/v1/annotations.csv, ants dataset/ants/eci30/annotations.csv), n_fg = M_t tokens per
  frame, identity check vs the stored scoring codes. Edge frames (first / last 2 per video) copy the nearest frame.
- `scripts/eci/v2/v2nes_nes.sh`: the v1 runners from a git archive, v1 settings; forced differences: `--poolings max`
  (v2 has no mean-pool codes), bouts `--compare-pooling max`, prefixes 128,256,512,2048,4096 (the v2 levels).
- `scripts/eci/v2/v2nes_tables.py`: v1 vs v2 per contrast -> compare.json, tables.md; `--check-v1` compares the v1
  rerun with the published summary.csv.
- `scripts/eci/v2/v2nes_page.py`: the draft atlas page (atlas/draft_v2_temporary/).

## Jobs (2026-11-26 ~12:15)
- rows (srun 3232481): mice 432 videos, 2,592,000 layout frames, 2,590,272 non-edge rows; ants 232 videos, 1,824,000
  frames, 1,823,072 non-edge rows. All videos found in the token store with matching frame counts.
- smoke: ants v2/a1 2,996 frames in 49 s; mice wt_ash1l_m_1_S_O 4,496 frames in 145 s (1 dictionary, 8 CPUs).
- chain: see jobs.txt (codes arrays 3232491 mice x108 / 3232492 ants x58 -> layout 3232493 / 3232494 -> NES mean
  3232495 / 3232496 -> mice bouts 3232497 + latency 3232498, ants bouts 3232499).
- v1 code-path check: jobs 3232488 (mice), 3232489 (ants).

## Sanity checks
- **v1 code path, ants (job 3232489):** today's runner on the v1 codes reproduces the published v1 summary.csv: 611 of
  611 rows match on every key column (analysis, setting, round, neuron, direction, ...), max |delta tau| 1.3e-13,
  max |delta p| 5e-16. The only text difference is the reworded 'confound' label (commit a781b20).
