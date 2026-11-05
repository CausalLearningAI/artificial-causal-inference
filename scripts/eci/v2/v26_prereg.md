# ECI v2.6 pre-registration: static-input SAEs, time only through the loss

Written 2026-11-05 ~09:30, BEFORE any v2.6 SAE was trained or scored. Seen before writing: all v2.5 results (no deploy
candidate; static wide N0w|dist was the best-looking option: nose-tail precision@top3/video 0.171 vs V 0.086, +0.085
[+0.012, +0.159]; nose-nose AUROC 0.822 vs 0.805; summing / concatenating motion into the INPUT hurt ants AUROC).

## Why
User (v2.6, quick): confirm the static wide recipe (N0w) on fresh seeds, and test temporal context in a way that cannot
displace static appearance: every arm's INPUT is x_t only; time enters only through the training loss. Inference /
readout for every arm = threshold-mode codes of x_t (same as N0w).

## Data (= v2.5)
Store v2.1_m5, DINOv3-L/16 block 20, native grid, 5 fps, mask v5 M_t tokens; edge rule frames 2 <= t < T - 2 (training and
scoring); L = 2 frames (+-0.4 s); x_{t+2} = the same grid position at t + 2 (always present). Training stream = v2.5 pass-0
order (windows of 100 frames, chunks of 256, rng 20261105, per-seed permutation): all arms of a seed see the SAME batches
(paired with N0w-c). Seeds 2601, 2602, 2603. Code: `scripts/eci/v2/v26_train.py`; scoring = the v2.5 code (`v25_eval.py`,
`v26_eval.py` wrapper), unchanged frame sets / metrics / bootstrap.

**Training length (budget: 4 H100-hours for v2.6; 5 arms x 3 seeds per domain in one job):** mice 50,000 steps x 4096
tokens (~0.27 pass over the 747.7M edge-excluded tokens), ants 25,561 steps = 1 pass. Same lr / batch / warm-up as V
(cosine to 10% over these steps). NOTE: N0w-c on mice is therefore a SHORTER run than v2.5 N0w (1 pass); v2.5 N0w is
reported next to it as a reference. This is a budget choice, stated before any result.

## Arms (all d_in 1024 = x_t; Matryoshka BatchTopK, k = 16, AuxK, TokenNorm one scalar)
| arm | latents / prefixes | loss |
|---|---|---|
| **N0w-c** (code N0wc) | 4096 / 128,256,512,2048,4096 | Matryoshka reconstruction of x_t (= v2.5 N0w recipe) |
| **PRED** | 4096 / same | L(x_t) + lambda L1(x_{t+2}); L1 = same Matryoshka prefix sum, second decoder D1 (W_dec1, b_dec1) on the SAME code z_t; **lambda = 1**; W_dec1 initialised = W_dec, b_dec1 = b_dec; x_{t+2} normalised with the arm's TokenNorm; D1 unit-norm rows + parallel-gradient removal like D0 |
| **SMOOTH** | 4096 / same | L(x_t) + mu mean_tokens ||z_t - z_{t+2}||_1; z_{t+2} = the batch-top-k code of x_{t+2}; gradient through both |
| **RW** | 4096 / same | per-token weighted reconstruction sum_m mean_i w_i ||r_m,i||^2 (AuxK unweighted); sampling unchanged |
| **N0x** | 8192 / 128,256,512,2048,4096,8192 | plain reconstruction |

- **mu rule (label-free, measured in the training job BEFORE the SMOOTH SAEs are created):** a throwaway plain 4096-latent
  SAE (seed 2600) is trained for 2,000 steps on the first chunk; over steps 1,500-1,999, R = mean Matryoshka reconstruction
  loss (sum over prefixes) and C = mean ||z_t - z_{t+2}||_1; **mu = 0.1 R / C** (penalty ~10% of reconstruction early in
  training). Logged in `results/vision/eci_v2_6/train/<dom>_mu.json`.
- **RW weights (label-free, step kmeans, before training):** k-means k = 256 (25 Lloyd iterations, init = 256 random
  tokens, rng 26) on 500k tokens of the v2.5 rng-6 sample (2M tokens of x); cluster frequencies f_c on the 2M sample;
  w_c = f_c^{-1/2}, normalised so the token mean is 1, then clipped at 10 (not renormalised; the post-clip token mean is
  reported); a training token's weight = w of its nearest centroid (raw x space). `<dom>_rw.json` / `.npz`.

Checks: featcheck (loader x_t, x_{t+2} == independent dict lookup, bit-exact, 50 windows incl. 10 at the edge boundary);
GPU == numpy on the first 16 windows; codes forward check; SAE health (FVE of x_t via D0, dead latents).

## Scoring (= v2.5)
Mice: edge-excluded ext frames (863,424), select on annotator SG's 11 pools, measure on the 12 non-SG pools; nose-nose
(code nose_nose) and nose-tail (code nt); rules **auc** and **dist** (nose-tail != nose-nose pick, discrimination AUROC
>= 0.6). Ants: v3 + v4, select on half A, measure on B (primary), B -> A reported; grooming-to-focal (any); best AUROC.
Metrics: AUROC, precision@top3/video (+ chance), distinct picks, discrimination AUROC, per-video MAE, Pearson r. Bootstrap
1,000 (mice pools + seeds, ants videos within strata + seeds); v2.6 arm vs N0w-c PAIRED (same seed index, same batches);
vs V / v1 unpaired. Baselines V (s1001-1006) and v1 s0: the v2.5 edge-excluded caches / tags.

## Pre-registered decisions
- **Confirmation of N0w** (N0w-c, rule dist, vs V|auc): nose-tail precision@top3/video lower 95% CI > 0; nose-nose
  AUROC lower CI >= -0.02; ants AUROC (A -> B) lower CI >= -0.02; distinct picks 3/3.
- **Temporal B** (PRED or SMOOTH vs N0w-c, paired, same rule; evaluated for rules dist and auc): lower CI > 0 on at
  least one of {mice nose-tail AUROC, mice nose-tail precision@top3/video, ants AUROC, ants precision@top3/video}, AND
  mice nose-nose AUROC lower CI >= -0.02 AND ants AUROC lower CI >= -0.02 (both vs N0w-c). Plus non-inferiority vs V
  (nose-nose AUROC and ants AUROC lower CI >= -0.02). All contrasts reported.
- **RW, N0x vs N0w-c** (paired): the same non-inferiority (nose-nose AUROC, ants AUROC lower CI >= -0.02 vs N0w-c) plus a
  gain with lower CI > 0 on mice nose-tail precision@top3/video or mice nose-nose AUROC.
- **Deploy pick**: if a temporal arm passes B (and its non-inferiority), deploy it (user preference for temporal context;
  ties -> higher nose-tail precision@top3/video). Otherwise the best of RW / N0x passing its rule, else N0w-c if confirmed,
  else none. Rule for the deploy = dist unless stated.

## Figures
Side-by-side sheets: v1 s0 (mice #611 / #414, ants #90) | N0w-c | best temporal arm (by the B contrasts: highest mice
nose-tail precision@top3/video under rule dist) | deploy pick if different. Panels: mice nose-nose, mice nose-tail, ants
grooming-to-focal (any); 12 held-out frames, at most 1 per video, raw-code ranking, heat maps, labels; best seed = highest
held-out nose-tail precision@top3/video (ants: grooming precision@top3/video), ties -> lower seed. Described in words.

## Measured before training (12:13-12:16, CPU jobs 2068213-2068218; nothing trained or scored)
- Featcheck (x_t, x_{t+2}) bit-exact: mice 50 windows / 28,979 tokens, ants 50 / 8,187 (incl. 10 at the edge boundary).
- RW k-means (k = 256): cluster frequencies mice 0.00092-0.00748, ants 0.00154-0.00678, no empty cluster; weights mice
  0.728-2.076, ants 0.764-1.606; no cluster clipped (token mean 1.000); effective sample size 98.6% of tokens. The
  clusters are nearly balanced, so RW is a MILD reweighting by construction (stated before any result).
- mu: measured inside the training job (rule above), before the SMOOTH SAEs exist.
