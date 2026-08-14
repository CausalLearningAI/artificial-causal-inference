# Retired runs — what they established

35 run directories were deleted on 2026-08-14. They were scored against validation splits that
are no longer used, so their numbers could not be compared against current results and their
presence made the directory unreadable. This file is what survives them.

`results/` is gitignored, so the deletion is permanent. `retired_runs.json` next to this file
holds the full config + metric dump of all 35, so every number quoted below stays checkable
even though the checkpoints are gone.

**All numbers here are best MONITOR macro AP** — the pre-2026-08-12 schema's headline. That is
a different quantity from the full-val `ap_report` macro AP that current runs report, on a
different split. **Never compare a number in this file to a current result.** They are kept only
so a settled question is not re-run by accident.

---

## 5-pool split — `rd11_2 rd32 rd34 rd35_2 rd41_3` (23 runs)

The main exploration era. Best: `res448_fastdecay` 0.3427.

| finding | evidence |
|---|---|
| **Resolution helps, monotonically** | res336 0.3147 → res448 0.3380 |
| **Focal loss is much worse than BCE** | balanced_focal 0.0812 vs balanced_1to1 0.2439 at identical settings; also fulldata_1M_focal 0.2453 vs fulldata_1M 0.2875. Settled — do not revisit. |
| **Explicit regularization did nothing** | reg_augonly 0.2465, reg_bottleneckonly 0.2408, reg_light 0.2321, reg_heavy 0.2277 — all at or below the 0.2464 baseline, and monotonically *worse* with more of it. |
| **Temporal stride does not help** | stride2 0.2376, stride3 0.2356, stride4 0.2450, none beating the ~0.246 baseline. Independently reconfirmed on the current split (stride 2 cost −10%), so this is settled across two splits. |
| **DINOv3 did not beat DINOv2** | patchgrid256_dinov3 0.2096 vs 0.2465 comparable DINOv2. Note this ran at neg_ratio 20, so it is not a perfectly clean comparison. |
| **Aggressive negative undersampling hurts (at this ratio)** | ratio2 0.1541 vs 0.2439 at neg_ratio 15. Superseded: current runs use neg_ratio 1 successfully with `pos_weight`, so read this as "the old recipe was sensitive to it", not "low ratios are bad". |
| More training frames gave little | fulldata_1M 0.2875 vs the ~0.246 baseline was a real gain, but later shown to buy negative *diversity* only — all positives were already included. |
| LayerNorm / pos-weight variants were flat | fulldata_1M_layernorm 0.2489, _modposw 0.2468, _noposw 0.2744, all within noise of 0.2875. |

## 4x4-era split — `rd14 rd19 rd29 rd35_3` (6 runs)

The original coarse 4×4 (16-token) patch grid. Best: 0.2080.

| finding | evidence |
|---|---|
| **Spatial patch tokens beat CLS pooling** | patchgrid4x4_dinov2 0.1971 vs cls 0.1580. This is the decision that produced the entire patch-grid line of work. |
| **DINOv3 lost here too** | patchgrid4x4_dinov3 0.1344 vs 0.1971 — consistent with the 5-pool result on a different split. |
| **Concatenating encoders was much worse** | patchgrid4x4_concat 0.0708. |
| More frames marginal | 1Mframes 0.2080 vs 0.1971. |

The winning trial's hyperparameters from this era (`n_heads` 8, `hidden_dim` 384, `dropout` 0.4,
`lr` 3e-4, `weight_decay` 1e-4) were read at runtime out of `patchgrid4x4_dinov2/config.json` by
six scripts, including the live trainer. They now live in `src/mice_behavior/head_cfg.py` with a
provenance warning — every condition they were tuned under has since changed.

## Unrecorded split (6 runs)

`val_pools` was never written to these configs, so their provenance is unrecoverable.

**These runs are the source of a wrong conclusion and are the reason this cleanup happened.**
The standing claim "504px beats 448px by 4.7%" came from comparing `res504` (0.3543, *this*
group, split unknown) against `res448` (0.3380, *5-pool* group). Cross-split, therefore
meaningless. On the current split at matched `context_k`, 448 and 504 are a tie — 0.4381 vs
0.4315. Nothing in this group should be cited.

Also here: `res448_blur224` 0.3492, `res448_cosinelr` 0.3486, `capacity_cut` 0.3324,
`combined_best` 0.3160.

---

## What carries forward

Two findings are settled well enough not to re-test: **focal loss loses to BCE**, and
**temporal stride > 1 does not help** (confirmed twice, on two different splits). Two are
worth re-testing under the current recipe because every condition has changed: **DINOv3 vs
DINOv2** (lost twice, but both times frozen, at low resolution, and once at an unmatched
negative ratio — the fine-tuning result changes what "encoder quality" even means), and
**negative sampling ratio**.
