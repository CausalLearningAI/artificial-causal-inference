"""
ECI v2 vs v1 DIAGNOSIS: why does the v2 representation not beat v1 on annotated behaviours, and which recipe does?

==================================================================================================================
PRE-REGISTRATION (written 2026-10-23 BEFORE any token of this study was encoded and before any score of any arm below
existed. Known before writing: the v2 thin-mask release check, results/vision/eci_v2_mask2/ab_table.md: M0 (= v2
recipe) minus v1, cross-fitted normalised R-precision R~ on the atlas split, full dictionaries: mice nose_nose +0.090,
mice nose_tail -0.007 [-0.018, +0.003], ants v2 groom_any -0.014, ants v3 groom_any -0.033, ants v3 on_lid_any +0.041;
at prefix 256: on_lid_any -0.29 (v2's lid neuron lies outside 256 in most seeds).)

QUESTION. Which of the pipeline differences between v1 (DINOv2-B/14 last layer at 448 px = 32 x 32 grid, dilated
feature+dark mask, Matryoshka BatchTopK 1024 trained on ~68M / ~7M tokens for 5 epochs, EMA threshold) and v2
(DINOv3-L/16 block 20 at native side = 64 grid mice / 48-52 grid ants, tight dark-pixel mask, Matryoshka 2048 on the
~4M / ~1.2M-token burst pool for 30k steps, calibrated theta*) explain the v2 - v1 differences per label, and does a
SIMPLER-or-equal recipe exist that is >= v1 - 0.01 on every label?

FRAMES, LABELS, SCORER: exactly those of the v2 release check (scripts/eci/v2_mask2/release_ab.py): mice = the 43,200
diagnosis frames restricted to the atlas window; ants = the step-2 scored frames of v2 (13,200) and v3 (32,100). SAE
training pools = the step-2 pools (mice 'pool' bursts; ants 'pool_v2' + 'pool_v3', one SAE per domain), roles
train / calib / eval as stored (theta* on calib videos, FVE on eval videos). Codes = frame max over the arm's kept
tokens, UNSMOOTHED (as release_ab). R~ = src/eci/v2/score.py; PRIMARY value = R~ of the cross-fitted best neuron on the
ATLAS split ('R_atlas', release_ab), the 10-random-halves mean 'R_rnd' reported.
CELLS (5): mice nose_nose, mice nose_tail, ants v2 groom_any, ants v3 groom_any, ants v3 on_lid_any; at the full
dictionary and at prefix 256 (both decide); nn_mutual / np_directional reported.

TOKEN SETS
    E0     the existing step-2 M0 store (eci_codes/v2_mask2): DINOv3-L/16 block 20 + final LN, native side (mice 1024 ->
           64 grid; ants 768 / 832 -> 48 / 52 grid), tight mask (>= 25% frozen-cue animal pixels per 16 px patch).
    32-grid sets, encoded here ONCE (step 'prep') on exactly the same frames, from the same decoded canonical frame
    (native side S) resized on the GPU (bicubic, antialias) to 512 px (patch-16 models) / 448 px (patch-14 models), so
    every set has a 32 x 32 grid = v1's token density; output of block L passed through the model's final LayerNorm,
    CLS / registers dropped:
        d3l16 / d3l20 / d3l24   DINOv3-L/16 (rev ea8dc28) blocks 16 / 20 / 24 (bf16 autocast, as v2)
        d3b12                   DINOv3-B/16 block 12 (last)               (bf16)
        d2b10 / d2b12           DINOv2-B/14 blocks 10 / 12 (12 = last = v1's encoder) (fp32, as v1)
        d2l24                   DINOv2-L/14 block 24 (last)               (bf16)
    32-grid masks: 'tight' = >= 25% frozen-cue animal pixels in the 32-grid cell (the v2 rule at v1's grid; pixel
    fraction pooled from the native-side pixel mask); 'dil' = tight dilated by one cell (3 x 3) = v1's one-patch ring.

ARMS (each: Matryoshka BatchTopK, frame max readout; the v2 SAE recipe W 2048, prefixes 128/512/2048, k 16, 30,000 steps
x 4096, lr 5e-4, global centring + scalar scale (TokenNorm), calibrated theta*, unless stated). SEEDS 300-304 (5 per
arm per domain).
  Stage A (SAE knobs, on E0):
    A_ref    the v2 recipe re-trained with this script (identity check against the 20 / 33 existing M0 seeds)
    A_w1024  W 1024, prefixes 128/256/512/1024 (v1's)
    A_w4096  W 4096, prefixes 128/512/2048/4096
    A_k8, A_k32           k 8 / 32
    A_s90k   90,000 steps (v1 mice saw ~82k steps)
    A_half   half of the training frames (seeded) -> data-size slope
    A_z      per-dimension z-scored tokens (mean / sd of 250k training tokens) instead of global centring + scale
    A_ema    the A_ref SAEs read out with their training EMA threshold instead of theta* (v1's readout); no new SAE
  Stage B (tokens, on the 32-grid sets, tight mask unless stated):
    B_d3l20  density: vs E0 (same encoder / layer, 32 vs native grid)
    B_d3l24, B_d3l16       layer (vs B_d3l20)
    B_d2b12  v1's encoder in the v2 pipeline (vs B_d3l20: DINOv3-L b20 -> DINOv2-B last)
    B_d2b10  layer for DINOv2-B (vs B_d2b12)
    B_d3b12  family at size B: vs B_d2b12          B_d2l24  family at size L (last layer): vs B_d3l24
    B_d3l20dil  mask: one-cell dilation (vs B_d3l20)
    B_v1like    d2b12 + dil + W 1024 (v1 prefixes) + EMA readout = v1's recipe on the v2 pool (vs v1 itself: the
                remaining difference is training data / steps)
CEILINGS (step 'probe', supervised, labels used only to measure): per token set (E0 and the seven 32-grid sets, tight
    mask) and cell: T-max = one linear direction on per-dimension standardised tokens, frame score = max over kept
    tokens (scripts/eci/nnnt/r2.py fit_probe 'max', 1500 steps, balanced batches); best raw channel = the best single
    standardised token dimension (either sign), frame max, selected by cross-fitting exactly like a neuron. Both on the
    atlas split (R~ held out, both directions). Reading: T-max(set) >> best neuron(set) = dictionary bottleneck;
    T-max(set) differs across sets = encoder / density bottleneck.

EFFECT OF A FACTOR on a cell = mean(arm) - mean(its reference) over seeds, Welch 95% CI. References: Stage A arms vs
    the existing M0 seeds (mice 70-89, ants 70-102; A_ref must lie inside their 95% range, else the A references
    switch to A_ref and this is reported); A_ema vs A_ref (paired by seed, t-CI); B arms as listed.
    A factor EXPLAINS part of the v1 - v2 gap on a cell iff moving it from the v2 level to the v1 level shifts R~
    toward v1 by >= 0.01 with the CI excluding 0.

COMBINED RECIPE (step 3; the recipe is fixed from the stage A/B results, then pre-registered in a second dated block
    here before any confirmation SAE is trained). Per factor, a non-v2 level is adopted iff (a) it improves at least one
    of the 10 decision cells (5 cells x {full, 256}) with the CI excluding 0, and (b) no decision cell gets worse with
    the CI upper bound < 0 AND the mean drop > 0.01, in BOTH domains' cells. TIE-BREAK (user rule 2026-10-23): when
    two levels are not separated by this rule, the SIMPLER one is chosen, simpler = (in order) fewer tokens per frame
    (coarser grid), smaller encoder, the encoder's last layer (no truncation), no mask dilation, global
    centring + scale (no per-dimension statistics), smaller dictionary, fewer training steps, fewer frames to encode.
    theta* stays (principled mean L0 = k, user 2026-10-14) unless EMA wins by the rule. The recipe must stay the
    one-line pipeline: encoder -> animal mask -> Matryoshka SAE -> readout -> NES/NESS (nothing added).
CONFIRMATION: fresh seeds 400-419 (mice) / 400-432 (ants) of the combined recipe vs v1 (deployed codes, release_ab
    refs): release criteria (A) distinct best nose-nose vs nose-tail neurons at full and prefix 256 (deployment seed =
    highest held-out FVE among 400-402, and >= 80% of seeds), (B) every decision cell's 95% t-CI lower bound of
    (recipe - v1) >= -0.01 at full and 256; background-neuron share <= 0.01 (scripts/eci/v2_mask2 definition);
    tail / odour-box discovery scores reported (scripts/eci/v2_mask2/discovery.py definitions).
DEVIATION FROM THE DEV / TEST POLICY (stated before scoring): factors are attributed on the lab labels, not MABe22 /
    CalMS21. Reasons: the question is why v2 trails v1 on these labels; the user allowed all mice v1 labels for this goal
    (project_nnnt_first_goal 2026-10-16); held-out evidence for the encoder factor already exists (CalMS21, 2026-10-15:
    size-matched DINOv3-L = DINOv2-L at a 64 grid, +0.017 primary AUROC; DINOv3-L wins at 32, +0.052). The confirmation
    uses fresh seeds but the same frames, so it is NOT an independent test of the recipe choice; this is said in the
    report. Rotation is identical in v1 and v2 (mice odour corner top right, ants none) and was measured neutral on
    2026-10-20, so it is not an arm here.
AMENDMENT 2026-10-23 ~04:30 (after the stage A/B scores of A_ref, A_w1024, A_w4096, A_s90k, A_ema (mice), B_d3l20,
    B_d3l24, B_d3l16, B_d2b12, B_d3b12, B_d2l24, B_v1like (mice) were seen; BEFORE any score of the arms below):
    (1) DROPPED for the GPU budget (25 GPU-h; ~15 used by the encode and the first arms): A_k8, A_k32 (k is not a
        v1 / v2 difference), B_d2b10. Not run.
    (2) BUG, no rule change: the first stage A/B jobs passed the arm list with commas, which sbatch --export splits,
        so only the first arm of each list ran; the rest were resubmitted with ':' separators.
    (3) ADDED, because B_v1like (v1's recipe on the v2 pool) does NOT reach v1 on mice nose_tail (0.047 vs v1 0.103)
        while width, steps and threshold do not explain it, leaving the training data / training mask:
        A_x2     the v2 recipe trained on the FULL reference pools (both burst halves: mice nnnt/r2/pool 62,208 frames,
                 ants v2/freeze pool_v2 + pool_v3 99,180 frames; tokens identical to M0, max |diff| 0 in the step-2
                 identity checks) = 2x the frames of the step-2 pools; same 30k steps, same roles. vs M0.
        B_v1sae  the DEPLOYED v1 SAE (mice fg448al_s0, ants antsfg_s0, its own stored threshold) read out on this
                 study's d2b12 'dil' tokens: a scoring identity check (does this token set reproduce v1's numbers?),
                 one SAE, reported next to v1 and B_v1like (B_v1like - B_v1sae = v1's training data / training mask /
                 steps, at identical scoring tokens).
COMBINED RECIPE, PRE-REGISTERED 2026-10-23 ~05:00 (after all stage A/B scores, BEFORE any confirmation SAE). The
    rule above applied factor by factor (results/vision/eci_v2_diag/ablation.json; 5 seeds per arm):
      encoder   DINOv3-L/16 block 20 KEPT: every alternative is worse on >= 1 decision cell with the CI excluding 0
                (d2b12 mice nn -0.036, lid -0.167; d3b12 nn -0.076, lid -0.076; d2l24 nn -0.047, v3 groom -0.059,
                lid -0.435; d3l24 v2 / v3 groom -0.107 / -0.081, lid -0.027; d3l16 nn -0.051, v3 groom -0.092).
      grid      32 x 32 (512 px input) BY THE TIE-BREAK: d3l20@32 vs native not separated on any of the 10 cells
                (largest: nn -0.019 [-0.044, +0.007], v2 groom -0.018 [-0.051, +0.014]); fewer tokens = simpler.
                NOTE: this departs from the 'native resolution' prior; the token ceiling IS higher at native (T-max
                nn 0.349 vs 0.291, nt 0.326 vs 0.209), the dictionaries just do not use it.
      mask      tight (>= 25% animal pixels per cell) KEPT: one-cell dilation is not better (nt @256 -0.059 sig.).
      norm      global centring + scale KEPT (z-score not separated; simpler).
      threshold theta* KEPT (EMA identical on every cell to 3 decimals).
      width     Matryoshka 1024, prefixes 128/256/512/1024, BY THE TIE-BREAK (1024 vs 2048 not separated; 4096 loses
                nn @256 -0.038 sig.).
      steps     30,000 KEPT (90k not separated).
      data      HALF of the pool's training frames (frac 0.5, seeded per SAE) BY THE TIE-BREAK (half vs pool and half
                vs 2x pool not separated on any cell) - read literally from 'fewer frames to encode'.
    => R_comb: DINOv3-L/16 block 20 at 512 px (32 x 32), tight mask, Matryoshka BatchTopK 1024 (128/256/512/1024) k 16,
       30k steps on half the pool frames, theta*, frame max. Interactions between the tie-break choices were NOT
       measured; the confirmation is the test of the combination.
    CONFIRMATION: seeds 400-419 (mice) / 400-432 (ants), criteria (A) and (B) exactly as stated above, vs v1 (release
    refs). Background-neuron share measured on seeds 400-404 per domain only (budget). Tail / odour-box discovery
    scores NOT measured (budget; needs re-encodes of MABe22 at the new grid).
P1 = v1-STYLE TRAINING POOL, PRE-REGISTERED 2026-10-23 ~06:30 (user approval: "yes, estimate cost then run"; written
    BEFORE any pool1 token was encoded and before any P1 score).
    WHY: v1's recipe re-trained on the v2 burst pool lost mice nose_tail (0.047 vs 0.115 for the deployed v1 SAE on the
    same tokens), while 2x more burst-pool frames did not help v2. The v2 pools (since rollout r4, 2026-10-15) are
    bursts of consecutive frames at random starts (r4: z_t / z_t-10 pairs for the since-dropped change channel and a
    12k-frame prepass budget; r2 / freeze: 10 consecutive frames, 16 bursts per mice video, 124 / 52 per ants v2 / v3
    video); v1 trained on 1 fps of every video.
    POOL (step 'pool1'): every video of the series the current v2 SAE trains on (mice v1: 432 videos, full length;
    ants v2 44 + v3 107 videos, full length; v4 not used, as M0), 1 fps = every 5th 5 fps frame with
    (k - off) % 5 == 2 (mice off = the video's fps1 offset, ants off = 0), so NO pool frame is a scored frame (mice
    diagnosis frames sit on (k - off) % 10 == 0, ants scored frames on k % 10 == 0). Mice 518,400 frames, ants 219,000.
    Tokens = the frozen v2 encoder (src/eci/v2/encode.py) and tight mask (fd >= 0.25 = M0), unchanged. Per-token keep
    probability p (seeded per frame, crc32(obs), k): mice 0.2, ants 1.0, from the cap of <= 16M training tokens (GPU
    memory; the step-2 rates give ~78M tight mice tokens and ~7.9M ants tokens); a uniform token subset keeps every
    frame (the diversity under test), as v1's ff448al pf25 store did. Roles = the plan's video roles (as M0).
    ARMS (everything else = the current v2 / M0 recipe: Matryoshka BatchTopK 2048, 128/512/2048, k 16, global norm,
    theta*, frame max; scored on the M0 tokens of the release-check frames):
        P1_30k   30,000 steps (matched to M0)          P1_90k   90,000 steps (the pool is 4-6x larger than M0's)
    SEEDS 500-519 (mice) / 500-532 (ants), fresh; deployment seed = highest held-out FVE among 500-502.
    CRITERIA per arm vs v1 (release refs): (A) and (B) exactly as above (full and @256, 5 decision cells);
    background-neuron share <= 0.01 (seeds 500-504); each arm - M0 (Welch 95% CI, M0 = step-2 seeds 70-89 / 70-102)
    reported. Tail / odour-box discovery scores (scripts/eci/v2_mask2/discovery.py definitions) for the deployment seed
    of the adopted arm if the budget allows (60 GPU-h for this step). DECISION: an arm is a v2 release candidate iff
    (A) and (B) and the background criterion pass. If both pass, the SIMPLER one (30k steps) is chosen unless P1_90k
    is better than P1_30k on >= 1 decision cell with the Welch CI excluding 0 and worse on none. No DINO-feature mask
    cue is added (the ants grooming gap is attributed to v1's readout mask, not tested here).
V1M = v1's ANIMAL MASK ON v2, PRE-REGISTERED 2026-10-23 ~15:30 (user decision; written BEFORE any token of this test was
    encoded and before any V1M score; mask definitions and the DINOv3 adaptation: scripts/eci/v2_diag/v1mask.py docstring).
    ARMS (the current v2 recipe: DINOv3-L/16 block 20 at native resolution, Matryoshka BatchTopK 2048 128/512/2048, k 16,
    30k steps, global norm, theta*, frame max; ONLY the animal mask changes, for training tokens AND readout):
        V1M_d3 (primary, the simplest: no second encoder) v1's rule with the feature cue on the DINOv3 tokens.
        V1M_d2 (attribution only; a release candidate only if it alone passes) the exact v1 mask (DINOv2-B feature cue,
               v1 backgrounds, 32 grid mapped to the native grid by the cell holding each token centre).
    TRAINING POOL = the step-2 burst pools (mice 'pool', ants 'pool_v2' + 'pool_v3'), NOT the 1 fps pool. Reason: P1 was
        not better overall for the current mask (vs M0: mice nose_nose +0.012 / +0.021 but ants v2 groom -0.023 / -0.011,
        other cells within noise, and (A) @256 weaker), so by the simplicity tie-break (fewer frames to encode) the bursts
        stay; re-encoding the 1 fps pool with the new masks (~17 GPU-h) would also exceed the 25 GPU-h budget.
    RE-ENCODE: needed, because the stored tokens cover only M0 / N2 patches, not v1's dilation ring / feature patches. One
        pass per frame set (pool + scored frames) stores M0 | d2 | d3 tokens with per-token flags and the mask2 fractions.
    SEEDS 600-619 (mice) / 600-632 (ants), fresh; deployment seed = highest held-out FVE among 600-602.
    CRITERIA per arm vs v1: (A) and (B) as above (full and @256); background-neuron share <= 0.01 in BOTH domains (seeds
        600-604; the dilated mask makes background neurons a real risk); arm - M0 (Welch) reported; lab tail-token and
        odour-box discovery scores (discovery.py 'lab', seeds 600-604; MABe22 tail NOT measured: neither v1 mask has
        backgrounds for MABe22 clips). Release candidate iff (A), (B) and background pass. Tie-break: if V1M_d3 and
        V1M_d2 both pass, V1M_d3 (no second encoder) unless V1M_d2 is better on >= 1 decision cell with the Welch CI
        excluding 0 and worse on none.
    AMENDMENT 2026-10-23 ~16:30 (user decision after the cost gate; before any V1M token store or score): V1M_d3 is NOT
        run (its DINOv3 backgrounds measured ~205 s per mice video, full plan ~47 GPU-h > budget). V1M_d2 (the exact v1
        mask) becomes the TESTED arm and a release candidate on its own, with the criteria above unchanged (A, B at full
        and @256 vs v1, vs M0 reported, background share in both domains on seeds 600-604, lab discovery). The tie-break
        between d3 and d2 no longer applies. Budget 20 GPU-h.
    COST ESTIMATE (written before running; from this study's measured rates): DINOv3 backgrounds (mice 432 x 200 frames,
        ants 365 windows x 200, seek decode) ~3 GPU-h; re-encode of pool + scored frames (as the step-2 prep, + DINOv2-B
        at 448) ~9 GPU-h; 106 SAEs + scoring ~5 GPU-h; neurons / discovery ~1.5 GPU-h; selftests ~0.5 -> ~19 GPU-h (<= 25).
MATCHED = TRAIN/TEST-MATCHED v1 vs v2, PRE-REGISTERED 2026-10-23 ~19:00 (coordinator's fairness check; written after the
    overlap measurement and BEFORE any matched SAE / score; code scripts/eci/v2_diag/matched.py, overlap.py).
    OVERLAP (results/vision/eci_v2_diag/overlap.json): the deployed v1 SAEs were TRAINED ON the scored frames:
        mice: 87.5% of the 32,400 scored atlas-window frames are exact v1 training frames (91.1% of the nose_nose
              positives, 87.0% of the nose_tail positives); the other 12.5% are in v1's 8 held-out val pools.
        ants: v2 scored frames 18.2% exact (grooming positives 25.6%), v3 0.9% exact (grooming positives 19.2%); 88-91%
              of the scored frames are from v1's training videos.
        v2 pools (M0 bursts, P1 1 fps): 0% exact frames, 100% same videos.
    ARMS (>= 10 seeds each; same scorer = release_ab R~ atlas split, same frames):
        V1R_full  v1's exact recipe (scripts/eci/train_sae_fg.py with the deployed args: DINOv2-B 448 fg448al / antsfg
                  stores, Matryoshka 1024, k 16, 5 epochs, its val split) retrained, seeds 1-10 (deployed = seed 0).
        V1R_ex    the same with every scored frame removed from training (--exclude-rows; mice: the 43,200 diagnosis
                  rows; ants: the scored frames matched to v1's row space by frame_path).
        M0        the current v2 (step-2 seeds 70-89 / 70-102): never saw a scored frame.
        M0_plus   the current v2 recipe with the scored frames of TRAIN-role videos added to its training tokens,
                  seeds 700-709 (calib / eval unchanged).
        v1-recipe SAEs are scored on v1's own tokens of the scored frames (matched.py v1tok: dataset JPG -> DINOv2-B 448
        -> v1 mask), identity-checked by re-reading the DEPLOYED v1 SAE against its stored codes_max.
    MATCHED PAIRS: neither_seen = M0 vs V1R_ex; both_seen = M0_plus vs V1R_full. Per pair, (B) on the 5 decision cells x
        {full, 256}: Welch 95% CI lower bound of (v2 arm - v1 arm) >= -0.01. (A) reported for M0 (known) and M0_plus.
        Seen-frame effects reported: V1R_full - V1R_ex, M0_plus - M0; deployed v1 vs the V1R_full seed distribution.
    READING: if v2 passes (B) in the neither_seen pair but not against the deployed v1, the earlier gaps were (partly)
        a train/test artefact of v1.
V2-FULL (TRANSDUCTIVE v2), PRE-REGISTERED 2026-10-23 ~19:30 (user direction: "train v2 the best way possible"; labels
    never enter training, as for v1; written before any V2-full encode or grid SAE).
    OVERLAP (context only): see MATCHED above (v1 trained on 87.5% of the mice scored frames). The MATCHED arms are NOT run
    (superseded by this direction).
    DATA: all frames we analyse, label-free: 1 fps of EVERY video ((k - off) % 5 == 2) of mice v1 (432 videos) and ants
        v2 + v3 + v4 (232 videos, all windows) + every scored frame (step-2 stores), frozen v2 encoder; token keep p:
        mice 0.2, ants 1.0 (16M-token GPU cap; scored-frame tokens thinned with the same p). Store 'full1' holds M0 | N2
        tokens with the mask2 fractions (v4 needs new mask2 backgrounds: step 'bgsall', CPU).
    GRID (label-free choice; on the already-encoded P1 pool + scored frames, M0 mask, v4 absent; 2 seeds 800-801):
        data {half of the training frames, full} x steps {30k, 90k} x width {2048 (128/256/512/2048),
        4096 (128/256/512/2048/4096)} - prefix 256 is a Matryoshka level in every cell (the discovery set; CHANGE vs
        M0's 128/512/2048, stated).
        RULE: on FULL data, the (steps, width) cell with the highest held-out FVE at prefix 256 (eval videos, theta*
        codes), averaged over seeds and over the two domains, among cells with never-firing latents on eval <= 1% in
        both domains; cells within 0.002 of the best are tied -> the SIMPLER (fewer steps, then smaller width).
        Loss / FVE histories are checked for convergence (reported). MORE DATA: if FVE@256(full) - FVE@256(half) >
        0.003 at the chosen cell in either domain, a denser token set (keep p = 1 at 1 fps) is judged plausibly
        useful and proposed (not run without a cost estimate). Lab labels are NOT used for any grid choice.
    CANDIDATES (the chosen cell, trained on 'full1'): V2F_M0 (tight mask) and V2F_N2 (thin-part mask); seeds 900-919
        (mice) / 900-932 (ants), fresh; deployment seed = highest held-out FVE among 900-902.
    CONFIRMATION per candidate vs v1 (deployed): (A) and (B) as above (full and @256, 5 decision cells), - M0 reported;
        background-neuron share <= 0.01 (both domains, seeds 900-904); discovery: lab tail-token share and odour-box R~
        (seeds 900-904) and MABe22 tail share. Release candidate iff (A), (B), background pass. If both pass: V2F_M0
        (simpler cue) unless V2F_N2 is better on >= 1 decision cell (Welch CI excluding 0) and worse on none, or the
        tail criterion (user release blocker 2026-10-22) is decisive - reported, decided with the user.
    An interim report follows the grid pick, before the confirmation.
    GRID PICK 2026-10-24 (by the rule above; results/vision/eci_v2_diag/grid_pick.json; no lab label read): full data,
        held-out FVE@256 averaged over domains and 2 seeds: 30k/2048 0.8354, 90k/2048 0.8354, 30k/4096 0.8318,
        90k/4096 0.8321; the 2048 cells tie (within 0.002) -> the simpler 30k steps. Dead latents <= 0.34% everywhere.
        Converged: the training FVE is flat over the last 10k of 30k steps (mice 0.8700 / 0.8700). More data: FVE@256
        full - half at 30k/2048 = mice +0.0001, ants -0.0003 (< 0.003) -> no denser token set. => V2F arms: Matryoshka
        2048 (128/256/512/2048), k 16, 30k steps on 'full1' + scored frames. The 'full1' mice store holds 24.9M M0 | N2
        tokens (> the 16M figure used to set p; the N2 arm trains on an 80 GB GPU instead of lowering p).
==================================================================================================================

Steps (scripts/eci/v2_diag/job.sh; staging in $LOCAL_DIR):
    prep  --dom D --kind K --shard i --nshard n [--selftest]  (GPU) 32-grid multi-encoder store -> BEE/<D>/<K>/s<i>/
    train --dom D --arm A [--seeds a-b]                       (GPU) SAEs -> BEE/<D>/sae/<A>_s<seed>
    arm   --dom D --arms A,B [--seeds a-b]                    (GPU) train + score each arm
    score --dom D --arm A [--seeds]                           (GPU) -> OUT/<D>/ab/<A>_s<seed>.json
    probe --dom D --set S                                     (GPU) -> OUT/<D>/probe/<S>.json
    neurons --dom D --arm A [--seeds]                         (GPU) background / appendage / odour neuron counts
    decide                                                    (CPU) -> OUT/ablation.json, OUT/ablation_table.md
    pool1 --dom D --shard i --nshard n [--selftest]          (GPU) 1 fps v1-style pool -> BEE/<D>/pool1/s<i>/
    confirm / p1decide                                        (CPU) -> OUT/confirm*.json, OUT/p1_<arm>*.json
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
for p in ('scripts/eci', 'scripts/eci/nnnt', 'scripts/eci/v2', 'scripts/eci/v2_mask', 'scripts/eci/v2_mask2'):
    sys.path.insert(0, str(REPO / p))
import freeze as FZ  # noqa: E402
import mask_test as MT  # noqa: E402
import mask2_test as T2  # noqa: E402
import release_ab as RA  # noqa: E402

N, R2, R3, R4 = FZ.N, FZ.R2, FZ.R3, FZ.R4
log = N.log
OUT = Path(os.environ.get('V2D_OUT') or REPO / 'results/vision/eci_v2_diag')
BEE = Path(os.environ.get('V2D_BEE') or '/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_diag')
G32 = 32
SEEDS = tuple(range(300, 305))
CONF_SEEDS = {'mice': tuple(range(400, 420)), 'ants': tuple(range(400, 433))}
P1_SEEDS = {'mice': tuple(range(500, 520)), 'ants': tuple(range(500, 533))}
V1M_SEEDS = {'mice': tuple(range(600, 620)), 'ants': tuple(range(600, 633))}
KINDS = {'mice': ('pool', 'diag'), 'ants': ('pool_v2', 'pool_v3', 'scored_v2', 'scored_v3')}

# ---------------------------------------------------------------------------------------------- encoders (32 grid)
MODELS = {  # name -> (HF id, revision or None, input side, patch, dtype, layers kept)
    'd3l': ('facebook/dinov3-vitl16-pretrain-lvd1689m', 'ea8dc2863c51be0a264bab82070e3e8836b02d51', 512, 16, 'bf16',
            (16, 20, 24)),
    'd3b': ('facebook/dinov3-vitb16-pretrain-lvd1689m', None, 512, 16, 'bf16', (12,)),
    'd2b': ('facebook/dinov2-base', None, 448, 14, 'fp32', (10, 12)),
    'd2l': ('facebook/dinov2-large', None, 448, 14, 'bf16', (24,)),
}
SETS = {f'{m}{L}': (m, L) for m, c in MODELS.items() for L in c[5]}  # d3l16 d3l20 d3l24 d3b12 d2b10 d2b12 d2l24
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

BASE = dict(W=2048, prefixes=(128, 512, 2048), k=16, steps=30000, norm='global', frac=1.0, readout='theta')
ARMS = {
    'A_ref': dict(src='E0'), 'A_w1024': dict(src='E0', W=1024, prefixes=(128, 256, 512, 1024)),
    'A_w4096': dict(src='E0', W=4096, prefixes=(128, 512, 2048, 4096)), 'A_k8': dict(src='E0', k=8),
    'A_k32': dict(src='E0', k=32), 'A_s90k': dict(src='E0', steps=90000), 'A_half': dict(src='E0', frac=0.5),
    'A_z': dict(src='E0', norm='z'), 'A_ema': dict(src='E0', reuse='A_ref', readout='ema'),
    **{f'B_{s}': dict(src=s, mask='tight') for s in SETS},
    'B_d3l20dil': dict(src='d3l20', mask='dil'),
    'B_v1like': dict(src='d2b12', mask='dil', W=1024, prefixes=(128, 256, 512, 1024), readout='ema'),
    'A_x2': dict(src='E0full'),
    'R_comb': dict(src='d3l20', mask='tight', W=1024, prefixes=(128, 256, 512, 1024), frac=0.5),
    'P1_30k': dict(src='P1'), 'P1_90k': dict(src='P1', steps=90000),
    'V1M_d3': dict(src='V1M_d3'), 'V1M_d2': dict(src='V1M_d2'),
    'M0_plus': dict(src='E0', plus=True),
    'V2F_M0': dict(src='F1', mask='M0', steps=30000, W=2048, prefixes=(128, 256, 512, 2048)),  # grid pick 2026-10-24
    'V2F_N2': dict(src='F1', mask='N2', steps=30000, W=2048, prefixes=(128, 256, 512, 2048)),
    'G_half_30k_w2048': dict(src='G', half=True, steps=30000, W=2048, prefixes=(128, 256, 512, 2048)),
    'G_half_30k_w4096': dict(src='G', half=True, steps=30000, W=4096, prefixes=(128, 256, 512, 2048, 4096)),
    'G_half_90k_w2048': dict(src='G', half=True, steps=90000, W=2048, prefixes=(128, 256, 512, 2048)),
    'G_half_90k_w4096': dict(src='G', half=True, steps=90000, W=4096, prefixes=(128, 256, 512, 2048, 4096)),
    'G_full_30k_w2048': dict(src='G', half=False, steps=30000, W=2048, prefixes=(128, 256, 512, 2048)),
    'G_full_30k_w4096': dict(src='G', half=False, steps=30000, W=4096, prefixes=(128, 256, 512, 2048, 4096)),
    'G_full_90k_w2048': dict(src='G', half=False, steps=90000, W=2048, prefixes=(128, 256, 512, 2048)),
    'G_full_90k_w4096': dict(src='G', half=False, steps=90000, W=4096, prefixes=(128, 256, 512, 2048, 4096)),
    'B_v1sae': dict(src='d2b12', mask='dil', readout='own',
                    sae_path={'mice': 'dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448al_s0/sae.pt',
                              'ants': 'dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt'}),
}
REFS = {  # arm -> reference arm for the one-factor effect ('M0' = existing step-2 M0 seeds)
    **{a: 'M0' for a in ARMS if a.startswith('A_') and a != 'A_ema'}, 'A_ema': 'A_ref',
    'B_d3l20': 'M0', 'B_d3l24': 'B_d3l20', 'B_d3l16': 'B_d3l20', 'B_d2b12': 'B_d3l20', 'B_d2b10': 'B_d2b12',
    'B_d3b12': 'B_d2b12', 'B_d2l24': 'B_d3l24', 'B_d3l20dil': 'B_d3l20', 'B_v1like': 'v1', 'A_x2': 'M0',
    'B_v1sae': 'v1',
}


def cfg_of(arm):
    return {**BASE, **ARMS[arm]}


def gpu_line():
    return MT.gpu_line()


# ---------------------------------------------------------------------------------------------- prep
class Multi:
    """The four encoders; __call__(x at side S uint8) -> {set: (B, 1024, d) fp16} at the 32 grid."""

    def __init__(self, dev):
        import torch
        from transformers import AutoModel
        self.dev, self.m, self.gpu_s, self.snap = dev, {}, 0.0, {}
        for nm, (mid, rev, side, patch, dt, layers) in MODELS.items():
            kw = {'revision': rev} if rev else {}
            m = AutoModel.from_pretrained(mid, attn_implementation='sdpa', **kw).to(dev).eval().requires_grad_(False)
            blocks = m.layer if hasattr(m, 'layer') else m.encoder.layer
            normm = m.norm if hasattr(m, 'norm') else m.layernorm
            npre = 1 + int(getattr(m.config, 'num_register_tokens', 0) or 0)
            cap = {}
            for L in layers:
                blocks[L - 1].register_forward_hook(
                    lambda mod, i, o, L=L, cap=cap: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
            self.m[nm] = dict(m=m, side=side, patch=patch, dt=dt, layers=layers, norm=normm, npre=npre, cap=cap,
                              nblocks=len(blocks))
            self.snap[nm] = getattr(m.config, '_commit_hash', None)
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.std = torch.tensor(STD, device=dev)[None, :, None, None]

    def __call__(self, x):
        import torch
        import torch.nn.functional as F
        t0 = time.time()
        out = {}
        xf = x.float()
        with torch.no_grad():
            for nm, M in self.m.items():
                xi = F.interpolate(xf, size=(M['side'], M['side']), mode='bicubic', antialias=True,
                                   align_corners=False).clamp(0, 255)
                xin = (xi / 255 - self.mean) / self.std
                M['cap'].clear()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=M['dt'] == 'bf16'):
                    last = M['m'](pixel_values=xin).last_hidden_state
                for L in M['layers']:
                    h = last if L == M['nblocks'] else M['norm'](M['cap'][L])
                    z = h[:, M['npre']:].float()
                    assert z.shape[1] == G32 * G32 and torch.isfinite(z).all(), (nm, L, z.shape)
                    out[f'{nm}{L}'] = z.half()
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0
        return out

    def last_check(self, x):
        """norm(hook of the last block) == last_hidden_state (the hook / LN convention used for the inner layers)."""
        import torch
        import torch.nn.functional as F
        res = {}
        with torch.no_grad():
            for nm, M in self.m.items():
                L = M['nblocks']
                blocks = M['m'].layer if hasattr(M['m'], 'layer') else M['m'].encoder.layer
                cap = {}
                hk = blocks[L - 1].register_forward_hook(
                    lambda mod, i, o, cap=cap: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
                xi = F.interpolate(x[:2].float(), size=(M['side'], M['side']), mode='bicubic', antialias=True,
                                   align_corners=False).clamp(0, 255)
                xin = (xi / 255 - self.mean) / self.std
                last = M['m'](pixel_values=xin).last_hidden_state.float()
                hk.remove()
                res[nm] = float((M['norm'](cap['h']).float() - last).abs().max())
        return res


def cmd_prep(args):
    import torch
    import torch.nn.functional as F
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    from src.eci.v2 import mask2 as M2
    dev = torch.device('cuda')
    dom, kind = args.dom, args.kind
    K = MT.KINDS[dom][kind]
    h = T2.h_of(dom, K['series'])
    enc = Multi(dev)
    se = Series(T2.series_cfg(dom, K['series']), dev, load_encoder=False)
    S = se.S
    cell = S // G32
    assert cell * G32 == S, (S, G32)
    Fr = MT.frame_list(dom, kind)
    obs_all = list(dict.fromkeys(Fr.obs.values))
    mine = set(obs_all[args.shard::args.nshard])
    Fr = Fr[Fr.obs.isin(mine)].reset_index(drop=True)
    items = []
    for o, g in Fr.groupby('obs', sort=False):
        idx = dict(zip(g.k.values, g.index.values))
        for w in sorted({T2.window_of(dom, k) for k in g.k.values}):
            kw = np.array([k for k in g.k.values if T2.window_of(dom, k) == w])
            for k0, n, every, ks in MT.runs_of(kw):
                items.append((o, w, k0, n, every, ks, [idx[k] for k in ks]))
    if args.selftest:
        items = items[:3]
    jobs = [((i,), se.job(se.V[o], k0, n, every)) for i, (o, w, k0, n, every, ks, _) in enumerate(items)]
    loc = N.local_dir() / f'v2d_{dom}_{kind}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = {s: open(loc / f'tok_{s}.f16', 'wb') for s in SETS}
    meta = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_new', 'fr_app', 'fr_any')}
    recs, t0, bgc = [], time.time(), {}
    bs = 16 if S >= 1000 else 32
    checks = {}
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=se.run_job)
    for (i,), fr in pf:
        o, w, k0, n, every, ks, fidx = items[i]
        v = se.V[o]
        if (o, w) not in bgc:
            bgc.clear()
            bgc[(o, w)] = T2.load_pack(dom, K['series'], o, w, dev)
        pack = bgc[(o, w)]
        grid = k0 + every * np.arange(n)
        fr = fr[np.isin(grid, ks)]
        assert len(fr) == len(ks)
        for b in range(0, len(ks), bs):
            kb = ks[b:b + bs]
            with torch.no_grad():
                x, _ = se.frames(v, fr[b:b + bs], kb)
                core0 = se.mask.pixels(x, v['src_lo'] + kb)
                r = M2.cue(x, pack, core0, h)
                _, app = M2.appendage(r['pix'], h)
                anyp = core0 | (r['d'] > M2.T_LO)
                pool = lambda m: F.avg_pool2d(m.float()[:, None], cell).flatten(1)  # noqa: E731
                raw, frn, fapp, fany = pool(core0), pool(r['pix']), pool(app), pool(anyp)
                tight = raw >= 0.25
                dil = F.max_pool2d(tight.float().view(-1, 1, G32, G32), 3, 1, 1).flatten(1) > 0.5
                Z = enc(x)
                if not checks and args.shard == 0:
                    checks['last_layer_hook_vs_last_hidden_state_maxabs'] = enc.last_check(x)
                    if dom == 'mice':  # 32-grid tight fraction == 2 x 2 mean of the 64-grid M0 fractions
                        r64 = F.avg_pool2d(core0.float()[:, None], 16)
                        checks['pool_consistency_maxabs'] = float((F.avg_pool2d(r64, 2).flatten(1) - raw).abs().max())
            for ii in range(len(kb)):
                pos = torch.nonzero(dil[ii]).flatten()
                fid = len(recs)
                for s_ in SETS:
                    ftok[s_].write(Z[s_][ii, pos].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.cpu().numpy().astype(np.int16))
                for nm, t_ in (('fr_raw', raw), ('fr_new', frn), ('fr_app', fapp), ('fr_any', fany)):
                    meta[nm].append(t_[ii, pos].cpu().numpy().astype(np.float16))
                r0 = Fr.iloc[fidx[b + ii]]
                rec = {'obs': o, 'k': int(kb[ii]), 'ref_f': int(r0.ref_f), 'n_tight': int(tight[ii].sum()),
                       'n_dil': int(len(pos))}
                for c_ in ('row', 'role', 'series'):
                    if c_ in Fr:
                        rec[c_] = r0[c_]
                recs.append(rec)
        if i % 200 == 0:
            el = time.time() - t0
            log(f'prep {dom} {kind} s{args.shard}: run {i}/{len(items)}, {len(recs):,} frames, {el:.0f}s '
                f'({len(recs) / max(el, 1):.1f} fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); '
                f'{gpu_line()}')
    for f_ in ftok.values():
        f_.close()
    for k_, x_ in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x_))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    # same frames, same order as the step-2 store shard (identity of the frame sets)
    ref = T2.BEE / dom / kind / f's{args.shard}' / 'frames.parquet'
    if ref.exists() and not args.selftest and args.nshard == len(list((T2.BEE / dom / kind).glob('s*'))):
        Fr2 = pd.read_parquet(ref)
        same = len(Fr2) == len(Fm) and (Fr2.obs.values == Fm.obs.values).all() and (Fr2.k.values == Fm.k.values).all()
        checks['same_frames_as_step2_shard'] = bool(same)
        assert same, f'frame order differs from {ref}'
    el = time.time() - t0
    info = {'dom': dom, 'kind': kind, 'shard': args.shard, 'nshard': args.nshard, 'S': S, 'n_frames': len(Fm),
            'n_stored_tokens': int(Fm.n_dil.sum()), 'tight_per_frame': float(Fm.n_tight.mean()),
            'dil_per_frame': float(Fm.n_dil.mean()), 'wall_s': el, 'gpu_s': enc.gpu_s, 'fps_wall': len(Fm) / el,
            'decode_thread_s': pf.dec_s, 'gpu': torch.cuda.get_device_name(0), 'snapshots': enc.snap,
            'checks': checks, 'dims': {s: int(Z[s].shape[-1]) for s in SETS}}
    log(json.dumps(info, default=float))
    od = OUT / dom / 'prep'
    od.mkdir(parents=True, exist_ok=True)
    tag = '_selftest' if args.selftest else ''
    (od / f'{kind}_s{args.shard}{tag}.json').write_text(json.dumps(info, indent=1, default=float))
    (loc / 'info.json').write_text(json.dumps(info, default=float))
    if not args.selftest:
        R4.atomic_copy(loc, BEE / dom / kind / f's{args.shard}')


# ---------------------------------------------------------------------------------------------- stores
class Store:
    """A 32-grid token set on frames of one or more kinds: the FZ.TokStore interface (F, tf, pos, T, starts, nf, G,
    dev, chunks) plus fd / app / any / raw per token for neuron counting."""

    def __init__(self, dom, kinds, tset, mask, dev, on_gpu=True):
        Fs, parts, o = [], [], 0
        d = None
        for kind in kinds:
            root = BEE / dom / kind
            for sp in sorted([p for p in root.glob('s*') if not p.name.endswith('.tmp')], key=lambda p: int(p.name[1:])):
                info = json.loads((sp / 'info.json').read_text())
                d = info['dims'][tset]
                F_ = pd.read_parquet(sp / 'frames.parquet')
                tf = np.load(sp / 'tok_frame.npy')
                raw = np.load(sp / 'fr_raw.npy').astype(np.float32)
                keep = raw >= 0.25 if mask == 'tight' else np.ones(len(tf), bool)
                mm = np.memmap(sp / f'tok_{tset}.f16', np.float16, 'r', shape=(len(tf), d))
                parts.append(dict(tf=tf[keep] + o, pos=np.load(sp / 'tok_pos.npy').astype(np.int64)[keep],
                                  raw=raw[keep], **{k: np.load(sp / f'fr_{k}.npy').astype(np.float32)[keep]
                                                    for k in ('new', 'app', 'any')},
                                  tok=np.asarray(mm[keep])))
                Fs.append(F_)
                o += len(F_)
        assert parts, (dom, kinds)
        self.F = pd.concat(Fs, ignore_index=True)
        cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
        self.tf, self.pos, self.raw, self.fd, self.app, self.any = (cat(k) for k in ('tf', 'pos', 'raw', 'new', 'app',
                                                                                      'any'))
        self.tok = cat('tok')
        self.d, self.nf, self.G, self.dev = d, len(self.F), G32, dev
        self.starts = np.r_[0, np.cumsum(np.bincount(self.tf, minlength=self.nf))]
        self.dist = np.zeros(len(self.tf), np.int8)
        self.T = R2.tokens_to(__import__('torch'), self.tok, dev) if on_gpu else None

    chunks = FZ.TokStore.chunks


def _armstore(dom, kind, mask, dev, on_gpu):
    St = T2.arm_store(T2.Union(dom, kind), mask, dev, on_gpu=on_gpu)
    St.d = 1024
    return St


def e0_store(dom, kind, dev, on_gpu=True):
    St = T2.arm_store(T2.Union(dom, kind), 'M0', dev, on_gpu=on_gpu)
    St.d = 1024
    return St


def store_of(dom, kind, cfg, dev, on_gpu=True):
    if cfg['src'] == 'E0full' and kind.startswith('pool'):  # the full reference pool (both burst halves)
        St = FZ.TokStore(MT.KINDS[dom][kind]['ref'], 0, dev, on_gpu=on_gpu)
        St.d = 1024
        return St
    if cfg['src'] in ('G', 'F1'):  # V2-full arms are scored on the step-2 stores of their mask
        return _armstore(dom, kind, cfg.get('mask', 'M0'), dev, on_gpu)
    if cfg['src'].startswith('V1M'):  # v1's animal mask on the v2 tokens (scripts/eci/v2_diag/v1mask.py)
        import v1mask
        return v1mask.store(dom, kind, cfg['src'], dev, on_gpu)
    if cfg['src'] == 'P1' and kind.startswith('pool'):  # the v1-style 1 fps pool (one store per domain)
        return P1Store(dom, dev)
    if cfg['src'] in ('E0', 'E0full', 'P1'):
        return e0_store(dom, kind, dev, on_gpu)
    return Store(dom, (kind,), cfg['src'], cfg.get('mask', 'tight'), dev, on_gpu)


def inputs(St, sel, device=None):
    """fp16 rows (on St.dev, or `device`) of the tokens where sel (numpy bool per token) holds."""
    import torch
    ix = np.flatnonzero(sel)
    out = torch.empty((len(ix), St.d), dtype=torch.float16, device=device or St.dev)
    for a in range(0, len(ix), 1_000_000):
        out[a:a + 1_000_000] = torch.from_numpy(np.ascontiguousarray(St.tok[ix[a:a + 1_000_000]])).to(St.dev)
    return out


def codes(St, sae, norm):
    """frame codes (nf, W) fp16 cuda = max over the store's kept tokens (threshold mode)."""
    import torch
    W = sae.n_latents
    X = torch.zeros((St.nf, W), dtype=torch.float16, device=St.dev)
    fp = max(4, min(48, int(2**27 // W // 160)))
    with torch.no_grad():
        for f0, f1, a, b, z in St.chunks(fp):
            fl = torch.from_numpy(St.tf[a:b] - f0).to(St.dev)
            c = sae.encode(norm(z.float()), mode='threshold')
            X[f0:f1] = torch.zeros((f1 - f0, W), device=St.dev).index_reduce_(0, fl, c.float(), 'amax',
                                                                               include_self=True).half()
    return X


# ---------------------------------------------------------------------------------------------- train
def sae_dir(dom, arm, s):
    return BEE / dom / 'sae' / f'{arm}_s{s}'


def train_one(torch, cfg, seed, Xtr, Xca, Xev, od, dev):
    """= scripts/eci/nnnt/r3.py train_arm (uniform sampler, 'id' prep) with: width / prefixes / k / steps from cfg,
    norm 'global' (TokenNorm, as r3) or 'z' (per-dimension mean / sd), and the training EMA threshold recorded."""
    from src.eci.sae import MatryoshkaBatchTopKSAE, TokenNorm, _train_step, geometric_median, save_checkpoint
    from src.eci.sae_calibrate import calibrate_threshold, set_threshold, sparsity_stats
    od.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    din = Xtr.shape[1]
    g = np.random.default_rng(seed)
    ii = torch.from_numpy(np.sort(g.choice(len(Xtr), min(250_000, len(Xtr)), replace=False))).to(dev)
    init = Xtr[ii].float()
    norm = TokenNorm(din).to(dev).fit_blocks(init, [din], seed=seed)
    if cfg['norm'] == 'z':
        with torch.no_grad():
            norm.mean.copy_(init.mean(0))
            norm.scale = (1.0 / (init.std(0) + 1e-6)).to(dev)
    sae = MatryoshkaBatchTopKSAE(d_in=din, n_latents=cfg['W'], prefixes=tuple(cfg['prefixes']), k=cfg['k'],
                                 seed=seed).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(geometric_median(norm(init[:100_000])))
    del init
    opt = torch.optim.Adam(sae.parameters(), lr=5e-4, betas=(0.9, 0.999))
    gen = torch.Generator(device=dev).manual_seed(seed)
    Nn, bs, steps = len(Xtr), 4096, cfg['steps']
    perm, p, hist, t0 = torch.randperm(Nn, generator=gen, device=dev), 0, [], time.time()
    for step in range(steps):
        if p + bs > Nn:
            perm, p = torch.randperm(Nn, generator=gen, device=dev), 0
        b = perm[p:p + bs]
        p += bs
        logs, _ = _train_step(sae, opt, norm(Xtr[b].float()), step, steps, 5e-4, 500, 1.0)
        if step % 5000 == 0 or step == steps - 1:
            logs.update(step=step, elapsed_s=round(time.time() - t0, 1))
            hist.append(logs)
    sae.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    tt = time.time() - t0
    th_ema = float(sae.threshold)
    full = lambda x: norm(x.float())  # noqa: E731
    th, _ = calibrate_threshold(sae, Xca, norm=full)
    set_threshold(sae, th)
    stt = sparsity_stats(sae, Xev, th, norm=full)
    pre_fve, fired = {}, torch.zeros(cfg['W'], dtype=torch.bool, device=dev)
    with torch.no_grad():  # held-out FVE of each Matryoshka prefix (theta* codes) and never-firing latents on eval
        s1 = s2 = 0
        sse = {q: 0 for q in cfg['prefixes']}
        n_ev = 0
        for a in range(0, len(Xev), 32768):
            x = full(Xev[a:a + 32768].to(dev))
            zc = sae.encode(x, mode='threshold')
            fired |= (zc > 0).any(0)
            s1 = s1 + x.double().sum(0)
            s2 = s2 + x.double().pow(2).sum(0)
            n_ev += len(x)
            for q in cfg['prefixes']:
                sse[q] = sse[q] + (x - sae.decode(zc, q)).double().pow(2).sum()
        tss = float((s2 - s1.pow(2) / n_ev).sum())
        pre_fve = {str(q): float(1 - float(sse[q]) / tss) for q in cfg['prefixes']}
    save_checkpoint(od / 'sae_theta.pt', sae, norm, extra={'config': 'v2_diag', 'cfg': cfg, 'seed': seed,
                                                            'theta_ema': th_ema})
    return {'train_time_s': tt, 'steps_per_s': steps / tt, 'n_train_tokens': Nn, 'theta_star': th, 'theta_ema': th_ema,
            'eval_fve': stt['fve'], 'eval_l0': stt['l0'], 'eval_fve_prefix': pre_fve,
            'eval_dead_frac': float(1 - fired.float().mean()), 'dead_at_end_train': int((sae.tokens_since_fired >= sae.dead_tokens).sum()), 'history': hist}


def pool_kinds(dom, cfg=None):
    if cfg is not None and cfg['src'] == 'P1':
        return ('pool',)
    return ('pool',) if dom == 'mice' else ('pool_v2', 'pool_v3')


def cmd_train(args):
    import torch
    dev = torch.device('cuda')
    dom, arm = args.dom, args.arm
    cfg = cfg_of(arm)
    if 'reuse' in cfg or 'sae_path' in cfg:
        return
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    todo = [s for s in seeds if not (sae_dir(dom, arm, s) / 'sae_theta.pt').exists()]
    if not todo:
        return
    sts = f1_train_stores(dom, cfg, dev) if cfg['src'] in ('G', 'F1') else \
        [store_of(dom, k, cfg, dev, on_gpu=False) for k in pool_kinds(dom, cfg)]
    if cfg.get('plus'):  # MATCHED: + the scored frames of TRAIN-role videos (training side only; calib / eval unchanged)
        for kind in ({'mice': ('diag',), 'ants': ('scored_v2', 'scored_v3')}[dom]):
            St = e0_store(dom, kind, dev, on_gpu=False)
            plan = json.loads((REPO / T2.series_cfg(dom, MT.KINDS[dom][kind]['series'])['plan']).read_text())
            role = {v['obs']: v['role'] for v in plan['videos']}
            St.F = St.F.copy()
            St.F['role'] = [role[o] if role[o] == 'train' else 'none' for o in St.F.obs.values]
            log(f'{dom} plus {kind}: {int((St.F.role == "train").sum()):,} scored frames of train videos added')
            sts.append(St)
    X = {}
    for r in ('calib', 'eval'):
        X[r] = torch.cat([inputs(St, St.F.role.values[St.tf] == r, 'cpu') for St in sts])
    tr_sel = []
    for St in sts:
        m = St.F.role.values[St.tf] == 'train'
        tr_sel.append(m)
    log(f'{dom} {arm}: calib {len(X["calib"]):,}, eval {len(X["eval"]):,}; {gpu_line()}')
    od = OUT / dom / 'train'
    od.mkdir(parents=True, exist_ok=True)
    for s in todo:
        sel = []
        for St, m in zip(sts, tr_sel):
            if cfg['frac'] < 1:  # seeded half of the TRAINING FRAMES (all tokens of a kept frame)
                fk = np.random.default_rng(s).random(St.nf) < cfg['frac']
                m = m & fk[St.tf]
            sel.append(m)
        if len(sts) == 1:
            Xtr = inputs(sts[0], sel[0])  # no 2x copy for a single store (P1 mice: 24 GiB)
        else:  # several stores: concatenate on the host, one copy to the GPU
            Xtr = torch.cat([inputs(St, m, 'cpu') for St, m in zip(sts, sel)]).to(dev)
        ti = train_one(torch, cfg, s, Xtr, X['calib'], X['eval'], sae_dir(dom, arm, s), dev)
        rec = {k: ti[k] for k in ('train_time_s', 'n_train_tokens', 'theta_star', 'theta_ema', 'eval_fve', 'eval_l0',
                                  'dead_at_end_train', 'eval_fve_prefix', 'eval_dead_frac', 'history')}
        log(f'{dom} {arm}_s{s}: {ti["train_time_s"]:.0f}s, {ti["n_train_tokens"]:,} train tokens, eval FVE '
            f'{ti["eval_fve"]:.4f}, L0 {ti["eval_l0"]:.2f}, dead {ti["dead_at_end_train"]}, theta* '
            f'{ti["theta_star"]:.3f} / ema {ti["theta_ema"]:.3f}; {gpu_line()}')
        (od / f'{arm}_s{s}.json').write_text(json.dumps(rec, indent=1, default=float))
        del Xtr
        torch.cuda.empty_cache()


def load_arm_sae(dom, arm, s, dev):
    from src.eci.sae import load_sae
    from src.eci.sae_calibrate import set_threshold
    cfg = cfg_of(arm)
    src_arm = cfg.get('reuse', arm)
    if 'sae_path' in cfg:  # a fixed deployed SAE (its own stored threshold)
        sae, norm, _ = load_sae(REPO / cfg['sae_path'][dom], dev)
        return sae, norm
    sae, norm, _ = load_sae(sae_dir(dom, src_arm, s) / 'sae_theta.pt', dev)
    if cfg['readout'] == 'ema':
        set_threshold(sae, float(json.loads((OUT / dom / 'train' / f'{src_arm}_s{s}.json').read_text())['theta_ema']))
    return sae, norm


def fve_of(dom, arm, s):
    f = OUT / dom / 'train' / f'{cfg_of(arm).get("reuse", arm)}_s{s}.json'
    return json.loads(f.read_text()).get('eval_fve', np.nan) if f.exists() else np.nan


# ---------------------------------------------------------------------------------------------- score
def scored_stores(dom, cfg, dev):
    if dom == 'mice':
        return {'v1': store_of('mice', 'diag', cfg, dev)}
    return {ser: store_of('ants', f'scored_{ser}', cfg, dev) for ser in ('v2', 'v3')}


def cmd_score(args):
    import torch
    dev = torch.device('cuda')
    dom, arm = args.dom, args.arm
    cfg = cfg_of(arm)
    seeds = FZ.parse_seeds(args.seeds, SEEDS)
    od = OUT / dom / 'ab'
    od.mkdir(parents=True, exist_ok=True)
    src_arm = cfg.get('reuse', arm)
    if 'sae_path' in cfg:
        seeds = [] if (od / f'{arm}_s{SEEDS[0]}.json').exists() else [SEEDS[0]]
    else:
        seeds = [s for s in seeds if not (od / f'{arm}_s{s}.json').exists()
                 and (sae_dir(dom, src_arm, s) / 'sae_theta.pt').exists()]
    if not seeds:
        return
    sts = scored_stores(dom, cfg, dev)
    if dom == 'mice':
        St = sts['v1']
        Y, sets, half, _, _ = MT.mice_sets(St)
        Yg = torch.from_numpy(Y).to(dev)
    else:
        lab = {}
        for ser, St_ in sts.items():
            Y, hh, labs = MT.ants_set(St_, ser)
            lab[ser] = (torch.from_numpy(Y).to(dev), hh, labs)
    for s in seeds:
        sae, norm = load_arm_sae(dom, arm, s, dev)
        if dom == 'mice':
            ab = RA.ab_mice(codes(St, sae, norm), Yg, half['all'], St, FZ.MLAB)
            log(f'mice {arm}_s{s}: R_atlas full nn {ab["full"]["nose_nose"]["R_atlas"]:.4f} nt '
                f'{ab["full"]["nose_tail"]["R_atlas"]:.4f} | 256 nn {ab["256"]["nose_nose"]["R_atlas"]:.4f} nt '
                f'{ab["256"]["nose_tail"]["R_atlas"]:.4f}; {gpu_line()}')
        else:
            ab = {}
            for ser, St_ in sts.items():
                Yg_, hh, labs = lab[ser]
                ab[ser] = RA.ab_ants(codes(St_, sae, norm), Yg_, hh, labs)
            log(f'ants {arm}_s{s}: R_atlas full v2 groom {ab["v2"]["full"]["groom_any"]["R_atlas"]:.4f} v3 groom '
                f'{ab["v3"]["full"]["groom_any"]["R_atlas"]:.4f} lid {ab["v3"]["full"]["on_lid_any"]["R_atlas"]:.4f} '
                f'| 256 lid {ab["v3"]["256"]["on_lid_any"]["R_atlas"]:.4f}; {gpu_line()}')
        ab['eval_fve'] = fve_of(dom, arm, s)
        (od / f'{arm}_s{s}.json').write_text(json.dumps(ab, indent=1, default=float))
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- probe (ceilings)
def cmd_probe(args):
    import torch
    from src.eci.v2.score import crossfit
    dev = torch.device('cuda')
    dom, tset = args.dom, args.set
    cfg = {'src': tset, 'mask': 'tight'}
    sts = scored_stores(dom, cfg, dev)
    res = {'set': tset, 'note': 'T-max: one linear direction on standardised tokens, frame max (r2 fit_probe max); '
                               'chan: best single standardised dimension (either sign), frame max; atlas split'}
    jobs = []
    if dom == 'mice':
        St = sts['v1']
        Y, sets, half, _, _ = MT.mice_sets(St)
        keep, hh = RA.mice_window(St, half['all'])
        jobs.append(('mice', St, keep, Y[keep][:, :2], hh['atlas'], ['nose_nose', 'nose_tail'], hh))
    else:
        for ser, St_ in sts.items():
            Y, hh, labs = MT.ants_set(St_, ser)
            want = ['groom_any'] + (['on_lid_any'] if ser == 'v3' else [])
            jobs.append((ser, St_, np.arange(St_.nf), Y[:, [labs.index(w) for w in want]], hh['r4'], want, hh))
    for ser, St, keep, Yk, h_at, labs, hall in jobs:
        # frame subset (atlas window for mice) -> local token bags
        fsel = np.zeros(St.nf, bool)
        fsel[keep] = True
        newid = -np.ones(St.nf, np.int64)
        newid[keep] = np.arange(len(keep))
        tsel = fsel[St.tf]
        T = inputs(St, tsel)
        tf = newid[St.tf[tsel]]
        ii = torch.from_numpy(np.random.default_rng(0).choice(len(T), min(1_000_000, len(T)), replace=False)).to(dev)
        mu, sd = T[ii].float().mean(0), T[ii].float().std(0) + 1e-3
        for a in range(0, len(T), 500_000):
            T[a:a + 500_000] = ((T[a:a + 500_000].float() - mu) / sd).half()
        nfk = len(keep)
        bags = R2.Bags(torch, T, tf, nfk, dev)
        # best single channel: frame max of +z and of -z (shifted >= 0 for the fp16 scorer)
        tfg = torch.from_numpy(tf).to(dev)
        mx = torch.full((nfk, St.d), -1e4, device=dev)
        mn = torch.full((nfk, St.d), -1e4, device=dev)
        for a in range(0, len(T), 500_000):
            z = T[a:a + 500_000].float()
            mx.index_reduce_(0, tfg[a:a + 500_000], z, 'amax', include_self=True)
            mn.index_reduce_(0, tfg[a:a + 500_000], -z, 'amax', include_self=True)
        empty = bags.cnt == 0
        Xc = torch.cat([mx, mn], 1)
        Xc[empty] = float('nan')
        Xc = Xc - torch.nan_to_num(Xc, nan=1e4).min(0).values
        Xc = torch.nan_to_num(Xc, nan=0.0).clamp(0, 6e4).half()
        Ykg = torch.from_numpy(Yk).to(dev)
        cf = crossfit(Xc, Ykg, {'atlas': h_at, 'rnd_atlas': h_at}, labs)  # crossfit's mean needs an rnd* key
        out = {}
        for li, lb in enumerate(labs):
            y = Yk[:, li].astype(bool)
            recs = []
            for a_, b_ in ((0, 1), (1, 0)):
                tr = torch.from_numpy(np.flatnonzero(h_at == a_)).to(dev)
                te = np.flatnonzero(h_at == b_)
                f = R2.fit_probe(torch, 'max', bags, None, y, tr, dev, steps=1500)
                recs.append(R2.metrics(f(torch.from_numpy(te).to(dev)), y[te]))
            out[lb] = {'tmax_R': float(np.mean([r['R'] for r in recs])), 'tmax_A': float(np.mean([r['A'] for r in recs])),
                       'chan_R': cf['splits']['atlas'][lb]['R'], 'chan_A': cf['splits']['atlas'][lb]['A'],
                       'base': float(y.mean()), 'n': int(len(y))}
            # robustness (added after the first probe run, reported only): T-max on the random halves rnd0-4 too
            rr = []
            for sn in [f'rnd{i}' for i in range(5)]:
                hs = hall[sn]
                for a_, b_ in ((0, 1), (1, 0)):
                    tr = torch.from_numpy(np.flatnonzero(hs == a_)).to(dev)
                    te = np.flatnonzero(hs == b_)
                    f = R2.fit_probe(torch, 'max', bags, None, y, tr, dev, steps=1500)
                    rr.append(R2.metrics(f(torch.from_numpy(te).to(dev)), y[te])['R'])
            out[lb]['tmax_R_rnd5'] = float(np.mean(rr))
            out[lb]['tmax_R_rnd5_sd'] = float(np.std(rr))
            log(f'probe {dom} {ser} {tset} {lb}: T-max R~ {out[lb]["tmax_R"]:.4f} AUROC {out[lb]["tmax_A"]:.4f} | '
                f'best channel R~ {out[lb]["chan_R"]:.4f}; {gpu_line()}')
        res[ser] = out
        del T, bags, Xc, mx, mn
        torch.cuda.empty_cache()
    od = OUT / dom / 'probe5'
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{tset}.json').write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------------------------------------- neurons
def cmd_neurons(args):
    """Background / appendage / odour-box neuron counts (scripts/eci/v2_mask2/mask2_test.py definitions, selection half
    rnd0 == 0); no sheets."""
    import torch
    dev = torch.device('cuda')
    dom, arm = args.dom, args.arm
    cfg = cfg_of(arm)
    sts = scored_stores(dom, cfg, dev)
    if dom == 'mice':
        Y, sets, half, uidx, SPL = MT.mice_sets(sts['v1'])
        hA = {'v1': SPL['rnd0'][uidx] == 0}
        odor = T2.odour_corner(sts['v1'].G * 16)
    else:
        hA = {ser: MT.ants_set(St, ser)[1]['rnd0'] == 0 for ser, St in sts.items()}
        odor = None
    od = OUT / dom / 'neurons'
    od.mkdir(parents=True, exist_ok=True)
    for s in FZ.parse_seeds(args.seeds, SEEDS):
        f = od / f'{arm}_s{s}.json'
        if f.exists() or not (sae_dir(dom, cfg.get('reuse', arm), s) / 'sae_theta.pt').exists():
            continue
        sae, norm = load_arm_sae(dom, arm, s, dev)
        accs = {ser: T2.token_stats(St, sae, norm, hA[ser], odor)[0] for ser, St in sts.items()}
        acc = {k: sum(accs[ser][k] for ser in sts) for k in accs[next(iter(sts))]}
        ok = acc['fire'] >= T2.MIN_FIRE
        off = np.where(ok, acc['off'] / np.maximum(acc['hit'], 1), np.nan)
        odr = np.where(ok, acc['odor'] / np.maximum(acc['hit'], 1), np.nan)
        app = np.where(ok & (acc['den'] > 0), acc['num'] / np.maximum(acc['den'], 1e-12), np.nan)
        r = {'n_scored': int(ok.sum()), 'n_background': int(np.nansum(off > 0.5)),
             'share_background': float(np.nansum(off > 0.5) / max(ok.sum(), 1)),
             'n_app': int(np.nansum(app > T2.APP_NEURON)), 'n_odor': int(np.nansum(odr > T2.ODOR_FRAC))}
        log(f'{dom} {arm}_s{s}: background {r["n_background"]}/{r["n_scored"]}, appendage {r["n_app"]}, odour '
            f'{r["n_odor"]}; {gpu_line()}')
        f.write_text(json.dumps(r, indent=1, default=float))
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- decide
DCELLS = [('mice', None, 'nose_nose'), ('mice', None, 'nose_tail'), ('ants', 'v2', 'groom_any'),
          ('ants', 'v3', 'groom_any'), ('ants', 'v3', 'on_lid_any')]
RCELLS = [('mice', None, 'nn_mutual'), ('mice', None, 'np_directional')]


def cell_vals(dom, arm, ser, lb, p, key='R_atlas', seeds=None):
    """per-seed values of one cell -> {seed: value}."""
    if arm == 'M0':
        d, ss = T2.OUT / dom / 'ab', T2.SEEDS[dom]
    else:
        d, ss = OUT / dom / 'ab', seeds or SEEDS
    out = {}
    for s in ss:
        f = d / f'{arm}_s{s}.json'
        if f.exists():
            r = json.loads(f.read_text())
            out[s] = (r[p] if ser is None else r[ser][p])[lb][key]
    return out


def v1_val(dom, ser, lb, p, key='R_atlas'):
    refs = json.loads((T2.OUT / 'ab_refs.json').read_text())
    r = refs[dom] if dom == 'mice' else refs[dom][ser]
    return r['v1'][p][lb][key]


def welch_ci(a, b):
    from src.eci.v2.score import welch
    return welch(a, b)


def cmd_decide(args):
    res, L = {}, []
    L.append('| arm | ref | cell | prefix | arm mean ± sd (n) | arm - ref (Welch 95% CI) | arm - v1 |')
    L.append('|---|---|---|---|---|---|---|')
    for arm in ARMS:
        ref = REFS[arm]
        for dom, ser, lb in DCELLS + RCELLS:
            for p in ('full', '256'):
                a = cell_vals(dom, arm, ser, lb, p)
                if len(a) < 2:
                    continue
                v1 = v1_val(dom, ser, lb, p)
                av = np.array(list(a.values()))
                rec = {'mean': float(av.mean()), 'sd': float(av.std(ddof=1)), 'n': len(av), 'v1': v1,
                       'minus_v1': RA.tci(av - v1)}
                if ref == 'v1':
                    rec['minus_ref'] = rec['minus_v1']
                elif ref == 'A_ref':  # paired by seed
                    b = cell_vals(dom, ref, ser, lb, p)
                    common = sorted(set(a) & set(b))
                    if len(common) >= 2:
                        rec['minus_ref'] = RA.tci(np.array([a[s] - b[s] for s in common]))
                else:
                    b = cell_vals(dom, ref, ser, lb, p)
                    if len(b) >= 2:
                        rec['minus_ref'] = list(welch_ci(av, np.array(list(b.values()))))
                nm = f'{dom}{" " + ser if ser else ""} {lb}'
                res.setdefault(arm, {})[f'{nm} | {p}'] = rec
                mr = rec.get('minus_ref')
                L.append(f'| {arm} | {ref} | {nm} | {p} | {rec["mean"]:.4f} ± {rec["sd"]:.4f} ({rec["n"]}) | '
                         + (f'{mr[0]:+.4f} [{mr[1]:+.4f}, {mr[2]:+.4f}]' if mr else '-') + ' | '
                         f'{rec["minus_v1"][0]:+.4f} [{rec["minus_v1"][1]:+.4f}, {rec["minus_v1"][2]:+.4f}] |')
    # M0 reference rows
    for dom, ser, lb in DCELLS:
        for p in ('full', '256'):
            a = np.array(list(cell_vals(dom, 'M0', ser, lb, p).values()))
            res.setdefault('M0', {})[f'{dom}{" " + ser if ser else ""} {lb} | {p}'] = {
                'mean': float(a.mean()), 'sd': float(a.std(ddof=1)), 'n': len(a), 'v1': v1_val(dom, ser, lb, p)}
    probes = {}
    for dom in ('mice', 'ants'):
        for f in sorted((OUT / dom / 'probe5').glob('*.json')) if (OUT / dom / 'probe5').exists() else []:
            probes.setdefault(dom, {})[f.stem] = json.loads(f.read_text())
    res['_probes'] = probes
    (OUT / 'ablation.json').write_text(json.dumps(res, indent=1, default=float))
    P = ['', '| token set | cell | T-max R~ (AUROC) | best channel R~ |', '|---|---|---|---|']
    for dom, d in probes.items():
        for tset, r in d.items():
            for ser, cells in r.items():
                if not isinstance(cells, dict):
                    continue
                for lb, c in cells.items():
                    P.append(f'| {tset} | {dom} {ser} {lb} | {c["tmax_R"]:.4f} ({c["tmax_A"]:.3f}) | {c["chan_R"]:.4f} |')
    (OUT / 'ablation_table.md').write_text('\n'.join(L + P))
    print('\n'.join(L + P))


def cmd_probes(args):
    """--set a,b,c: one probe per token set in sequence."""
    import torch
    for tset in args.set.split(','):
        a = argparse.Namespace(**{**vars(args), 'set': tset})
        if (OUT / args.dom / 'probe5' / f'{tset}.json').exists():
            continue
        cmd_probe(a)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- 1 fps pool (v1-style)
P1_CAP = 16_000_000  # training-token cap: per-token keep probability p = min(1, P1_CAP / N_tight) (GPU memory)
P1_P = {'mice': 0.2, 'ants': 1.0}  # fixed before encoding from the step-2 token rates (see the P1 block above)
P1_SERIES = {'mice': ('v1',), 'ants': ('v2', 'v3')}
P1_NSHARD = {'mice': 24, 'ants': 8}


def p1_frames(dom, ser, v, fps1_off):
    """1 fps from the whole video, disjoint from every scored frame: 5 fps index k with (k - off) % 5 == 2
    (mice off = the video's fps1 offset, whose grid holds the diagnosis frames; ants off = 0, scored k = 0 mod 10)."""
    off = fps1_off if dom == 'mice' else 0
    return np.arange((off + 2) % 5, v['n'], 5)


def cmd_pool1(args):
    """v1-style training pool for the frozen v2 token stream: 1 fps of every video (all phases, full length), the frozen
    v2 encoder (src/eci/v2/encode.py) and tight mask (fd >= 0.25 = the M0 rule), a seeded per-token keep p (P1_P).
    -> BEE/<dom>/pool1/s<i>/ (FZ.TokStore layout: frames.parquet obs / k / row / role / series, tok_frame, tok_pos,
    tok.f16, info.json)."""
    import torch
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = args.dom
    off = {v['obs']: v['fps1_off'] for v in N.load_plan()['videos']} if dom == 'mice' else {}
    items, SE = [], {}
    for ser in P1_SERIES[dom]:
        se = SE.setdefault(ser, Series(T2.series_cfg(dom, ser), dev, load_encoder=False))
        for v in se.videos:
            items.append((ser, v))
    items = items[args.shard::args.nshard]
    if args.selftest:
        items = items[:2]
    from src.eci.v2.encode import Encoder
    enc = Encoder(dev)
    for se in SE.values():
        se.enc = enc
    jobs, meta_j = [], []
    for ser, v in items:
        ks = p1_frames(dom, ser, v, off.get(v['obs'], 0))
        if args.selftest:
            ks = ks[:120]
        for c0 in range(0, len(ks), 240):  # decode runs of <= 240 kept frames (every 5th 5 fps frame)
            kk = ks[c0:c0 + 240]
            jobs.append(((len(meta_j),), SE[ser].job(v, int(kk[0]), len(kk), 5)))
            meta_j.append((ser, v, kk))
    loc = N.local_dir() / f'v2d_p1_{dom}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    tfs, poss, recs = [], [], []
    p = P1_P[dom]
    n_tight, t0 = 0, time.time()
    bs = 16 if max(se.S for se in SE.values()) >= 1000 else 32
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=Series.run_job)
    for (j,), fr in pf:
        ser, v, kk = meta_j[j]
        se = SE[ser]
        assert len(fr) == len(kk), (v['obs'], len(fr), len(kk))
        for b in range(0, len(kk), bs):
            kb = kk[b:b + bs]
            with torch.no_grad():
                x, fd = se.frames(v, fr[b:b + bs], kb)
                z, keep = se.tokens(x, fd)
            for ii in range(len(kb)):
                pos = torch.nonzero(keep[ii]).flatten().cpu().numpy()
                n_tight += len(pos)
                if p < 1:
                    g = np.random.default_rng([zlib.crc32(v['obs'].encode()), int(kb[ii])])
                    pos = pos[g.random(len(pos)) < p]
                fid = len(recs)
                ftok.write(z[ii, torch.from_numpy(pos).to(dev)].cpu().numpy().tobytes())
                tfs.append(np.full(len(pos), fid, np.int32))
                poss.append(pos.astype(np.int16))
                recs.append({'obs': v['obs'], 'k': int(kb[ii]), 'row': int(v['src_lo'] + kb[ii]), 'role': v['role'],
                             'series': ser, 'n_tight': int(keep[ii].sum()), 'n_kept': int(len(pos))})
        if j % 50 == 0:
            el = time.time() - t0
            log(f'pool1 {dom} s{args.shard}: run {j}/{len(jobs)}, {len(recs):,} frames, {el:.0f}s ({len(recs) / max(el, 1):.1f}'
                f' fps; GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); {gpu_line()}')
    ftok.close()
    np.save(loc / 'tok_frame.npy', np.concatenate(tfs) if tfs else np.zeros(0, np.int32))
    np.save(loc / 'tok_pos.npy', np.concatenate(poss) if poss else np.zeros(0, np.int16))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'dom': dom, 'shard': args.shard, 'nshard': args.nshard, 'n_videos': len(items), 'n_frames': len(Fm),
            'n_tight_tokens': int(n_tight), 'n_stored_tokens': int(Fm.n_kept.sum()), 'p': p, 'wall_s': el,
            'gpu_s': enc.gpu_s, 'fps_wall': len(Fm) / max(el, 1), 'decode_thread_s': pf.dec_s,
            'gpu': torch.cuda.get_device_name(0)}
    if args.selftest:  # identity: overlapping frames of the step-2 / reference pools -> same tokens at the same positions
        ref = {'mice': [MT.KINDS['mice']['pool']['ref']], 'ants': [MT.KINDS['ants'][k]['ref'] for k in ('pool_v2', 'pool_v3')]}[dom]
        chk = []
        tf_all, pos_all = np.concatenate(tfs), np.concatenate(poss)
        mine = np.memmap(loc / 'tok.f16', np.float16, 'r', shape=(len(tf_all), FZ.D))
        key = {(o, k): i for i, (o, k) in enumerate(zip(Fm.obs.values, Fm.k.values))}
        for r_ in ref:
            St = FZ.TokStore(r_, 0, dev, on_gpu=False)
            for f_, (o, k) in enumerate(zip(St.F.obs.values, St.F.k.values)):
                if (o, int(k)) in key:
                    i = key[(o, int(k))]
                    a_ = np.flatnonzero(tf_all == i)
                    b_ = np.arange(St.starts[f_], St.starts[f_ + 1])
                    pa, pb = pos_all[a_], St.pos[b_]
                    com, ia, ib = np.intersect1d(pa, pb, return_indices=True)
                    d_ = float(np.abs(mine[a_[ia]].astype(np.float32) - St.tok[b_[ib]].astype(np.float32)).max()) if len(com) else None
                    chk.append({'obs': o, 'k': int(k), 'n_mine': len(pa), 'n_ref': len(pb), 'n_common': len(com),
                                'subset_of_ref': bool(np.isin(pa, pb).all()), 'max_abs_diff': d_})
        info['identity_vs_ref_pool'] = chk[:20]
        info['identity_n_overlap'] = len(chk)
    log(json.dumps(info, default=float)[:3000])
    od = OUT / dom / 'pool1'
    od.mkdir(parents=True, exist_ok=True)
    (od / f's{args.shard}{"_selftest" if args.selftest else ""}.json').write_text(json.dumps(info, indent=1, default=float))
    if not args.selftest:
        R4.atomic_copy(loc, BEE / dom / 'pool1' / f's{args.shard}')


class P1Store:
    """All pool1 shards of a domain as one FZ.TokStore-like object (training only)."""

    def __init__(self, dom, dev):
        Fs, tfs, poss, toks, o = [], [], [], [], 0
        root = BEE / dom / 'pool1'
        for sp in sorted([q for q in root.glob('s*') if not q.name.endswith('.tmp')], key=lambda q: int(q.name[1:])):
            F_ = pd.read_parquet(sp / 'frames.parquet')
            tf = np.load(sp / 'tok_frame.npy')
            toks.append(np.asarray(np.memmap(sp / 'tok.f16', np.float16, 'r', shape=(len(tf), FZ.D))))
            tfs.append(tf + o)
            poss.append(np.load(sp / 'tok_pos.npy'))
            Fs.append(F_)
            o += len(F_)
        assert len(Fs) == P1_NSHARD[dom], (dom, len(Fs))
        self.F = pd.concat(Fs, ignore_index=True)
        self.tf, self.pos, self.tok = np.concatenate(tfs), np.concatenate(poss).astype(np.int64), np.concatenate(toks)
        self.d, self.nf, self.dev, self.T = FZ.D, len(self.F), dev, None


def confirm_arm(arm, seeds, deploy_from, name):
    """arm on fresh seeds vs v1: criteria (A) and (B) (release_ab definitions); + vs M0 (Welch)."""
    res, L = {'arm': arm, 'rule': 'diag.py docstring'}, []
    R = {d: {s: json.loads((OUT / d / 'ab' / f'{arm}_s{s}.json').read_text()) for s in seeds[d]
             if (OUT / d / 'ab' / f'{arm}_s{s}.json').exists()} for d in seeds}
    fv = {s: fve_of('mice', arm, s) for s in deploy_from}
    dep = max(fv, key=lambda s: fv[s] if np.isfinite(fv[s]) else -1)
    A = {}
    for p in ('full', '256', '128'):
        ds = {s: RA.distinct(r[p]) for s, r in R['mice'].items()}
        A[p] = {'share_seeds_distinct': float(np.mean(list(ds.values()))), 'deploy_seed': dep,
                'deploy_distinct': ds.get(dep), 'deploy_best': {lb: R['mice'][dep][p][lb]['modal_neuron']
                                                                for lb in ('nose_nose', 'nose_tail')}}
    A['PASS'] = bool(all(A[p]['deploy_distinct'] and A[p]['share_seeds_distinct'] >= 0.8 for p in ('full', '256')))
    res['A'] = A
    L += [f'| criterion | cell | prefix | v1 | {arm} mean ± sd (n) | {arm} - v1 (95% t-CI) | pass | {arm} - M0 (Welch) |', '|---|---|---|---|---|---|---|---|']
    for p in ('full', '256', '128'):
        L.append(f'| A distinct | nn vs nt best neuron | {p} | - | {A[p]["share_seeds_distinct"]:.2f} of seeds | '
                 f'deploy s{dep}: {A[p]["deploy_best"]} distinct={A[p]["deploy_distinct"]} | '
                 f'{A["PASS"] if p != "128" else "(reported)"} | |')
    ok, B = True, {}
    for dom, ser, lb in DCELLS + RCELLS:
        for p in ('full', '256'):
            xs = np.array([(r[p] if ser is None else r[ser][p])[lb]['R_atlas'] for r in R[dom].values()])
            v1 = v1_val(dom, ser, lb, p)
            m, lo, hi = RA.tci(xs - v1)
            decides = (dom, ser, lb) in DCELLS
            ps = bool(lo >= -0.01)
            ok &= ps or not decides
            nm = f'{dom}{" " + ser if ser else ""} {lb}'
            m0 = np.array(list(cell_vals(dom, 'M0', ser, lb, p).values()))
            vm = list(welch_ci(xs, m0)) if len(m0) > 1 else None
            B[f'{nm} | {p}'] = {'v1': v1, 'mean': float(xs.mean()), 'sd': float(xs.std(ddof=1)), 'n': len(xs),
                                'minus_v1': [m, lo, hi], 'pass': ps, 'decides': decides, 'minus_M0': vm}
            L.append(f'| B R_atlas | {nm} | {p} | {v1:.4f} | {xs.mean():.4f} ± {xs.std(ddof=1):.4f} ({len(xs)}) | '
                     f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {ps if decides else "(reported: " + str(ps) + ")"} | '
                     + (f'{vm[0]:+.4f} [{vm[1]:+.4f}, {vm[2]:+.4f}]' if vm else '-') + ' |')
    B['PASS'] = bool(ok)
    res['B'] = B
    bg = {}
    for d in ('mice', 'ants'):
        fs = [json.loads(f.read_text()) for f in sorted((OUT / d / 'neurons').glob(f'{arm}_s*.json'))]
        if fs:
            bg[d] = {'n_seeds': len(fs), 'share_background_mean': float(np.mean([f['share_background'] for f in fs])),
                     'n_background': [f['n_background'] for f in fs], 'n_app_mean': float(np.mean([f['n_app'] for f in fs])),
                     'n_odor_mean': float(np.mean([f['n_odor'] for f in fs]))}
    res['background'] = bg
    res['background_PASS'] = bool(bg and all(v['share_background_mean'] <= 0.01 for v in bg.values()))
    L += ['', f'A PASS: {A["PASS"]}', f'B PASS (5 decision cells x full/256): {B["PASS"]}',
          f'background share <= 0.01: {res["background_PASS"]} {json.dumps(bg)}']
    (OUT / f'{name}.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / f'{name}_table.md').write_text('\n'.join(L))
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------- V2-full (transductive)
F1_SERIES = {'mice': ('v1',), 'ants': ('v2', 'v3', 'v4')}
F1_NSHARD = {'mice': 24, 'ants': 12}
F1_SEEDS = {'mice': tuple(range(900, 920)), 'ants': tuple(range(900, 933))}
GRID_SEEDS = (800, 801)


def f1_frames(dom):
    """{series: DataFrame obs, k} of the 1 fps grid of every video ((k - off) % 5 == 2, as P1)."""
    from src.eci.v2.pipeline import Series
    import torch
    off = {v['obs']: v['fps1_off'] for v in N.load_plan()['videos']} if dom == 'mice' else {}
    out = {}
    for ser in F1_SERIES[dom]:
        se = Series(T2.series_cfg(dom, ser), torch.device('cpu'), load_encoder=False)
        rows = [(v['obs'], int(k)) for v in se.videos for k in p1_frames(dom, ser, v, off.get(v['obs'], 0))]
        out[ser] = pd.DataFrame(rows, columns=['obs', 'k'])
    return out


def h_ser(dom, ser):
    try:
        return T2.h_of(dom, ser)
    except KeyError:  # ants v4: no calibration entry; same species / rig as v3
        return T2.h_of(dom, 'v3')


def cmd_bgsall(args):
    """mask2 per-(video, window) median backgrounds (mask2_test.cmd_bgs, unchanged) for every window of every video of
    the V2-full series (v4 and any window the step-2 frame sets did not cover). CPU."""
    fr = f1_frames(args.dom)
    kinds = {f'f1_{ser}': {'series': ser} for ser in fr}
    old_k, old_f = T2.KINDS, MT.frame_list
    T2.KINDS = {args.dom: kinds}
    MT.frame_list = lambda dom, kind: fr[kinds[kind]['series']]
    try:
        T2.cmd_bgs(args)
    finally:
        T2.KINDS, MT.frame_list = old_k, old_f


def cmd_full1(args):
    """V2-full token store: 1 fps of every video of F1_SERIES, frozen v2 encoder, tokens of M0 | N2 (mask2_test.Union
    layout: fr_raw / fr_new / fr_app / fr_any / n_app), per-token keep p = P1_P[dom] (seeded per frame and position)."""
    import torch
    import torch.nn.functional as F
    from src.eci.v2 import mask2 as M2
    from src.eci.v2.encode import Encoder
    from src.eci.v2.pipeline import Series
    from src.eci.v2.video import Prefetch
    dev = torch.device('cuda')
    dom = args.dom
    fr = f1_frames(dom)
    enc = Encoder(dev)
    SE = {ser: Series(T2.series_cfg(dom, ser), dev, encoder=enc) for ser in fr}
    vids = [(ser, o) for ser in fr for o in dict.fromkeys(fr[ser].obs.values)]
    vids = vids[args.shard::args.nshard]
    if args.selftest:
        vids = vids[:2]
    jobs, meta_j = [], []
    for ser, o in vids:
        ks = np.sort(fr[ser][fr[ser].obs == o].k.values)
        if args.selftest:
            ks = ks[:120]
        for w in sorted({T2.window_of(dom, k) for k in ks}):
            kw = ks[np.array([T2.window_of(dom, k) == w for k in ks])]
            for c0 in range(0, len(kw), 240):
                kk = kw[c0:c0 + 240]
                jobs.append(((len(meta_j),), SE[ser].job(SE[ser].V[o], int(kk[0]), len(kk), 5)))
                meta_j.append((ser, o, w, kk))
    p = P1_P[dom]
    loc = N.local_dir() / f'f1_{dom}_s{args.shard}'
    loc.mkdir(parents=True, exist_ok=True)
    ftok = open(loc / 'tok.f16', 'wb')
    meta = {k: [] for k in ('tok_frame', 'tok_pos', 'fr_raw', 'fr_new', 'n_app', 'fr_app', 'fr_any')}
    recs, t0, bgc = [], time.time(), {}
    pf = Prefetch(jobs, max(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) - 4), fn=Series.run_job)
    for (j,), frm in pf:
        ser, o, w, kk = meta_j[j]
        se = SE[ser]
        v = se.V[o]
        h = h_ser(dom, ser)
        bs = 16 if se.S >= 1000 else 32
        if (ser, o, w) not in bgc:
            bgc.clear()
            bgc[(ser, o, w)] = T2.load_pack(dom, ser, o, w, dev)
        pack = bgc[(ser, o, w)]
        assert len(frm) == len(kk)
        for b in range(0, len(kk), bs):
            kb = kk[b:b + bs]
            with torch.no_grad():
                x, _ = se.frames(v, frm[b:b + bs], kb)
                core0 = se.mask.pixels(x, v['src_lo'] + kb)
                r = M2.cue(x, pack, core0, h)
                _, app = M2.appendage(r['pix'], h)
                anyp = core0 | (r['d'] > M2.T_LO)
                pool = lambda m: F.avg_pool2d(m.float()[:, None], 16).flatten(1)  # noqa: E731
                raw, frn, fapp, fany = pool(core0), pool(r['pix']), pool(app), pool(anyp)
                ng = fapp * 256
                keep = (raw >= 0.25) | M2.keep(frn, ng, 'r25t')
                z = enc(x)
            for ii in range(len(kb)):
                pos = torch.nonzero(keep[ii]).flatten().cpu().numpy()
                if p < 1:
                    g = np.random.default_rng([zlib.crc32(o.encode()), int(kb[ii])])
                    pos = pos[g.random(len(pos)) < p]
                pt = torch.from_numpy(pos).to(dev)
                fid = len(recs)
                ftok.write(z[ii, pt].cpu().numpy().tobytes())
                meta['tok_frame'].append(np.full(len(pos), fid, np.int32))
                meta['tok_pos'].append(pos.astype(np.int16))
                for nm, t_ in (('fr_raw', raw), ('fr_new', frn), ('fr_app', fapp), ('fr_any', fany)):
                    meta[nm].append(t_[ii, pt].cpu().numpy().astype(np.float16))
                meta['n_app'].append(ng[ii, pt].round().cpu().numpy().astype(np.int16))
                recs.append({'obs': o, 'k': int(kb[ii]), 'row': int(v['src_lo'] + kb[ii]), 'role': v['role'],
                             'series': ser, 'n_M0': int((raw[ii] >= 0.25).sum()), 'n_store': int(len(pos))})
        if j % 50 == 0:
            el = time.time() - t0
            log(f'full1 {dom} s{args.shard}: run {j}/{len(jobs)}, {len(recs):,} frames, {el:.0f}s ({len(recs) / max(el, 1):.1f} fps;'
                f' GPU busy {enc.gpu_s:.0f}s; decode thread-s {pf.dec_s:.0f}); {gpu_line()}')
    ftok.close()
    for k_, x_ in meta.items():
        np.save(loc / f'{k_}.npy', np.concatenate(x_) if x_ else np.zeros(0))
    Fm = pd.DataFrame(recs)
    Fm.to_parquet(loc / 'frames.parquet')
    el = time.time() - t0
    info = {'dom': dom, 'shard': args.shard, 'nshard': args.nshard, 'n_videos': len(vids), 'n_frames': len(Fm),
            'n_stored_tokens': int(Fm.n_store.sum()), 'p': p, 'wall_s': el, 'gpu_s': enc.gpu_s,
            'fps_wall': len(Fm) / max(el, 1), 'gpu': torch.cuda.get_device_name(0)}
    log(json.dumps(info, default=float))
    od = OUT / dom / 'full1'
    od.mkdir(parents=True, exist_ok=True)
    (od / f's{args.shard}{"_selftest" if args.selftest else ""}.json').write_text(json.dumps(info, indent=1, default=float))
    if not args.selftest:
        R4.atomic_copy(loc, BEE / dom / 'full1' / f's{args.shard}')


def f1_train_stores(dom, cfg, dev):
    """Training stores of a V2-full arm: the 1 fps pool (P1 for the grid, full1 for the candidates) + the scored frames
    (step-2 stores), mask M0 / N2, scored-frame tokens thinned with the same keep p as the pool."""
    mask = cfg.get('mask', 'M0')
    sts = []
    if cfg['src'] == 'G':  # grid: the P1 pool (M0 only, v4 absent)
        St = P1Store(dom, dev)
        if cfg.get('half'):
            fk = np.random.default_rng(0).random(St.nf) < 0.5
            St.F = St.F.copy()
            St.F.loc[(~fk) & (St.F.role.values == 'train'), 'role'] = 'none'  # half the TRAINING frames; calib / eval kept
        sts.append(St)
    else:  # full1 store (M0 | N2)
        old = T2.BEE
        T2.BEE = BEE
        try:
            U = T2.Union(dom, 'full1')
        finally:
            T2.BEE = old
        St = T2.arm_store(U, mask, dev, on_gpu=False)
        St.d = 1024
        sts.append(St)
    p = P1_P[dom]
    for kind in ({'mice': ('diag',), 'ants': ('scored_v2', 'scored_v3')}[dom]):
        St = T2.arm_store(T2.Union(dom, kind), mask, dev, on_gpu=False)
        St.d = 1024
        plan = json.loads((REPO / T2.series_cfg(dom, MT.KINDS[dom][kind]['series'])['plan']).read_text())
        role = {v['obs']: v['role'] for v in plan['videos']}
        St.F = St.F.copy()
        St.F['role'] = [role[o] for o in St.F.obs.values]
        if p < 1:  # thin the scored frames' tokens with the pool's keep p (seeded)
            keep = np.random.default_rng(1).random(len(St.tf)) < p
            St.tf, St.pos, St.tok = St.tf[keep], St.pos[keep], St.tok[keep]
            St.starts = np.r_[0, np.cumsum(np.bincount(St.tf, minlength=St.nf))]
        sts.append(St)
    return sts


def cmd_confirm(args):
    confirm_arm('R_comb', CONF_SEEDS, (400, 401, 402), 'confirm')


def cmd_v2fdecide(args):
    for arm in ('V2F_M0', 'V2F_N2'):
        confirm_arm(arm, F1_SEEDS, (900, 901, 902), f'v2f_{arm}')


def cmd_v1mdecide(args):
    for arm in ('V1M_d2',):  # amendment 2026-10-23: V1M_d3 not run
        confirm_arm(arm, V1M_SEEDS, (600, 601, 602), f'v1m_{arm}')


def cmd_p1decide(args):
    for arm in ('P1_30k', 'P1_90k'):
        confirm_arm(arm, P1_SEEDS, (500, 501, 502), f'p1_{arm}')


def cmd_arm(args):
    """train then score (one job); --arms a,b,c runs several arms in sequence."""
    import torch
    for arm in (args.arms or args.arm).split(','):
        args.arm = arm
        cmd_train(args)
        torch.cuda.empty_cache()
        if not args.train_only:  # grid arms: no lab score (label-free choice)
            cmd_score(args)
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=('prep', 'train', 'score', 'arm', 'probe', 'neurons', 'decide', 'confirm', 'pool1', 'p1decide', 'v1mdecide', 'bgsall', 'full1', 'v2fdecide'))
    p.add_argument('--dom', choices=('mice', 'ants'))
    p.add_argument('--kind')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--arm')
    p.add_argument('--arms')
    p.add_argument('--set')
    p.add_argument('--seeds')
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--train-only', action='store_true')
    a = p.parse_args()
    {'prep': cmd_prep, 'train': cmd_train, 'score': cmd_score, 'arm': cmd_arm, 'probe': cmd_probes, 'neurons': cmd_neurons, 'decide': cmd_decide, 'confirm': cmd_confirm, 'pool1': cmd_pool1,
     'p1decide': cmd_p1decide, 'v1mdecide': cmd_v1mdecide, 'bgsall': cmd_bgsall,
     'full1': cmd_full1, 'v2fdecide': cmd_v2fdecide}[a.step](a)


if __name__ == '__main__':
    main()
