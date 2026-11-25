# ECI v2.8 alignment cards: does one neuron track one labelled behaviour?

One card per behaviour. Each card compares the **old representation (v1)** with the **new representation (v2)**. Both
models use the same frames, the same videos and the same axes. The neurons were already picked on the selection frames
before this figure; nothing is selected here. This is a POST HOC illustration.

| Card | Old representation (v1) | New representation (v2) |
|---|---|---|
| `mice_nose_nose.png` | v1 seed 0, neuron #611 | PREDk seed 2703, neuron #3593 |
| `mice_nose_tail.png` | v1 seed 0, neuron #414 | PREDk seed 2703, neuron #3009 |
| `ants_grooming.png` | v1 seed 0, neuron #90 | PREDk seed 2703, neuron #33 |

Every card also has a `.pdf` and a `.json`. The `.json` holds the numbers shown on the card and the cross-check against the audit.

**Frames.**
- **Mice:** the held-out set, 12 pools not annotated by annotator SG. That is 72 videos and 431,712 frames at 5 fps (edge-excluded "ext" frames of the v2.7 table). Nose-nose is in 1.57% of frames, nose-tail in 0.85%.
- **Ants:** half B of series v3 + v4. That is 73 videos x the first 10 min at 1 fps = 43,800 frames. Grooming-to-focal (any) is in 38.5% of frames.

**Code.** `scripts/eci/v2/v28_card.py` with the sbatch wrapper `v28_card.sh`. To make a card for another dictionary:

    sbatch -J card scripts/eci/v2/v28_card.sh all --dom mice --beh nose_nose \
        --rows 'v1=v1_s0:611,v2=KEY:NEURON' --out NAME

- `--beh` is `nose_nose`, `nt` or `groom_any`. A row label other than v1 / v2 is printed as given.
- The mice data step takes about 20 min, mostly reading the 13 GB on-probability cache.
- `plot --out NAME` re-draws from `figs/data/NAME.pkl` in under a minute. The data files are not committed.

## What each panel shows (one block per model)

1. **Activity over time**, in 4-5 held-out videos.
   - The filled trace is the neuron's on-probability, P(on). This is a 3-state hidden Markov model on its frame activity, the readout the reports use. For v1 the same model is fitted to its raw code.
   - Mice: the trace is the maximum over 3-s bins, so that a single 0.2-s spike stays visible on a 30-min axis.
   - Red bars are the annotated bouts. The grey line marks the scored frames; gaps are not scored.
   - The videos are chosen by a fixed rule on the labels only, and are the same for every model. The rule takes the 25th / 50th / 75th / 95th percentile of annotated time among videos with the behaviour, plus the alphabetically first held-out video without it.
   - Mice nose-nose has no video without nose-nose, so it shows 4 videos.
2. **Precision-recall over all held-out frames**, with the other model's curve in light grey and chance = prevalence.
   - Frames are ranked as in the reports: v2 by P(on), with raw activity breaking ties; v1 by raw activity.
   - AUROC and average precision are printed above the panel.
3. **Precision of each video's top 3 / 10 / 30 frames.** Frames are ranked by raw activity, earliest frame first, over videos with at least one labelled frame. This is the audit's precision@top-k/video. Chance is the prevalence in those videos.
4. **Predicted vs annotated time in the behaviour per video, in seconds,** with the identity line.
   - Prediction = c x the summed P(on) x the frame duration.
   - c is fitted on the selection frames only: annotator-SG pools for mice, half A for ants.
   - Printed: Pearson r, the mean absolute error, and the error of a constant-rate guess (the selection-set rate x the video length).
5. **The 6 strongest held-out frames, one per video, by raw activity.**
   - The heat map is the neuron's activity per image patch; the cyan box is the strongest patch.
   - A green border means the frame is labelled with the behaviour; a red border means it is not.
   - Each tile's strongest patch equals the stored frame activity on every tile (max difference 0.0000).

## Numbers on the cards, and the cross-check

All values are point estimates on the held-out frames above.

| Behaviour | Model, neuron | AUROC | Average precision | Precision top 3 / 10 / 30 per video | Time per video: Pearson r | Time per video: mean abs. error (constant guess) | Top 6 frames labelled |
|---|---|---|---|---|---|---|---|
| nose-nose | v1 #611 | 0.691 | 0.040 | 0.074 / 0.069 / 0.075 | 0.55 | 15.7 s (13.9 s) | 0 of 6 |
| nose-nose | v2 #3593 | 0.790 | 0.230 | 0.528 / 0.451 / 0.335 | 0.68 | 18.8 s (13.9 s) | 5 of 6 |
| nose-tail | v1 #414 | 0.777 | 0.047 | 0.127 / 0.111 / 0.098 | 0.30 | 9.5 s (11.4 s) | 1 of 6 |
| nose-tail | v2 #3009 | 0.833 | 0.206 | 0.397 / 0.330 / 0.239 | 0.69 | 7.0 s (11.4 s) | 6 of 6 |
| grooming-to-focal (any) | v1 #90 | 0.875 | 0.790 | 0.788 / 0.79 / 0.80 | 0.80 | 85 s (136 s) | 6 of 6 |
| grooming-to-focal (any) | v2 #33 | 0.864 | 0.803 | 0.848 / 0.86 / 0.85 | 0.76 | 83 s (136 s) | 6 of 6 |

**Cross-check** (the `crosscheck` block in each `.json`):

- **Mice:** every AUROC, average precision and precision@top3/10/30 equals the v2.8 audit to 3-4 decimals. The sources are `audit/work/mice_<key>.pkl` and `audit/final.json`.
- **Ants v1 #90:** AUROC 0.8749 and precision@top3 0.788 equal the v2.7 ants report exactly. There is no audit pickle for v1.
- **Ants v2 #33:**
  - Average precision 0.803 equals the audit (0.8034), and precision@top3 0.848 equals the audit and the report.
  - AUROC is **0.864 on the card vs 0.867 in the report and audit.** This is a tie-handling difference, checked directly on the same 43,800 frames. 15.5% of the ants P(on) values are tied with another frame, mostly saturated at 1 (13.9% of frames have P(on) >= 0.999). Averaging the tied values gives 0.8670, exactly the report and audit value. Breaking ties by raw activity, as the card and the audit's `final.json` do, gives 0.8642. The card states both. Mice and v1 are not affected: mice P(on) rarely ties, and v1 is ranked by raw activity.

## Honest reading: is v2 visibly better aligned than v1?

**Nose-tail: yes, clearly.**
- v1 #414 fires at a low level almost all the time, in every video, including the video with no nose-tail. Its strongest frames are mostly two mice lying side by side in a huddle; only 1 of 6 is labelled.
- v2 #3009 is silent most of the time and fires in short bursts. Many bursts line up with annotated bouts, for example in wt_ash1l_m_1_S_H and het_kdm6b_m_1_S_H. All 6 of its strongest frames are labelled, with the heat map on the nose-to-rump contact.
- Still weak:
  - It misses many bouts entirely, for example both bouts in het_kdm6b_f_3_F_H.
  - In het_kdm6b_m_1_S_H it fires in dense trains well beyond the annotated bouts.
  - It has a few spikes in the video with no nose-tail.
- Time per video is tracked moderately: r 0.69, error 7.0 s vs 11.4 s for a constant guess. Two videos are over-predicted at about 55 s against about 40 s annotated.

**Nose-nose: yes for precision, no for time per video.**
- v1 #611 is a near-constant "mice are close" signal. It is on through long stretches of every video, and none of its 6 strongest frames is labelled.
- v2 #3593 is precise but low-recall:
  - It is silent most of the time.
  - Its spikes often fall on or next to annotated bouts.
  - Over half of each video's top 3 frames are labelled (0.53 vs chance 0.016), and 5 of its 6 strongest frames are labelled.
  - But many annotated bouts get no spike at all, and several spikes fall between bouts; see het_kmt5b_m_1_F_H and het_ash1l_f_1_S_H.
- Its per-video time is not better than a guess:
  - The mean absolute error is 18.8 s, worse than the constant-rate guess at 13.9 s, even though the correlation is higher (0.68 vs 0.55).
  - The scatter shows systematic over-prediction in many videos (up to about 180 s predicted against about 50 s annotated).
  - So the calibration fitted on the SG pools does not transfer: what v2 fires on outside the labelled frames still inflates the total.
- The one unlabelled top frame, wt_kdm6b_f_1_F_H 06:40, is two heads close together. It may be a missed label or a near-miss; we did not check this.

**Grooming-to-focal (any): no visible difference.**
- Both neurons switch on during annotated grooming and stay almost silent in the video with no grooming; v2 has one spike there.
- Both miss parts of long bouts. v1 gives smooth on-blocks, while v2 flickers within bouts.
- v2 is slightly more precise at the top (top-3 precision 0.85 vs 0.79), but AUROC is about equal (0.864 vs 0.875).
- Time per video is equally good for both (error about 84 s vs 136 s for a constant guess). Both under-predict the most-groomed videos.

**Overall.**
- On the mice behaviours the v2 neurons are much more *specific*. Their strongest moments are the labelled behaviour, while v1's are generic proximity or huddling. This is the gain that AUROC alone hides: AUROC moves only by +0.06 to +0.10, while average precision rises 4-6x.
- The gap is recall and counting: the v2 nose-nose neuron misses most bouts and miscounts time per video.
- Ants were already well aligned in v1, and v2 is on par.
