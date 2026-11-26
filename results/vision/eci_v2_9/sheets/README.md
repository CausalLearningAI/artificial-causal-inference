# ECI v2.9 top-frame sheets and the mice PREDk16_s2901 catalogue

Same tool and rules as `results/vision/eci_v2_8/sheets/README.md` (`scripts/eci/v2/v28_sheets.py --ver 29`): one picture per
sparse-autoencoder neuron with the 12 held-out frames where it fires hardest (raw frame code, at most 2 per video, >= 2 s
apart), a heat map of the neuron's per-token code and a cyan box on the strongest token. Held-out mice frames = the 12
non-SG pools (72 videos, 431,712 edge-excluded frames, annotated for nose-nose and nose-tail).

## Picked behaviour neurons (rule sgap; by-eye counts from `results/vision/eci_v2_9/LOG.md`)

| dictionary, neuron | target | by eye | labelled | what the other frames show |
|---|---|---|---|---|
| PREDk8_s2901 #3422 | nose-nose | 9/12 | 7/12 | #2 and #7 are 3-4 mouse huddles with the max token at a head among bodies; #6 is snout to the other's flank |
| PREDk8_s2901 #1027 | nose-tail | 12/12 | 9/12 | all are head-to-rear trains, the 3 unlabelled included (#10 the least clear) |
| PREDk16_s2901 #3461 | nose-nose | 11/12 | 10/12 | #12 is snout to the other's shoulder / flank; #4 (unlabelled) looks like a true head-to-head |
| PREDk16_s2901 #1780 | nose-tail | 11/12 | 8/12 | #11 is ambiguous (vertical train, snout near the rear but partly occluded) |
| PREDk16_s2902 #3811 | nose-nose | 2/12 | 0/12 | mostly snout at another's flank, side-by-side resting and corner huddles; not a nose-nose neuron |
| PREDk16_s2902 #2455 | nose-tail | 11/12 | 9/12 | #12 is snout to the other's side / rear |

## Catalogue: mice PREDk16_s2901 (40 neurons; POST HOC, by eye)

- Job 3232256 (`defaultp`, CPU, 26 min 49 s, max RSS 22 GB). Output `mice/PREDk16_s2901/catalogue/n<neuron>.png` +
  `index.json`. Identity checks in `index.json`: tile max == stored frame code on all 480 tiles (max abs difference 0.0);
  200 random held-out frames x all 4,096 latents, max abs difference 0.0.
- Selection (same as the v2.8 catalogues): held-out firing rate 0.2%-20%, >= 8 videos among the top-200 frames, not a
  picked behaviour neuron of any rule (#135, #1487, #1780, #3461 excluded); ranked by sharpness = mean of the top-50 held-out
  frame codes / median of the non-zero held-out codes; 8 per Matryoshka level, empty slots filled by sharpness.
  2,196 candidates; levels 1-2 have only 0 and 7 candidates (most of those neurons fire on > 20% of frames), so the 40 are
  7 L2, 8 L3, 17 L4, 8 L5.
- AUROC in the index = raw-code AUROC on held-out frames (not the P(on) readout of the reports). No catalogue neuron has
  raw-code AUROC above 0.68 for either behaviour; the highest are #287 nose-tail 0.672 (a tail detector, see below),
  #1521 nose-tail 0.578 (tail), #3708 nose-tail 0.569 and #314 nose-nose 0.557 (spout).
- Judged by one observer from the sheets (~200 px tiles), labels read after the description. "clear" = 10-12 of 12 tiles
  show the same thing; "plausible" = most do; "unclear" = no consistent picture. Self-grooming cannot be confirmed: the
  paws are not visible at this resolution, so "hunched / curled" describes the posture only.

| neuron | level | firing | what the top frames show | confidence | flag |
|---|---|---|---|---|---|
| #183 | 2 | 4.8% | a mouse with its snout at the tip of the tube on the left wall | clear | spout (mouse at spout) |
| #232 | 2 | 18.5% | one token on the head / neck of a single mouse, mixed scenes | unclear | |
| #129 | 2 | 18.7% | the back of a lone mouse walking in the open, body stretched | clear | lone mouse / locomotion |
| #171 | 2 | 8.0% | the tube tip on the left wall with one or more mice next to it | clear | spout |
| #136 | 2 | 2.1% | the rear of a mouse sitting at a wall or in a corner | unclear | |
| #218 | 2 | 8.5% | 2-4 mice packed together, heat spread over the bodies | clear | huddle |
| #152 | 2 | 1.7% | mice packed in a corner or in the nest, heat on the pile | plausible | huddle (corner / nest) |
| #396 | 3 | 5.9% | one token on the far (top) wall above a mouse standing below it; sharpness 8.7, the highest | clear | artefact: wall reflection (like PREDk s2703 #416, PRED26 #328) |
| #472 | 3 | 11.7% | small single tokens at the edge of a mouse (foot, tail base) | unclear | |
| #435 | 3 | 9.1% | a mouse's head at the left-wall grid | clear | hopper (left-wall grid) |
| #287 | 3 | 7.5% | a tail lying along or across another mouse's body, mice in a line; 4/12 labelled nose-tail | clear | tail on another mouse (raw nose-tail AUROC 0.672; not snout contact) |
| #257 | 3 | 19.7% | mice in or on the paper nest, heat on the paper around them | clear | nest material |
| #317 | 3 | 12.8% | a tail on the open floor | clear | tail |
| #454 | 3 | 12.4% | shredded nest paper around and over resting mice | clear | nest material |
| #314 | 3 | 15.5% | a mouse with its head at the tube tip on the left wall | clear | spout |
| #1520 | 4 | 3.5% | a lone, compact mouse, heat over the whole body | plausible | hunched posture (possibly self-grooming) |
| #1411 | 4 | 4.4% | the head of a mouse pushed into bedding or nest material | plausible | snout in bedding / nest |
| #1521 | 4 | 15.5% | a tail on the floor or over a white block | clear | tail |
| #942 | 4 | 13.9% | the contact point between 2-3 mice (head on another's body) | plausible | social contact (generic) |
| #1127 | 4 | 6.5% | a token at the tail base of a mouse, tail on the floor | plausible | tail base |
| #1793 | 4 | 12.8% | a curled, round mouse, heat on its back | plausible | hunched / curled posture |
| #1795 | 4 | 10.5% | floor tokens next to resting groups of mice | unclear | huddle / position |
| #540 | 4 | 12.4% | a lone mouse in the top-right corner next to the odour bag, heat over its body; all phases (H / O / P), not only odour videos | clear | bag corner (position) |
| #1985 | 4 | 7.8% | tails of mice sitting at the left-wall grid, pointing into the arena | clear | tail + hopper |
| #1217 | 4 | 7.9% | a tail lying along a wall | clear | tail on the wall |
| #1630 | 4 | 6.2% | the back of one mouse inside a resting group in a corner | clear | huddle |
| #1771 | 4 | 15.8% | the first part of a curved tail behind a lone mouse | clear | tail |
| #901 | 4 | 0.9% | a mouse up on the far or right wall (climbing or rearing), body foreshortened, heat over the whole body | clear | rearing / climbing |
| #1496 | 4 | 10.1% | the tail base of mice sitting at the left-wall grid | clear | tail + hopper |
| #1567 | 4 | 12.9% | a point inside a resting group of 3-4 mice | plausible | huddle |
| #1255 | 4 | 11.5% | the back / shoulder of a mouse inside a group | plausible | huddle |
| #1911 | 4 | 15.5% | the rear half of a compact mouse, often next to others | unclear | |
| #3468 | 5 | 2.5% | a lone mouse in the open, heat over the whole body | plausible | lone mouse / locomotion |
| #2766 | 5 | 2.0% | the back / rear of a lone compact mouse | plausible | hunched posture (possibly self-grooming) |
| #2737 | 5 | 3.9% | head and shoulders of a mouse pressed against the right wall or the bag | plausible | wall / bag sniffing or rearing |
| #3008 | 5 | 3.7% | a curled, round mouse alone, heat over the whole body | clear | curled posture (possibly self-grooming) |
| #3708 | 5 | 1.1% | two mice side by side, one mouse's head at the other's flank or rear; 5/12 labelled nose-tail | plausible | snout to flank / rear (raw nose-tail AUROC 0.569) |
| #4007 | 5 | 2.6% | a mouse with its snout at the tube tip on the left wall | clear | spout |
| #3933 | 5 | 14.5% | a mouse's snout pressed into the right wall or a corner | clear | wall sniffing / rearing |
| #3976 | 5 | 7.1% | a tail on nest paper or the floor | clear | tail |

Mice flags summary (PREDk16_s2901): spout #183, #171, #314, #4007; hopper (left-wall grid) #435 (+ tails at the grid
#1985, #1496); huddle #218, #1630 (+ #152, #1567, #1255; #1795 unclear); tail #317, #1521, #1771, #1217, #3976, #1127,
and #287 (tail on another mouse); nest material #257, #454 (+ #1411 snout in nest); rearing / climbing #901, wall
sniffing #3933 (+ #2737); hunched / curled posture #3008 (+ #1520, #1793, #2766); lone-mouse locomotion #129 (+ #3468);
bag corner #540 (position); generic contact #942, #3708; artefact #396 (wall reflection); unclear #232, #136, #472,
#1911. As for PREDk s2703: no tail-CONTACT neuron (the snout-at-tail neuron is the picked #1780) and no nose-nose neuron
among the 40 (#3461 was excluded as a pick).

## Ants (no new catalogue)

The ants dictionary of the v2 candidate is v2.7 PREDk s2703; its 40-neuron catalogue is in
`results/vision/eci_v2_8/sheets/README.md` ("Catalogue: ants PREDk s2703"). Clear neurons there: antenna movement #1331,
#3388, #3843; antenna on the wall #660, #1957, #1780, #4002; climbing / on the wall #149, #215; fast walking / blur #1399;
paint dot #1720; rim artefact #143. No neuron separates grooming from plain contact.

## Page gallery (draft `atlas/repr_page_draft/`)

Thumbnails used on the representation page: mice #183 (spout), #218 (huddle), #901 (rearing / climbing), #1771 (tail);
ants #1331 (antenna movement), #660 (antenna on the wall).
