# ECI v2.8 top-frame sheets and neuron catalogues

What this is: one picture per sparse-autoencoder (SAE) neuron showing the 12 held-out frames where it fires hardest, with a
heat map of WHERE in the frame it fires, so we can judge by eye whether a neuron means something.

Tool: `scripts/eci/v2/v28_sheets.py` (+ Slurm wrapper `scripts/eci/v2/v28_sheets.sh`, CPU only). It generalises
`v25_sheet.py` / `v26_sheet.py` (same token store, same SAE encode path, same frame decoding and crop/heat-map style).

## How the sheets are made

- Frame code = max over the frame's animal-mask (v5) tokens of the neuron's SAE code (the stored per-frame codes).
- Held-out frames: mice = the 12 non-SG pools (72 videos, 431,712 edge-excluded frames, annotated for nose-nose and
  nose-tail); ants = half B of v3/v4 (73 videos, 43,800 frames; the A -> B direction of `v27_eval`).
- Top 12 frames by raw frame code, code > 0; pass 1 takes one frame per video, pass 2 allows a second frame per video;
  two frames of one video are always >= 2 s (10 frames) apart. Every sheet below ended up with 12 frames from 12
  different videos.
- Tile = crop (half the frame side) centred on the strongest token; heat = the neuron's per-token code (recomputed from
  the token store through the SAE); cyan box = strongest token; caption = video, mm:ss, code, ground-truth labels (mice:
  nose-nose, nose-tail; ants: yellow / blue nestmate grooms focal). Green caption = a label is YES.
- Header: dictionary, neuron, Matryoshka level, held-out firing rate; for picked neurons the pre-registered held-out
  AUROC (HMM P(on) readout) and precision@top3/video from the version's `report.json`; and the held-out AUROC of the RAW
  frame code for the main labels (computed here).
- Identity checks (every job): (1) every tile, max of the recomputed per-token map == stored frame code: max abs
  difference 0 on all 1,320 tiles; (2) 200 random held-out frames per job, max over tokens of ALL latents == the stored
  code row: max abs difference <= 0.0039 (one fp16 step for codes > 8; exactly 0 for 13 of the 17 dictionaries, non-zero only for
  the four PRED26 jobs). V s1002 (v2.1 input
  path x + (x_{t+1} - x_{t-1}) / 2) passes both checks too.

## Picked behaviour neurons

"By eye" = my count of tiles that show the target behaviour (nose-nose: two snouts touching; nose-tail: one mouse's
snout at another mouse's rear / tail base). These are judgements on ~250 px crops, roughly +-1. "Labelled" = tiles
whose ground-truth label is positive. AUROC / P@3 = pre-registered held-out AUROC and precision@top3/video from the
report (rule(s) in brackets; a neuron picked under several rules has the same numbers under each).

### Mice, nose-tail picks

| dictionary | neuron (level, firing) | rules | AUROC / P@3 | by eye | labelled | what the top frames show |
|---|---|---|---|---|---|---|
| PRED26 s2601 | #2785 (L5, 4.6%) | auc, dist, dist2 | 0.848 / 0.386 | 11/12 | 10/12 | one mouse's snout at another mouse's rear / tail base; heat on the contact point |
| PRED26 s2602 | #3882 (L5, 3.7%) | auc, dist, dist2 | 0.872 / 0.365 | 11/12 | 11/12 | snout at the tail base of another mouse, mostly mice in a line; heat small, on the contact |
| PRED26 s2603 | #2255 (L5, 4.8%) | auc, dist, dist2 | 0.850 / 0.354 | 12/12 | 11/12 | snout at another mouse's rear; the one unlabelled tile (#8) also looks like nose-tail to me |
| PREDc s2701 | #443 (L3, 55.4%) | auc (also its nose-nose pick) | 0.856 / 0.122 | 4/12 | 5/12 | head of one mouse touching another's rear or flank, pairs and triads; generic contact point |
| PREDc s2701 | #3441 (L5, 5.3%) | dist, dist2 | 0.843 / 0.397 | 11/12 | 10/12 | snout at another mouse's rear; heat blob on the rear of the sniffed mouse |
| PREDc s2702 | #1661 (L4, 9.6%) | auc, dist, dist2 | 0.846 / 0.143 | 0/12 | 1/12 | tightly packed 2-3 mouse HUDDLES, often in a corner; heat on a small point between bodies |
| PREDc s2703 | #212 (L2, 50.6%) | auc, dist, dist2 | 0.816 / 0.069 | 1/12 | 1/12 | TAILS: heat runs along tails lying on the floor or across other mice (a tail detector) |
| PREDd s2701 | #2262 (L5, 4.6%) | auc, dist, dist2 | 0.874 / 0.381 | 11/12 | 9/12 | snout at another mouse's rear / tail base; heat on the sniffed mouse's rear |
| PREDd s2702 | #2949 (L5, 14.8%) | auc, dist, dist2 | 0.830 / 0.132 | 1/12 | 1/12 | 2-3 mice HUDDLED side by side, many near the left-wall grid or the bag; small heat between bodies |
| PREDd s2703 | #1686 (L4, 35.2%) | auc, dist, dist2 | 0.849 / 0.090 | 2/12 | 2/12 | mice HUDDLED in corners / against walls; heat between bodies |
| PREDh s2701 | #3324 (L5, 11.8%) | auc | 0.849 / 0.116 | 2/12 | 3/12 | mice next to the left wall near the grid / wall tube; heat at a head near the wall or another mouse |
| PREDh s2701 | #3457 (L5, 4.9%) | dist, dist2 | 0.847 / 0.249 | 8/12 | 4/12 | snout at another mouse's rear; 4 tiles look like nose-tail to me but are labelled no |
| PREDh s2702 | #1408 (L4, 26.8%) | auc (also its nose-nose pick) | 0.850 / 0.090 | 1/12 | 1/12 | 2-3 mice with heads together at the wall or in corners; heat at head contact |
| PREDh s2702 | #3741 (L5, 3.5%) | dist, dist2 | 0.846 / 0.217 | 10/12 | 7/12 | snout at another mouse's rear / tail base; mice in a line |
| PREDh s2703 | #1360 (L4, 14.6%) | auc | 0.843 / 0.116 | 1/12 | 1/12 | mice near the left-wall tube, heads close together; mixed head-to-head and huddle |
| PREDh s2703 | #3132 (L5, 4.9%) | dist, dist2 | 0.846 / 0.392 | 11/12 | 10/12 | snout at another mouse's rear; big heat blob on the sniffed mouse's rear |
| V s1002 | #270 (L3, 19.7%) | auc, dist, dist2 | 0.814 / 0.132 | 2/12 | 2/12 | TAILS: heat along tails (on the floor and over bodies), like PREDc s2703 #212 |

### Mice, nose-nose picks

| dictionary | neuron (level, firing) | rules | AUROC / P@3 | by eye | labelled | what the top frames show |
|---|---|---|---|---|---|---|
| PRED26 s2601 | #287 (L3, 53.1%) | auc, dist | 0.814 / 0.153 | 2/12 | 2/12 | head-to-body contact points in 2-4 mouse clusters (head on another's flank / neck), mostly not face-to-face |
| PRED26 s2602 | #841 (L4, 24.7%) | auc, dist | 0.831 / 0.255 | 5/12 | 4/12 | two heads close together; about half face-to-face, the rest head-to-neck |
| PRED26 s2603 | #403 (L3, 49.8%) | auc, dist | 0.799 / 0.083 | 1/12 | 1/12 | 2-3 mice HUDDLED side by side near the wall grid or the bag |
| PREDc s2701 | #443 (L3, 55.4%) | auc, dist | 0.789 / 0.157 | 1/12 | 0/12 | head touching another mouse's rear or flank (see nose-tail table) |
| PREDc s2702 | #16 (L1, 88.7%) | auc, dist | 0.761 / 0.181 | 3/12 | 3/12 | centre of 3-4 mouse clusters; heat at head-to-body contacts |
| PREDc s2703 | #422 (L3, 59.5%) | auc, dist | 0.780 / 0.069 | 0/12 | 0/12 | 3-4 mice HUDDLED in a corner, often next to the bag; heat over the huddle |
| PREDd s2701 | #421 (L3, 61.1%) | auc, dist | 0.794 / 0.315 | 4/12 | 3/12 | contact point where one head meets another mouse (head-to-head or head-to-flank) |
| PREDd s2702 | #448 (L3, 55.6%) | auc, dist | 0.779 / 0.245 | 5/12 | 2/12 | two mice approaching head-on, heads touching or nearly; best-looking nose-nose of the PRED arms besides V |
| PREDd s2703 | #208 (L2, 70.3%) | auc, dist | 0.796 / 0.278 | 5/12 | 3/12 | two mice end to end, contact point between them; head-to-head or head-to-rear |
| PREDh s2701 | #155 (L2, 82.8%) | auc, dist | 0.788 / 0.153 | 3/12 | 3/12 | 3-4 mouse clusters / HUDDLES; heat spread over the central contact |
| PREDh s2702 | #1408 (L4, 26.8%) | auc, dist | 0.759 / 0.111 | 4/12 | 4/12 | 2-3 heads together at the wall or in corners |
| PREDh s2703 | #246 (L2, 66.7%) | auc, dist | 0.805 / 0.171 | 2/12 | 1/12 | head / neck junctions inside 2-4 mouse clusters |
| V s1002 | #436 (L3, 50.9%) | auc, dist | 0.782 / 0.278 | 7/12 | 6/12 | two mice face-to-face or head-to-neck, heat on the meeting point; the most nose-nose-like pick |

### Ants, grooming-to-focal (any)

| dictionary | neuron (level, firing) | rule | AUROC / P@3 | by eye | labelled | what the top frames show |
|---|---|---|---|---|---|---|
| PRED26 s2601 | #40 (L1, 72.1%) | best AUROC on half A | 0.863 / 0.803 | 12/12 tight 3-ant cluster on the focal | 10/12 | the three ants packed together, heat on the unpainted focal at the cluster centre. Grooming itself (mouthparts working) cannot be resolved at this resolution, so "by eye" counts contact, not grooming |
| PREDc s2701 | #181 (L2, 42.7%) | best AUROC on half A | 0.856 / 0.763 | 12/12 tight cluster (same caveat) | 12/12 | same picture as #40: painted nestmates on the focal, heat on the contact zone |

### What the picked sheets say

- Nose-tail: 8 of the 17 nose-tail picks are clean, recognisable nose-tail neurons (10-12/12 by eye): PRED26 #2785,
  #3882, #2255; PREDc s2701 #3441; PREDd s2701 #2262; PREDh s2702 #3741, s2703 #3132; PREDh s2701 #3457 is close (8/12).
  ALL of them are level-5 neurons (latents 2048-4095) firing on 3.5-5.3% of frames. Every nose-tail pick that is not
  like this is a huddle detector (PREDc s2702 #1661, PREDd s2702 #2949, s2703 #1686), a tail detector (PREDc s2703 #212,
  V #270) or a wall / tube-region neuron (PREDh s2701 #3324, s2703 #1360, s2702 #1408). Rule auc tends to pick these
  broad neurons; rule dist picks the clean one in PREDc s2701, PREDh s2701 / s2702 / s2703. So PRED-c's weaker
  reproduction of PRED26 (nose-tail precision@top3/video 0.203 vs 0.369) is visible here: in PREDc only seed 2701 has a
  clean nose-tail pick; in PRED26 all three seeds do.
- Labels miss real nose-tail events: PREDh s2701 #3457 (8 by eye vs 4 labelled) and PREDh s2702 #3741 (10 vs 7) show
  snout-at-rear frames labelled negative. Ground truth looks incomplete for nose-tail.
- Nose-nose: no PRED-family dictionary has a clean nose-nose neuron. Picks are levels 1-4, fire on 25-89% of frames,
  and fire on contact points in clusters or on huddles; best 5/12 by eye (PRED26 s2602 #841, PREDd s2702 #448, s2703
  #208). The v2.1 baseline V s1002 #436 is the best (7/12 by eye, 6/12 labelled).
- Ants: both grooming picks are "three ants in a tight cluster on the focal" neurons; the label agrees 10-12/12 but the
  sheets cannot tell grooming from plain contact.

## Catalogue: mice PRED26 s2601 (40 neurons)

Selection (`catalogue/index.json` has the stats of every chosen neuron): held-out firing rate 0.2%-20%, >= 8 videos
among the top-200 frames, not one of the picked behaviour neurons (#287, #2785, #3602); ranked by sharpness = mean of
the top-50 held-out frame codes / median of the non-zero held-out codes; 8 per Matryoshka level, empty slots filled by
global sharpness. 2,363 candidates; levels 1-2 have only 0 and 6 candidates (those neurons fire on > 20% of frames), so
the 40 are 6 L2, 8 L3, 15 L4, 11 L5. AUROC in the index = raw-code AUROC on held-out frames; no catalogue neuron has
raw-code AUROC above 0.60 for any mouse label (best: #3383 nose-tail 0.596).

| neuron | level | firing | what the top frames show | confidence | flag |
|---|---|---|---|---|---|
| #130 | 2 | 17.9% | huddle of 3-4 resting mice in a corner, heat over one body in the pile | clear | huddle |
| #212 | 2 | 15.9% | mice buried in / covered by nest material (bedding, paper) | clear | nest |
| #230 | 2 | 17.1% | one mouse at the left-wall grid, head into the grid, body along the floor | clear | hopper (grid on left wall; feeding / drinking not verifiable) |
| #149 | 2 | 2.5% | mice in the paper nest, partly covered | plausible | nest |
| #197 | 2 | 1.9% | a single mouse along the bottom or left wall, heat on its rear / flank | unclear | |
| #152 | 2 | 0.3% | small point on the back of a mouse inside a huddle (near a shave mark) | unclear | huddle |
| #328 | 3 | 11.1% | one token on the far (top) wall just above a mouse standing at that wall; sharpness 9.6, the highest | clear | artefact: wall token next to a mouse (reflection / mask edge) |
| #506 | 3 | 10.6% | huddles near the left-wall grid, heat at a head inside the huddle | plausible | huddle |
| #368 | 3 | 17.0% | strip on the near (bottom) wall below a mouse walking along it | clear | artefact: reflection / shadow on the near wall |
| #498 | 3 | 9.7% | a mouse facing the near (bottom) wall, body foreshortened, heat on the wall in front of it | clear | rearing against the near wall |
| #358 | 3 | 18.4% | mouse at the left-wall grid, heat on its head and back; several mice queueing at the grid | clear | hopper |
| #504 | 3 | 16.8% | a mouse with its back to the camera at the near wall, tail up, heat on the lower back | plausible | rearing against the near wall |
| #304 | 3 | 9.7% | the edge of a resting huddle (a head or flank at the rim) | plausible | huddle |
| #382 | 3 | 15.0% | a mouse at the far (top) wall with head up, heat on the wall above / around the head | clear | rearing against the far wall |
| #952 | 4 | 17.1% | one token on the left-wall grid where a mouse's head touches it; very high codes (~24) | clear | hopper |
| #685 | 4 | 5.4% | the snout of a single mouse, head down in the bedding | plausible | sniffing / digging the floor |
| #1229 | 4 | 16.7% | a floor token beside a resting huddle in the bottom-right corner | unclear | huddle / position |
| #1848 | 4 | 12.2% | scattered tokens on nest paper around mice | unclear | nest |
| #738 | 4 | 12.5% | nest paper sheets next to or under mice | plausible | nest |
| #584 | 4 | 18.4% | the floor at the near wall right next to a corner huddle | plausible | huddle / position |
| #1214 | 4 | 6.0% | a mouse's snout at the left-wall grid (single token on the grid) | clear | hopper |
| #1192 | 4 | 8.3% | a point inside a resting huddle | plausible | huddle |
| #923 | 4 | 8.6% | a mouse's snout pointed into a wall or corner | clear | wall sniffing / rearing |
| #1816 | 4 | 7.9% | the tube sticking out of the left wall with a mouse's head at its tip | clear | spout / port on the left wall (water spout or odour port; I cannot tell which from the image) |
| #707 | 4 | 6.9% | one point on the back / head of a resting mouse | unclear | |
| #1069 | 4 | 11.1% | middle of a mouse's back, near the shave mark | unclear | identity mark? |
| #2032 | 4 | 12.3% | a mouse facing the far (top) wall with head raised, heat on head and shoulders | clear | rearing against the far wall |
| #846 | 4 | 6.5% | tails / nest material around a huddle | unclear | tail / nest |
| #1888 | 4 | 4.9% | one curled mouse lying inside a huddle, heat on its whole body | plausible | huddle (resting) |
| #2506 | 5 | 19.1% | a head near another mouse's flank or tail base, mixed scenes | unclear | social contact (nose-tail AUROC 0.555) |
| #3046 | 5 | 6.4% | whole body of a lone mouse in the open, hunched / curved | unclear | possibly self-grooming |
| #2380 | 5 | 2.4% | a lone mouse curled up (hunched), heat on its hind body | plausible | self-grooming (hunched posture; I cannot see the paws) |
| #3024 | 5 | 4.1% | the TIP of a tail, often lying on the wall or the floor edge | clear | tail |
| #2305 | 5 | 10.3% | tail segments on the floor / nest paper | plausible | tail |
| #3383 | 5 | 4.2% | one mouse's head at another's flank or rear | plausible | nose-tail-like contact (6/12 labelled nose-tail; raw AUROC 0.596) |
| #2316 | 5 | 7.5% | a point near a mouse's head inside paper nest | unclear | nest |
| #3659 | 5 | 1.6% | the white shave mark on a mouse's head | plausible | identity mark |
| #2981 | 5 | 0.8% | a tail lying over or next to the white wooden block | clear | tail on the block |
| #3280 | 5 | 10.6% | the middle of a tail on the open floor | clear | tail |
| #2969 | 5 | 5.4% | a mouse at the BAG in the top-right corner, heat on the mouse at the bag; 11/12 tiles from odour-phase videos (`_O`) | clear | sniffing the box / bag |

Mice flags summary: bag sniffing #2969; huddle #130, #506, #304, #1192, #1888 (+ #152, #584, #1229 partly positional);
tail #3024, #3280, #2981, #2305 (+ #846); rearing #498, #504, #382, #2032 (+ wall sniffing #923); hopper / left-wall
grid #230, #358, #952, #1214; wall tube #1816; possible self-grooming #2380 (#3046 unclear); nest material #212, #149,
#738, #1848, #2316; artefacts (wall reflection / shadow) #328, #368. No nose-nose neuron in the catalogue.

## Catalogue: ants PRED26 s2601 (40 neurons)

Same selection (picked neuron #40 excluded). 2,968 candidates; level 1 has only 2 (most L1 neurons fire on > 20% of
frames), so the 40 are 2 L1, 8 L2, 8 L3, 14 L4, 8 L5. No catalogue neuron has raw-code AUROC above 0.58 for
grooming-to-focal (best #1705 0.578).

| neuron | level | firing | what the top frames show | confidence | flag |
|---|---|---|---|---|---|
| #87 | 1 | 9.0% | specks on the empty floor / rim, no ant | clear | artefact: debris / mask leak |
| #60 | 1 | 3.5% | ants at the dish wall, heat over the ant(s) at the rim | plausible | wall |
| #218 | 2 | 11.6% | the head end of an ant touching the dish wall | plausible | antenna / head at the wall |
| #245 | 2 | 13.1% | the wall right next to an ant's head | plausible | antenna / head at the wall |
| #193 | 2 | 11.1% | an ant on the far rim (top of the image) | plausible | wall (position) |
| #232 | 2 | 17.5% | head / thorax of a lone ant standing in the open | unclear | |
| #221 | 2 | 17.5% | an ant on the dish wall, seen foreshortened (climbing the wall) | clear | climbing the wall |
| #254 | 2 | 10.8% | a single ant in the open, compact posture, heat over the whole ant | unclear | possibly self-grooming |
| #182 | 2 | 16.1% | head and antennae of a standing ant | plausible | antenna |
| #200 | 2 | 17.1% | a blurred ant on the left dish wall (climbing, out of focus), heat over the body | clear | climbing the wall |
| #500 | 3 | 15.3% | an ant pressed along the dish wall, body elongated | clear | walking along the wall |
| #321 | 3 | 17.2% | an ant's head near another ant or its legs | plausible | contact |
| #307 | 3 | 16.2% | one token just off an ant (antenna or leg tip) on the floor | plausible | antenna |
| #288 | 3 | 7.3% | the TIP of an ant's antenna (single token, antenna stretched forward) | clear | antenna |
| #412 | 3 | 3.3% | the floor just ahead of an ant's head (antenna tips) | plausible | antenna |
| #346 | 3 | 6.3% | the gaster (abdomen) tip of an ant, painted or not | clear | body part: gaster |
| #287 | 3 | 10.8% | an ant's head at the bottom dish wall | plausible | head at the wall |
| #453 | 3 | 10.3% | the head of an ant (front of the body) | plausible | head |
| #1977 | 4 | 6.8% | single tokens off-ant, near antenna or leg tips | unclear | antenna? |
| #1242 | 4 | 5.7% | the head of one ant on another ant's body; 9/12 labelled grooming | plausible | contact (grooming-like) |
| #1355 | 4 | 12.9% | the contact point between two ants (head on the other's gaster / body); 11/12 labelled grooming | plausible | contact (grooming-like) |
| #1635 | 4 | 2.3% | the head-thorax junction of an ant | plausible | body part |
| #1255 | 4 | 0.7% | empty floor / rim tokens, no ant | clear | artefact: floor / rim |
| #1725 | 4 | 18.8% | an ant along the wall with its head against it | plausible | walking along the wall |
| #585 | 4 | 6.8% | a motion-blurred ant, heat over the blur | plausible | fast walking (blur) |
| #994 | 4 | 16.2% | the rear (gaster) of an ant touched by another ant's head; 7/12 labelled grooming | plausible | contact at the gaster |
| #1677 | 4 | 1.8% | a blurred / ghost ant | plausible | fast walking (blur) |
| #1705 | 4 | 10.3% | the contact point between two or three ants (head to body); 12/12 labelled grooming | plausible | contact (grooming-like) |
| #534 | 4 | 13.0% | the floor just in front of a walking ant's head | plausible | antenna sweep |
| #1006 | 4 | 4.4% | the dish rim next to an ant's head | plausible | antennae on the wall |
| #1023 | 4 | 9.6% | the head of an ant at the bottom / left rim | plausible | head at the wall |
| #987 | 4 | 4.0% | rim tokens next to an ant at the wall | plausible | wall |
| #3620 | 5 | 0.4% | the BLUE paint dot on a nestmate's gaster | clear | identity paint (blue) |
| #3304 | 5 | 5.7% | an ant lying along the dish wall | clear | walking along the wall |
| #3322 | 5 | 1.6% | the focal's thorax at the centre of a tight 3-ant cluster; 11/12 labelled grooming | plausible | contact (grooming-like) |
| #2807 | 5 | 5.0% | an antenna tip, often touching another ant | plausible | antennation (antenna contact) |
| #2461 | 5 | 1.2% | a single token just ahead of an ant's head (antenna tip) | plausible | antenna |
| #3443 | 5 | 8.2% | the head and front legs of a walking ant | unclear | walking / possibly self-grooming |
| #2645 | 5 | 4.1% | the ORANGE ("yellow") paint dot on a nestmate's gaster; 9/12 labelled grooming because that ant is near the focal | clear | identity paint (yellow) |
| #3420 | 5 | 3.4% | leg / contact region between ants | unclear | contact |

Ants flags summary: antenna (tips, sweeps, contact) #288, #307, #412, #534, #2461, #2807, #182 (+ #1977 unclear); wall /
climbing / walking along the wall #221, #200, #500, #3304, #1725, #60, #193, #218, #245, #287, #1006, #1023, #987;
walking (motion blur) #585, #1677 (#3443 unclear); contact / grooming-like #1242, #1355, #1705, #994, #3322, #321
(#3420 unclear); possible self-grooming #254 (unclear); identity paint #3620 (blue), #2645 (yellow); body parts #346
(gaster), #1635, #453; artefacts #87, #1255.

## Files and commands

- Sheets: `<dom>/<dictionary>/n<neuron>.png` + `sheets.json` (frames, labels, codes, identity numbers, header metrics);
  catalogues: `<dom>/PRED26_s2601/catalogue/n<neuron>.png` + `index.json`.
- Commit policy: picked-neuron PNGs (18.8 MB) and the catalogue sheets marked "clear" above are in git; the other
  catalogue PNGs (47.4 MB total for both catalogues) stay on disk only (repo size).
- Commands (CPU, `defaultp`, 8 cores, 64 GB; mice catalogue 110 GB):
  - `sbatch -J NAME scripts/eci/v2/v28_sheets.sh --ver 27 --dom mice --key PREDc_s2701 --rules auc,dist`
  - `sbatch -J NAME scripts/eci/v2/v28_sheets.sh --ver 26 --dom ants --key PRED_s2601 --name PRED26_s2601 --rules auc`
  - `sbatch --mem=110G -J NAME scripts/eci/v2/v28_sheets.sh --ver 27 --dom mice --key PRED26_s2601 --catalogue 40`
  - explicit neurons: `... --neurons 12,345`; random-frame self-check size: `--selfcheck N` (default 200).
- `--ver 27` covers PREDc / PREDd / PREDk / PREDh s2701-2703, PRED26 s2601-2603 (mice) and V s1001-1006 (mice);
  `--ver 26` covers the v2.6 arms (N0wc / PRED / SMOOTH / RW / N0x) in both domains (ants PRED26 = `PRED_s2601`).
  v1 dictionaries and ants V are not supported (no token path / cache in this code).
- Runtimes: picks jobs 3-16 min each (wall clock on a shared node; the reruns with the final headers); mice catalogue 20.6 min (read 6.7 GB raw cache once, AUROC of 4,096 neurons on
  431,712 frames, 40 sheets); ants catalogue 7.2 min.

## PREDk candidates (2026-11-25)

Why: the v2.8 audit (`results/vision/eci_v2_8/audit/tables.md` section 6b, rule "sgap" = POST HOC precision-based pick)
found sparse, specific-looking neurons in the v2.7 contact-weighted arm PREDk. These are their sheets, plus the sgap picks of
PREDk s2701 and PRED26 s2602 for comparison, the ants grooming picks of PREDk s2702 / s2703, and two 40-neuron catalogues
(mice and ants PREDk s2703). Same tool, same held-out frames and same tile rules as above.

- The audit json (`audit.json` -> `mice.sgap`) lists ONE sgap neuron per dictionary and behaviour, not a top 3, so the
  comparison dictionaries got that single pick (PRED26 s2602's nose-tail sgap pick is #3882, already sheeted above).
- Judging: I counted "by eye" before reading the label captions. Exception: for #3009 and #3776 I had already seen the
  labelled counts in the job log. Nose-nose counted strictly: two snouts together/touching. Snout on a flank or neck, or a
  pile-up of heads, does not count.
- Identity checks: every job had a per-tile max abs difference of 0. The 200-random-frame self-check (all 4,096 latents) gave 0 for every
  PREDk job (mice s2701 / s2702 / s2703, ants s2702 / s2703, both catalogues) and 0.0039 (one fp16 step) for PRED26
  s2602, as for the earlier PRED26 jobs.
- Jobs (all `defaultp`, CPU): 3198162 (mice PREDk s2703 picks), 3198163 (s2702), 3198164 (s2701), 3198165 (PRED26 s2602 ->
  folder `PRED26_s2602_sgap`), 3198166 / 3198167 (ants PREDk s2703 / s2702), 3198168 (mice catalogue), 3198169 (ants
  catalogue). Pick jobs 2-7 min, ants catalogue 10 min, mice catalogue 28.5 min.

### Picked neurons

AUROC / P@3 = held-out AUROC (HMM P(on) readout) / precision@top3/video from the audit; raw = held-out AUROC of the raw
frame code for that behaviour (from the sheet header).

| dictionary | neuron (level, firing) | target | AUROC / P@3 (raw AUROC) | by eye | labelled | what the top frames show |
|---|---|---|---|---|---|---|
| mice PREDk s2703 | #3593 (L5, 1.5%) | nose-nose | 0.790 / 0.528 (raw 0.610) | 9/12 | 9/12 | two mice head to head; small heat spot exactly where the two snouts meet. #4, #10, #12 doubtful (head at neck / flank); #2, #7, #8 look head to head but are labelled no |
| mice PREDk s2703 | #3009 (L5, 2.7%) | nose-tail | 0.833 / 0.397 (raw 0.674) | 11/12 | 10/12 | one mouse's snout at another's rear / tail base, mice in a line; heat blob on the sniffed mouse's rear. #8 looks like nose-tail but is labelled no |
| mice PREDk s2702 | #4027 (L5, 5.7%) | nose-nose | 0.813 / 0.556 (raw 0.678) | 6/12 | 8/12 | heads meeting inside 2-4 mouse clusters, often at the left-wall grid; about half are head to neck or pile-ups rather than snout to snout |
| mice PREDk s2702 | #2063 (L5, 0.54%) | nose-tail | 0.830 / 0.360 (raw 0.584) | 11/12 | 8/12 | snout at another mouse's rear / tail base; small heat on the contact point. Several unlabelled tiles look like real nose-tail |
| mice PREDk s2701 | #359 (L3, 61.5%) | nose-nose | 0.698 / 0.069 (raw 0.664) | 2/12 | 1/12 | generic: heads / backs inside 2-3 mouse clusters, often near the grid. Not nose-nose |
| mice PREDk s2701 | #3776 (L5, 3.1%) | nose-tail | 0.844 / 0.418 (raw 0.696) | 11/12 | 11/12 | clean nose-tail, heat on the sniffed mouse's rear. Shares top frames with s2703 #3009 (e.g. het_kdm6b_m_1_S_H 04:28.2 is #1 in both): the same neuron in two seeds |
| mice PRED26 s2602 | #2545 (L5, 10.4%) | nose-nose | 0.685 / 0.093 (raw 0.619) | 2/12 | 1/12 | left-wall grid neuron: two mice stacked at the grid, heat over a head / neck. Not nose-nose |
| ants PREDk s2703 | #33 (L1, 63.8%) | grooming-to-focal (any) | 0.867 / 0.848 (raw 0.848) | 12/12 contact with the focal; 10/12 tight 3-ant cluster | 12/12 | painted nestmates on the unpainted focal, heat on the focal / contact zone. Grooming itself cannot be resolved; "by eye" counts contact |
| ants PREDk s2702 | #72 (L1, 78.2%) | grooming-to-focal (any) | 0.866 / 0.783 (raw 0.863) | 12/12 contact; 11/12 tight cluster | 11/12 | same picture as #33, larger heat blob over the cluster |

Verdict per dictionary ("clean" = at least 8/12 by eye):

- mice PREDk s2703: has a clean, distinct nose-nose AND nose-tail neuron: YES. #3593 (nose-nose, 9/12) and #3009
  (nose-tail, 11/12) fire on different places (snout meets snout vs snout at rear) and share no top frames. This is the
  first nose-nose neuron in any v2 dictionary at >= 8/12 by eye (previous best V s1002 #436, 7/12). Caveat: it was found
  by a POST HOC rule (sgap) over 34 dictionaries, and its held-out AP is 0.230; one seed only.
- mice PREDk s2702: NO. Nose-tail #2063 is clean (11/12), but nose-nose #4027 is 6/12 and partly a "heads together in a
  cluster" neuron.
- mice PREDk s2701: NO. Nose-tail #3776 is clean (11/12); the nose-nose sgap pick #359 is generic (2/12).
- mice PRED26 s2602: NO for nose-nose (#2545 is a wall-grid neuron, 2/12); its nose-tail #3882 is clean (11/12, above).
- ants PREDk s2702 / s2703: both grooming picks are level-1 "tight 3-ant cluster on the focal" neurons, as for PRED26 /
  PREDc; they agree with the label 11-12/12 but cannot tell grooming from plain contact.

### Catalogue: mice PREDk s2703 (40 neurons)

Same selection as the PRED26 catalogue. The excluded picked neurons are #149, #1730, #1765 and #3593 (the rule auc / dist / dist2 picks). The sgap
nose-tail pick #3009 is NOT one of these, so it was eligible and was selected; its catalogue sheet is the same as its pick sheet. 2,210
candidates; levels 1-2 have only 0 and 4 (most fire on > 20% of frames), so the 40 are 4 L2, 8 L3, 16 L4, 12 L5. Apart
from #3009 (raw nose-tail AUROC 0.674, 10/12 labelled), no catalogue neuron has raw-code held-out AUROC above 0.58 for
either behaviour (best #925 nose-nose 0.578, which is a spout neuron).
Job 3198168: 28.5 min (30 min wall clock).

| neuron | level | firing | what the top frames show | confidence | flag |
|---|---|---|---|---|---|
| #143 | 2 | 19.7% | whole body of a lone mouse walking in the open, heat over its back | clear | lone mouse / locomotion |
| #144 | 2 | 15.1% | the TIP of the tube on the left wall, with or without a mouse at it | clear | spout |
| #234 | 2 | 11.7% | mice in or under the paper nest, speckled heat over the bodies | clear | nest material |
| #248 | 2 | 8.7% | the tube tip on the left wall with a mouse's head at it | clear | spout (mouse at spout) |
| #416 | 3 | 7.3% | one token on the far (top) wall just above a mouse below it; sharpness 9.0 | clear | artefact: wall reflection / shadow (like PRED26 #328) |
| #499 | 3 | 15.9% | the head and shoulders of a lone mouse walking LEFT in the open | clear | lone mouse, one heading (locomotion) |
| #504 | 3 | 7.1% | the tube tip on the left wall (mice nearby or not) | clear | spout |
| #511 | 3 | 12.4% | a mouse with its head into the BAG in the top-right corner; 12/12 tiles from odour-phase videos (`_O`) | clear | bag / box sniffing |
| #448 | 3 | 17.0% | a mouse along the left wall at the grid, heat on its back by the grid | clear | hopper (left-wall grid) |
| #323 | 3 | 13.1% | a tail on the floor near the near (bottom) wall | plausible | tail |
| #348 | 3 | 10.2% | a tail lying over or next to a white wooden block | clear | tail on the block |
| #389 | 3 | 17.5% | tail base and first part of the tail of a mouse whose head is in the top-right (bag) corner | clear | tail (bag-corner position) |
| #733 | 4 | 8.1% | one mouse's back inside a resting huddle | clear | huddle |
| #513 | 4 | 7.3% | tail bases / tails on the floor between 2-3 nearby mice | plausible | tail (between mice; not labelled nose-tail) |
| #1791 | 4 | 13.9% | the tail base of mice sitting at the left-wall grid (tails pointing into the arena) | clear | tail + hopper |
| #698 | 4 | 3.3% | the white shave mark on a mouse's head | clear | identity mark |
| #1134 | 4 | 12.5% | a hunched mouse in the top-right (bag) corner, heat over its back | plausible | bag-corner position; possibly self-grooming |
| #1593 | 4 | 17.0% | the middle of a tail on the open floor | clear | tail |
| #812 | 4 | 12.7% | a mouse's snout pressed into the right wall | clear | wall sniffing / rearing against the right wall |
| #1606 | 4 | 9.0% | the floor just ahead of a mouse's snout (snout tip pointing left / down) | plausible | sniffing the floor |
| #1773 | 4 | 4.7% | a mouse partly under nest paper, small heat on the visible part | plausible | nest material |
| #1177 | 4 | 15.8% | the floor at the near (bottom) wall right in front of mice resting there | plausible | huddle / position |
| #1496 | 4 | 18.0% | a tail lying over nest paper | plausible | tail (on nest) |
| #1944 | 4 | 5.3% | the TIP of a tail lying against a wall (left grid wall, top or bottom wall) | clear | tail tip on the wall (like PRED26 #3024) |
| #867 | 4 | 0.8% | the middle of the back of a mouse with a large white shave patch | plausible | identity mark (back) |
| #810 | 4 | 14.5% | a compact, rounded (hunched) mouse, heat on its back / head | plausible | possibly self-grooming (hunched; paws not visible) |
| #925 | 4 | 17.7% | a mouse with its head at the tube tip on the left wall | clear | spout (raw nose-nose AUROC 0.578 is the spout, not nose-nose) |
| #1571 | 4 | 13.4% | a mouse facing the right wall, head foreshortened against it | clear | rearing against the right wall |
| #3587 | 5 | 3.3% | a lone compact mouse, heat over the whole body | plausible | possibly self-grooming (hunched) |
| #3856 | 5 | 1.0% | a mouse up on the far / right wall (climbing or rearing), body on the wall | clear | rearing / climbing |
| #3586 | 5 | 5.8% | the first part of a tail on the floor, a little behind the tail base | clear | tail |
| #2659 | 5 | 2.9% | the hindquarters of a lone mouse heading right | plausible | lone mouse, one heading |
| #2454 | 5 | 2.4% | the rear / back of a compact, hunched mouse | plausible | possibly self-grooming (hunched) |
| #2953 | 5 | 1.5% | nest paper / bedding next to resting mice in the bottom-left corner | plausible | nest material / huddle |
| #2293 | 5 | 0.8% | the white shave mark on a mouse's head (single token) | clear | identity mark |
| #3692 | 5 | 0.5% | the tail base of mice sitting at the left-wall grid | clear | tail + hopper |
| #3273 | 5 | 3.2% | a tail next to a resting / curled mouse, often near the nest | plausible | tail |
| #3009 | 5 | 2.7% | snout at another mouse's rear / tail base (same sheet as the pick above); 10/12 labelled nose-tail | clear | nose-tail (the sgap pick) |
| #2828 | 5 | 2.1% | a floor token near the near (bottom) wall, often with no mouse under it | plausible | artefact / position |
| #3548 | 5 | 10.2% | a tail lying across a white wooden block | clear | tail on the block |

Mice flags summary: bag / box sniffing #511 (+ bag-corner position #389, #1134); huddle #733, #1177, #2953 (resting,
next to the nest); tail #323, #348, #389, #513, #1593, #1496, #1944, #3586, #3273, #3548, and tail bases at the
hopper #1791, #3692. None of these is tail CONTACT; the only snout-at-tail neuron is #3009. Rearing / wall #812, #1571, #3856; possible
self-grooming (hunched posture only) #810, #3587, #2454, #1134; hopper #448 (+ #1791, #3692); spout #144, #248, #504,
#925; nest material #234, #1773, #2953; identity marks #698, #2293, #867; lone-mouse locomotion / heading #143, #499,
#2659; artefacts #416 (wall reflection), #2828 (floor). No nose-nose neuron among the 40 (#3593 was excluded as a
pick).

### Catalogue: ants PREDk s2703 (40 neurons)

Same selection as the PRED26 catalogues (picked neuron #33 excluded). 2,756 candidates; level 1 has 0 (every L1 neuron
fires on > 20% of frames), so the 40 are 8 L2, 8 L3, 16 L4, 8 L5. No catalogue neuron has raw-code held-out AUROC above
0.59 for grooming-to-focal (best #1720 0.590). Lab = tiles labelled grooming-to-focal (any), out of 12.

| neuron | level | firing | what the top frames show | confidence | flag |
|---|---|---|---|---|---|
| #206 | 2 | 8.4% | a single token on empty floor a short way from an ant (no body part under it) | plausible | artefact: floor token near an ant |
| #143 | 2 | 14.5% | a token ON the dish rim at the left / right edge, no ant under it | clear | artefact: rim |
| #153 | 2 | 14.3% | the near (bottom) rim directly below an ant standing at the bottom wall | clear | wall (position: bottom rim) |
| #234 | 2 | 20.0% | a lone UNPAINTED ant (the focal), compact, heat over its whole body; 0/12 lab | plausible | focal identity (no paint); possibly self-grooming (compact posture, legs not resolved) |
| #149 | 2 | 17.7% | an ant on the far (top) rim / wall, foreshortened | clear | climbing / on the wall |
| #207 | 2 | 12.0% | the head / thorax of an ant, often next to another ant | plausible | body part: head |
| #215 | 2 | 16.4% | a blurred, out-of-focus ant on the left dish wall; big heat over the blur | clear | climbing the wall (out of focus) |
| #164 | 2 | 17.7% | the head of an ant (front of the body, antenna base) | plausible | body part: head / antenna |
| #352 | 3 | 10.1% | head and antennae of an ant touching the dish wall | plausible | antenna on the wall |
| #290 | 3 | 14.8% | a token just ahead of an ant's head (antenna tips on the floor) | plausible | antenna movement |
| #398 | 3 | 19.6% | a compact, foreshortened ant (seen head-on / curled), heat over the small body | plausible | possibly self-grooming (curled posture) |
| #510 | 3 | 13.6% | a single token on a painted nestmate's body (gaster / paint region), mixed | unclear | paint dot? |
| #335 | 3 | 8.7% | the front end of a lone ant (head and front legs), compact | plausible | possibly self-grooming (front legs) |
| #434 | 3 | 6.5% | the near (bottom) rim under the head / gaster of an ant at the wall | clear | wall (bottom rim) |
| #345 | 3 | 12.1% | ants at the bottom rim, heat on the ant against the wall | clear | wall (bottom rim) |
| #391 | 3 | 16.2% | the head end of a walking ant | plausible | walking / head |
| #660 | 4 | 1.7% | a single token where an ant's antenna tip touches the dish wall (rim) | clear | antenna on the wall |
| #538 | 4 | 17.4% | a token at the head / antenna of an ant among other ants | unclear | antenna? |
| #1331 | 4 | 7.9% | the floor just ahead of a lone walking ant's head (antenna sweep), mostly blue-painted ants | clear | antenna movement |
| #1399 | 4 | 3.8% | a faint / ghost ant (motion blur or out of focus near the wall) | clear | fast walking / blur |
| #1780 | 4 | 0.9% | a far-rim token next to the head of an ant at the top wall | clear | antenna / head on the wall |
| #958 | 4 | 8.9% | the head of an ant close to another ant or the wall | unclear | contact? |
| #2017 | 4 | 3.1% | a single floor token ahead of an ant's head (antenna tip) | plausible | antenna movement |
| #1957 | 4 | 8.4% | a rim token where an ant's antennae touch the dish wall | clear | antenna on the wall |
| #908 | 4 | 2.4% | 2-3 ants packed against the dish wall, heat over the contact; 9/12 lab | plausible | contact at the wall (grooming-like) |
| #1245 | 4 | 1.1% | a rim token next to an ant's antennae (right wall); low codes after tile #1 | plausible | antenna on the wall |
| #1720 | 4 | 14.5% | the ORANGE ("yellow") paint dot of a nestmate in a chain of ants, the dot touching another ant's head; 11/12 lab | clear | paint dot (yellow) + contact |
| #565 | 4 | 16.2% | the focal's body at the centre of a 2-3 ant group; 11/12 lab | plausible | contact (grooming-like) |
| #1461 | 4 | 13.8% | the gaster end of an ant, often with a pinkish paint mark; 0/12 lab | plausible | body part: gaster / paint |
| #1980 | 4 | 12.9% | an ant with its head or gaster pressed against the near / right rim | plausible | wall |
| #1563 | 4 | 1.4% | a token between ants in a loose 3-ant group; 7/12 lab | plausible | contact / proximity |
| #783 | 4 | 8.3% | the front of a single ant (head and front legs), compact | plausible | possibly self-grooming |
| #3388 | 5 | 1.8% | the TIP of a stretched antenna (single token), often near the wall | clear | antenna movement |
| #3843 | 5 | 3.1% | a single floor token just ahead of a walking ant's head (antenna tip); 0/12 lab | clear | antenna movement |
| #2099 | 5 | 3.9% | a near-rim token below an ant at the wall | plausible | wall |
| #3650 | 5 | 7.6% | the gaster tip of the orange-painted nestmate (paint dot), often touching another ant; 6/12 lab | plausible | paint dot (yellow) |
| #4002 | 5 | 2.3% | head / antennae of an ant facing the far (top) rim | clear | antenna / head on the wall |
| #3943 | 5 | 0.9% | a token near the legs / gaster of ants in contact; low codes; 6/12 lab | unclear | contact? |
| #2805 | 5 | 4.3% | a leg of a walking ant (single token at the leg) | plausible | walking (leg) |
| #2178 | 5 | 1.2% | a middle leg of a lone ant standing in the open; low codes | plausible | walking / standing (leg) |

Ants flags summary: antenna movement #290, #1331, #2017, #3388, #3843 (+ #538 unclear); antenna / head on the wall
#352, #660, #1245, #1780, #1957, #4002; wall / climbing / rim position #149, #153, #215, #345, #434, #1980, #2099;
walking / blur #1399, #391, #2805, #2178; possible self-grooming #398, #335, #783, #234 (posture only, no paws resolved);
contact (grooming-like) #908, #565, #1563 (+ #958, #3943 unclear); paint dots #1720 (yellow, at a contact), #3650
(yellow), #1461 (gaster / pinkish mark), #510 unclear; artefacts #143 (rim), #206 (floor). No blue-paint neuron this
time (PRED26 had one, #3620). As in PRED26, no neuron separates grooming from plain contact.
