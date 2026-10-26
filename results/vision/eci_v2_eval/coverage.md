# Label coverage audit (2026-10-26)

This is an audit, not a new pre-registration. The counts come from scripts/eci/v2_eval/coverage.py, which writes coverage_counts.json. "Bouts" are runs of consecutive positive 5 fps frames inside one video. "Halves" are the two halves of the atlas split.

## User's question: why is there no yellow-on-lid or focal-on-lid score on ants?

The annotations do exist, but there are very few lid events:

- **Focal on lid (FOL).** Ants v3 has 3 bouts in 3 videos (1,069 positive 5 fps frames out of 321,000). v4 and v4_infested have 0. That is too few to score.
- **Yellow on lid (YOL).** Ants v3 has 23 bouts in 12 videos. v4 and v4_infested have 0. The annotations cover only the first 10 min of each video (3,000 5 fps frames).
- **Blue on lid (BOL).** v3 has 15 bouts in 7 videos, v4 has 2 bouts, v4_infested has 1.

Until now, no scorer reported the lid colours one by one:

- The release / diag scorer computes yellow / blue / focal lid labels but reports only the pooled `on_lid_any`. See scripts/eci/v2/freeze.py:123 and 537-538, and scripts/eci/v2_mask2/release_ab.py:66 (ACELLS).
- The v2_prob metric audit hard-codes 5 cells, with `on_lid_any` = YOL | BOL. See scripts/eci/v2_prob/analysis.py:134-137.
- The atlas page scores YOL and BOL but has no FOL entry. See scripts/eci/atlas_align.py:50-51.

So the answer is: yes, all three are annotated. FOL is too rare to score. YOL is now scored (eval.py), but it fails the eligibility rule: one half of the atlas split holds only 3 positive videos and 10 bouts. Its numbers are in table.md under "below the rule".

## Eligibility rule

The coordinator gave the rule ">= 20 bouts AND >= 3 positive videos in EACH half of the atlas split". It did not exist in code. The rules that did exist were:

- atlas_align.py:54: >= 100 labelled and >= 10 positive frames.
- v2_prob and release_ab: fixed cell lists, no count rule.

eval.py now applies the stated rule, counted on each row's scored frames. Under this rule, the old Tier B cell "ants v3 on_lid_any" (6 bouts, 3 videos per half on the 1-min probe windows) would NOT have been eligible. The same holds for its v2_prob / release numbers (scored on 1-min windows).

## Every annotated behaviour

Counts are on all labelled frames. "Scored frames" means: mice = Tier B probe, 144 x 2 min at 5 fps; ants = labelled 1 fps frames (full1), 600 per video.

| domain / series | behaviour | positive 5 fps frames (all) | bouts (all) | positive videos | bouts per half on the scored frames | measured before? | measured now? |
|---|---|---|---|---|---|---|---|
| mice v1 | nn (mutual) | 4,661 | 1,679 | 130 / 144 | 132 / 100 | R~ only, release tables (reported, not deciding) | yes, eligible |
| mice v1 | np (directional) | 8,813 | 2,116 | 133 | 205 / 107 | R~ only, release tables | yes, eligible |
| mice v1 | nose_nose = nn or np | 13,463 | 3,778 | 135 | 334 / 207 | yes (all scorers) | yes, eligible |
| mice v1 | nt | 6,385 | 1,058 | 119 | 67 / 54 | yes | yes, eligible |
| ants v2 | Y2F (groomed by yellow) | 17,711 | 131 | 32 / 44 | 59 / 61 | atlas page only | yes, eligible |
| ants v2 | B2F | 14,914 | 110 | 36 | 45 / 55 | atlas page only | yes, eligible |
| ants v2 | groom_any | 29,488 | 201 | 41 | 81 / 99 | yes | yes, eligible |
| ants v2 | FOL / YOL / BOL | not annotated in v2 (column empty) | - | - | - | - | not possible |
| ants v3 | Y2F | 63,030 | 420 | 83 / 107 | 178 / 173 | atlas page only | yes, eligible |
| ants v3 | B2F | 60,975 | 356 | 81 | 148 / 169 | atlas page only | yes, eligible |
| ants v3 | groom_any | 105,832 | 602 | 97 | 241 / 275 | yes | yes, eligible |
| ants v3 | FOL | 1,069 | 3 | 3 | 1 / 2 | never (no FOL entry in atlas_align.py:50; pooled only in freeze.py:537) | scored, below the rule |
| ants v3 | YOL | 8,772 | 23 | 12 | 13 / 10 (3 positive videos in one half) | atlas page only | scored, below the rule |
| ants v3 | BOL | 2,103 | 15 | 7 | 2 / 13 | atlas page only | scored, below the rule |
| ants v3 | onlid_any = YOL or BOL | 10,875 | 38 | 15 | 15 / 23 | yes (v2_prob, Tier B: 6 bouts) | scored, below the rule |
| ants v3 | lid_any3 = FOL, YOL or BOL | 11,887 | 39 | 15 | 16 / 23 | yes (release / diag 'on_lid_any', freeze.py:537) | scored, below the rule |
| ants v4 | Y2F | 37,727 | 267 | 36 / 40 | 114 / 141 | atlas page only (prefix 128/256) | yes, eligible |
| ants v4 | B2F | 36,563 | 216 | 34 | 84 / 128 | atlas page only | yes, eligible |
| ants v4 | groom_any | 58,594 | 308 | 39 | 142 / 152 | atlas page only; release/diag NOT (no step-2 store, release_ab.py:26); v2_prob NOT (study.py:10 'v4 not used (cost)') | yes, eligible |
| ants v4 | BOL / onlid | 945 | 2 | 1 | 0 / 2 | atlas page | not scorable (one half has no positive) |
| ants v4 | FOL, YOL | 0 | 0 | 0 | - | - | nothing to score |
| ants v4_infested | Y2F | 13,946 | 68 | 16 / 21 | 29 / 36 | atlas page (pooled with v4) | yes, eligible |
| ants v4_infested | B2F | 11,896 | 55 | 14 | 23 / 32 | atlas page (pooled with v4) | yes, eligible |
| ants v4_infested | groom_any | 21,591 | 90 | 18 | 42 / 45 | atlas page (pooled with v4) | yes, eligible |
| ants v4_infested | BOL | 308 | 1 | 1 | 0 / 1 | atlas page | not scorable |

## Omissions found and fixed

1. **Per-colour grooming (Y2F, B2F) and v4 / v4_infested were never part of the v1-vs-v2 decision scorers.** Cause: fixed cell lists (analysis.py:134, release_ab.py:66, diag.py:961) and no v4 token store (study.py:10, release_ab.py:26). Fix: eval.py scores all of them on the labelled 1 fps frames, using the existing full1 N2 tokens. The only new compute was a v1 token encode (0.76 GPU-h).
2. **FOL was never scored on its own anywhere.** atlas_align.py:50-51 has no FOL entry. It is now scored, but it has only 3 bouts.
3. **`on_lid_any` means two different things.** In freeze.py:537 (release / diag tables) it is YOL | BOL | FOL. In analysis.py:137 and atlas_align.py:51 (v2_prob, atlas) it is YOL | BOL. The difference is small (1 extra bout, about 1,000 frames). Both versions are now reported: lid_any3 and onlid_any.
4. **The Tier B lid cell has too few events.** On the 1-min v3 probe windows it rests on 6 bouts (3 positive videos per half). The earlier "v2 beats v1 on on_lid_any" rows (v2_prob A3, release tables) should be read with that in mind.
