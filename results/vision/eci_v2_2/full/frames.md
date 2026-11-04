| frame set | pools | videos | frames (per video min/median/max) | nose-nose (nose_nose): positive frames / bouts / videos with >= 1 bout | nose-nose-mutual (nn): positive frames / bouts / videos with >= 1 bout | nose-nose-passive (np): positive frames / bouts / videos with >= 1 bout | nose-tail (nt): positive frames / bouts / videos with >= 1 bout |
|---|---|---|---|---|---|---|---|
| old_2min|all | 24 | 144 | 86,400 (600/600/600) | 1,852 / 541 / 109 | 608 / 232 / 79 | 1,248 / 312 / 88 | 728 / 121 / 55 |
| old_2min|sg | 11 | 66 | 39,600 (600/600/600) | 1,030 / 314 / 56 | 342 / 136 / 41 | 692 / 181 / 46 | 345 / 82 / 32 |
| old_2min|nonsg | 12 | 72 | 43,200 (600/600/600) | 718 / 188 / 48 | 235 / 82 / 34 | 483 / 106 / 37 | 370 / 34 / 20 |
| old_2min|unknown | 1 | 6 | 3,600 (600/600/600) | 104 / 39 / 5 | 31 / 14 / 4 | 73 / 25 / 5 | 13 / 5 / 3 |
| full|all | 24 | 144 | 648,000 (4500/4500/4500) | 13,463 / 3,778 / 135 | 4,661 / 1,679 / 130 | 8,813 / 2,116 / 133 | 6,385 / 1,058 / 119 |
| full|sg | 11 | 66 | 297,000 (4500/4500/4500) | 7,896 / 2,259 / 64 | 2,977 / 1,020 / 61 | 4,930 / 1,251 / 64 | 3,426 / 678 / 57 |
| full|nonsg | 12 | 72 | 324,000 (4500/4500/4500) | 4,568 / 1,278 / 66 | 1,568 / 603 / 64 | 3,000 / 680 / 64 | 2,696 / 331 / 57 |
| full|unknown | 1 | 6 | 27,000 (4500/4500/4500) | 999 / 241 / 5 | 116 / 56 / 5 | 883 / 185 / 5 | 263 / 49 / 5 |

```
{
 "rows_outside_window": 1944000,
 "rows_outside_window_with_any_label_value": 216000,
 "videos_with_labels_not_in_window_set": [],
 "nose_nose": {
  "bouts_annotations_csv": 5205,
  "bouts_fully_inside": 3777,
  "bouts_fully_outside": 1427,
  "bouts_straddling_boundary": 1,
  "positive_frames_outside": 4806,
  "positive_frames_outside_H_before_4500": 4806
 },
 "nn": {
  "bouts_annotations_csv": 2430,
  "bouts_fully_inside": 1679,
  "bouts_fully_outside": 751,
  "bouts_straddling_boundary": 0,
  "positive_frames_outside": 1855,
  "positive_frames_outside_H_before_4500": 1855
 },
 "np": {
  "bouts_annotations_csv": 2805,
  "bouts_fully_inside": 2115,
  "bouts_fully_outside": 689,
  "bouts_straddling_boundary": 1,
  "positive_frames_outside": 2959,
  "positive_frames_outside_H_before_4500": 2959
 },
 "nt": {
  "bouts_annotations_csv": 1600,
  "bouts_fully_inside": 1058,
  "bouts_fully_outside": 542,
  "bouts_straddling_boundary": 0,
  "positive_frames_outside": 2647,
  "positive_frames_outside_H_before_4500": 2647
 }
}
```
