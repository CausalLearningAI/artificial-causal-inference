"""Render results/vision/benchmark_calms21/inventory/inventory.json (+ plan/eval_plan.json, mask/maskqa.json when present)
to inventory.md (login safe: reads small JSONs only)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN  # noqa: E402

I = json.loads((OUT / 'inventory' / 'inventory.json').read_text())
L = ['# CalMS21 inventory', '', 'Source: CaltechDATA s0vdx-0k302 (CC BY 4.0); data at dataset/benchmark/calms21 (BeeGFS).', '',
     '## Files (size + md5 verified against the record API)', '', '| file | bytes | md5 verified |', '|---|---|---|']
for k, v in I['files'].items():
    if v['verified_marker']:
        L.append(f"| {k} | {v['record_size']:,} | yes |")
L += ['', 'Unzip: ' + '; '.join(f"{k}: {v['members']} members, bad {len(v['bad'])}" for k, v in I['unzip'].items()
                               if isinstance(v, dict)), '', '## Videos (task 1, the only videos in the record)', '',
      '| split | videos | width x height | fps | codec | frames total | frames min / median / max | duration total (h) | '
      'median duration (min) |', '|---|---|---|---|---|---|---|---|---|']
for s, v in I['videos'].items():
    L.append(f"| {s} | {v['n']} | {v['width']} x {v['height']} | {v['avg_frame_rate']} | {v['codec']} {v['pix_fmt']} | "
             f"{v['frames_total']:,} | {v['frames_min']:,} / {v['frames_median']:,.0f} / {v['frames_max']:,} | "
             f"{v['duration_s_total'] / 3600:.2f} | {v['duration_min_median']:.1f} |")
L += ['', '## Task 1 labels (frame-level, every frame annotated, one label per frame)', '']
for s, t in I['task1'].items():
    L += [f"**{s}**: {t['n_seqs']} sequences, {t['frames']:,} frames, label values {t['label_values']}, vocab "
          f"{t['vocab']}, annotators {t['annotator_ids']}; keypoint x range {[round(x) for x in t['kp_x_range']]}, "
          f"y range {[round(x) for x in t['kp_y_range']]}; video frames - keypoint frames per sequence: "
          f"{t['video_minus_kp_frames']}; sequences without video {t['seqs_without_video']}", '',
          '| label | frames | positive rate | bouts | bout length median / mean / p90 (frames) | sequences with a positive |',
          '|---|---|---|---|---|---|']
    for b, x in t['labels'].items():
        L.append(f"| {b} | {x['frames']:,} | {x['pos_rate']:.4f} | {x['bouts']:,} | {x['bout_len_median']} / "
                 f"{x['bout_len_mean'] and round(x['bout_len_mean'], 1)} / {x['bout_len_p90']} | "
                 f"{x['seqs_with_pos']} / {t['n_seqs']} |")
    L.append('')
L += ['## Tasks 2 and 3: which task-1 video does each sequence come from?', '',
      'Keypoint fingerprint (nose + neck of both mice, 0.01 px) at 64 evenly spaced frames looked up in all task-1 '
      'frames; a sequence is matched when >= 48 of 64 fingerprints hit one task-1 sequence at one constant offset.', '',
      '| task | group | split | sequences | frames | positives | matched to a task1 TRAIN video | to a task1 TEST video |',
      '|---|---|---|---|---|---|---|---|']
for task, G in I['tasks23'].items():
    for g, S in G.items():
        for s, rows in S.items():
            pos = {}
            for r in rows.values():
                for k, v in (r['pos'] or {}).items():
                    pos[k] = pos.get(k, 0) + v
            mt = [r['match'] for r in rows.values() if r['match'] and r['match'][3] >= 48]
            L.append(f"| {task} | {g} | {s} | {len(rows)} | {sum(r['frames'] for r in rows.values()):,} | "
                     f"{ {k: v for k, v in pos.items() if k != 'other'} } | {sum(m[0] == 'train' for m in mt)} | "
                     f"{sum(m[0] == 'test' for m in mt)} |")
A = I['alignment']
L += ['', '## Label / keypoint index -> video frame', '',
      f"Rule: video frame = keypoint / label index + **{A['offset']}**. Mean grey (512 x 285 grey decode) under the "
      'keypoints of MOVING mice (> 3 px / frame), 8 videos, first 9,000 frames:', '',
      '| d | ' + ' | '.join(str(d) for d in A['mean_grey_under_moving_black_kp']) + ' |',
      '|---|' + '---|' * len(A['mean_grey_under_moving_black_kp']),
      '| resident (black) | ' + ' | '.join(f'{v:.1f}' for v in A['mean_grey_under_moving_black_kp'].values()) + ' |',
      '| intruder (white) | ' + ' | '.join(f'{v:.1f}' for v in A['mean_grey_under_moving_white_kp'].values()) + ' |',
      '', f"Darkest d per video (black mouse): {A['per_video_darkest_black']}; brightest d per video (white mouse): "
      f"{A['per_video_brightest_white']}; white-mouse best d agrees: {A['agree']}.", '']
ap = [(q, v['floor_bg_median_grey'], v['resident_black_grey_under_kp_median'], v['intruder_white_grey_under_kp_median'])
      for q, v in A['videos'].items()]
L += ['Appearance (grey 0-255): ' + '; '.join(f'{q}: floor {f:.0f}, black mouse {b:.0f}, white mouse {w:.0f}'
                                             for q, f, b, w in ap), '',
      'Examples: ' + ', '.join(f"examples/{e['label']}_{e['seq']}_kp{e['kp_index']}.jpg (bout {e['bout_len']} frames)"
                               for e in I['examples']), '']
if (PLAN / 'eval_plan.json').exists():
    E = json.loads((PLAN / 'eval_plan.json').read_text())
    L += ['## Frame plans (eci_plan.py)', '', f"SAE pool (label-free): task1 TRAIN videos, {E['train']['n_train']:,} "
          f"train + {E['train']['n_calib']:,} calibration frames from {E['train']['n_videos']} videos (uniform over "
          f"{E['train']['grid_population']:,} grid frames). Eval: task1 TEST, {E['n_frames']:,} frames from "
          f"{E['n_videos']} videos, strata {E['strata']}; scored labels {E['scored_labels']}.", '',
          '| label | weighted base rate | population rate | positives in sample | videos with a positive |',
          '|---|---|---|---|---|']
    for b in E['n_pos']:
        L.append(f"| {b} | {E['weighted_base_rate'][b]:.4f} | {E['population_base_rate'][b]:.4f} | {E['n_pos'][b]} | "
                 f"{E['n_videos_with_pos'][b]} |")
(OUT / 'inventory' / 'inventory.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
