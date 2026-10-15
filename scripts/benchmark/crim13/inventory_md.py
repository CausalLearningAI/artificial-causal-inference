"""Render results/vision/benchmark_crim13/inventory/inventory.json as inventory.md (tables, alignment, body length,
and the pre-registered behaviour-selection rules applied with halves = the released train / test videos).
usage: python -I inventory_md.py <inventory dir>"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np

D = sys.argv[1]
inv = json.load(open(os.path.join(D, 'inventory.json')))
V = inv['videos']
L = ['# CRIM13 inventory (top view)', '',
     f"Source: {inv['source']}; data at dataset/benchmark/crim13 (BeeGFS). Measured from the files by "
     'scripts/benchmark/crim13/inventory.py unless marked "from paper" / "from record".', '',
     '## Files (size + md5 verified against the record API)', '', '| file | bytes | downloaded | md5 verified |',
     '|---|---|---|---|']
for f in inv['files']:
    L.append(f"| {f['key']} | {f['bytes']:,} | {'yes' if f['downloaded'] else 'no (features, skipped)'} | "
             f"{'yes' if f['md5_verified'] else '-'} |")
L += ['', 'Extraction (python zipfile, CRC checked, zip deleted after a clean extraction):', '']
for z, e in inv['extraction'].items():
    L.append(f"- {z}: {e['members']} members, {e['extracted']} extracted, {e['skipped']} skipped "
             f"({e['skipped_bytes'] / 1e9:.1f} GB; skip rule `{e['skip_regex']}`), bad {len(e['bad'])}, "
             f"zip deleted {e['deleted_zip']}")
vs = inv['validation_seq_vs_main']
L += ['', f"validation.zip videos: {vs['identical']} / {vs['n']} .seq members have the same size and CRC32 as a "
      f"member of the train/test zips (missing from main: {vs['missing_in_main']}), so its videos were not kept.", '']

L += ['## Splits', '']
for sp in ['train', 'test']:
    s = inv['splits'][sp]
    di = s.get('dinfo', {})
    L.append(f"- **{sp}**: {s['top_seq']} top-view .seq, {s['label_txt']} label files; label without video "
             f"{s['txt_without_seq']}, video without label {s['seq_without_txt']}; dinfo.mat lists "
             f"{di.get('num_videos')} videos, not on disk: {di.get('not_on_disk')}")
for sub in ['train', 'test']:
    same = inv['splits'][f'validation/{sub}_vs_main']
    L.append(f"- **validation/{sub}**: {len(same)} videos, from main splits {dict(Counter(x[1] for x in same))}; "
             f"label file byte-identical to the main copy: {sum(x[2] for x in same)} / {len(same)}")
L.append('')

L += ['## Videos (top view)', '', '| split | videos | width x height | header fps | fps from timestamps (median) | '
      'frames min / median / max | hours | frames == track == seek.mat | timestamp gaps > 1.5x |',
      '|---|---|---|---|---|---|---|---|---|']
tot = 0
for sp in ['train', 'test']:
    vv = [v for v in V if v['split'] == sp]
    fr = [v['frames'] for v in vv]
    h = sum(v['ts']['duration_s'] for v in vv) / 3600
    tot += h
    wh = Counter(f"{v['header']['width']}x{v['header']['height']}" for v in vv)
    eq = sum(v['frames'] == v['track']['T'] and v.get('seek_mat', {}).get('equal_to_walk', False) for v in vv)
    L.append(f"| {sp} | {len(vv)} | {dict(wh)} | {np.median([v['header']['fps_header'] for v in vv]):.2f} "
             f"({min(v['header']['fps_header'] for v in vv):.2f}-{max(v['header']['fps_header'] for v in vv):.2f}) | "
             f"{np.median([v['ts']['fps_from_ts'] for v in vv]):.2f} | {min(fr):,} / {int(np.median(fr)):,} / {max(fr):,} | "
             f"{h:.2f} | {eq} / {len(vv)} | {sum(v['ts']['gaps_gt_1p5x'] for v in vv)} in "
             f"{sum(v['ts']['gaps_gt_1p5x'] > 0 for v in vv)} videos |")
L += ['', f'Total: {tot:.2f} h (timestamps). Codec: JPEG frames in Norpix .seq (image format '
      f"{sorted({v['header']['image_format'] for v in V})}, 102 = jpg), 8-bit grayscale.", '']
mism = [v['name'] for v in V if v['frames'] != v['track']['T'] or not v.get('seek_mat', {}).get('equal_to_walk', False)]
L += [f'Videos where frames != track length or seek.mat != walked offsets: {mism}', '']

L += ['## Label files', '',
      '- `<name>_t.txt`, "Caltech Behavior Annotator - Annotation File": a configuration block (behaviour name + '
      'key) then streams `S1:`, `S2:` ... of `start end type` rows, 1-based inclusive frame indices. '
      'CRIM13GetInfo.m reads them with `behaviorData(...).getLbls()` into a 1 x nFrames vector, one label per frame. '
      'CodeExamples.m reads frame i (1-based) as `seek(i - 1)`: label frame L <-> 0-based frame L - 1.',
      '- `<name>_t-track.mat`: `Y`, 1 x 1 x nFrames cell; each cell 0, 1 or 2 rows of (x, y); row 1 = resident, '
      'row 2 = intruder (CRIM13GetInfo.m); empty when the intruder is absent or the tracker failed.',
      '- `<name>_t-seek.mat`: byte offset of every frame record in the .seq.', '']
voc = Counter(tuple(v['vocab']) for v in V)
st = Counter(tuple(sorted(v['streams'])) for v in V)
L.append(f"- vocabularies: {dict((', '.join(k), n) for k, n in voc.items())}")
L.append(f"- streams per file: {dict((', '.join(k), n) for k, n in st.items())}")
unc = sum(v['labels']['frames_uncovered'] for v in V)
mul = sum(v['labels']['frames_multiply_covered'] for v in V)
L.append(f"- dense: frames not covered by any S1 interval {unc:,} (in {sum(v['labels']['frames_uncovered'] > 0 for v in V)} videos); "
         f"frames covered twice {mul:,}; intervals past the last video frame {sum(v['labels']['beyond_video'] for v in V)}; "
         f"first labelled frame values {dict(Counter(v['labels']['first'] for v in V))}; "
         f"last labelled frame - video frames: {dict(Counter(v['labels']['last'] - v['frames'] for v in V))}; "
         f"unknown names {sorted({x for v in V for x in v['labels']['unknown_names']})}")
L.append('')

behs = V[0]['vocab']
L += ['## Per behaviour (S1 stream; one label per frame)', '', '| behaviour | split | frames | positive rate | bouts | '
      'bout length median (frames) | videos with a positive |', '|---|---|---|---|---|---|---|']
RULE = {}
for b in behs:
    for sp in ['train', 'test']:
        vv = [v for v in V if v['split'] == sp]
        fr = sum(v['per_beh'][b]['frames'] for v in vv)
        n = sum(v['frames'] for v in vv)
        bo = sum(v['per_beh'][b]['bouts'] for v in vv)
        ln = [x for v in vv for x in v['per_beh'][b]['lens']]
        mv = sum(v['per_beh'][b]['frames'] > 0 for v in vv)
        RULE[(b, sp)] = (bo, mv)
        L.append(f"| {b} | {sp} | {fr:,} | {fr / n:.4f} | {bo:,} | {np.median(ln) if ln else '-'} | {mv} / {len(vv)} |")
L.append('')

L += ['## Animals and arena (measured where stated)', '']
res = Counter(re.match(r'\d+_(A\d+)_', v['name']).group(1) if re.match(r'\d+_(A\d+)_', v['name']) else '?' for v in V)
L.append(f"- file names `<date>_<A##>_Block<k>_<intruder>`: {len(res)} distinct `A##` codes {dict(res)}")
for sp in ['train', 'test']:
    vv = [v for v in V if v['split'] == sp]
    codes = dict(Counter(re.match(r'\d+_(A\d+)_', v['name']).group(1) for v in vv))
    L.append(f"  - {sp}: A codes {codes}")
intr = Counter(re.sub(r'\d+$', '', v['name'].split('_')[-1]) for v in V)
L.append(f"- intruder field (digits stripped): {dict(intr)}")
L.append(f"- track frames with 0 / 1 / 2 mice: {sum(v['track']['frames_0'] for v in V):,} / "
         f"{sum(v['track']['frames_1'] for v in V):,} / {sum(v['track']['frames_2'] for v in V):,}")
fg = [v['fg_at_track'] for v in V if v['fg_at_track']['n']]
L.append(f"- |frame - background| at the track positions, median over videos {np.median([f['median_absdiff_at_track'] for f in fg]):.1f} "
         f"vs at random pixels {np.median([f['median_absdiff_random'] for f in fg]):.1f} (tracks sit on moving foreground)")
L.append('')

L += ['## Alignment', '', '(b) video vs track: Pearson r between track speed at frame t and motion energy '
      '(mean |I_t - I_t-1|, 1/4-size decode) at frame t + lag, per video:', '']
lags = sorted(int(k) for k in V[0]['align_track_vs_video']['r_by_lag'])
rr = {l: [v['align_track_vs_video']['r_by_lag'][str(l)] for v in V if v['align_track_vs_video']['r_by_lag'][str(l)] is not None]
      for l in lags}
L.append('| lag | ' + ' | '.join(str(l) for l in lags) + ' |')
L.append('|---|' + '---|' * len(lags))
L.append('| median r | ' + ' | '.join(f'{np.median(rr[l]):.3f}' for l in lags) + ' |')
L.append('')
L.append(f"best lag per video: {dict(Counter(v['align_track_vs_video']['best_lag'] for v in V))}")
L += ['', '(c) labels vs video: bout-onset-triggered mean motion energy, label start L read as 0-based frame L - 1:', '']
for b in behs:
    n = sum(v['eta_sum'][b][0] for v in V if b in v['eta_sum'])
    if not n:
        continue
    s = np.sum([v['eta_sum'][b][1] for v in V if b in v['eta_sum']], 0) / n
    d = np.diff(s)
    k = int(np.argmax(np.abs(d)))
    L.append(f"- {b} ({n} onsets): mean at offsets -3..+3 " + ', '.join(f'{x:.2f}' for x in s[12:19])
             + f"; steepest change between offsets {k - 15} and {k - 14}")
L.append('')

bm = [x for v in V for x in v['body']['major']]
ar = [x for v in V for x in v['body']['area']]
L += ['## Body length (native pixels, label-free)', '',
      f"Major axis of the foreground blob at the resident's track position, on frames with exactly one tracked mouse: "
      f"n = {len(bm)} blobs from {sum(v['body']['n'] > 0 for v in V)} videos, median {np.median(bm):.1f} px "
      f"(IQR {np.percentile(bm, 25):.1f}-{np.percentile(bm, 75):.1f}); blob area median {np.median(ar):.0f} px^2. "
      'NOT USABLE as a body length: this |I - bg| blob merges the mouse with its head cable, the dark cage walls and '
      'moved bedding (example frames show the resident at roughly 100-120 px nose to tail base).', '']
bp = os.path.join(D, 'bodylen.json')
if os.path.exists(bp):
    b = json.load(open(bp))['summary']
    L += ['Second pass (scripts/benchmark/crim13/bodylen.py: pixels darker than the background by > 40 inside the '
          'pale arena, 11 x 11 opening to cut tail and cable, the component under the resident\'s track point, frames '
          f"with one tracked mouse): {b['blobs']} blobs from {b['videos_with_blobs']} / {b['videos']} videos; major "
          f"axis 4 sqrt(l1) median {b['major_median']:.1f} px (IQR {b['major_iqr'][0]:.1f}-{b['major_iqr'][1]:.1f}); "
          f"extent along the major axis median {b['extent_median']:.1f} px; per-video medians IQR "
          f"{b['per_video_major_median_iqr'][0]:.1f} / {b['per_video_major_median_iqr'][1]:.1f} / "
          f"{b['per_video_major_median_iqr'][2]:.1f} px. Checked against trajectories only in that the blob must sit "
          'on the track point (the tracks carry positions, no size).', '']

L += ['## Pre-registered behaviour rules (halves = released train / test videos)', '',
      'Rule: frame-level and dense; >= 20 positive bouts AND >= 10 videos with a positive in EACH half; drop '
      '"other" and proximity supersets and condition labels.', '',
      '| behaviour | train bouts / videos | test bouts / videos | passes counts | kept |', '|---|---|---|---|---|']
for b in behs:
    (a, m), (c, n) = RULE[(b, 'train')], RULE[(b, 'test')]
    ok = a >= 20 and m >= 10 and c >= 20 and n >= 10
    L.append(f"| {b} | {a} / {m} | {c} / {n} | {'yes' if ok else 'no'} | {'no (other)' if b == 'other' else ('yes' if ok else 'no')} |")
L.append('')
L += ['## Animals, appearance, arena (looked at the example frames)', '',
      '- Two mice: the resident (track row 1) is dark and carries a head-mounted cable; the intruder (row 2) is pale '
      'when the file name says BC and dark when it says C57 (seen in the example frames; BC = BALB/c, C57 = C57BL/6 '
      'is the usual reading of these codes, not stated in the files). A dark C57 intruder and the resident are told '
      'apart only by the cable. Intruder prefixes ana / cast / ovt are condition codes in the file name.',
      '- Rectangular home cage on pale bedding, top view, a water spout on the right; 640 x 480 or 656 x 490 px, 25 fps.',
      '- The intruder is absent at the start and end of a video (track frames with one mouse, above).', '',
      '## Proposed validation protocol', '',
      '- Cross-fitting unit: the video. Primary: the released halves as two folds. Fold A trains the SAE on the 103 '
      'train videos (labels ignored), picks each behaviour\'s neuron on the train labels, scores it on the 133 test '
      'videos; fold B swaps the roles. The four residents (A22 / A24 / A25 / A29) appear in both halves, so the '
      'released split is not animal-disjoint; sensitivity analysis: leave-one-resident-out (4 folds).',
      '- Frame-level target from the S1 stream, label frame L -> 0-based frame L - 1 (verified above); labels are '
      'dense and mutually exclusive (one label per frame, `other` fills the gaps), so each behaviour is scored one '
      'vs rest over all frames.',
      '- Candidate behaviour set (all pass the >= 20 bouts and >= 10 videos per half rule): approach, attack, '
      'copulation, chase, circle, drink, eat, clean, sniff, up, walk_away. Dropped: `other` (catch-all) and `human` '
      '(the experimenter\'s hand, an event of the protocol, not mouse behaviour). Flags: circle is the thinnest '
      '(95 / 98 bouts, 18 / 12 videos); approach and walk_away are defined by a change of distance, so a proximity '
      'neuron may partly align with them; eat and drink happen at fixed places (food, spout), so a position neuron '
      'may align with them.',
      '- Preprocessing must handle the two frame sizes (640 x 480 and 656 x 490).', '']
open(os.path.join(D, 'inventory.md'), 'w').write('\n'.join(L) + '\n')
print('wrote', os.path.join(D, 'inventory.md'))
