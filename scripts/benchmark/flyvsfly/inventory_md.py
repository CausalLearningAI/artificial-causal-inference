"""Render results/vision/benchmark_flyvsfly/inventory/inventory.json as inventory.md (tables + the pre-registered
behaviour-selection rules applied per context, halves = the released train / test movies).
usage: python -I inventory_md.py <inventory dir>"""
import json
import os
import sys

import numpy as np

D = sys.argv[1]
inv = json.load(open(os.path.join(D, 'inventory.json')))
L = ['# Fly-vs-Fly inventory', '',
     f"Source: {inv['source']}; data at dataset/benchmark/flyvsfly (BeeGFS). Everything below is measured from the "
     'files by scripts/benchmark/flyvsfly/inventory.py unless marked "from paper" / "from record".', '',
     '## Files (size + md5 verified against the record API)', '', '| file | bytes | md5 verified |', '|---|---|---|']
for f in inv['files']:
    L.append(f"| {f['key']} | {f['bytes']:,} | {'yes' if f['md5_verified'] else 'NO'} |")
L += ['', 'Extraction (python zipfile, CRC checked; `__MACOSX/` and `.DS_Store` skipped; zip deleted after a clean '
      'extraction): ' + '; '.join(f"{z}: {e['members']} members, {e['extracted']} extracted, {e['skipped']} skipped, "
                                   f"bad {len(e['bad'])}, zip deleted {e['deleted_zip']}" for z, e in inv['extraction'].items()), '']

L += ['## Videos', '', '| context | split | movies | width x height | fps | codec | frames per movie (min / median / max) '
      '| hours | track frames == video frames |', '|---|---|---|---|---|---|---|---|---|']
tot_h = 0
for ctx, C in inv['contexts'].items():
    for sp in ['train', 'test']:
        ms = [m for m in C['movies'].values() if m['split'] == sp]
        if not ms:
            continue
        pr = [m['probe'] for m in ms]
        fr = [p['frames'] for p in pr]
        h = sum(p['frames'] / eval(p['r_frame_rate']) for p in pr) / 3600
        tot_h += h
        wh = sorted({f"{p['width']}x{p['height']}" for p in pr})
        fps = sorted({p['r_frame_rate'] for p in pr})
        cod = sorted({f"{p['codec']}/{p['pix_fmt']}" for p in pr})
        tr = sum(m['track']['shape'][1] == m['probe']['frames'] for m in ms)
        L.append(f"| {ctx} | {sp} | {len(ms)} | {', '.join(wh)} | {', '.join(fps)} | {', '.join(cod)} | "
                 f"{min(fr):,} / {int(np.median(fr)):,} / {max(fr):,} | {h:.2f} | {tr} / {len(ms)} |")
L += ['', f'Total: {tot_h:.2f} h of video.', '']
L += ['Per movie: frames (video, demuxed packets) / track frames / last labelled frame / duration:', '']
for ctx, C in inv['contexts'].items():
    L.append(f"- **{ctx}**: " + '; '.join(
        f"{n} ({m['split']}) {m['probe']['frames']:,}/{m['track']['shape'][1]:,}/"
        f"{m['label_frame_range'][1] if m['label_frame_range'] else '-'} {m['probe']['duration_s'] / 60:.1f} min"
        for n, m in C['movies'].items()))
L.append('')

L += ['## Label files', '',
      '- `<movie>/<movie>_actions.mat`: `behs` (behaviour names) and `bouts`, a cell array (fly x behaviour); each '
      'cell is an n x 3 array `[start, end, c]`, frame indices 1-based and inclusive. The release\'s `visualizer.m` '
      'reads them this way: display frame `fr = frame + 1`, image `read(vidobj, fr)` (MATLAB 1-based, i.e. 0-based '
      'video frame `fr - 1`), track `trk.data(:, fr, :)`, labels `idxmap(fly, beh, start:end)`. So label frame L '
      '<-> 0-based video frame L - 1 <-> 0-based track index L - 1; the measured checks below test this.',
      '- Non-bout frames are negatives: the format is bout lists, so the label is dense over the annotated range '
      'iff the whole movie was annotated (record: "fully annotated by experts", from record).',
      '- `<movie>_track.mat`: `trk.data` (fly x frame x 17), `trk.names`, `trk.flag_frames`.',
      '- `traintest.mat` per context: `behs`, `movienames`, `trainids` / `testids` (1-based), `n_flies`.',
      '- `<movie>_second_actions.mat` on some movies: a second annotation (see "annotators").', '']
for ctx, C in inv['contexts'].items():
    t = C['traintest']
    m0 = next(iter(C['movies'].values()))
    col3 = sorted({v for m in C['movies'].values() for v in m['col3_values']})
    bad = sum(m['bad_bouts'] for m in C['movies'].values())
    L.append(f"- **{ctx}**: traintest behs {t['behs']}, n_flies {t['n_flies']}, ext {t['movie_ext']}; "
             f"actions.mat behs {m0['behs']}, bouts cell {m0['bouts_cell_shape']}; third column values {col3}; "
             f"invalid bouts (end < start, < 1 or > frames) {bad}; track fields {m0['track']['names']}; "
             f"flag_frames non-zero (summed) {sum(m['track']['flag_frames_nonzero'] for m in C['movies'].values()):,}")
L.append('')

L += ['## Per behaviour (frame positive = labelled on EITHER fly; bouts = runs of that either-fly mask)', '']
RULE = {}
for ctx, C in inv['contexts'].items():
    behs = C['movies'][next(iter(C['movies']))]['behs']
    L += [f'### {ctx}', '', '| behaviour | split | frames | positive rate | bouts (either fly) | bouts annotated '
          '(per fly, summed) | bout length median (frames) | movies with a positive |', '|---|---|---|---|---|---|---|---|']
    for b in behs:
        for sp in ['train', 'test']:
            ms = [m for m in C['movies'].values() if m['split'] == sp]
            fr = sum(m['labels'][b]['frames'] for m in ms if b in m['labels'])
            n = sum(m['probe']['frames'] for m in ms)
            ba = sum(m['labels'][b]['bouts_any'] for m in ms if b in m['labels'])
            bp = sum(sum(m['labels'][b]['bouts_per_fly']) for m in ms if b in m['labels'])
            ln = [x for m in ms if b in m['labels'] for x in m['labels'][b]['bout_len_any']]
            mv = sum(m['labels'][b]['frames'] > 0 for m in ms if b in m['labels'])
            RULE[(ctx, b, sp)] = (ba, mv)
            L.append(f"| {b} | {sp} | {fr:,} | {fr / n:.4f} | {ba:,} | {bp:,} | "
                     f"{np.median(ln) if ln else '-'} | {mv} / {len(ms)} |")
    L.append('')

L += ['## Label timing', '']
for ctx, C in inv['contexts'].items():
    cc = {}
    for m in C['movies'].values():
        for b, d in m.get('col3_counts', {}).items():
            for v, n in d.items():
                cc.setdefault(b, {}).setdefault(v, 0)
                cc[b][v] += n
    L.append(f"- **{ctx}** third bout column, value counts per behaviour (meaning not documented in the release): {cc}")
cop = {n: m['copulation'] for n, m in inv['contexts']['Courtship']['movies'].items() if 'copulation' in m}
if cop:
    st = [c['first_start'] for c in cop.values()]
    en = [c['last_end'] for c in cop.values()]
    tot = {}
    for c in cop.values():
        for b, r in c['others'].items():
            for k, v in r.items():
                tot.setdefault(b, {}).setdefault(k, 0)
                tot[b][k] += v
    L.append(f"- **Courtship**: copulation found in {len(cop)} / {len(inv['contexts']['Courtship']['movies'])} movies; "
             f"first copulation frame median {int(np.median(st)):,} (min {min(st):,}, max {max(st):,}) of 54,000; "
             f"copulation end median {int(np.median(en)):,}. Other bouts before / during / after copulation: {tot}. "
             'No other behaviour is labelled during copulation (the two flies are joined); courtship actions are '
             'labelled both before it and after it ends.')
L.append('')

L += ['## Annotators', '']
for ctx, C in inv['contexts'].items():
    sec = {n: m['second_annotation'] for n, m in C['movies'].items() if 'second_annotation' in m}
    if not sec:
        L.append(f'- **{ctx}**: one annotation per movie (no second_actions files).')
        continue
    L.append(f"- **{ctx}**: second annotation on {len(sec)} / {len(C['movies'])} movies ({', '.join(sec)}). "
             'Frame IoU (either fly, first vs second): ' + '; '.join(
                 f"{n}: " + ', '.join(f"{b} {a['iou']:.2f}" for b, a in s['agreement'].items() if a['iou'] is not None)
                 for n, s in sec.items()))
L.append('')

L += ['## Alignment', '', '(b) video vs track: median distance (px) between the blob centroid segmented from decoded frame i '
      '(one sequential decode pass from frame 0, no seeking) and the trk pos x/y at track index i + lag, for the '
      'faster fly, on frames where the flies are apart (> 1.5 body lengths) and that fly is in the movie\'s top 5 % '
      'of speed:', '',
      '| movie | frames scored | median speed (px/frame) | ' + ' | '.join(f'lag {k}' for k in range(-5, 6)) + ' | best lag |',
      '|---|---|---|' + '---|' * 12]
for k, a in inv['alignment_video_vs_track'].items():
    vals = {int(x): y for x, y in a['median_err_px_by_lag'].items()}
    sp = a['median_speed_px_per_frame_scored_fly']
    L.append(f"| {k} | {a['frames_scored']} | {sp:.2f} | " if sp is not None else f"| {k} | 0 | - | ")
    L[-1] += ' | '.join(f"{vals[l]:.2f}" if vals.get(l) is not None else '-' for l in range(-5, 6)) + f" | {a['best_lag']} |"
L += ['', '(c) labels vs track: bout-onset-triggered mean of a track signal; label start L read as track index L - 1. '
      'Offsets where the mean changes most steeply:', '']
for ctx, E in inv['alignment_labels_vs_track'].items():
    for b, e in E.items():
        L.append(f"- {ctx} / {b} ({e['signal']}, {e['n_onsets']} onsets): steepest change between offsets "
                 f"{e['steepest_change_between_offsets']}; mean at -3..+3: "
                 + ', '.join(f"{x:.2f}" for x in e['mean'][12:19]))
L.append('')

L += ['## Body length (native pixels, label-free)', '',
      '| movie | blobs | blob major axis median (IQR) | trk major axis len median, same frames |', '|---|---|---|---|']
for k, b in inv['body_length'].items():
    L.append(f"| {k} | {b['n_blobs']} | {b['blob_major_median']:.1f} ({b['blob_major_iqr'][0]:.1f}-{b['blob_major_iqr'][1]:.1f}) "
             f"| {b['trk_major_median']:.1f} |" if b['n_blobs'] else f"| {k} | 0 | - | - |")
L += ['', 'trk "major axis len" over all movies (every 10th frame): ' + '; '.join(
    f"{c}: median {v['median']:.1f} (p10 {v['p10']:.1f}, p90 {v['p90']:.1f})" for c, v in inv['trk_major_axis_len'].items()), '']

L += ['## Pre-registered behaviour rules (halves = released train / test movies, per context)', '',
      'Rule: frame-level and dense; >= 20 positive bouts AND >= 10 movies with a positive in EACH half; drop '
      '"unknown" / "other" supersets and proximity labels.', '',
      '| context | behaviour | train bouts / movies | test bouts / movies | passes |', '|---|---|---|---|---|']
for (ctx, b, sp), (ba, mv) in RULE.items():
    if sp != 'train':
        continue
    tb, tm = RULE[(ctx, b, 'test')]
    ok = ba >= 20 and mv >= 10 and tb >= 20 and tm >= 10 and b not in ('unknown', 'other', 'none', 'touch')
    L.append(f'| {ctx} | {b} | {ba} / {mv} | {tb} / {tm} | {"yes" if ok else "no"} |')
L.append('')
L += ['## Animals and arena', '',
      '- Two flies per movie (`n_flies` 2 in the track files of every context). Appearance (looked at the example '
      'frames, and measured as darkness): dark, unmarked bodies on a bright back-lit floor; the foreground threshold '
      '(half the median background-minus-frame darkness at the track centres) came out at '
      + ', '.join(f"{k} {a['threshold']:.0f}" for k, a in inv['alignment_video_vs_track'].items()) + ' grey levels.',
      '- Aggression and Courtship: 144 x 144 crops of one round chamber (the chamber rim is visible); Boy-meets-Boy: '
      'a 640 x 480 rectangular arena with a static dark square in the middle, 200 fps.',
      '- Identity: the track keeps two identities (fly 1 / fly 2); Courtship labels are on one fly only '
      '(traintest `n_flies` 1; which fly that is - presumably the male - is not stated in the files).', '',
      '## Proposed validation protocol', '',
      '- Unit of cross-fitting: the movie (video-level), with the two released halves as the two folds. Fold A: the '
      'SAE is trained on the train movies with labels ignored, each behaviour picks its neuron on the train movies, '
      'and that neuron is scored on the test movies. Fold B swaps the roles. Report both folds and their mean.',
      '- Frame-level target: a frame is positive if the behaviour is labelled on either fly (bouts end-inclusive, '
      'label frame L -> 0-based video frame L - 1, verified above). All frames of the scored movies are scored; '
      'there is no "other" class to exclude.',
      '- Candidate behaviour set (rules applied in the table above): Courtship `wing_extension`, `circle`, '
      '`copul_attempt`. Dropped: `unknown` (catch-all), `copulation` (a 15-20 min state, median bout ~31,000 frames, '
      'and only 16 / 15 bouts per half), `touch` (proximity / contact). Every Aggression and Boy-meets-Boy behaviour '
      'fails the >= 10 movies per half rule (5 / 5 and 3 / 3 movies), so lunge, wing threat, charge, hold, tussle '
      'are descriptive only (no pre-registered claim). Pooling Aggression with Boy-meets-Boy would still give at most '
      '8 movies per half, and mixes 144 px / 30 fps with 640 x 480 / 200 fps.',
      '- Caveats for scoring: nothing but copulation is labelled during the long copulation bout (Label timing); `copul_attempt` '
      'bouts are short (median 8-9 frames); the second annotation agrees only moderately (Annotators), which bounds '
      'any AUROC that can be reached.', '']
open(os.path.join(D, 'inventory.md'), 'w').write('\n'.join(L) + '\n')
print('wrote', os.path.join(D, 'inventory.md'))
