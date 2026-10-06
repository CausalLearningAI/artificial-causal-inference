"""Switch-audit renders: overview frame with boxes + identity labels + state, and a column of
full-resolution insets (one per identity, then any unassigned tracks).

Selection per video and variant:
  random      20 random frames (seeded) among frames with track rows
  post_merge  the frame ~1 s after the end of each of the 20 longest MERGE EPISODES (maximal runs
              of frames where some box touches another box, a box is oversize, or fewer than N
              boxes are detected)
index.csv lists every image with its frame, kind, and per-identity state / track; the columns
verdict_<identity> are left empty for a human (or the worker) to fill: correct / wrong / cant_tell.
"""
import cv2
import numpy as np
import pandas as pd

from common import read_frame, video_path

COLORS = {'m1_none': (60, 220, 60), 'm2_head': (40, 160, 255), 'm3_middle': (255, 120, 40),
          'm4_back': (220, 60, 220), 'focal': (230, 230, 230), 'blue': (255, 140, 30),
          'yellow': (20, 210, 255)}
ST = {'confirmed': 'C', 'inferred': 'I', 'unknown': '?'}


def merge_episodes(tr, N):
    f = tr.groupby('frame_src').agg(c=('in_contact', 'any'), o=('oversize', 'any'), n=('detected', 'sum'))
    bad = (f.c | f.o | (f.n < N)).values
    fr = f.index.values
    eps, s = [], None
    for i, b in enumerate(bad):
        if b and s is None:
            s = i
        if (not b or i == len(bad) - 1) and s is not None:
            e = i - 1 if not b else i
            eps.append((fr[s], fr[e]))
            s = None
    return sorted(eps, key=lambda t: -(t[1] - t[0]))


def select_frames(tr, N, n_random=20, n_merge=20, seed=0):
    rng = np.random.default_rng(seed)
    frames = np.unique(tr.frame_src.values)
    sel = [(int(f), 'random', '') for f in np.sort(rng.choice(frames, min(n_random, len(frames)), replace=False))]
    for s, e in merge_episodes(tr, N)[:n_merge]:
        k = np.searchsorted(frames, e + 30)
        if k < len(frames):
            sel.append((int(frames[k]), 'post_merge', f'{s}-{e} ({(e - s) / 30:.1f}s)'))
    return sel


def render(domain, vid, f, pf, tr, ids, title):
    img = read_frame(video_path(domain, vid), f)
    H, W = img.shape[:2]
    ov_scale = 1000 / W
    ov = cv2.resize(img, None, fx=ov_scale, fy=ov_scale, interpolation=cv2.INTER_AREA)
    rows_f = tr[tr.frame_src == f]
    p = pf[pf.frame_src == f].set_index('identity')
    assigned = set(p.track_id[p.state != 'unknown'].tolist())
    for r in rows_f.itertuples():
        x0, y0 = int((r.cx - r.w / 2) * ov_scale), int((r.cy - r.h / 2) * ov_scale)
        x1, y1 = int((r.cx + r.w / 2) * ov_scale), int((r.cy + r.h / 2) * ov_scale)
        if r.track_id not in assigned:
            cv2.rectangle(ov, (x0, y0), (x1, y1), (128, 128, 128), 1)
            cv2.putText(ov, f't{r.track_id} ?', (x0, max(y0 - 4, 12)), 0, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
    for i in ids:
        q = p.loc[i]
        if q.state == 'unknown' or np.isnan(q.cx):
            continue
        c = COLORS[i]
        x0, y0 = int((q.cx - q.w / 2) * ov_scale), int((q.cy - q.h / 2) * ov_scale)
        x1, y1 = int((q.cx + q.w / 2) * ov_scale), int((q.cy + q.h / 2) * ov_scale)
        cv2.rectangle(ov, (x0, y0), (x1, y1), c, 2 if q.state == 'confirmed' else 1)
        lab = f'{i} [{ST[q.state]} {q.conf:.2f}] t{int(q.track_id)}'
        cv2.putText(ov, lab, (x0, max(y0 - 5, 14)), 0, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ov, lab, (x0, max(y0 - 5, 14)), 0, 0.55, c, 1, cv2.LINE_AA)
    cv2.putText(ov, title, (8, 22), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ov, title, (8, 22), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # insets at full resolution
    side = 480 if domain == 'mice' else 120
    ins_px = 240
    tiles = []
    entries = [(i, p.loc[i]) for i in ids]
    for r in rows_f.itertuples():
        if r.track_id not in assigned:
            entries.append((f'unassigned t{r.track_id}', r))
    for name, q in entries:
        t = np.full((ins_px + 24, ins_px, 3), 40, np.uint8)
        if not np.isnan(q.cx):
            x0, y0 = int(q.cx - side / 2), int(q.cy - side / 2)
            big = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            c = big[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
            t[24:] = cv2.resize(c, (ins_px, ins_px), interpolation=cv2.INTER_AREA if side > ins_px else cv2.INTER_NEAREST)
            st = getattr(q, 'state', 'unknown') if not isinstance(q, pd.Series) else q.state
            lab = f'{name} [{ST.get(st, "?")}]' if name in ids else name
        else:
            lab = f'{name}: not located'
        col = COLORS.get(name, (160, 160, 160))
        cv2.putText(t, lab, (4, 17), 0, 0.5, col, 1, cv2.LINE_AA)
        tiles.append(t)
    ncol = 2
    while len(tiles) % ncol:
        tiles.append(np.full_like(tiles[0], 40))
    grid = np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)])
    h = max(ov.shape[0], grid.shape[0])
    canvas = np.full((h, ov.shape[1] + grid.shape[1] + 8, 3), 25, np.uint8)
    canvas[:ov.shape[0], :ov.shape[1]] = ov
    canvas[:grid.shape[0], ov.shape[1] + 8:] = grid
    return canvas


def make_audit(domain, vid, variant, tr, pf, ids, N, outdir, seed=0):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, kind, note in select_frames(tr, N, seed=seed):
        title = f'{vid} {variant} frame {f} ({f / 30:.1f}s) {kind} {note}'
        im = render(domain, vid, f, pf, tr, ids, title)
        fn = f'{variant}_{kind}_{f:06d}.png'
        cv2.imwrite(str(outdir / fn), im)
        p = pf[pf.frame_src == f].set_index('identity')
        rows.append({'file': fn, 'video': vid, 'variant': variant, 'kind': kind, 'frame_src': f,
                     'merge_episode': note,
                     **{f'state_{i}': p.loc[i].state for i in ids},
                     **{f'track_{i}': int(p.loc[i].track_id) for i in ids},
                     **{f'verdict_{i}': '' for i in ids}})
    idx = pd.DataFrame(rows)
    p = outdir / 'index.csv'
    if p.exists():  # keep verdicts already filled for other variants
        old = pd.read_csv(p)
        old = old[old.variant != variant]
        idx = pd.concat([old, idx], ignore_index=True)
    idx.to_csv(p, index=False)
    return idx
