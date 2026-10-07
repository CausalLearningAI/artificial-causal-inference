"""
Neuron-interpretation galleries for the ECI (Exploratory Causal Inference) pipeline on mice v1.

For every SAE neuron j we render one PNG:
    top row block     the 12 frames where j is most active (frame-pooled code, 'mean' by default,
                      'max' as an option), at most max_per_video frames per video (default 1;
                      if > 1, frames of one video are >= min_gap_s apart), with a per-patch
                      heatmap of j overlaid, so we see WHERE the concept fires;
    least block       12 frames where j is silent: random frames (fixed seed, 1 per video) with
                      activation exactly 0 when at least 12 videos contain such a frame, else the
                      12 lowest frames (1 per video). The figure says which rule was used;
    panel             per-video mean activation per stage (1-6) by genotype, mean and 95% t-CI
                      across pools; the density histogram of the frame activation; firing rate.
and an HTML index linking all figures (self-contained, relative paths, light / dark).

Heatmaps. The per-frame codes only store pooled vectors, so for the ~24 displayed frames per
neuron the 16x16 patch codes are recomputed: DINOv2 (src/eci/extract.py load_encoder, the same
preprocessing) -> fp16 rounding -> SAE (global threshold), exactly as in src/eci/encode.py.
Preprocessing = resize the shortest edge to 256 (bicubic), center crop 224. So on a W x H frame
the model sees the square [x0, x0 + c) x [y0, y0 + c) with s = 256 / min(W, H), c = 224 / s,
x0 = ((round(W s) - 224) // 2) / s (for 512 x 512 frames: x0 = y0 = 32, c = 448, 28 px / patch).
The heatmap is placed on that square; the dashed box marks it (the border is never seen).
Each recomputed patch map is checked against the stored pooled value (mean or max over patches).
Foreground SAEs ('fg448', src/eci/foreground.py): DINOv2 sees the WHOLE 512 frame resized to 448
(32 x 32 patches = 16 x 16 pixel blocks of the frame, box (0, 0, 512)); the SAE is applied only to
the foreground patches (the video's stored background + FG_RULE), the other patches are shown as 0.
The stored codes pool over foreground patches: codes_max = max, codes_mean = sum / n_fg.

Domains: load_full_codes, representation, make_patch_encoder and default_sae_path take an optional
domain (src/eci/domain.py); it replaces the subject / version paths (ants: dataset/ants/eci/) and the
per-video metadata (domain.video_meta, its meta_cols are carried into the video blocks). Without a
domain everything is mice v1 as before.

Candidate search over the (N, m) memmap: rows are grouped by video (contiguous in
annotations.csv and in the training sample), so ONE pass over video blocks gives, per neuron and
video, the top max_per_video frames (non-maximum suppression in time), a random zero frame, the
lowest frame, the video mean, the firing count and a fixed random subsample for the histogram.
With one frame per video this is exact (no top-C candidate truncation).

Functions / classes:
    video_meta              observation_id -> pool, stage, genotype, phase, odor
    CodeSource              pooled codes + row metadata (full codes or training sample)
    load_full_codes         dataset/mice/v1/eci/codes/{sae}/ (annotations.csv row order), or a domain's
    sample_codes            encode the training-sample patch tokens through the SAE (cached)
    scan_codes              the single pass above -> NeuronScan
    pick_top / pick_least   per-neuron frame selections from a NeuronScan
    PatchEncoder            DINOv2 + SAE, per-patch codes for a list of frames
    crop_box                region of the original frame seen by the model
    representation          'crop224' (patch SAEs above) or 'fg448' (foreground SAEs) of an SAE's codes
    PatchEncoderFG          fg448: whole frame at 448, SAE on foreground patches only (0 elsewhere); aligned
                            SAEs (fg448al) turn the frames first
    frame_rot90             per-row 90-degree turns of an aligned SAE's frames (None when not aligned)
    make_patch_encoder      the encoder matching an SAE's representation
    frame_box               heatmap placement (x0, y0, size) on the frame for a representation
    plot_neuron             one PNG per neuron
    write_index             HTML index
    gallery_for             end to end: scan, select, recompute heatmaps, plot, index
    video_view              one neuron in one video: activation time course + top frames of that video
    video_views             video_view for many (neuron, video) pairs, sharing one PatchEncoder
"""

import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats

from src.eci.encode import sae_pool
from src.eci.extract import STAGES, _FrameDataset, load_encoder
from src.eci.sae import load_sae

STAGE_LABEL = {v: f'{p},{o}' for (p, o), v in STAGES.items()}  # 1 -> 'H,S'
META_COLS = ('pool', 'stage', 'genotype')  # per-video columns of the mice video blocks
GENO_COLOR = {'wt': '#2a78b5', 'het': '#d9661f'}


# ---------------------------------------------------------------------- code sources
def video_meta(data_dir='data', subject='mice', version='v1'):
    """One row per observation: observation_id, pool, genotype, phase, odor, stage (1-6)."""
    exp = pd.read_csv(Path(data_dir) / subject / version / 'experiment.csv')
    exp['stage'] = [STAGES[(p, o)] for p, o in zip(exp['phase'], exp['odor'])]
    return exp[['observation_id', 'pool', 'genotype', 'phase', 'odor', 'stage']].reset_index(drop=True)


@dataclass
class CodeSource:
    """Pooled SAE codes and one metadata row per code row.
    codes: {'mean': (N, m) array, 'max': (N, m) array}, memmaps are fine.
    meta columns: observation_id, frame_idx, frame_path (relative to dataset_dir), pool, stage,
    genotype, fps."""
    name: str
    codes: dict
    meta: pd.DataFrame
    dataset_dir: Path
    info: dict = field(default_factory=dict)


def _eci_dir(dataset_dir, subject, version, domain):
    return Path(dataset_dir) / (domain.eci_rel if domain is not None else Path(subject) / version / 'eci')


def load_full_codes(sae_name, dataset_dir='dataset', data_dir='data', subject='mice', version='v1', codes_dir=None,
                    domain=None):
    ds = Path(dataset_dir)
    d = Path(codes_dir) if codes_dir else _eci_dir(ds, subject, version, domain) / 'codes' / sae_name
    if not (d / 'DONE').exists():
        raise FileNotFoundError(f'{d}/DONE missing: full codes not merged yet (use --source sample)')
    ann = pd.read_csv(ds / domain.ann_rel if domain is not None else ds / subject / version / 'annotations.csv',
                      usecols=['observation_id', 'frame_idx', 'fps', 'frame_path'])
    vm = domain.video_meta() if domain is not None else video_meta(data_dir, subject, version)
    meta = ann.merge(vm, on='observation_id', how='left', validate='m:1')
    cols = domain.meta_cols if domain is not None else META_COLS
    if meta[cols[1]].isna().any():
        raise ValueError('observations in annotations.csv missing from experiment.csv')
    codes = {k: np.load(d / f'codes_{k}.npy', mmap_mode='r') for k in ('mean', 'max')}
    for k, a in codes.items():
        if a.shape[0] != len(meta):
            raise ValueError(f'codes_{k} has {a.shape[0]} rows, annotations.csv has {len(meta)}')
    info = {'codes_dir': str(d), 'n_rows': len(meta)}
    if domain is not None:
        info['meta_cols'] = cols
    return CodeSource('full', codes, meta, ds, info)


@torch.no_grad()
def sample_codes(tokens_dir, sae_path, out_dir, device='cuda', batch_frames=256):
    """Pooled SAE codes of the training-sample frames, from their stored fp16 patch tokens.
    Writes out_dir/{codes_mean,codes_max}.npy (float16, same row order as metadata.parquet)
    and a DONE flag; skips if DONE exists."""
    out_dir = Path(out_dir)
    if (out_dir / 'DONE').exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sae, norm, _ = load_sae(sae_path, device)
    tok = np.load(Path(tokens_dir) / 'patch_tokens.npy', mmap_mode='r')
    N, m = tok.shape[0], sae.n_latents
    mm = {k: np.lib.format.open_memmap(out_dir / f'codes_{k}.npy.tmp', 'w+', np.float16, (N, m))
          for k in ('mean', 'max')}
    t0 = time.time()
    for a in range(0, N, batch_frames):
        x = torch.from_numpy(np.asarray(tok[a:a + batch_frames])).to(device)
        mean, mx = sae_pool(sae, norm, x)
        mm['mean'][a:a + len(x)] = mean.half().cpu().numpy()
        mm['max'][a:a + len(x)] = mx.half().cpu().numpy()
        if (a // batch_frames) % 20 == 0:
            print(f'  sample codes {a + len(x):6d}/{N}  {time.time() - t0:.0f}s', flush=True)
    for k, a in mm.items():
        a.flush()
        del a
    del mm
    for k in ('mean', 'max'):
        (out_dir / f'codes_{k}.npy.tmp').rename(out_dir / f'codes_{k}.npy')
    (out_dir / 'config.json').write_text(json.dumps({
        'tokens_dir': str(tokens_dir), 'sae_checkpoint': str(sae_path), 'n_rows': N,
        'pooling': 'src/eci/encode.py sae_pool (global threshold), mean / max over 256 patches',
        'elapsed_s': round(time.time() - t0, 1)}, indent=1))
    (out_dir / 'DONE').touch()
    return out_dir


def load_sample_codes(sae_name, dataset_dir='dataset', subject='mice', version='v1',
                      tokens='dinov2_base_l-1', device='cuda'):
    ds = Path(dataset_dir)
    eci = ds / subject / version / 'eci'
    tokens_dir = eci / 'train_tokens' / tokens
    d = sample_codes(tokens_dir, eci / 'sae' / sae_name / 'sae.pt', eci / 'codes_sample' / sae_name, device=device)
    meta = pd.read_parquet(tokens_dir / 'metadata.parquet')
    meta['fps'] = 5.0 if 'fps' not in meta else meta['fps']
    codes = {k: np.load(d / f'codes_{k}.npy', mmap_mode='r') for k in ('mean', 'max')}
    return CodeSource('sample', codes, meta, ds, {'codes_dir': str(d), 'n_rows': len(meta)})


# ---------------------------------------------------------------------- one pass over the codes
@dataclass
class NeuronScan:
    neurons: np.ndarray        # (s,) neuron ids
    videos: pd.DataFrame       # one row per video block: observation_id, pool, stage, genotype, lo, hi
    top_val: np.ndarray        # (V, K, s) best K frames per video (NMS in time), -inf if none
    top_row: np.ndarray        # (V, K, s) global row index
    zero_row: np.ndarray       # (V, s) a random frame with activation 0, -1 if none
    min_val: np.ndarray        # (V, s)
    min_row: np.ndarray        # (V, s)
    video_mean: np.ndarray     # (V, s)
    n_active: np.ndarray       # (s,) frames with activation > 0
    n_frames: int
    hist_sample: np.ndarray    # (n_sub, s) float16, fixed random frames
    key: str


def _video_blocks(meta, cols=META_COLS):
    """Contiguous row ranges per observation (raises if a video's rows are not contiguous), with the
    per-video columns cols (mice: pool, stage, genotype)."""
    obs = meta['observation_id'].values
    change = np.flatnonzero(obs[1:] != obs[:-1]) + 1
    lo = np.r_[0, change]
    hi = np.r_[change, len(obs)]
    ids = obs[lo]
    if len(set(ids)) != len(ids):
        raise ValueError('rows of a video are not contiguous')
    first = meta.iloc[lo]
    return pd.DataFrame({'observation_id': ids, **{c: first[c].values for c in cols}, 'lo': lo, 'hi': hi})


def scan_codes(source, neurons, key='mean', max_per_video=1, min_gap_s=2.0, hist_per_video=128, seed=0,
               log_every=50):
    """Single pass over video blocks of source.codes[key][:, neurons] (see module docstring)."""
    Z = source.codes[key]
    neurons = np.asarray(neurons, dtype=np.int64)
    cols = neurons if not np.array_equal(neurons, np.arange(neurons[0], neurons[-1] + 1)) \
        else slice(int(neurons[0]), int(neurons[-1]) + 1)
    vids = _video_blocks(source.meta, source.info.get('meta_cols', META_COLS))
    V, s, K = len(vids), len(neurons), max_per_video
    frame_idx = source.meta['frame_idx'].values
    fps = source.meta['fps'].values
    out = dict(top_val=np.full((V, K, s), -np.inf, np.float32), top_row=np.full((V, K, s), -1, np.int64),
               zero_row=np.full((V, s), -1, np.int64), min_val=np.zeros((V, s), np.float32),
               min_row=np.zeros((V, s), np.int64), video_mean=np.zeros((V, s), np.float32))
    n_active = np.zeros(s, np.int64)
    hist = []
    t0 = time.time()
    for v, (lo, hi) in enumerate(zip(vids['lo'].values, vids['hi'].values)):
        X = np.asarray(Z[lo:hi][:, cols], dtype=np.float32)  # (n_v, s)
        n_v = hi - lo
        rng = np.random.default_rng([seed, v])
        out['video_mean'][v] = X.mean(0)
        n_active += (X > 0).sum(0)
        a = X.argmin(0)
        out['min_row'][v], out['min_val'][v] = lo + a, X[a, np.arange(s)]
        r = rng.random(n_v)
        score = np.where(X == 0, r[:, None], -1.0)
        a = score.argmax(0)
        out['zero_row'][v] = np.where(score[a, np.arange(s)] >= 0, lo + a, -1)
        hist.append(X[np.sort(rng.choice(n_v, min(hist_per_video, n_v), replace=False))].astype(np.float16))
        # top-K per video with >= min_gap_s between picks (non-maximum suppression in time)
        t = frame_idx[lo:hi] / fps[lo:hi]
        Xw = X.copy()
        for k in range(K):
            a = Xw.argmax(0)
            val = Xw[a, np.arange(s)]
            ok = np.isfinite(val)
            out['top_val'][v, k] = np.where(ok, val, -np.inf)
            out['top_row'][v, k] = np.where(ok, lo + a, -1)
            if k + 1 < K:
                Xw[np.abs(t[:, None] - t[a][None, :]) < min_gap_s] = -np.inf
        if log_every and (v % log_every == 0 or v == V - 1):
            print(f'  scan video {v + 1}/{V}  {time.time() - t0:.0f}s', flush=True)
    return NeuronScan(neurons=neurons, videos=vids, n_active=n_active, n_frames=len(source.meta),
                      hist_sample=np.concatenate(hist), key=key, **out)


def pick_top(scan, i, n=12):
    """Rows of the n most active frames of neuron scan.neurons[i] (only activations > 0)."""
    val = scan.top_val[:, :, i].ravel()
    row = scan.top_row[:, :, i].ravel()
    order = np.argsort(-val, kind='stable')[:n]
    order = order[val[order] > 0]
    return row[order], val[order]


def pick_least(scan, i, n=12, seed=0):
    """(rows, rule): random zero frames, 1 per video, if >= n videos have one; else the n lowest
    frames, 1 per video. rule is 'random_zero' or 'bottom'."""
    zr = scan.zero_row[:, i]
    has = np.flatnonzero(zr >= 0)
    if len(has) >= n:
        pick = np.random.default_rng([seed, int(scan.neurons[i])]).choice(has, n, replace=False)
        return np.sort(zr[pick]), 'random_zero'
    order = np.argsort(scan.min_val[:, i], kind='stable')[:n]
    return scan.min_row[order, i], 'bottom'


# ---------------------------------------------------------------------- windows of consecutive frames
@dataclass
class WindowScan:
    """Per video block (V) and neuron (s), for windows of `length` consecutive frames of one video.
    Rows are global row indices of the window's FIRST frame (-1 when the video has no such window)."""
    neurons: np.ndarray
    videos: pd.DataFrame
    length: int
    best_start: np.ndarray    # (V, s) window with the highest mean activation (first one on ties)
    best_mean: np.ndarray     # (V, s)
    silent_start: np.ndarray  # (V, s) a random window with activation exactly 0 on every frame, -1 if none
    min_start: np.ndarray     # (V, s) window with the lowest mean activation
    min_mean: np.ndarray      # (V, s)
    key: str


def scan_windows(source, neurons, key='max', lengths=(1, 5, 15), seed=0, log_every=50, start=None):
    """One pass over the video blocks of source.codes[key][:, neurons] -> {length: WindowScan}.
    Windows are `length` consecutive rows of one video (rows of a video are consecutive frames);
    length 1 = single frames (best = the video's highest frame, silent = a random zero frame).
    The random silent window uses rng([seed, video, length]).
    start: optional (V,) first frame (offset within the video, in _video_blocks order) a window may cover, e.g.
    src/eci/domain.py Domain.window_start (mice: habituation clips only from minute 15 on); None = 0."""
    Z = source.codes[key]
    neurons = np.asarray(neurons, dtype=np.int64)
    vids = _video_blocks(source.meta, source.info.get('meta_cols', META_COLS))
    V, s = len(vids), len(neurons)
    out = {w: {k: np.full((V, s), -1 if k.endswith('start') else np.nan, np.int64 if k.endswith('start')
                          else np.float32) for k in ('best_start', 'best_mean', 'silent_start', 'min_start', 'min_mean')}
           for w in lengths}
    t0 = time.time()
    ar = np.arange(s)
    for v, (lo, hi) in enumerate(zip(vids['lo'].values, vids['hi'].values)):
        if start is not None:
            lo = lo + int(start[v])
        X = np.asarray(Z[lo:hi][:, neurons], dtype=np.float64)
        S = np.vstack([np.zeros((1, s)), np.cumsum(X, 0)])
        C = np.vstack([np.zeros((1, s), np.int64), np.cumsum(X > 0, 0)])
        for w in lengths:
            n_a = (hi - lo) - w + 1
            if n_a <= 0:
                continue
            m = (S[w:w + n_a] - S[:n_a]) / w                  # (n_a, s) window means
            silent = (C[w:w + n_a] - C[:n_a]) == 0              # every frame exactly 0
            o = out[w]
            a = m.argmax(0)
            o['best_start'][v], o['best_mean'][v] = lo + a, m[a, ar]
            a = m.argmin(0)
            o['min_start'][v], o['min_mean'][v] = lo + a, m[a, ar]
            r = np.random.default_rng([seed, v, w]).random(n_a)
            score = np.where(silent, r[:, None], -1.0)
            a = score.argmax(0)
            o['silent_start'][v] = np.where(score[a, ar] >= 0, lo + a, -1)
        if log_every and (v % log_every == 0 or v == V - 1):
            print(f'  window scan video {v + 1}/{V}  {time.time() - t0:.0f}s', flush=True)
    return {w: WindowScan(neurons=neurons, videos=vids, length=w, key=key, **o) for w, o in out.items()}


def subset_windows(ws, mask):
    """The WindowScan restricted to the videos where mask (V,) is True (e.g. the videos of one
    contrast), so pick_top_windows / pick_least_windows choose among those videos only."""
    from dataclasses import replace
    mask = np.asarray(mask, bool)
    return replace(ws, videos=ws.videos[mask].reset_index(drop=True),
                   **{k: getattr(ws, k)[mask] for k in ('best_start', 'best_mean', 'silent_start', 'min_start',
                                                         'min_mean')})


def pick_top_windows(ws, i, n=16):
    """(starts, means): the n videos with the highest best-window mean of neuron ws.neurons[i], that
    window in each (so at most one window per video; only means > 0)."""
    val = np.nan_to_num(ws.best_mean[:, i], nan=-np.inf)
    order = np.argsort(-val, kind='stable')[:n]
    order = order[val[order] > 0]
    return ws.best_start[order, i], val[order]


def pick_least_windows(ws, i, n=16, seed=0):
    """(starts, rule): if >= n videos contain a window where the neuron is exactly 0 on every frame,
    n of those videos at random (rng([seed, neuron, length])) with their random silent window,
    rule 'silent'; else the n videos with the lowest min-window mean and that window, rule 'lowest'
    (silent windows, mean 0, come first). At most one window per video."""
    st = ws.silent_start[:, i]
    has = np.flatnonzero(st >= 0)
    if len(has) >= n:
        pick = np.random.default_rng([seed, int(ws.neurons[i]), ws.length]).choice(has, n, replace=False)
        return np.sort(st[pick]), 'silent'
    val = np.nan_to_num(ws.min_mean[:, i], nan=np.inf)
    order = np.argsort(val, kind='stable')[:n]
    order = order[np.isfinite(val[order])]
    return ws.min_start[order, i], 'lowest'


def stage_genotype_table(scan, i):
    """Per (stage, genotype): mean over pools of the per-video means, 95% t-CI, n pools."""
    df = scan.videos[['pool', 'stage', 'genotype']].copy()
    df['y'] = scan.video_mean[:, i]
    rows = []
    for (st, g), grp in df.groupby(['stage', 'genotype']):
        y = grp.groupby('pool')['y'].mean().values
        se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
        h = stats.t.ppf(0.975, len(y) - 1) * se if len(y) > 1 else np.nan
        rows.append({'stage': st, 'genotype': g, 'mean': y.mean(), 'lo': y.mean() - h, 'hi': y.mean() + h,
                     'n_pools': len(y)})
    return pd.DataFrame(rows)


def neuron_stats(scan):
    """One row per neuron: firing rate (frames > 0), mean / max of the frame activation,
    fraction of videos with a zero frame, het - wt difference of pool means (all stages)."""
    rows = []
    for i, j in enumerate(scan.neurons):
        vm = scan.video_mean[:, i]
        geno = scan.videos['genotype'].values
        rows.append({'neuron': int(j), 'firing_rate': scan.n_active[i] / scan.n_frames,
                     'mean_act': float((vm * (scan.videos['hi'] - scan.videos['lo'])).sum() / scan.n_frames),
                     'max_act': float(scan.top_val[:, 0, i].max()),
                     'videos_with_zero': float((scan.zero_row[:, i] >= 0).mean()),
                     'het_minus_wt': float(vm[geno == 'het'].mean() - vm[geno == 'wt'].mean())
                     if {'het', 'wt'} <= set(geno) else np.nan})
    return pd.DataFrame(rows).set_index('neuron')


# ---------------------------------------------------------------------- patch heatmaps
def crop_box(width, height, resolution=224, shortest_edge=256):
    """(x0, y0, size) in original-frame pixels of the center crop the model sees
    (HF resize of the shortest edge + center crop, as in load_encoder)."""
    s = shortest_edge / min(width, height)
    rw, rh = int(round(width * s)), int(round(height * s))
    return ((rw - resolution) // 2) / s, ((rh - resolution) // 2) / s, resolution / s


class PatchEncoder:
    """DINOv2 + SAE -> per-patch codes of selected neurons for a list of frames."""

    def __init__(self, sae_path, encoder='dinov2_base', resolution=224, device='cuda'):
        self.device = torch.device(device)
        _, self.processor, self.model = load_encoder(encoder, resolution, self.device)
        self.sae, self.norm, _ = load_sae(sae_path, self.device)
        self.n_prefix = 1 + (getattr(self.model.config, 'num_register_tokens', 0) or 0)
        self.grid = resolution // self.model.config.patch_size
        self.resolution = resolution
        self.shortest_edge = self.processor.size['shortest_edge']

    @torch.no_grad()
    def patch_codes(self, paths, neurons, batch_size=64, num_workers=8):
        """(len(paths), grid, grid, len(neurons)) float32 patch codes."""
        loader = torch.utils.data.DataLoader(_FrameDataset(list(paths), self.processor), batch_size=batch_size,
                                             num_workers=num_workers, shuffle=False)
        nsel = torch.as_tensor(np.asarray(neurons), device=self.device)
        out = []
        for pix in loader:
            hs = self.model(pixel_values=pix.to(self.device)).last_hidden_state.float().half()
            tok = hs[:, self.n_prefix:]
            B, P, d = tok.shape
            z = self.sae.encode(self.norm(tok.reshape(B * P, d)), mode='threshold').view(B, P, -1)
            out.append(z[:, :, nsel].view(B, self.grid, self.grid, -1).float().cpu().numpy())
        return np.concatenate(out)


def representation(sae_name, dataset_dir='dataset', subject='mice', version='v1', domain=None):
    """'fg448' when the SAE's full codes come from the foreground pipeline (codes config.json has
    resolution 448, center_crop false, a foreground_rule), else 'crop224'."""
    cfg = _eci_dir(dataset_dir, subject, version, domain) / 'codes' / sae_name / 'config.json'
    if cfg.exists():
        c = json.loads(cfg.read_text())
        if c.get('resolution') == 448 and c.get('center_crop') is False and 'foreground_rule' in c:
            return 'fg448'
    return 'crop224'


def frame_box(rep, width, height):
    """(x0, y0, size) of the frame region the patch grid covers."""
    return (0.0, 0.0, float(width)) if rep == 'fg448' else crop_box(width, height)


class PatchEncoderFG:
    """fg448 representation: DINOv2 on the whole frame at 448 (32 x 32 patches), foreground mask from
    the video's stored background + the codes' foreground rule (src/eci/foreground.py RULES, default
    'fg448' = FG_RULE; FgBackgrounds), SAE on the foreground tokens only; non-foreground patches get
    code 0. Needs the annotations.csv row of every frame (to find its video's background and time block).
    Aligned SAEs (checkpoint 'align', e.g. 'odor' = fg448al): every frame is first turned by its row's
    90-degree turns (src/eci/foreground.py align_rot90, as src/eci/fg_encode.py does for the full codes) and
    the backgrounds must record the same alignment; the patch codes are then aligned-frame positions, so
    the frame they are drawn on must be turned the same way (self.rot, frame_rot90)."""

    rep = 'fg448'

    def __init__(self, sae_path, bg_dir, ann_path, device='cuda', rule_name='fg448'):
        from src.eci.foreground import GRID, RULES, FgBackgrounds, align_rot90, load_encoder_fg, obs_rows
        self.device = torch.device(device)
        _, self.processor, self.model = load_encoder_fg(device=self.device)
        self.sae, self.norm, ck = load_sae(sae_path, self.device)
        self.align = ck.get('align', 'none')
        self.rot = align_rot90(self.align, ann_path)  # None or (n_rows,) 90-degree CCW turns
        self.bgs = FgBackgrounds(bg_dir, obs_rows(ann_path), RULES[rule_name], self.device, align=self.align)
        self.grid = GRID

    @torch.no_grad()
    def patch_codes(self, paths, neurons, rows, batch_size=32, num_workers=8, return_mask=False):
        """(len(paths), 32, 32, len(neurons)) float32 patch codes (0 off the foreground)
        [, (len(paths), 32, 32) bool foreground mask]. Aligned SAEs: positions on the turned frame."""
        from src.eci.foreground import FrameDatasetFG, encode_batch
        rows = np.asarray(rows)
        rot = None if self.rot is None else self.rot[rows.astype(np.int64)]
        loader = torch.utils.data.DataLoader(FrameDatasetFG(list(paths), self.processor, rows, rot=rot),
                                             batch_size=batch_size, num_workers=num_workers, shuffle=False)
        nsel = torch.as_tensor(np.asarray(neurons), device=self.device)
        out, masks = [], []
        for pix, grey, r in loader:
            tok = encode_batch(self.model, pix, self.device)
            mask, _ = self.bgs.mask(tok, grey.to(self.device), r.numpy())
            B, P, _ = tok.shape
            fi, pi = torch.nonzero(mask, as_tuple=True)
            z = self.sae.encode(self.norm(tok[fi, pi]), mode='threshold')[:, nsel]
            full = torch.zeros(B, P, len(nsel), device=self.device)
            full[fi, pi] = z
            out.append(full.view(B, self.grid, self.grid, -1).cpu().numpy())
            masks.append(mask.view(B, self.grid, self.grid).cpu().numpy())
        pc = np.concatenate(out)
        return (pc, np.concatenate(masks)) if return_mask else pc


def frame_rot90(sae_name, dataset_dir='dataset', subject='mice', version='v1', domain=None):
    """Frame alignment of an SAE's codes (codes config.json 'align', absent = 'none'): None, or (n_rows,) int8
    90-degree counter-clockwise turns per annotations.csv row (src/eci/foreground.py align_rot90). Turn a frame
    with src/eci/foreground.py rotate_image before drawing that SAE's patch codes or arena map on it."""
    ds = Path(dataset_dir)
    cfg = _eci_dir(ds, subject, version, domain) / 'codes' / sae_name / 'config.json'
    align = json.loads(cfg.read_text()).get('align', 'none') if cfg.exists() else 'none'
    if align == 'none':
        return None
    from src.eci.foreground import align_rot90
    return align_rot90(align, ds / domain.ann_rel if domain is not None else ds / subject / version / 'annotations.csv')


def make_patch_encoder(sae_name, sae_path=None, dataset_dir='dataset', subject='mice', version='v1', device='cuda',
                       domain=None):
    """PatchEncoder (crop224) or PatchEncoderFG (fg448) for sae_name; .rep says which. Call
    patch_codes(paths, neurons) for crop224 and patch_codes(paths, neurons, rows) for fg448."""
    ds = Path(dataset_dir)
    eci = _eci_dir(ds, subject, version, domain)
    sae_path = sae_path or eci / 'sae' / sae_name / 'sae.pt'
    if representation(sae_name, ds, subject, version, domain) == 'fg448':
        c = json.loads((eci / 'codes' / sae_name / 'config.json').read_text())
        ann = ds / domain.ann_rel if domain is not None else ds / subject / version / 'annotations.csv'
        return PatchEncoderFG(sae_path, c['backgrounds'], ann, device, c.get('foreground_rule_name', 'fg448'))
    pe = PatchEncoder(sae_path, device=device)
    pe.rep = 'crop224'
    return pe


# ---------------------------------------------------------------------- plotting
def _tile(ax, img, hm, box, vmax, cmap, title, fontsize=6.5):
    from matplotlib.patches import Rectangle
    ax.imshow(img)
    if hm is not None and vmax > 0:
        x0, y0, c = box
        rgba = cmap(np.clip(hm / vmax, 0, 1))
        rgba[..., 3] = 0.65 * np.clip(hm / vmax, 0, 1) ** 0.7
        ax.imshow(rgba, extent=(x0, x0 + c, y0 + c, y0), interpolation='bilinear')
        ax.add_patch(Rectangle((x0, y0), c, c, fill=False, ls='--', lw=0.6, ec='white', alpha=0.7))
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize, pad=2, linespacing=1.1)


def plot_neuron(path, j, key, top, least, least_rule, table, hist, firing_rate, n_frames, source_name,
                extra=None, sae_name='', max_per_video=1, min_gap_s=2.0):
    """top / least: lists of dicts {img, hm, box, val, other, obs, frame, stage, genotype}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('turbo')
    other = 'max' if key == 'mean' else 'mean'
    ncol = 6
    fig = plt.figure(figsize=(15, 18), dpi=90)
    gs = fig.add_gridspec(7, ncol, height_ratios=[1, 1, 0.13, 1, 1, 0.06, 1.05], hspace=0.3, wspace=0.05,
                          left=0.035, right=0.985, top=0.905, bottom=0.04)
    vmax = max([t['hm'].max() for t in top if t['hm'] is not None] + [0])
    head = (f'neuron {j}  ({sae_name}, codes: {source_name}, n = {n_frames:,} frames)   '
            f'firing rate {firing_rate:.2%}   ranking: codes_{key}')
    if extra:
        head += '\n' + '   '.join(f'{k} = {v:.4g}' if isinstance(v, (float, np.floating)) else f'{k} = {v}'
                                   for k, v in extra.items())
    fig.suptitle(head, fontsize=11, y=0.99)

    def cap(t):
        return (f'{key} {t["val"]:.3f} | {other} {t["other"]:.3f}\n{t["obs"]} f{t["frame"]}\n'
                f'stage {t["stage"]} ({STAGE_LABEL[t["stage"]]}) | {t["genotype"]}')

    def block(tiles, row0, label):
        for k in range(2 * ncol):
            ax = fig.add_subplot(gs[row0 + k // ncol, k % ncol])
            if k < len(tiles):
                t = tiles[k]
                _tile(ax, t['img'], t['hm'], t['box'], vmax, cmap, cap(t))
            else:
                ax.axis('off')
            if k == 0:
                pos = ax.get_position()
                fig.text(pos.x0, pos.y1 + 0.036, label, fontsize=10, fontweight='bold', va='bottom')

    per = '1 per video' if max_per_video == 1 else f'<= {max_per_video} per video, >= {min_gap_s:g} s apart'
    block(top, 0, f'Top {len(top)} frames by codes_{key} ({per}); heatmap = per-patch code, '
                  f'shared scale 0-{vmax:.2f} (dashed = 224 crop seen by DINOv2)')
    rule = ('random frames with activation exactly 0 (seed 0, 1 per video)' if least_rule == 'random_zero'
            else 'bottom 12 frames (1 per video): fewer than 12 videos have a zero frame')
    block(least, 3, f'Least activated: {rule}; same heatmap scale')

    sub = gs[6, :].subgridspec(1, 2, wspace=0.14)
    ax = fig.add_subplot(sub[0])
    for g in ('wt', 'het'):
        d = table[table.genotype == g].sort_values('stage')
        if d.empty:
            continue
        off = -0.08 if g == 'wt' else 0.08
        ax.errorbar(d.stage + off, d['mean'], yerr=[d['mean'] - d.lo, d.hi - d['mean']], marker='o', ms=4,
                    capsize=3, lw=1.3, color=GENO_COLOR[g], label=f'{g} (n={int(d.n_pools.max())} pools)')
    ax.set_xticks(range(1, 7), [f'{s}\n{STAGE_LABEL[s]}' for s in range(1, 7)], fontsize=8)
    ax.set_xlabel('stage (phase, odor)', fontsize=8)
    ax.set_ylabel(f'per-video mean of codes_{key}', fontsize=8)
    ax.set_title('Mean activation per stage by genotype (mean over pools, 95% t-CI)', fontsize=9)
    ax.axvline(3.5, color='0.6', lw=0.6, ls=':')
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)

    ax = fig.add_subplot(sub[1])
    pos = hist[hist > 0]
    if len(pos):
        ax.hist(pos, bins=60, color='#555', alpha=0.85)
        ax.set_yscale('log')
    ax.set_title(f'Density of codes_{key} over {len(hist):,} random frames: {1 - len(pos) / max(len(hist), 1):.1%} '
                 f'exactly 0 (not shown), histogram of the > 0 values', fontsize=9)
    ax.set_xlabel(f'codes_{key}', fontsize=8)
    ax.set_ylabel('frames (log)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(path)
    plt.close(fig)


def _thumb(tiles, path, size=150, n=3):
    """Strip of the first n heatmap tiles (already rendered RGBs) for the index."""
    ims = [Image.fromarray(t['overlay']).resize((size, size), Image.BILINEAR) for t in tiles[:n]]
    strip = Image.new('RGB', (size * max(len(ims), 1), size), (128, 128, 128))
    for k, im in enumerate(ims):
        strip.paste(im, (k * size, 0))
    strip.save(path, quality=85)


def _overlay_rgb(img, hm, box, vmax, cmap):
    """Heatmap blended onto the frame, as an RGB uint8 array (for thumbnails)."""
    out = img.astype(np.float32) / 255
    if hm is not None and vmax > 0:
        x0, y0, c = box
        H = np.array(Image.fromarray(hm.astype(np.float32)).resize((int(round(c)), int(round(c))), Image.BILINEAR))
        x0, y0 = int(round(x0)), int(round(y0))
        w = np.clip(H / vmax, 0, 1)
        rgba = cmap(w)
        a = (0.65 * w ** 0.7)[..., None]
        sub = out[y0:y0 + H.shape[0], x0:x0 + H.shape[1]]
        out[y0:y0 + H.shape[0], x0:x0 + H.shape[1]] = (1 - a) * sub + a * rgba[..., :3]
    return (out * 255).astype(np.uint8)


def write_index(out_dir, table, sort_by='firing_rate', ascending=False, title='SAE neuron gallery', note=''):
    """out_dir/index.html: one card per neuron with thumbnail, stats and a link to the PNG.
    table: DataFrame indexed by neuron; every column is shown and sortable."""
    out_dir = Path(out_dir)
    table = table.sort_values(sort_by, ascending=ascending) if sort_by in table else table
    cols = list(table.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.3g}' if abs(v) >= 1e-3 or v == 0 else f'{v:.2e}'
        return html.escape(str(v))
    cards = []
    for j, r in table.iterrows():
        data = ' '.join(f'data-{c.lower().replace("_", "-")}="{r[c]}"' for c in cols
                        if isinstance(r[c], (int, float, np.integer, np.floating)))
        rows = ''.join(f'<tr><td>{html.escape(str(c))}</td><td>{fmt(r[c])}</td></tr>' for c in cols)
        cards.append(f'<a class="card" href="neuron_{j:04d}.png" data-neuron="{j}" {data}>'
                     f'<img loading="lazy" src="thumbs/neuron_{j:04d}.jpg" alt="neuron {j}">'
                     f'<div class="n">neuron {j}</div><table>{rows}</table></a>')
    opts = ''.join(f'<option value="{c.lower().replace("_", "-")}"{" selected" if c == sort_by else ""}>{c}</option>'
                   for c in ['neuron'] + cols)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fafaf8; --fg:#1d1d1f; --card:#fff; --line:#ddd; --muted:#666; color-scheme: light dark; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --fg:#e6e6e6; --card:#212226; --line:#383a40; --muted:#9a9a9a; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px system-ui, sans-serif; margin:0; padding:16px 20px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} .note {{ color:var(--muted); margin:0 0 12px; max-width:70em; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(460px, 100%), 1fr)); gap:12px; }}
.card {{ display:block; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px;
        color:inherit; text-decoration:none; }}
.card:hover {{ border-color:#2a78b5; }} .card img {{ width:100%; max-width:450px; display:block; border-radius:4px; }}
.n {{ font-weight:600; margin:6px 0 2px; }} table {{ font-size:12px; border-collapse:collapse; }}
td {{ padding:0 10px 0 0; }} td:first-child {{ color:var(--muted); }}
select {{ font:inherit; }}
</style></head><body>
<h1>{html.escape(title)}</h1><p class="note">{note}</p>
<p>Sort by <select id="key">{opts}</select> <label><input type="checkbox" id="asc"> ascending</label>
&nbsp; {len(table)} neurons. Click a card for the full figure.</p>
<div class="grid" id="grid">{''.join(cards)}</div>
<script>
const grid = document.getElementById('grid'), key = document.getElementById('key'), asc = document.getElementById('asc');
function sortCards() {{
  const k = key.value, s = asc.checked ? 1 : -1;
  const cards = [...grid.children];
  cards.sort((a, b) => s * ((parseFloat(a.dataset[k.replace(/-([a-z])/g, (m, c) => c.toUpperCase())]) || 0)
                          - (parseFloat(b.dataset[k.replace(/-([a-z])/g, (m, c) => c.toUpperCase())]) || 0)));
  cards.forEach(c => grid.appendChild(c));
}}
key.onchange = sortCards; asc.onchange = sortCards;
</script></body></html>"""
    (out_dir / 'index.html').write_text(page)
    return out_dir / 'index.html'


# ---------------------------------------------------------------------- end to end
def default_sae_path(sae_name, dataset_dir='dataset', subject='mice', version='v1', domain=None):
    return _eci_dir(dataset_dir, subject, version, domain) / 'sae' / sae_name / 'sae.pt'


def _tile_dicts(source, rows, vals, maps, idx, box_cache):
    meta = source.meta
    tiles = []
    for r, v in zip(rows, vals):
        m = meta.iloc[int(r)]
        path = source.dataset_dir / m['frame_path']
        with Image.open(path) as im:
            img = np.asarray(im.convert('RGB'))
        if img.shape[:2] not in box_cache:
            box_cache[img.shape[:2]] = crop_box(img.shape[1], img.shape[0])
        tiles.append({'img': img, 'hm': maps[int(r)][..., idx], 'box': box_cache[img.shape[:2]], 'val': float(v),
                      'other': np.nan, 'row': int(r),
                      'obs': m['observation_id'], 'frame': int(m['frame_idx']), 'stage': int(m['stage']),
                      'genotype': m['genotype']})
    return tiles


def gallery_for(neurons, stats_table=None, source=None, sae_name='matryoshka_btk_1024_k16_ep20_s0', out_dir=None,
                key='mean', n_tiles=12, max_per_video=1, min_gap_s=2.0, device='cuda', seed=0,
                sort_by='firing_rate', ascending=False, title=None, note='', dataset_dir='dataset',
                scan=None, patch_encoder=None, sae_path=None):
    """Render figures + index.html for `neurons` into out_dir.

    sae_path: SAE checkpoint used for the heatmaps (default dataset/mice/v1/eci/sae/<sae_name>/sae.pt);
    it must be the SAE that produced source.codes (the heatmap check in gallery.json would show otherwise).

    stats_table: optional DataFrame indexed by neuron id (e.g. NES output: tau, p-value, round);
    its columns are added to the figure headers and to the index, where they can be sorted on.
    source: a CodeSource (default: full codes if merged, else the training sample).
    Returns (out_dir, per-neuron stats DataFrame)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('turbo')
    ds = Path(dataset_dir)
    if source is None:
        try:
            source = load_full_codes(sae_name, ds)
        except FileNotFoundError:
            source = load_sample_codes(sae_name, ds, device=device)
    neurons = np.asarray(sorted(int(j) for j in neurons), dtype=np.int64)
    out_dir = Path(out_dir) if out_dir else Path('results/vision/mice/eci/galleries') / sae_name
    (out_dir / 'thumbs').mkdir(parents=True, exist_ok=True)
    other = 'max' if key == 'mean' else 'mean'
    t0 = time.time()

    if scan is None:
        scan = scan_codes(source, neurons, key, max_per_video, min_gap_s, seed=seed)
    t_scan = time.time() - t0
    table = neuron_stats(scan)

    picks = {}
    for i, j in enumerate(neurons):
        tr, tv = pick_top(scan, i, n_tiles)
        lr, rule = pick_least(scan, i, n_tiles, seed)
        picks[j] = (tr, tv, lr, rule)
    all_rows = np.unique(np.concatenate([np.r_[p[0], p[2]] for p in picks.values()])).astype(np.int64)
    t1 = time.time()
    pe = patch_encoder or PatchEncoder(sae_path or default_sae_path(sae_name, ds), device=device)
    pc = pe.patch_codes([str(source.dataset_dir / p) for p in source.meta['frame_path'].values[all_rows]], neurons)
    maps = dict(zip(all_rows.tolist(), pc))
    t_enc = time.time() - t1

    # check: recomputed patch maps pooled == stored pooled codes
    pool = (lambda a: a.mean((0, 1))) if key == 'mean' else (lambda a: a.max((0, 1)))
    stored = np.asarray(source.codes[key][np.sort(all_rows)][:, neurons], dtype=np.float32)
    recomp = np.stack([pool(maps[int(r)]) for r in np.sort(all_rows)])
    check = {'n_frames': int(len(all_rows)), 'max_abs_diff': float(np.abs(stored - recomp).max()),
             'max_stored': float(stored.max()),
             'active_set_mismatch_frac': float(((stored > 0) != (recomp > 0)).mean())}
    print(f'  heatmap check vs stored codes_{key}: {check}', flush=True)

    key_vals = dict(zip(np.sort(all_rows).tolist(), stored))
    other_vals = dict(zip(np.sort(all_rows).tolist(),
                          np.asarray(source.codes[other][np.sort(all_rows)][:, neurons], dtype=np.float32)))
    rules = {}
    t2 = time.time()
    box_cache = {}
    for i, j in enumerate(neurons):
        tr, tv, lr, rule = picks[j]
        rules[int(j)] = rule
        top = _tile_dicts(source, tr, tv, maps, i, box_cache)
        least = _tile_dicts(source, lr, [key_vals[int(r)][i] for r in lr], maps, i, box_cache)
        for t in top + least:
            t['other'] = float(other_vals[t['row']][i])
        extra = None
        if stats_table is not None and j in stats_table.index:
            extra = stats_table.loc[j].to_dict()
        plot_neuron(out_dir / f'neuron_{j:04d}.png', int(j), key, top, least, rule, stage_genotype_table(scan, i),
                    scan.hist_sample[:, i].astype(np.float32), table.loc[j, 'firing_rate'], scan.n_frames,
                    source.name, extra, sae_name, max_per_video, min_gap_s)
        vmax = max([t['hm'].max() for t in top] + [0])
        for t in top:
            t['overlay'] = _overlay_rgb(t['img'], t['hm'], t['box'], vmax, cmap)
        _thumb(top, out_dir / 'thumbs' / f'neuron_{j:04d}.jpg')
    t_plot = time.time() - t2

    table['least_rule'] = pd.Series(rules)
    if stats_table is not None:
        table = table.join(stats_table, how='left', rsuffix='_nes')
    table.to_csv(out_dir / 'neurons.csv')
    timing = {'scan_s': round(t_scan, 1), 'patch_encode_s': round(t_enc, 1), 'plot_s': round(t_plot, 1),
              'per_neuron_s': round((t_enc + t_plot) / len(neurons), 2), 'n_neurons': len(neurons)}
    (out_dir / 'gallery.json').write_text(json.dumps({
        'source': source.name, 'source_info': source.info, 'sae': sae_name, 'key': key, 'n_tiles': n_tiles,
        'max_per_video': max_per_video, 'min_gap_s': min_gap_s, 'seed': seed, 'neurons': neurons.tolist(),
        'heatmap_check': check, 'timing': timing}, indent=1))
    note = note or (f'SAE {sae_name}; codes: {source.name} ({scan.n_frames:,} frames); ranking by codes_{key}; '
                    f'top tiles at most {max_per_video} per video. Thumbnails: top-3 frames with the per-patch '
                    f'heatmap. least_rule: random_zero = random frames with activation 0, bottom = 12 lowest.')
    write_index(out_dir, table, sort_by, ascending, title or f'Neuron gallery: {sae_name}', html.escape(note))
    print(f'  gallery: {len(neurons)} neurons -> {out_dir}  timing {timing}', flush=True)
    return out_dir, table


# ---------------------------------------------------------------------- within-video view
def video_top_frames(t, y, n=6, min_gap_s=2.0):
    """Indices of the n largest y (> 0) with pairwise |t_a - t_b| >= min_gap_s (greedy NMS)."""
    y = np.asarray(y, dtype=np.float64).copy()
    picks = []
    for _ in range(n):
        a = int(np.argmax(y))
        if not np.isfinite(y[a]) or y[a] <= 0:
            break
        picks.append(a)
        y[np.abs(t - t[a]) < min_gap_s] = -np.inf
    return np.array(picks, dtype=np.int64)


def video_view(neuron, observation_id, source, out_path, sae_name='matryoshka_btk_1024_k16_ep20_s0', sae_path=None,
               key='mean', n_top=6, min_gap_s=2.0, smooth_s=10.0, device='cuda', patch_encoder=None, extra=None):
    """One PNG: activation of `neuron` over the whole video `observation_id` (frame values and a
    centred rolling mean over smooth_s), the n_top most active frames of THAT video (>= min_gap_s
    apart) marked on the curve and shown below with the per-patch heatmap.
    extra: optional dict (e.g. NES stats) printed in the header. Returns (out_path, picks DataFrame)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta = source.meta
    rows = np.flatnonzero(meta['observation_id'].values == observation_id)
    if len(rows) == 0:
        raise KeyError(f'{observation_id} not in the code source')
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    if hi - lo != len(rows):
        raise ValueError(f'rows of {observation_id} are not contiguous')
    m = meta.iloc[lo]
    fps = float(m['fps'])
    t = meta['frame_idx'].values[lo:hi] / fps
    y = np.asarray(source.codes[key][lo:hi, int(neuron)], dtype=np.float32)
    other = 'max' if key == 'mean' else 'mean'
    y_other = np.asarray(source.codes[other][lo:hi, int(neuron)], dtype=np.float32)
    picks = video_top_frames(t, y, n_top, min_gap_s)
    sparse = len(t) < 0.5 * (t[-1] - t[0]) * fps  # training sample: frames far apart, no smoothing

    pe = patch_encoder or PatchEncoder(sae_path or default_sae_path(sae_name, source.dataset_dir), device=device)
    maps = pe.patch_codes([str(source.dataset_dir / p) for p in meta['frame_path'].values[lo + picks]], [int(neuron)]) \
        if len(picks) else np.zeros((0, 16, 16, 1))
    vmax = float(maps.max()) if len(picks) else 0.0
    cmap = plt.get_cmap('turbo')

    ncol = max(n_top, 1)
    fig = plt.figure(figsize=(2.5 * ncol, 6.6), dpi=90)
    gs = fig.add_gridspec(2, ncol, height_ratios=[0.9, 1.2], hspace=0.42, wspace=0.05, left=0.05, right=0.99,
                          top=0.86, bottom=0.03)
    head = (f'neuron {neuron} in {observation_id}  (pool {m["pool"]}, stage {int(m["stage"])} '
            f'({STAGE_LABEL[int(m["stage"])]}), {m["genotype"]})   {sae_name}, codes_{key}   '
            f'video mean {y.mean():.3f}, fires in {(y > 0).mean():.1%} of {len(y):,} frames')
    if extra:
        head += '\n' + '   '.join(f'{k} = {v:.4g}' if isinstance(v, (float, np.floating)) else f'{k} = {v}'
                                   for k, v in extra.items())
    fig.suptitle(head, fontsize=10, y=0.985)
    ax = fig.add_subplot(gs[0, :])
    tm = t / 60
    if sparse:
        ax.plot(tm, y, '.-', ms=3, lw=0.6, color='0.35', label=f'codes_{key} (sampled frames)')
    else:
        ax.plot(tm, y, lw=0.4, color='0.7', label=f'codes_{key} per frame')
        w = max(1, int(round(smooth_s * fps)))
        ax.plot(tm, pd.Series(y).rolling(w, center=True, min_periods=1).mean().values, lw=1.4, color='#2a78b5',
                label=f'rolling mean ({smooth_s:g} s)')
    ax.axhline(y.mean(), color='#d9661f', lw=0.8, ls='--', label='video mean')
    for k, a in enumerate(picks):
        ax.plot(tm[a], y[a], 'v', color='#c0392b', ms=7)
        ax.annotate(str(k + 1), (tm[a], y[a]), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8,
                    color='#c0392b', fontweight='bold')
    ax.set_xlim(tm[0], tm[-1])
    ax.set_xlabel('time in video (min)', fontsize=8)
    ax.set_ylabel(f'codes_{key}', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(min(0, y.min()), 1.15 * max(y.max(), 1e-6))
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='lower right', bbox_to_anchor=(1, 1))
    ax.set_title(f'Activation over the video; numbered = top {len(picks)} frames of this video (>= {min_gap_s:g} s apart)',
                 fontsize=9, loc='left')

    box = None
    for k in range(ncol):
        ax = fig.add_subplot(gs[1, k])
        if k >= len(picks):
            ax.axis('off')
            continue
        r = lo + int(picks[k])
        with Image.open(source.dataset_dir / meta['frame_path'].values[r]) as im:
            img = np.asarray(im.convert('RGB'))
        box = box or crop_box(img.shape[1], img.shape[0])
        fi = int(meta['frame_idx'].values[r])
        _tile(ax, img, maps[k, ..., 0], box, vmax, cmap,
              f'#{k + 1}  t = {fi / fps / 60:.0f}:{fi / fps % 60:04.1f}  f{fi}\n'
              f'{key} {y[picks[k]]:.3f} | {other} {y_other[picks[k]]:.3f}', fontsize=7.5)
    fig.savefig(out_path)
    plt.close(fig)
    out = pd.DataFrame({'rank': np.arange(1, len(picks) + 1), 'row': lo + picks,
                        'frame_idx': meta['frame_idx'].values[lo + picks], 'time_s': t[picks],
                        f'codes_{key}': y[picks], f'codes_{other}': y_other[picks]})
    return out_path, out


def video_views(pairs, source, out_dir, sae_name='matryoshka_btk_1024_k16_ep20_s0', sae_path=None, key='mean',
                n_top=6, min_gap_s=2.0, device='cuda', stats_table=None, **kw):
    """video_view for each (neuron, observation_id) in pairs -> out_dir/video_nXXXX_<obs>.png.
    stats_table (indexed by neuron) adds that neuron's stats to the header. Returns the picks table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pe = PatchEncoder(sae_path or default_sae_path(sae_name, source.dataset_dir), device=device)
    res = []
    for j, obs in pairs:
        extra = stats_table.loc[j].to_dict() if stats_table is not None and j in stats_table.index else None
        _, df = video_view(j, obs, source, out_dir / f'video_n{int(j):04d}_{obs}.png', sae_name, key=key,
                           n_top=n_top, min_gap_s=min_gap_s, patch_encoder=pe, extra=extra, **kw)
        res.append(df.assign(neuron=int(j), observation_id=obs))
    table = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    table.to_csv(out_dir / 'video_views.csv', index=False)
    return table
