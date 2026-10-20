"""
Frames -> kept tokens (ECI v2), one class per SERIES (a set of videos with one plan, one input geometry, one mask cue).

  decode     src/eci/v2/video.py: the 5 fps grid decoded once from the native source (source frame offset + step k).
  canonical  rot90 (mice: odour corner to the top right, dataset/mice/v1/eci/odor_corner.csv) and hflip (frogs: the
             mutant side to the frog's right, experiment.csv hflip, applied in the ffmpeg chain as the lab standard).
  geometry   'frame': the whole frame at side S (native, multiple of 16, never upsampled).
             'crop':  a SIDE x SIDE source-px window following the animal (label-free blob track of the r4 tadpole
                      locate step, dead band 2 px), edge-padded, resized to S (bicubic) = tadpoles 96 px -> 384 px.
  mask       src/eci/v2/mask.py tight rule (>= 25% animal pixels per patch) with the series' pixel cue.
  encoder    src/eci/v2/encode.py DINOv3-L/16 block 20.

Series config keys (configs/eci_v2/<domain>.json 'series'): name, experiment(s), plan (r4 plan json: videos with obs,
src_lo, lay_lo, n, path, hflip, offset, role), ann (annotations.csv of src_lo), fps, step, offset, S, geometry,
rot ('odor' | null), mask {cue, bg, align, dark_abs, dark_rel}, crop {side, locate}, layout (eci dir for codes).
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.eci.v2.encode import Encoder
from src.eci.v2.mask import THR, TightMask, grey_of
from src.eci.v2.video import FFMPEG, decode

REPO = Path(__file__).resolve().parents[3]


def obs_ranges(ann_path):
    o = pd.read_csv(ann_path, usecols=['observation_id'])['observation_id'].values
    ch = np.r_[0, np.flatnonzero(o[1:] != o[:-1]) + 1, len(o)]
    return {i: (int(s), int(e)) for i, s, e in zip(o[ch[:-1]], ch[:-1], ch[1:])}


def decode_native(path, s0, n, step, w, h, fps, threads=4):
    """Native-size frames source s0 + step i (i < n) -> (n, h, w, 3) uint8 (no scaling)."""
    vf = f"select='not(mod(n\\,{step}))'"
    cmd = [str(FFMPEG), '-nostdin', '-v', 'error', '-threads', str(threads)]
    if s0 > 0:
        cmd += ['-ss', f'{(s0 - 0.5) / fps:.6f}']
    cmd += ['-i', str(path), '-an', '-sn', '-vf', vf, '-fps_mode', 'passthrough', '-frames:v', str(n),
            '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    p = subprocess.run(cmd, capture_output=True)
    need = n * w * h * 3
    if p.returncode != 0 or len(p.stdout) != need:
        raise RuntimeError(f'decode {path} s0 {s0} n {n}: rc {p.returncode}, {len(p.stdout)} of {need} bytes; '
                           f'{p.stderr.decode()[-400:]}')
    return np.frombuffer(p.stdout, np.uint8).reshape(n, h, w, 3)


def boxes_of(sx, sy, side, band=2):
    """Smoothed centroid -> crop box top-left with a dead band (scripts/eci/tadpole_r4.py boxes_of, unchanged)."""
    cx = np.empty(len(sx), np.int32)
    cy = np.empty(len(sy), np.int32)
    ux, uy = int(np.round(sx[0])), int(np.round(sy[0]))
    for i in range(len(sx)):
        if abs(sx[i] - ux) > band or abs(sy[i] - uy) > band:
            ux, uy = int(np.round(sx[i])), int(np.round(sy[i]))
        cx[i], cy[i] = ux, uy
    return cx - side // 2, cy - side // 2


def crop_np(frame, x0, y0, side, fill=None):
    """(h, w[, c]) array, box top-left (may be outside) -> (side, side[, c]); edge padding (fill None) or constant."""
    pad = side // 2
    h, w = frame.shape[:2]
    if fill is None and 0 <= x0 and 0 <= y0 and x0 + side <= w and y0 + side <= h:
        return frame[y0:y0 + side, x0:x0 + side]
    pw = ((pad, pad), (pad, pad)) + ((0, 0),) * (frame.ndim - 2)
    fp = np.pad(frame, pw, mode='edge') if fill is None else np.pad(frame, pw, mode='constant', constant_values=fill)
    xa, ya = int(np.clip(x0 + pad, 0, fp.shape[1] - side)), int(np.clip(y0 + pad, 0, fp.shape[0] - side))
    return fp[ya:ya + side, xa:xa + side]


class Series:
    def __init__(self, cfg, dev, encoder=None):
        self.cfg, self.dev = cfg, dev
        self.P = json.loads((REPO / cfg['plan']).read_text())
        self.videos = self.P['videos']
        self.V = {v['obs']: v for v in self.videos}
        self.S = cfg['S']
        self.G = self.S // 16
        self.enc = encoder or Encoder(dev)
        self.geometry = cfg.get('geometry', 'frame')
        m = cfg['mask']
        if self.geometry == 'frame':
            self.mask = TightMask(REPO / m['bg'], obs_ranges(REPO / cfg['ann']), m['cue'], dev, align=m.get('align', 'none'),
                                  dark_abs=m.get('dark_abs', 60), dark_rel=m.get('dark_rel', 40))
        self.dark_rel = m.get('dark_rel', 40)
        self.rot = {}
        if cfg.get('rot') == 'odor':
            from src.eci.foreground import ODOR_ROT90
            oc = pd.read_csv(REPO / 'dataset/mice/v1/eci/odor_corner.csv').set_index('observation_id')['odor_corner']
            self.rot = {o: int(ODOR_ROT90[oc[o]]) for o in self.V}
        self._loc = {}

    # ------------------------------------------------------------------ decoding
    def voff(self, v):
        return int(v.get('offset', self.cfg['offset']))

    def job(self, v, k0, n, every=1, threads=2):
        """Decode kwargs for n frames k0, k0 + every, ... of video v (5 fps grid)."""
        c = self.cfg
        if self.geometry == 'crop':
            return dict(fn='native', path=v['path'], s0=self.voff(v) + c['step'] * k0, n=n, step=c['step'] * every,
                        w=v['w'], h=v['h'], fps=c['fps'], threads=threads)
        return dict(fn='scaled', path=v['path'], s0=self.voff(v) + c['step'] * k0, n=n, step=c['step'] * every,
                    S=self.S, hflip=v['hflip'], fps=c['fps'], threads=threads)

    @staticmethod
    def run_job(fn, **kw):
        return decode_native(**kw) if fn == 'native' else decode(**kw)

    # ------------------------------------------------------------------ crops (tadpoles)
    def locate(self, obs):
        if obs not in self._loc:
            if len(self._loc) > 8:
                self._loc.clear()
            L = np.load(Path(self.cfg['crop']['locate']) / f'{obs}.npz')
            d = {k: L[k] for k in ('sx', 'sy', 'pix_bg', 'roi')}
            d['x0'], d['y0'] = boxes_of(d['sx'], d['sy'], self.cfg['crop']['side'])
            self._loc[obs] = d
        return self._loc[obs]

    def crops(self, v, frames, ks):
        """Native frames (B, h, w, 3) of frames ks -> (x (B, 3, S, S) uint8 cuda, pix_bg (B, S, S) float, roi (B, S, S))."""
        side = self.cfg['crop']['side']
        L = self.locate(v['obs'])
        cr = np.stack([crop_np(f, int(L['x0'][k]), int(L['y0'][k]), side) for f, k in zip(frames, ks)])
        bg = np.stack([crop_np(L['pix_bg'], int(L['x0'][k]), int(L['y0'][k]), side) for k in ks])
        roi = np.stack([crop_np(L['roi'], int(L['x0'][k]), int(L['y0'][k]), side, fill=False) for k in ks])
        x = torch.from_numpy(np.ascontiguousarray(cr)).to(self.dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(self.S, self.S), mode='bicubic', align_corners=False).round().clamp(0, 255)
        b = F.interpolate(torch.from_numpy(bg).to(self.dev).float()[:, None], size=(self.S, self.S), mode='bilinear',
                          align_corners=False)[:, 0]
        r = F.interpolate(torch.from_numpy(roi.astype(np.float32)).to(self.dev)[:, None], size=(self.S, self.S),
                          mode='nearest')[:, 0] > 0.5
        return x.to(torch.uint8), b, r

    # ------------------------------------------------------------------ frames -> tokens
    @torch.no_grad()
    def frames(self, v, fr, ks):
        """Decoded frames of video v (numpy) for 5 fps frames ks -> x (B, 3, S, S) uint8 cuda canonical encoder input,
        fd (B, G*G) animal-pixel fraction per patch."""
        ks = np.asarray(ks)
        if self.geometry == 'crop':
            x, b, r = self.crops(v, fr, ks)
            g = grey_of(x).float()
            px = ((b - g) > self.dark_rel) & r
            fd = F.avg_pool2d(px.float()[:, None], 16).flatten(1)
            return x, fd
        x = torch.from_numpy(np.ascontiguousarray(fr)).to(self.dev).permute(0, 3, 1, 2)
        k = self.rot.get(v['obs'], 0)
        if k:
            x = torch.rot90(x, k, dims=(2, 3))
        x = x.contiguous()
        fd = self.mask.fraction(x, v['src_lo'] + ks)
        return x, fd

    @torch.no_grad()
    def tokens(self, x, fd):
        """-> z (B, G*G, 1024) fp16, keep (B, G*G) bool."""
        return self.enc(x), fd >= THR
