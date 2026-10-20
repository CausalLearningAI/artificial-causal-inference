"""
Frame decoding (ECI v2): the 5 fps grid of the lab pipeline (annotations.csv rows) decoded ONCE from the native source
video with ffmpeg 6.1.1 (rgb24 through a pipe, no JPEG anywhere): source frame = OFFSET + STEP k (+ the video's
start frame), side S with the standardisation's chain [hflip,]scale=S:S:force_original_aspect_ratio=decrease,
pad=S:S:(ow-iw)/2:(oh-ih)/2. Identical to scripts/eci/rollout_r4.py decode / Prefetch (copied so the library has
no dependency on the rollout scripts).
"""
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FFMPEG = Path.home() / '.conda/envs/ffmpeg611/bin/ffmpeg'


def decode(path, s0, n, step, S, hflip, fps, threads=4):
    """Source frames s0 + step i (i < n) at side S -> (n, S, S, 3) uint8. Input seek to (s0 - 0.5) / fps (accurate seek:
    the first frame kept is the first with pts >= that time = source frame s0), then every step-th frame."""
    vf = f"select='not(mod(n\\,{step}))'" + (',hflip' if hflip else '') + \
        f',scale={S}:{S}:force_original_aspect_ratio=decrease,pad={S}:{S}:(ow-iw)/2:(oh-ih)/2'
    cmd = [str(FFMPEG), '-nostdin', '-v', 'error', '-threads', str(threads)]
    if s0 > 0:
        cmd += ['-ss', f'{(s0 - 0.5) / fps:.6f}']
    cmd += ['-i', str(path), '-an', '-sn', '-vf', vf, '-fps_mode', 'passthrough', '-frames:v', str(n),
            '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    p = subprocess.run(cmd, capture_output=True)
    need = n * S * S * 3
    if p.returncode != 0 or len(p.stdout) != need:
        raise RuntimeError(f'decode {path} s0 {s0} n {n}: rc {p.returncode}, {len(p.stdout)} of {need} bytes; '
                           f'{p.stderr.decode()[-400:]}')
    return np.frombuffer(p.stdout, np.uint8).reshape(n, S, S, 3)


class Prefetch:
    """Ordered decode jobs [(key, decode kwargs)] run `par` at a time in threads; iterating yields (key, frames)."""

    def __init__(self, jobs, par, fn=None):
        self.jobs, self.ex, self.par = list(jobs), ThreadPoolExecutor(par), par
        self.fn = fn or decode
        self.futs, self.next, self.dec_s = {}, 0, 0.0
        self.lock = threading.Lock()
        for _ in range(par + 1):
            self._submit()

    def _run(self, kw):
        t0 = time.time()
        x = self.fn(**kw)
        with self.lock:
            self.dec_s += time.time() - t0
        return x

    def _submit(self):
        if self.next < len(self.jobs):
            k, kw = self.jobs[self.next]
            self.futs[self.next] = (k, self.ex.submit(self._run, kw))
            self.next += 1

    def __iter__(self):
        for i in range(len(self.jobs)):
            k, f = self.futs.pop(i)
            x = f.result()
            self._submit()
            yield k, x
        self.ex.shutdown()
