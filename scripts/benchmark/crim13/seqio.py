"""Minimal reader for the Norpix .seq files of CRIM13 (Piotr Dollar's toolbox format, JPEG-compressed frames).
Measured layout (2026-10-15, from the files): 1024-byte header (magic 0xFEED, 'Norpix seq', version 3); width / height /
bit depth / image format (102 = jpg) / numFrames at bytes 548..584, fps (double) at 584. Each frame record is
uint32 n (bytes incl. these 4) + JPEG (n - 4 bytes) + timestamp (uint32 s, uint16 ms, uint16 us) + 8 bytes padding,
so frame k+1 starts at off_k + n_k + 16 (matches the released <name>-seek.mat offsets)."""
import struct

import cv2
import numpy as np


def header(path):
    with open(path, 'rb') as f:
        h = f.read(1024)
    magic = struct.unpack('<I', h[:4])[0]
    w, hh, bd, bdr, sb, fmt, nf, orig, tis = struct.unpack('<9I', h[548:584])
    fps = struct.unpack('<d', h[584:592])[0]
    return dict(magic=hex(magic), name=h[4:28].decode('utf-16-le', 'ignore').strip('\x00'),
                version=struct.unpack('<i', h[28:32])[0], header_size=struct.unpack('<I', h[32:36])[0], width=w,
                height=hh, bitdepth=bd, image_format=fmt, num_frames_header=nf, fps_header=fps)


def index(path):
    """Walk the frame records from byte 1024 to EOF. Returns offsets, sizes, timestamps (s)."""
    offs, ns, ts = [], [], []
    with open(path, 'rb') as f:
        f.seek(0, 2)
        end = f.tell()
        o = 1024
        while o + 4 <= end:
            f.seek(o)
            n = struct.unpack('<I', f.read(4))[0]
            if n < 4 or o + n + 8 > end:
                break
            f.seek(o + n)
            s, ms, us = struct.unpack('<IHH', f.read(8))
            offs.append(o)
            ns.append(n)
            ts.append(s + ms / 1e3 + us / 1e6)
            o += n + 16
    return np.array(offs, np.int64), np.array(ns, np.int64), np.array(ts), end - o


class Reader:
    def __init__(self, path, offs=None, ns=None):
        self.f = open(path, 'rb')
        if offs is None:
            offs, ns, _, _ = index(path)
        self.offs, self.ns = offs, ns

    def __len__(self):
        return len(self.offs)

    def raw(self, i):
        self.f.seek(int(self.offs[i]) + 4)
        return self.f.read(int(self.ns[i]) - 4)

    def gray(self, i, reduce=1):
        flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}[reduce]
        return cv2.imdecode(np.frombuffer(self.raw(i), np.uint8), flag)

    def close(self):
        self.f.close()
