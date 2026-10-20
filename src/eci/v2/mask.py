"""
Tight animal mask (ECI v2, accepted prior): a token is kept iff at least THR = 25% of the pixels of its 16 x 16 patch
of the encoder input are ANIMAL PIXELS by the domain's pixel cue. No dilation ring, no DINOv2 token cue, computed on
the fine (encoder) grid directly.

Pixel cues (one per domain; thresholds = the lab rules of src/eci/foreground.py):
  dark   (mice, ants: dark animals on light ground) grey < dark_abs (60) AND pix_bg - grey > dark_rel (40), pix_bg =
         the per-pixel background of the frame's video (or background window), the high (0.98) quantile of the video's
         grey levels (scripts/eci/fg_background.py), bilinearly resized from 512 px to the input side S. Static dark
         objects are dark in pix_bg too and never count.
  frog   (frogs juveniles, backlit dishes) pix_bg - grey > dark_rel (30) inside the dish ROI, pix_bg = the frog-free
         background of the video (scripts/eci/frogs_background.py). No SLEAP restriction (trackers are never part of
         the representation); see the v2 config for the measured effect.
grey = PIL convert('L') of the encoder input frame (after the canonical rotation / mirroring; the backgrounds of a
canonical domain are stored in canonical coordinates, e.g. mice fg448al).
"""
import numpy as np
import torch
import torch.nn.functional as F

THR = 0.25


def grey_of(x):
    """x (B, 3, H, W) uint8 -> (B, H, W) uint8 = PIL convert('L'): (R 19595 + G 38470 + B 7471 + 32768) >> 16."""
    t = x.int()
    return ((t[:, 0] * 19595 + t[:, 1] * 38470 + t[:, 2] * 7471 + 32768) >> 16).to(torch.uint8)


class TightMask:
    """Per-video pixel backgrounds -> animal-pixel fraction per encoder patch."""

    def __init__(self, bg_dir, obs_ranges, cue, dev, align='none', dark_abs=60, dark_rel=40, patch=16):
        from src.eci.foreground import RULES, FgBackgrounds
        rule = RULES['frogs'] if cue == 'frog' else RULES['fg448']
        self.bgs = FgBackgrounds(bg_dir, obs_ranges, rule, dev, align=align, cache=8)
        self.cue, self.dev, self.patch = cue, dev, patch
        self.dark_abs, self.dark_rel = dark_abs, dark_rel
        self._planes = {}

    def planes(self, rows, S):
        """-> list of (sel (n,) indices into rows, pix_bg (S, S) float, roi (S, S) bool or None) per background window."""
        rows = np.asarray(rows)
        k = self.bgs.window_index(rows)
        out = []
        for kk in np.unique(k):
            key = (self.bgs.wids[kk], S)
            if key not in self._planes:
                if len(self._planes) > 16:
                    self._planes.clear()
                b = self.bgs.get(self.bgs.wids[kk])
                pb = F.interpolate(b['pix_bg'].float()[None, None].to(self.dev), size=(S, S), mode='bilinear',
                                   align_corners=False)[0, 0]
                roi = None
                if 'roi' in b:
                    roi = F.interpolate(b['roi'].float()[None, None].to(self.dev), size=(S, S),
                                        mode='nearest')[0, 0] > 0.5
                self._planes[key] = (pb, roi)
            out.append((np.flatnonzero(k == kk), *self._planes[key]))
        return out

    @torch.no_grad()
    def pixels(self, x, rows):
        """x (B, 3, S, S) uint8 cuda (encoder input, canonical), rows (B,) annotations rows -> (B, S, S) bool."""
        S = x.shape[-1]
        g = grey_of(x).float()
        out = torch.zeros(g.shape, dtype=torch.bool, device=x.device)
        for sel, pb, roi in self.planes(rows, S):
            st = torch.from_numpy(sel).to(x.device)
            if self.cue == 'frog':
                px = (pb[None] - g[st]) > self.dark_rel
                if roi is not None:
                    px &= roi[None]
            else:
                px = (g[st] < self.dark_abs) & ((pb[None] - g[st]) > self.dark_rel)
            out[st] = px
        return out

    @torch.no_grad()
    def fraction(self, x, rows):
        """-> (B, G*G) float animal-pixel fraction per patch (G = S / patch)."""
        px = self.pixels(x, rows)
        return F.avg_pool2d(px.float()[:, None], self.patch).flatten(1)


def mask_sheet(items, path, G, tile=512):
    """items [(x (3, S, S) uint8 numpy, kept (G*G,) bool, title)] -> one JPEG (green tint = kept tokens)."""
    from PIL import Image, ImageDraw
    tiles = []
    for x, kept, title in items:
        im = np.asarray(Image.fromarray(x.transpose(1, 2, 0)).resize((tile, tile), Image.BILINEAR)).astype(np.float32)
        K = np.array(Image.fromarray(kept.reshape(G, G).astype(np.uint8) * 255).resize((tile, tile), Image.NEAREST)) > 0
        im[K] = 0.6 * im[K] + 0.4 * np.array([0, 255, 0])
        t = Image.fromarray(im.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(t).text((6, 6), title, fill=(255, 255, 0))
        tiles.append(t)
    cols = 4
    sh = Image.new('RGB', (cols * tile, tile * int(np.ceil(len(tiles) / cols))))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % cols) * tile, (i // cols) * tile))
    sh.save(path, quality=80)
