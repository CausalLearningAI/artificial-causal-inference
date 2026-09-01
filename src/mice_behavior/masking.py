"""Blank out a fixed region of every input frame, at training AND at inference.

WHY THIS EXISTS
===============
During the O phase a physical bag sits in one corner of the cage. build_derm.py's `phase_probe`
shows that this makes the treatment phase almost free to read off a QUIET frame -- one carrying no
scored behaviour and 5 s away from the nearest bout -- with a linear probe on a 32x32 grey
thumbnail, held out by pool:

    O vs not-O, whole frame        0.946 balanced accuracy   (chance 0.5)
    ... bottom-left quadrant only  0.903
    ... top-left / top-right / bottom-right   0.581 / 0.507 / 0.545
    ... the border ring            0.951
    ... centre only                0.656

and the per-pool peak of |mean(O) - mean(not-O)| lands in the bottom-left quadrant in all four
probed pools (rows 25-29 and columns 0-3 of 32, i.e. 78-93% down and 0-12% across).

That is the "bag shortcut" premise: a model can read the treatment without reading behaviour. This
module is the control that tests whether a given backbone actually USES it -- retrain with the bag's
corner blanked and see whether the estimand bias moves. Masking must be applied to the SAME region
at training and at evaluation, or the control changes the input distribution between the two and
measures that instead.

WHERE IN THE PIPELINE
=====================
On the decoded PIL frame, in ORIGINAL frame coordinates, BEFORE any D4 augmentation and before the
resize to the encoder's input size. Masking after D4 would blank a corner of the *rendering* while
leaving the bag itself wherever the rotation put it; masking after the resize is equivalent to
masking before it, but doing it first keeps one call site per decoder.

WHAT THE RESIDUAL MEANS
=======================
The same probe with `bottom_left` blanked still reads O vs not-O at 0.657. That residual is NOT a
failure of the mask. The bag also changes WHERE THE ANIMALS ARE, which is real behaviour and must
not be removed. The mask removes the non-behavioural cue and leaves the behavioural one.

CAVEAT ON COVERAGE: the probe was run on the four monitor pools (rd11_2, rd13, rd14, rd18) only.
The corner is the same in all four, but "the same corner in every pool" is an extrapolation from 4
of 24.
"""

# name -> tuple of (row0, row1, col0, col1) boxes in FRACTIONS of the frame edge.
#
# bottom_left is the bag corner at the largest size build_derm.py probed (side 0.25 of the edge =
# 6.25% of the frame area), which is the one that actually moves the probe: 0.946 -> 0.657 against
# 0.917 at side 0.125 and 0.783 at 0.1875. It covers every probed pool's peak cell with margin
# (peaks at rows 25-29/32 and columns 0-3/32; the box is rows 24-32, columns 0-8).
#
# four_corners is the AREA CONTROL, not used by default: it removes four times the pixels at all
# four corners, so "the masked arm changed" cannot be attributed to simply having fewer pixels.
REGIONS = {
    'bottom_left': ((0.75, 1.0, 0.0, 0.25),),
    'four_corners': ((0.0, 0.25, 0.0, 0.25), (0.0, 0.25, 0.75, 1.0),
                     (0.75, 1.0, 0.0, 0.25), (0.75, 1.0, 0.75, 1.0)),
}

# Black, matching build_derm.py's probe exactly, so its measured 0.946 -> 0.657 is a statement
# about the very pixels the trainer will see rather than about a differently-filled variant.
FILL = (0, 0, 0)


def region_boxes(name, width, height):
    """Fractional boxes -> integer pixel boxes (left, upper, right, lower) for PIL."""
    if name not in REGIONS:
        raise ValueError(f'unknown mask region {name!r}; known: {sorted(REGIONS)}')
    return [(int(round(c0 * width)), int(round(r0 * height)),
             int(round(c1 * width)), int(round(r1 * height)))
            for r0, r1, c0, c1 in REGIONS[name]]


def apply_mask(im, name, fill=FILL):
    """Blank `name`'s boxes on a PIL image, in place. `name` None/'' -> untouched, no copy."""
    if not name:
        return im
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for box in region_boxes(name, im.width, im.height):
        # PIL's rectangle is inclusive of the lower/right edge, so subtract one to land on
        # exactly (right - left) x (lower - upper) pixels.
        d.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=fill)
    return im


def frame_share(name):
    """Fraction of the frame the region removes, for logging."""
    if not name:
        return 0.0
    return sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in REGIONS[name])
