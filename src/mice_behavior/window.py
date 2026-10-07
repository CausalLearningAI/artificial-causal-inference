"""THE ESTIMATION WINDOW for the mice v1/v2 phase-transition estimand. One copy, imported everywhere.

Every recording is a separate video at 5 fps. Habituation (H) runs 30 minutes (9,000 frames);
exposure (O) and post-exposure (P) run 15 (4,500 frames). A phase mean over unequal stretches of a
decaying curve is not comparable across phases, so every outcome is measured on a 15-minute window
of each phase:

    H   minutes 15-30   (frames 4,500-9,000)   the LAST 15 minutes: the settled baseline that
                                               immediately precedes the exposure
    O   minutes  0-15   (frames     0-4,500)   the whole recording
    P   minutes  0-15   (frames     0-4,500)   the whole recording

DECIDED 2026-10-07 with the neuroscientists, replacing "the first 15 minutes of every phase". The
reason is biological: the exposure is meant to be compared against the baseline the animals have
settled into, not against the novelty of the first minutes in the cage. O->P is unaffected by the
choice (both phases are 15 minutes long); only H->O moves.

Bout onsets inside a window are measured from the WINDOW's start, so the `decay` unit (mean onset
minute) has the same 0-15 range, and the same flat-process null of 7.5, in every phase.

A bout already running when the window opens counts as a bout starting at the window's first
frame -- the same convention the first-15 window applied at its closing edge.
"""
from __future__ import annotations

import numpy as np

FPS = 5.0
WIN_MIN = 15
WIN = int(WIN_MIN * 60 * FPS)                 # 4,500 frames
PHASE_FRAMES = {'H': 2 * WIN, 'O': WIN, 'P': WIN}

# [start, stop) in frames, per phase
PHASE_WINDOW = {'H': (WIN, 2 * WIN), 'O': (0, WIN), 'P': (0, WIN)}
PHASE_WINDOW_MIN = {p: (a / FPS / 60, b / FPS / 60) for p, (a, b) in PHASE_WINDOW.items()}

# For prose and figure notes, so no sentence spells the rule by hand.
WINDOW_NAME = 'last 15 min of H'
WINDOW_TEXT = ('the last 15 minutes of habituation (minutes 15&ndash;30) against all 15 minutes '
               'of exposure and of post-exposure')


def phase_of(observation_id: str) -> str:
    """The phase letter, read off the observation_id suffix (`..._S_H`). Verified against
    experiment.csv's `phase` column for all 432 v1 and 216 v2 observations."""
    p = str(observation_id)[-1]
    if p not in PHASE_WINDOW:
        raise ValueError(f'cannot read a phase off observation_id {observation_id!r}')
    return p


def bounds(phase: str) -> tuple[int, int]:
    return PHASE_WINDOW[phase]


def window_mask(phase: str, frame_idx) -> np.ndarray:
    """Boolean mask over 0-based frame indices of one observation."""
    a, b = PHASE_WINDOW[phase]
    f = np.asarray(frame_idx)
    return (f >= a) & (f < b)


def window_slice(phase: str) -> slice:
    """Slice for a per-frame array whose index IS the frame index (stride 1, starting at 0)."""
    a, b = PHASE_WINDOW[phase]
    return slice(a, b)


def take(arr, observation_id: str):
    """`arr` restricted to its observation's window. Refuses a recording too short to hold it,
    rather than silently measuring a shorter stretch."""
    ph = phase_of(observation_id)
    a, b = PHASE_WINDOW[ph]
    if len(arr) < b:
        raise ValueError(f'{observation_id}: {len(arr)} frames, the {ph} window needs {b}')
    return arr[a:b]
