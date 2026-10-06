"""Step 1: make the frame-exact "analysis copy" AMADEUS tracks, and record source facts.

    python prepare_video.py <video_id> [--max-frames N]

Writes into results/tracking_pilot/{species}/{video_id}/amadeus/:
  work/<video_id>.mp4   trimmed to the source window, scaled by SETUPS[species]['scale'],
                        H.264 crf 12 / GOP 30 (the same encoder settings AMADEUS's own
                        conversion uses, main/video_compat.py), constant 30 fps.
  prep.json             ffprobe facts of source and analysis copy, the frame mapping,
                        and the result of the frame-alignment check.

Frame mapping: analysis frame i == source frame start_frame + i (trim on frame numbers,
no fps change). This is verified, not assumed: a few source frames are decoded exactly
(ffmpeg select on n, sequential decode) and compared with analysis frames i-1, i, i+1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from pilot import FFMPEG, SETUPS, VIDEOS, ffprobe_facts, out_dir, source_path


def grab_frames(path: Path, indices: list[int], scale_to: tuple[int, int] | None) -> dict[int, np.ndarray]:
    """Decode exact frame numbers (sequential decode, so no seek inaccuracy)."""
    with tempfile.TemporaryDirectory() as tmp:
        expr = "+".join(f"eq(n\\,{i})" for i in indices)
        vf = f"select='{expr}'"
        if scale_to:
            vf += f",scale={scale_to[0]}:{scale_to[1]}:flags=area"
        subprocess.run([str(FFMPEG), "-v", "error", "-i", str(path), "-vf", vf, "-fps_mode", "passthrough",
                        f"{tmp}/f_%04d.png"], check=True)
        files = sorted(Path(tmp).glob("f_*.png"))
        assert len(files) == len(indices), (len(files), indices)
        return {i: cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) for i, f in zip(sorted(indices), files)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: only keep the first N frames of the window")
    ap.add_argument("--suffix", default="", help="debug: name suffix for the analysis copy")
    args = ap.parse_args()

    vid = args.video_id
    v = VIDEOS[vid]
    setup = SETUPS[v["species"]]
    src = source_path(vid)
    od = out_dir(vid)
    (od / "work").mkdir(parents=True, exist_ok=True)

    src_facts = ffprobe_facts(src)
    start = int(v["start_frame"])
    end = int(v["end_frame"]) if int(v["end_frame"]) >= 0 else src_facts["nb_frames_decoded"] - 1
    end = min(end, src_facts["nb_frames_decoded"] - 1)
    if args.max_frames:
        end = min(end, start + args.max_frames - 1)
    n_expected = end - start + 1

    scale = float(setup["scale"])
    w = int(round(src_facts["width"] * scale)) // 2 * 2
    h = int(round(src_facts["height"] * scale)) // 2 * 2
    dst = od / "work" / f"{vid}{args.suffix}.mp4"
    vf = [f"trim=start_frame={start}:end_frame={end + 1}", "setpts=PTS-STARTPTS", f"fps={src_facts['r_frame_rate']}"]
    if (w, h) != (src_facts["width"], src_facts["height"]):
        vf.append(f"scale={w}:{h}:flags=area")
    vf.append("format=yuv420p")
    cmd = [str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-an", "-sn",
           "-dn", "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p",
           "-g", "30", "-keyint_min", "30", "-r", src_facts["r_frame_rate"], "-movflags", "+faststart",
           "-map_metadata", "-1", str(dst)]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    encode_sec = time.time() - t0
    dst_facts = ffprobe_facts(dst)
    if dst_facts["nb_frames_decoded"] != n_expected:
        raise SystemExit(f"frame count mismatch: analysis copy has {dst_facts['nb_frames_decoded']}, "
                         f"expected {n_expected}")

    # Frame-alignment check at 3 points (start, middle, near end of the window).
    checks = []
    probe_rel = sorted({1, n_expected // 2, n_expected - 2})
    src_frames = grab_frames(src, [start + i for i in probe_rel], (w, h))
    ana_idx = sorted({j for i in probe_rel for j in (i - 1, i, i + 1)})
    ana_frames = grab_frames(dst, ana_idx, None)
    for i in probe_rel:
        diffs = {d: float(np.abs(src_frames[start + i] - ana_frames[i + d]).mean()) for d in (-1, 0, 1)}
        checks.append(dict(analysis_frame=i, source_frame=start + i, mad_prev=diffs[-1], mad_same=diffs[0],
                           mad_next=diffs[1], aligned=bool(diffs[0] < min(diffs[-1], diffs[1]))))
    ok = all(c["aligned"] for c in checks)

    prep = dict(video_id=vid, species=v["species"], source_path=str(src), source=src_facts,
                window=dict(start_frame=start, end_frame=end, n_frames=n_expected, inclusive=True),
                analysis_path=str(dst), analysis=dst_facts, scale=scale,
                scale_xy=[dst_facts["width"] / src_facts["width"], dst_facts["height"] / src_facts["height"]],
                frame_mapping="frame_src = start_frame + frame_analysis (same fps, frame-exact trim)",
                encode_cmd=cmd, encode_sec=encode_sec, alignment_checks=checks, alignment_ok=ok)
    (od / f"prep{args.suffix}.json").write_text(json.dumps(prep, indent=2))
    print(json.dumps({k: prep[k] for k in ("video_id", "window", "scale_xy", "encode_sec", "alignment_ok")}))
    for c in checks:
        print(c)
    if not ok:
        raise SystemExit("frame alignment check FAILED")


if __name__ == "__main__":
    main()
