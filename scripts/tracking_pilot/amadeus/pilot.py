"""Shared definitions for the AMADEUS tracking pilot (6 fixed videos).

Every other script in this folder imports from here, so paths and per-setup
choices live in exactly one place.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AMADEUS_DIR = Path(os.environ.get("AMADEUS_DIR", "/nfs/scistore19/locatgrp/rcadei/tools/AMADEUS"))
AMADEUS_PY = AMADEUS_DIR / ".venv" / "bin" / "python"
FFMPEG = AMADEUS_DIR / ".ffmpeg-hardware" / "btbn-2026-08-31-N-126342-linux64" / "ffmpeg"
RESULTS = REPO / "results" / "tracking_pilot"

# start_frame / end_frame are SOURCE frame indices, inclusive on both ends
# (src/data/standardize.py: duration = (end - start + 1) / fps). For mice they
# come from data/mice/v1/experiment.csv; ants use the whole video.
VIDEOS = {
    "rd25": dict(species="mice", path="data/mice/source/2024-12-02_15-34-27_BHVScreen_rd25_SocialOdor2_Habit1.mp4",
                 observation_id="wt_ash1l_m_1_S_H", start_frame=0, end_frame=54000),
    "rd32": dict(species="mice", path="data/mice/source/2025-02-07_12-18-47_BHVScreen_rd32_SocialOdor_Test.mp4",
                 observation_id="het_kdm6b_m_2_S_O", start_frame=0, end_frame=27000),
    "rd18": dict(species="mice", path="data/mice/source/2024-10-14_16-00-16_BHVScreen_rd18_SocialOdor_Post.mp4",
                 observation_id="het_kmt5b_m_1_S_P", start_frame=0, end_frame=27000),
    "3_1_1": dict(species="ants", path="data/ants/v3/observations/source/3_1_1.mkv", treatment=2,
                  start_frame=0, end_frame=-1),
    "3_6_6": dict(species="ants", path="data/ants/v3/observations/source/3_6_6.mkv", treatment=6,
                  start_frame=0, end_frame=-1),
    "3_21_2": dict(species="ants", path="data/ants/v3/observations/source/3_21_2.mkv", treatment=8,
                   start_frame=0, end_frame=-1),
}

# One setting per setup (species). scale = analysis px / source px.
SETUPS = {
    "mice": dict(num_objects=4, scale=0.5),
    "ants": dict(num_objects=3, scale=1.0),
}


def out_dir(video_id: str) -> Path:
    v = VIDEOS[video_id]
    return RESULTS / v["species"] / video_id / "amadeus"


def source_path(video_id: str) -> Path:
    return REPO / VIDEOS[video_id]["path"]


def ffprobe_facts(path: Path) -> dict:
    """Resolution, fps, duration and an exact decoded frame count."""
    def probe(*args):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", *args, "-of", "json", str(path)],
                             check=True, capture_output=True, text=True).stdout
        return json.loads(out)

    s = probe("-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt")
    s = s["streams"][0]
    fmt = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
                                     str(path)], check=True, capture_output=True, text=True).stdout)["format"]
    counted = probe("-count_frames", "-show_entries", "stream=nb_read_frames")["streams"][0]["nb_read_frames"]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(codec=s["codec_name"], width=int(s["width"]), height=int(s["height"]), pix_fmt=s.get("pix_fmt"),
                r_frame_rate=s["r_frame_rate"], fps=num / den, nb_frames_header=s.get("nb_frames"),
                nb_frames_decoded=int(counted), duration_sec=float(fmt["duration"]))
