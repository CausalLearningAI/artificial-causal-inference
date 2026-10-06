"""Step 3: write the AMADEUS config.yaml that the Easy Tracking GUI would write.

    python make_config.py <video_id> [--analysis PATH] [--session DIR] [--variant idcorr|noidcorr]
                          [--workers N] [--epochs N] [--num-images N]

Mirrors gui/gui_easy_tracking.py::_build_config for a fresh session with these answers:
  number of animals  = SETUPS[species]['num_objects']
  overlap severity   = "super_heavy" (GUI label "Very heavy"): MIN/MAX_OVERLAP 0.01/0.5 and
                       EMBEDDING.ENABLE = True -> contrastive identity correction in refinement
  backward movement  = yes  (ants walk backwards when dragging, mice back up) -> the
                       "refine blobs through tracking" stage runs
  variable count / without-direction = no
Training/analysis video are the same analysis copy (work/<video_id>.mp4).

The "noidcorr" variant is NOT made here: run_video.sh copies the config.yaml that the main run
left behind (initial_tracking.py writes its auto-derived parameters into it), turns every stage
off except refinement and sets EMBEDDING.ENABLE=False. Refinement then rebuilds the plain
"filled" result from the same forward-tracking buffers (refinement.py writes either filled OR
id_resolved, never both).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import yaml

from pilot import AMADEUS_DIR, SETUPS, VIDEOS, out_dir

sys.path.insert(0, str(AMADEUS_DIR / "main"))
from segmentation_metadata import read_segmentation_metadata, segmentation_paths_for_session  # noqa: E402
from tracking_constants import DEFAULT_INTERACT_IOU  # noqa: E402
from experiment_utils import DEFAULT_LR0, DEFAULT_LRF  # noqa: E402

SKIPS = ["skip_initial_tracking", "skip_trajectory_direction_filtering", "skip_refine_blobs_through_tracking",
         "skip_paste_blobs_with_crossing", "skip_paste_blobs_clustered", "skip_cropping",
         "skip_creating_direction_dataset", "skip_training", "skip_detection", "skip_id_tracking",
         "skip_refinement", "skip_creating_video"]


def img_size_from_short(short: int) -> int:          # gui_easy_tracking._img_size_from_short
    return 1024 if short >= 1200 else max(32, (short // 32) * 32)


def geometry_flags(w: int, h: int, image_size: int) -> tuple[bool, int, bool]:   # _derive_dataset_geometry_flags
    short, long_ = min(w, h), max(w, h)
    if w == h and (0 <= short - image_size < 32 or short <= image_size):
        return True, 1, True
    tiles = max(1, min(4, max(1, long_ // image_size) * max(1, short // image_size)))
    return False, max(1, math.ceil(tiles / 2)), short >= 1200


def build(video_id: str, analysis: str, session: str, variant: str, workers: int, epochs: int,
          num_images: int) -> dict:
    species = VIDEOS[video_id]["species"]
    cap = cv2.VideoCapture(analysis)
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    img_size = img_size_from_short(min(vw, vh))
    skip_crop, num_crops, include_full = geometry_flags(vw, vh, img_size)
    pickle_path, background_path = segmentation_paths_for_session(session, analysis)
    meta = read_segmentation_metadata(pickle_path)
    images_per_frame = (0 if skip_crop else num_crops) + (1 if include_full else 0)
    cluster_frames = max(1, math.ceil(num_images * 0.05 / max(1, images_per_frame)))

    cfg = {
        "SESSION_PATH": session, "TRAINING_VIDEO_PATH": analysis, "TRACKING_VIDEO_PATH": analysis,
        "TRACKING_VIDEO_FILES": [], "NUM_OBJECTS": SETUPS[species]["num_objects"], "TRAIN_IMG_SIZE": img_size,
        "TRACKING_VIDEO_PATH_IS_DIR": False, "AUTO_PARAMS": True, "LOCALIZED_RATIO": 0.9,
        "MIN_OVERLAP": 0.01, "MAX_OVERLAP": 0.5, "DIR_MIN_SEC": 0.5, "WIDTH_SCALE_MIN": 0.9, "WIDTH_SCALE_MAX": 1.1,
        "NUM_CROPS": num_crops, "LOCALIZED": False, "USE_FULL": include_full, "USE_CROP": not skip_crop,
        "CLUSTER_FRAMES": cluster_frames, "SINGLE_PASTE": False,
        "NUM_WORKERS": workers, "NUM_PREVIEW_FRAMES": 20, "PREVIEW_INTERVAL": 100,
        "FRAME_INTERVAL": int(meta["training_frame_interval"]), "NUM_IMAGES": num_images, "RANDOM_SEED": 0,
        "delete_tmp_files": True,
        "INIT_MAX_GAP": 1, "SKIP_INIT_PREVIEW": False,
        "TRAJ_MAX_DIST": 2.0, "TRAJ_MAX_JUMP": 2.0, "MIN_ASPECT": 1.1, "DIR_MIN_DISP": 1.0, "SKIP_DIR_PREVIEW": False,
        "REFINE_FRAME_RATIO": 1.0, "REFINE_EPOCHS": 5, "REFINE_BATCH": "auto", "REFINE_ITERS": 1,
        "REFINE_MODEL": "yolo11n-obb", "REFINE_CONF": 0.2, "RUN_DELETE_RATIO": 0.4, "SKIP_REFINE_PREVIEW": False,
        "RATIO_SINGLE": 0.1, "RATIO_P2": 0.5, "RATIO_P3": 0.4, "FREE_SCALE": 0.25, "FREE_RATIO_SINGLE": 0.0,
        "FREE_RATIO_P2": 1.0, "FREE_RATIO_P3": 1.0,
        "MAX_TRIES": 100, "PASTE_SCALE_MIN": 0.9, "PASTE_SCALE_MAX": 1.1, "PASTE_MIN_ASPECT": 1.1,
        "PASTE_LAYER_MODE": "mixed", "UNDER_PASTE_PROB": 0.5, "OCCLUDER_MARGIN": -2, "ALPHA_MODE": "distance",
        "FEATHER_MIN": 2, "FEATHER_MAX": 4, "EDGE_BLUR_KSIZE": 7, "EDGE_BLUR_SIGMA": 11.0,
        "BRIGHT_MIN": 0.90, "BRIGHT_MAX": 1.10, "CONTRAST_MIN": 0.90, "CONTRAST_MAX": 1.10,
        "CLUSTERED_RATIO": 0.05, "CLUSTER_COUNT": 12, "CLUSTER_FIT_LONG": 0.8, "CLUSTER_FIT_SHORT": 0.8,
        "CLUSTER_BREAK_PROB": 0.20, "VAL_RATIO": 0.05,
        "training": {"EPOCHS": epochs, "LR0": DEFAULT_LR0, "LRF": DEFAULT_LRF, "SAVE_PERIOD": 5, "BATCH_SIZE": "auto",
                     "PRETRAINED_MODEL": "yolo11n-obb", "DEVICE": "auto",
                     "FIRST_FRAME": int(meta["training_frame_start"]), "LAST_FRAME": int(meta["training_frame_end"])},
        "analysis": {"DEVICE": "auto", "BATCH_SIZE": "auto", "WEIGHT": "last", "CONF": 0.1, "NMS_IOU": 0.8,
                     "SKIP_DETECT_PREVIEW": False, "MATCH_IOU": 0.5, "MATCH_ANGLE": 90.0, "MAX_AXIS_ERR": 45.0,
                     "MAX_AGE": 10, "FLIP_SEC": 5.0,
                     "TRACKING_COST": {"IOU_WEIGHT": 1.0, "DIRECTION_WEIGHT": 1.0, "MISS_WEIGHT": 1.0,
                                       "DISTANCE_WEIGHT": 1.0},
                     "FIRST_FRAME": 0, "LAST_FRAME": -1},
        "EMBEDDING": {"ENABLE": variant == "idcorr", "DEVICE": "auto", "IMG_SIZE": "auto", "PREVIEW_COUNT": 100,
                      "INTERACT_IOU": DEFAULT_INTERACT_IOU},
        "create_video": {"WEIGHT": "last", "FRAME_STEP": 1, "FPS": None, "ACCELERATION": "cpu", "EXPORT_RAW": False,
                         "EXPORT_IMAGES": False, "IMAGE_FORMAT": "jpeg", "DRAW_OBB": True, "DRAW_MODE": "obb",
                         "DRAW_LABELS": False, "DRAW_ARROW": True, "OBB_WIDTH": 1, "ARROW_WIDTH": 0,
                         "ARROW_ALPHA": 0.3, "ARROW_SCALE": 1.2, "LABEL_SCALE": 0.5, "LABEL_THICKNESS": 1},
        "PICKLE_PATH": pickle_path, "BACKGROUND_PATH": background_path,
        "VARIABLE_NUM_OBJECTS": False, "WITHOUT_DIRECTION_ESTIMATION": False,
    }
    flags = {k: False for k in SKIPS}
    flags["skip_cropping"] = skip_crop              # GUI sets this from the video geometry
    flags["skip_refine_blobs_through_tracking"] = False   # backward movement = yes
    flags["skip_creating_video"] = True             # we render our own sanity PNGs instead
    cfg.update(flags)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--variant", default="idcorr", choices=["idcorr", "noidcorr"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num-images", type=int, default=10000)
    ap.add_argument("--out", default=None, help="config path (default <session>/config.yaml)")
    args = ap.parse_args()
    od = out_dir(args.video_id)
    analysis = os.path.abspath(args.analysis or od / "work" / f"{args.video_id}.mp4")
    session = os.path.abspath(args.session or od / "session")
    cfg = build(args.video_id, analysis, session, args.variant, args.workers, args.epochs, args.num_images)
    out = Path(args.out or Path(session) / "config.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(out)


if __name__ == "__main__":
    main()
