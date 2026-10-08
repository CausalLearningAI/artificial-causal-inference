#!/usr/bin/env python3
"""
Add the `annotation_end_frame` column to ants experiment.csv files (and, for
versions extended to 30 min, raise `end_frame`).

`annotation_end_frame` is in source-video frame units (30 fps, same coordinates
as start_frame / end_frame and the annotation files' Beginning-/End-frame).
get_annotations.py turns it into a per-frame `eval_window` flag and NaN outcomes
past it; an empty cell means "no limit" (versions without annotations).

Values (from the annotation files, see PLAN):
  v1      per row = end_frame   30-min annotations, row-specific start (600-1500)
  v2-v5   18000                 10-min annotations (max End-frame 18000; v5 17999)
  v6, vA  empty                 no annotations
  v5 also gets end_frame 18000 -> 54000 (30-min rebuild).

Edits are text-level: the column is appended at the end of each line, row order,
cell text and CRLF line endings (and a missing final newline) are kept. The
result is re-parsed and compared cell by cell against the original: only the
intended cells may change. Before writing, the original is copied to
<archive-root>/<version>/data/ants/<version>/experiment.csv (outside data/, mirroring
the repo-relative path; refuses to overwrite a different archived copy).

Usage:
  python scripts/05_deploy/add_annotation_window.py                 # dry run, all versions
  python scripts/05_deploy/add_annotation_window.py v5 --apply      # archive + write v5
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COL = "annotation_end_frame"

# version -> (annotation_end_frame: int | "end_frame" | "", new end_frame or None)
PLAN = {
    "v1": ("end_frame", None),
    "v2": (18000, None),
    "v3": (18000, None),
    "v4": (18000, None),
    "v5": (18000, 54000),
    "v6": ("", None),
    "vA": ("", None),
}


def _rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))


def edit(raw: bytes, ann_end, new_end) -> tuple[bytes, dict]:
    """Return the edited file bytes and a report. Raises if the file is not plain CSV."""
    text = raw.decode("utf-8")
    eol = "\r\n" if "\r\n" in text else "\n"
    if eol == "\r\n" and text.replace("\r\n", "").count("\n"):
        raise ValueError("mixed line endings")
    if '"' in text:
        raise ValueError("quoted cells: text-level edit not supported")
    final_eol = text.endswith(eol)
    lines = text[: -len(eol)].split(eol) if final_eol else text.split(eol)
    header = lines[0].split(",")
    if COL in header:
        raise ValueError(f"{COL} already present")
    i_end = header.index("end_frame")
    if ann_end == "end_frame" and new_end is not None:
        raise ValueError("annotation_end_frame=end_frame with a new end_frame is ambiguous")

    out = [lines[0] + "," + COL]
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) != len(header):
            raise ValueError(f"row has {len(cells)} cells, header {len(header)}: {line!r}")
        if new_end is not None:
            cells[i_end] = str(new_end)
        val = cells[i_end] if ann_end == "end_frame" else str(ann_end)
        out.append(",".join(cells + [val]))
    new_text = eol.join(out) + (eol if final_eol else "")
    new = new_text.encode("utf-8")

    # ── verify cell by cell against the original ──────────────────────────────
    old_rows, new_rows = _rows(text), _rows(new_text)
    assert len(old_rows) == len(new_rows), "row count changed"
    assert new_rows[0] == old_rows[0] + [COL], "header changed beyond the new column"
    changed = {"end_frame": 0, COL: 0, "other": 0}
    for o, n in zip(old_rows[1:], new_rows[1:]):
        assert len(n) == len(o) + 1
        for j, (a, b) in enumerate(zip(o, n)):
            if a != b:
                changed["end_frame" if j == i_end else "other"] += 1
        changed[COL] += 1
    assert changed["other"] == 0, f"unintended cells changed: {changed}"
    if new_end is None:
        assert changed["end_frame"] == 0
    assert new.count(b"\r\n") == raw.count(b"\r\n") and new.count(b"\n") == raw.count(b"\n"), \
        "line endings changed"
    assert new.endswith(b"\n") == raw.endswith(b"\n"), "final newline changed"

    values = sorted({r[-1] for r in new_rows[1:]}, key=lambda s: (s == "", s))
    report = {"n_rows": len(old_rows) - 1, "eol": repr(eol), "final_eol": final_eol,
              "cells_changed": changed, f"{COL}_values": values[:6] + (["..."] if len(values) > 6 else []),
              "header": new_rows[0], "old_sample": old_rows[1:4], "new_sample": new_rows[1:4]}
    return new, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("versions", nargs="*", default=list(PLAN))
    ap.add_argument("--apply", action="store_true", help="Archive the original and write the edit")
    ap.add_argument("--archive-root", type=Path, default=ROOT / "archive" / "10min" / "ants")
    args = ap.parse_args()

    for v in args.versions:
        ann_end, new_end = PLAN[v]
        path = ROOT / "data" / "ants" / v / "experiment.csv"
        raw = path.read_bytes()
        new, rep = edit(raw, ann_end, new_end)
        print(f"== {v}  ({path.relative_to(ROOT)})  rows={rep['n_rows']}  eol={rep['eol']}  "
              f"final_eol={rep['final_eol']}")
        print(f"   cells changed: {rep['cells_changed']}   {COL} values: {rep[f'{COL}_values']}")
        print(f"   header: {','.join(rep['header'])}")
        for o, n in zip(rep["old_sample"], rep["new_sample"]):
            print(f"   - {','.join(o)}\n   + {','.join(n)}")
        if not args.apply:
            continue
        arch = args.archive_root / v / path.relative_to(ROOT)
        if arch.exists() and arch.read_bytes() != raw:
            sys.exit(f"[ERROR] {arch} exists and differs from {path}; not overwriting the archive")
        arch.parent.mkdir(parents=True, exist_ok=True)
        if not arch.exists():
            shutil.copy2(path, arch)
        assert arch.read_bytes() == raw
        tmp = path.with_suffix(".csv.tmp")
        tmp.write_bytes(new)
        os.replace(tmp, path)
        assert path.read_bytes() == new
        print(f"   archived -> {arch.relative_to(ROOT)}; written {path.relative_to(ROOT)}")
    if not args.apply:
        print("\n[dry run] nothing written; pass --apply to archive and write")


if __name__ == "__main__":
    main()
