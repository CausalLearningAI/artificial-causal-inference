"""Render inventory.md from inventory.json (+crc_check.json) and measure on-disk size (run in a job)."""
import json, os
O = "results/vision/benchmark_mabe22/inventory"; D = "dataset/benchmark/mabe22"
inv = json.load(open(f"{O}/inventory.json")); crc = json.load(open(f"{O}/crc_check.json"))
def dsize(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs: t += os.path.getsize(os.path.join(r, f))
    return t
sizes = {k: dsize(f"{D}/{k}") for k in ["raw", "videos"]}; inv["disk_bytes"] = sizes; inv["crc_check"] = crc
json.dump(inv, open(f"{O}/inventory.json", "w"), indent=1)
G = lambda b: f"{b/1e9:.2f} GB"
md = ["# MABe22 inventory (measured from the downloaded files unless marked 'from paper')", "",
      f"On disk: raw/ {G(sizes['raw'])}, videos/ {G(sizes['videos'])}.", "", "## Files (all md5 + size matched the CaltechDATA record, checked in download job 828077)", "", "| file | bytes |", "|---|---|"]
md += [f"| {f['name']} | {f['size']:,} |" for f in inv["files"]]
md += ["(video zips were deleted after extraction was verified; their md5s matched the record before deletion)", "", "## Video (ffprobe on every extracted clip)", "", "| set | clips | WxH | codec/container | fps | packets per clip | total packets |", "|---|---|---|---|---|---|---|"]
for z, v in inv["video"].items():
    md.append(f"| {z} | {v['n_files']} | {list(v['width'])[0]}x{list(v['height'])[0]} | {list(v['codec'])[0]}/{list(v['container'])[0]} {list(v['pix_fmt'])[0]} | {list(v['avg_frame_rate'])[0]} | {v['packets_per_clip_min']}-{v['packets_per_clip_max']} | {v['total_packets']:,} |")
md += ["", "CRC-failed zip members (the zips themselves matched the record md5, so the corruption is upstream):", ""]
for k, v in crc.items(): md.append(f"- {k}: stored {v['stored_size']} B, CRC in zip {v['crc_in_zip']} vs recomputed {v['crc_computed']}; ffmpeg decode rc={v['decode_rc']}, frames={v['ffprobe']['streams'][0]['nb_read_frames']}, decoder messages: {v['decode_stderr'].strip()[:120] or 'none'}")
md += ["", "These 4 are kept under videos/crc_failed/ and excluded from videos/<set>/. One further mouse clip (R6R4SNHGFM0ZOOH9A1TN) is in the labels/frame map but not in the zip at all."]
def table(rows, cols):
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows: out.append("| " + " | ".join(f(r) if r.get("kind") == "binary" or c in ("name",) else "" for c, f in cols) + " |")
    return out
cols = [("row", lambda r: str(r["row"])), ("name", lambda r: r["name"]), ("type", lambda r: str(r["task_type"])), ("labelled frames", lambda r: f"{r['labelled_frames']:,}"),
        ("labelled clips", lambda r: str(r["labelled_clips"])), ("positive frames", lambda r: f"{r['positive_frames']:,}"), ("pos rate", lambda r: f"{r['positive_rate']:.4f}"),
        ("clips w/ pos", lambda r: str(r["clips_with_positive"])), ("bouts", lambda r: str(r.get("n_bouts", 0))), ("bout median/mean/p90 (frames)", lambda r: f"{r.get('bout_median',0):.0f}/{r.get('bout_mean',0):.1f}/{r.get('bout_p90',0):.0f}"),
        ("frac clips constant", lambda r: f"{r['frac_clips_constant']:.2f}")]
def sect(title, ls):
    o = [f"### {title}", f"clips {ls['n_clips']}, clip length {ls['clip_len']}, contiguous map {ls['contiguous']}", ""]
    o += table(ls["rows"], cols)
    cont = [r for r in ls["rows"] if r["kind"] != "binary"]
    for r in cont: o.append(f"\nrow {r['row']} {r['name']}: non-binary, {r['n_unique']} unique values, min {r['min']:.4g}, max {r['max']:.4g}, NaN frames {r['nan_frames']}, constant within clip for {r['frac_clips_constant']:.2f} of clips")
    return o + [""]
for dom in ["mouse", "beetle"]:
    d = inv["domains"][dom]; md += ["", f"## {dom}", "", f"label_array shape {d['label_array_shape']}; frame_number_map clips {d['frame_map_n_clips']}; frame map identical inside labels file and standalone file: {d['frame_map_equals_labels_frame_map']}", "",
        f"submission consistency: {json.dumps({k: (v if not isinstance(v, list) or len(v) < 6 else f'{len(v)} items, e.g. {v[:2]}') for k, v in d['submission_consistency'].items()})}", "",
        f"userTrain consistency: {json.dumps({k: (v if not isinstance(v, list) or len(v) < 6 else f'{len(v)} items, e.g. {v[:2]}') for k, v in d['user_train_consistency'].items()})}", "",
        f"keypoints submission: {d['submission_keypoints']}", f"keypoints userTrain: {d['user_train_keypoints']}", "",
        f"identical label rows in submission file: {json.dumps(d['row_equalities'])}"]
    if dom == "beetle": md.append("CORRECTED ROW MAPPING: the file's vocabulary list puts 'duration' at index 14 and 'idle_object_same' at 20, but row 20 is the only continuous row (task_type[20]=Continious; monotone 0..1 within every clip = time in trial) so it is duration; the names are rewritten as vocab[:14]+vocab[15:]+['duration'] here. Rows 14-19 (the *_same rows) are exact duplicates of rows 8-13 in the released array. Evidence: row_mapping_check.json. Rows 8-13 names rest on file order + paper list only.")
    if dom == "beetle": md.append(f"NaN pattern identical across rows 8-19: {d['nan_pattern_identical_rows_8_19']}")
    md += [""] + sect(f"{dom} SUBMISSION-set labels (videos available)", d["submission_labels"]) + sect(f"{dom} USERTRAIN labels (vocabulary {d['user_train_vocab']})", d["user_train_labels"])
    if dom == "mouse": md += sect("mouse pose-only TEST labels (no video released; mouse_triplets_test_labels.npy)", d["test_pose_only_labels"])
    md.append("examples: " + ", ".join(f"{e['png'].split('/')[-1]} ({e['clip']} f{e['frame']} {e['label']}={'pos' if e['positive'] else 'neg'})" for e in d["examples"]))
fl = inv["domains"]["fly"]["labels"]
md += ["", "## fly (labels only)", "", f"label_array {fl['label_array_shape']}, clips {fl['n_clips']} x {fl['clip_len']} frames; vocabulary length {fl['vocab_len']} but task_type length {fl['task_type_len']} (mismatch, unresolved)", ""]
md += ["| row | name | labelled frames | pos frames | pos rate | clips w/ pos |", "|---|---|---|---|---|---|"]
for r in fl["rows"]:
    md.append(f"| {r['row']} | {r['name']} | {r['labelled_frames']:,} | " + (f"{r['positive_frames']:,} | {r['positive_rate']:.4f} | {r['clips_with_positive']} |" if r["kind"] == "binary" else f"non-binary ({r['n_unique']} values) | | |"))
open(f"{O}/inventory.md", "w").write("\n".join(md) + "\n"); print("ok", sizes)
