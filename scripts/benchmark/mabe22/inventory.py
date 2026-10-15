"""MABe22 inventory (run inside a Slurm job). Writes inventory.json, inventory.md, examples/*.png."""
import json, os, subprocess, zipfile, hashlib, re, tempfile, shutil
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw

D = "dataset/benchmark/mabe22"; R = f"{D}/raw"; V = f"{D}/videos"
O = "results/vision/benchmark_mabe22/inventory"; os.makedirs(f"{O}/examples", exist_ok=True)
L = lambda f: np.load(f"{R}/{f}", allow_pickle=True)

# ---------------- 1. files ----------------
rec = {e["key"]: e for e in json.load(open(f"{R}/record_files.json"))["entries"]}
files = []
for f in sorted(os.listdir(R)):
    if f == "record_files.json": continue
    sz = os.path.getsize(f"{R}/{f}")
    files.append(dict(name=f, size=sz, record_size=rec[f]["size"], size_match=sz == rec[f]["size"], record_md5=rec[f]["checksum"],
                      md5_verified="yes (download job 828077 recomputed md5 and matched record)"))

# ---------------- 2. video probing ----------------
def probe(path):
    try:
        j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration:format=format_name", "-of", "json", path],
            capture_output=True, text=True, timeout=120).stdout)
        s = j["streams"][0]; s["format"] = j["format"]["format_name"]; return s
    except Exception as e:
        return dict(error=repr(e))
def probe_dir(z):
    d = f"{V}/{z}"; names = sorted(os.listdir(d))
    with Pool(8) as p: res = p.map(probe, [f"{d}/{n}" for n in names], chunksize=16)
    return {n[:-4]: r for n, r in zip(names, res)}
vid = {z: probe_dir(z) for z in ["mouse_userTrain", "mouse_submission", "beetle_userTrain", "beetle_submission"]}
def vsummary(pr):
    ok = [r for r in pr.values() if "error" not in r]
    c = lambda k: {str(a): sum(1 for r in ok if r.get(k) == a) for a in sorted({r.get(k) for r in ok}, key=str)}
    nf = np.array([int(r["nb_read_packets"]) for r in ok])
    return dict(n_files=len(pr), n_probe_errors=len(pr) - len(ok), width=c("width"), height=c("height"), codec=c("codec_name"), pix_fmt=c("pix_fmt"),
                r_frame_rate=c("r_frame_rate"), avg_frame_rate=c("avg_frame_rate"), container=c("format"),
                packets_per_clip_min=int(nf.min()), packets_per_clip_max=int(nf.max()), total_packets=int(nf.sum()),
                header_nb_frames_equals_packets=sum(1 for r in ok if r.get("nb_frames") == r.get("nb_read_packets")))
video = {z: vsummary(p) for z, p in vid.items()}

bad_members = 'see crc_check.json (zips deleted after extraction)'

# ---------------- 3. labels ----------------
def bouts(x):  # x: (n_clips, T) bool -> run lengths (within-clip; truncated at clip edges)
    n, T = x.shape
    pad = np.zeros((n, T + 2), np.int8); pad[:, 1:-1] = x
    d = np.diff(pad, axis=1)
    s = np.argwhere(d == 1); e = np.argwhere(d == -1)
    return (e[:, 1] - s[:, 1])
def label_stats(arr, vocab, ttype, fmap, ids_with_video, name_override=None):
    """arr: (n_rows, N) float; fmap: dict clip->(start,end). Returns per-row stats."""
    ids = sorted(fmap, key=lambda k: fmap[k][0]); st = np.array([fmap[k][0] for k in ids]); en = np.array([fmap[k][1] for k in ids])
    lens = en - st; T = int(lens[0]); contiguous = bool(np.all(st[1:] == en[:-1]) and st[0] == 0 and en[-1] == arr.shape[1]); uniform = bool(np.all(lens == T))
    out = dict(clip_len=T if uniform else None, uniform=uniform, contiguous=contiguous, n_clips=len(ids), rows=[])
    for r, v in enumerate(vocab):
        x = arr[r].reshape(len(ids), T) if uniform else None
        valid = ~np.isnan(x); u = np.unique(x[valid])
        row = dict(row=r, name=v, task_type=ttype[r] if r < len(ttype) else None, n_unique=int(len(u)), unique_head=[float(a) for a in u[:6]],
                   labelled_frames=int(valid.sum()), nan_frames=int((~valid).sum()), labelled_clips=int(valid.any(1).sum()),
                   fully_labelled_clips=int(valid.all(1).sum()))
        if len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}:
            pos = (x == 1); row.update(kind="binary", positive_frames=int(pos.sum()), positive_rate=float(pos.sum() / max(valid.sum(), 1)),
                clips_with_positive=int(pos.any(1).sum()),
                clips_all_positive=int((pos & valid).all(1).sum()),
                clips_with_positive_and_video=int(sum(1 for k, a in zip(ids, pos.any(1)) if a and k in ids_with_video)))
            b = bouts(pos)
            if len(b): row.update(n_bouts=int(len(b)), bout_median=float(np.median(b)), bout_mean=float(b.mean()), bout_p90=float(np.percentile(b, 90)), bout_max=int(b.max()))
            # constant-within-clip (a condition-like label) vs varying
            row["frac_clips_constant"] = float(np.mean([(c[~np.isnan(c)].min() == c[~np.isnan(c)].max()) if (~np.isnan(c)).any() else True for c in x]))
        else:
            row.update(kind="continuous_or_multivalued", min=float(np.nanmin(x)), max=float(np.nanmax(x)),
                       frac_clips_constant=float(np.mean([np.nanmin(c) == np.nanmax(c) for c in x])))
        out["rows"].append(row)
    return out, ids, x if False else None
def kp_stats(sk):
    ks = list(sk)[:200]; a = np.stack([sk[k]["keypoints"] for k in ks]); return dict(shape_per_clip=list(sk[ks[0]]["keypoints"].shape), dtype=str(a.dtype),
        min=float(a.min()), max=float(a.max()), n_nan=int(np.isnan(a.astype(float)).sum()) if a.dtype.kind == "f" else 0, sample_clips=len(ks))

inv = dict(files=files, video=video, crc_bad_members=bad_members, domains={})
vsets = {z: set(p) for z, p in vid.items()}
for dom in ["mouse", "beetle"]:
    ut = L(f"{dom}_user_train.npy").item(); sub = L(f"{dom}_submission_labels.npy").item(); fm = L(f"{dom}_frame_number_map.npy").item()
    sk = L(f"{dom}_submission_keypoints.npy").item()["sequences"]
    orig_vocab = list(sub["vocabulary"])
    if dom == "beetle":  # file's vocabulary is misaligned: 'duration' (idx 14) belongs at idx 20 (the only continuous row); see row_mapping_check.py
        sub["vocabulary"] = orig_vocab[:14] + orig_vocab[15:] + [orig_vocab[14]]
    d = dict(user_train_vocab=ut["vocabulary"], submission_vocab=sub["vocabulary"], submission_vocab_as_in_file=orig_vocab, submission_task_type=sub["task_type"])
    d["frame_map_equals_labels_frame_map"] = fm == sub["frame_number_map"]
    d["frame_map_n_clips"] = len(fm); d["label_array_shape"] = list(sub["label_array"].shape)
    ls, ids, _ = label_stats(sub["label_array"], sub["vocabulary"], sub["task_type"], fm, vsets[f"{dom}_submission"])
    d["submission_labels"] = ls
    # consistency: video packets vs map length
    T = ls["clip_len"]; pk = vid[f"{dom}_submission"]
    d["submission_consistency"] = dict(fmap_clips_without_video=sorted(set(fm) - set(pk)), videos_without_fmap=sorted(set(pk) - set(fm)),
        videos_with_packets_ne_maplen=[k for k, r in pk.items() if k in fm and int(r["nb_read_packets"]) != fm[k][1] - fm[k][0]],
        keypoint_ids_equal_fmap=set(sk) == set(fm), keypoint_len_ne_maplen=[k for k in fm if sk[k]["keypoints"].shape[0] != fm[k][1] - fm[k][0]][:5])
    d["submission_keypoints"] = kp_stats(sk)
    # user train
    seqs = ut["sequences"]; uids = sorted(seqs); T2 = seqs[uids[0]]["annotations"].shape[1]
    ann = np.stack([seqs[k]["annotations"] for k in uids])  # (n, nvocab, T)
    arr = ann.transpose(1, 0, 2).reshape(ann.shape[1], -1)
    ufm = {k: (i * T2, (i + 1) * T2) for i, k in enumerate(uids)}
    uls, _, _ = label_stats(arr, ut["vocabulary"], ["?"] * len(ut["vocabulary"]), ufm, vsets[f"{dom}_userTrain"])
    d["user_train_labels"] = uls
    d["user_train_consistency"] = dict(n_sequences=len(uids), clip_len=T2, ids_without_video=sorted(set(uids) - vsets[f"{dom}_userTrain"]), videos_without_ids=sorted(vsets[f"{dom}_userTrain"] - set(uids)),
        ids_overlap_with_submission=len(set(uids) & set(fm)),
        keypoint_len_ne_T=int(sum(seqs[k]["keypoints"].shape[0] != T2 for k in uids)),
        videos_with_packets_ne_T=[k for k, r in vid[f"{dom}_userTrain"].items() if int(r["nb_read_packets"]) != T2])
    d["user_train_keypoints"] = kp_stats({k: seqs[k] for k in uids})
    # beetle row-equality check (vocabulary order ambiguity)
    la = sub["label_array"]
    d["row_equalities"] = [dict(a=sub["vocabulary"][i], b=sub["vocabulary"][j], identical=bool(np.array_equal(la[i], la[j], equal_nan=True)))
                           for i in range(la.shape[0]) for j in range(i + 1, la.shape[0]) if np.array_equal(la[i], la[j], equal_nan=True)]
    if dom == "beetle":
        nanmask = np.isnan(la[8:20]); d["nan_pattern_identical_rows_8_19"] = bool(np.all(nanmask == nanmask[0]))
    # user_train vs submission: also do the labelled positives in userTrain per clip
    inv["domains"][dom] = d
    # ---- examples ----
    rows = [r for r in ls["rows"] if r["kind"] == "binary" and r["frac_clips_constant"] < 1.0 and r["name"] in (
        ["approaches","chases","close","contact","huddles","oral_ear_contact","oral_genital_contact","oral_oral_contact","watching"] if dom == "mouse" else
        ["grooming_self","grooming_object","exploring_alone","exploring_object","idle_alone","idle_object"]) and r["clips_with_positive_and_video"] > 0]
    rows.sort(key=lambda r: r["positive_rate"]); rare, common = rows[0], rows[-1]
    names = sub["vocabulary"]; srt = ids
    def pick(rowname, pos=True):
        ri = names.index(rowname)
        for k in srt:
            if k not in pk or k in (set(fm) - set(pk)): continue
            s, e = fm[k]; seg = la[ri, s:e]
            if pos and (seg == 1).any(): return k, int(np.argmax(seg == 1))
            if not pos and np.all(la[:, s:e][[names.index(n) for n in names if n in ["chases","huddles","oral_genital_contact","oral_oral_contact"] or n.startswith(("grooming","exploring","idle"))]] == 0): return k, (e - s) // 2
        return None, None
    ex = [(rare["name"], True), (common["name"], True), (rare["name"], False)]
    d["examples"] = []
    for n, (rn, pos) in enumerate(ex):
        k, f = pick(rn, pos)
        if k is None and not pos:  # no all-negative frame exists (beetle: every labelled frame has some behaviour) -> second-rarest label, positive
            rn = rows[1]["name"]; pos = True; k, f = pick(rn, True)
        if k is None: continue
        s, e = fm[k]; png = f"{O}/examples/{dom}_{n}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{V}/{dom}_submission/{k}.avi", "-vf", f"select=eq(n\\,{f})", "-vsync", "0", "-frames:v", "1", png], check=True)
        im = Image.open(png).convert("RGB"); W = im.size[0]; im = im.resize((448, 448), Image.NEAREST) if W < 400 else im
        dr = ImageDraw.Draw(im); active = [names[i] for i in range(la.shape[0]) if not np.isnan(la[i, s + f]) and la[i, s + f] == 1]
        txt = f"{dom} clip {k} frame {f}/{e - s}\n{'POSITIVE' if pos else 'NEGATIVE'} for {rn}\nactive labels: {', '.join(active) or 'none'}"
        dr.rectangle([0, 0, im.size[0], 48], fill=(0, 0, 0)); dr.text((4, 2), txt, fill=(255, 255, 0))
        im.save(png); d["examples"].append(dict(png=png, clip=k, frame=f, label=rn, positive=pos, size_in_file=[W, W], active_labels=active))
    d["example_rare_label"] = rare["name"]
# mouse pose-only test labels and fly
mt = L("mouse_triplets_test_labels.npy").item()
mls, _, _ = label_stats(mt["label_array"], mt["vocabulary"], mt["task_type"], mt["frame_number_map"], set())
inv["domains"]["mouse"]["test_pose_only_labels"] = mls; inv["domains"]["mouse"]["test_pose_only_labels"]["label_array_shape"] = list(mt["label_array"].shape)
fl = L("fly_groups_test_labels.npy").item()
fls, _, _ = label_stats(fl["label_array"], fl["vocabulary"], fl["task_type"], fl["frame_number_map"], set())
fls["label_array_shape"] = list(fl["label_array"].shape); fls["vocab_len"] = len(fl["vocabulary"]); fls["task_type_len"] = len(fl["task_type"])
fls["vocabulary"] = fl["vocabulary"]
inv["domains"]["fly"] = dict(labels=fls, frame_map_equals=(L("fly_frame_number_map.npy").item() == fl["frame_number_map"]))
json.dump(inv, open(f"{O}/inventory.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("DONE")
