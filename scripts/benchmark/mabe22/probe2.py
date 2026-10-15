"""Second probe: vocabularies, task types, annotation/keypoint shapes, frame map semantics, id overlaps."""
import numpy as np, os
R = "dataset/benchmark/mabe22/raw"
L = lambda f: np.load(f"{R}/{f}", allow_pickle=True)
for dom in ["mouse", "beetle"]:
    ut = L(f"{dom}_user_train.npy").item()
    print("=====", dom, "user_train vocab", ut["vocabulary"])
    ids = list(ut["sequences"]); s = ut["sequences"][ids[0]]
    print("n seq", len(ids), "kp", s["keypoints"].shape, s["keypoints"].dtype, "ann", s["annotations"].shape, s["annotations"].dtype)
    for k in ids[:3]:
        a = ut["sequences"][k]["annotations"]; print(k, a.shape, np.unique(a, return_counts=True), ut["sequences"][k]["keypoints"].shape)
    print("keys of seq", list(s))
    sub = L(f"{dom}_submission_labels.npy").item()
    print("submission vocab", sub["vocabulary"]); print("task_type", sub["task_type"])
    fm = sub["frame_number_map"]; ks = list(fm)[:3]; print("fmap sample", [(k, fm[k]) for k in ks])
    la = sub["label_array"]; print("label_array", la.shape, la.dtype)
    for i, v in enumerate(sub["vocabulary"]):
        u, c = np.unique(la[i], return_counts=True)
        print(i, v, sub["task_type"][i], "nuniq", len(u), dict(zip(u[:8].tolist(), c[:8].tolist())) if len(u) <= 8 else (u[:3], u[-3:]), "nan", int(np.isnan(la[i]).sum()))
    sk = L(f"{dom}_submission_keypoints.npy").item()["sequences"]
    k0 = list(sk)[0]; print("sub kp", sk[k0]["keypoints"].shape, sk[k0]["keypoints"].dtype)
    print("ids: userTrain∩submission", len(set(ut["sequences"]) & set(sub["frame_number_map"])), "kp ids == fmap ids", set(sk) == set(fm))
    vdirs = {z: set(f[:-4] for f in os.listdir(f"dataset/benchmark/mabe22/videos/{z}")) for z in [f"{dom}_userTrain", f"{dom}_submission"]}
    print({z: len(v) for z, v in vdirs.items()}, "sub fmap ids not in videos", sorted(set(fm) - vdirs[f"{dom}_submission"])[:5], "videos not in fmap", sorted(vdirs[f"{dom}_submission"] - set(fm))[:5], "ut ids not in videos", len(set(ut["sequences"]) - vdirs[f"{dom}_userTrain"]))
mt = L("mouse_triplets_test_labels.npy").item()
print("mouse test vocab", mt["vocabulary"], mt["task_type"]); fm = mt["frame_number_map"]; print([(k, fm[k]) for k in list(fm)[:3]])
fl = L("fly_groups_test_labels.npy").item(); print("fly vocab", fl["vocabulary"]); print("fly task_type", fl["task_type"]); print(fl["keypoint_vocabulary"][:3])
