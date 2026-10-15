"""Check label-row <-> vocabulary alignment (beetle shift reported by coordinator; mouse sanity)."""
import numpy as np, json
R = "dataset/benchmark/mabe22/raw"; L = lambda f: np.load(f"{R}/{f}", allow_pickle=True).item()
out = {}
b = L("beetle_submission_labels.npy"); la = b["label_array"]; v = b["vocabulary"]; fm = b["frame_number_map"]
ids = sorted(fm, key=lambda k: fm[k][0]); X = la.reshape(21, len(ids), 900)
out["beetle_vocab_in_file"] = v; out["beetle_task_type"] = b["task_type"]
out["beetle_continuous_rows_by_data"] = [r for r in range(21) if len(np.unique(la[r][~np.isnan(la[r])])) > 2]
out["beetle_continuous_idx_by_task_type"] = [i for i, t in enumerate(b["task_type"]) if t != "Discrete"]
d = X[20]; dd = np.diff(d, axis=1)
out["row20_within_clip_monotone_nondecreasing_frac"] = float((dd >= -1e-9).all(1).mean())
out["row20_step_median"] = float(np.median(dd)); out["row20_min"] = float(d.min()); out["row20_max"] = float(d.max()); out["1/216000"] = 1 / 216000
out["row20_clip_start_ordering_unique_starts"] = int(len(np.unique(d[:, 0])))
out["identical_pairs"] = [(i, j) for i in range(21) for j in range(i + 1, 21) if np.array_equal(la[i], la[j], equal_nan=True)]
# co-occurrence of the 6 behaviour rows (8..13) on labelled frames
B = la[8:14]; ok = ~np.isnan(B[0]); P = (B[:, ok] == 1).astype(int)
names = v[8:14]; out["beh_names_8_13"] = names
out["beh_cooccurrence_counts"] = (P @ P.T).tolist(); out["frames_with_zero_behaviours"] = int((P.sum(0) == 0).sum()); out["frames_labelled"] = int(ok.sum())
# interactor condition rows: sums
C = la[:8]; out["cond_rows_0_7_sum_per_frame_counts"] = {int(k): int(c) for k, c in zip(*np.unique(C.sum(0), return_counts=True))}
# mouse
m = L("mouse_submission_labels.npy"); t = L("mouse_triplets_test_labels.npy"); u = L("mouse_user_train.npy")
ml = m["label_array"]; tl = t["label_array"]
out["mouse_vocab_equal_submission_vs_test"] = m["vocabulary"] == t["vocabulary"]
out["mouse_continuous_by_data"] = [m["vocabulary"][r] for r in range(13) if len(np.unique(ml[r])) > 2]
out["mouse_continuous_by_task_type"] = [m["vocabulary"][i] for i, x in enumerate(m["task_type"]) if x != "Discrete"]
out["mouse_pos_rate_sub_vs_test"] = {n: [float(np.mean(ml[i] == 1)), float(np.mean(tl[i] == 1))] for i, n in enumerate(m["vocabulary"]) if m["task_type"][i] == "Discrete"}
ua = np.stack([s["annotations"] for s in u["sequences"].values()])
out["mouse_userTrain_pos_rate"] = {n: float(np.mean(ua[:, i] == 1)) for i, n in enumerate(u["vocabulary"])}
# mouse: structural sanity, strain/lights constant per clip, close>=contact? contact subset of close?
nm = {n: ml[i] for i, n in enumerate(m["vocabulary"])}
out["mouse_P(close | contact)"] = float(np.mean(nm["close"][nm["contact"] == 1]))
out["mouse_P(contact | oral_oral)"] = float(np.mean(nm["contact"][nm["oral_oral_contact"] == 1]))
out["mouse_P(contact | oral_genital)"] = float(np.mean(nm["contact"][nm["oral_genital_contact"] == 1]))
out["mouse_P(contact | oral_ear)"] = float(np.mean(nm["contact"][nm["oral_ear_contact"] == 1]))
out["mouse_P(close | huddles)"] = float(np.mean(nm["close"][nm["huddles"] == 1]))
json.dump(out, open("results/vision/benchmark_mabe22/inventory/row_mapping_check.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
