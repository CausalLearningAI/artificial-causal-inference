"""Tests of the ECI domain layer (src/eci/domain.py) and the domain-generic contrasts
(src/eci/contrasts.py two_sample, frame_matrix_rows, size_adjusted_round1 with analyses).

What is asserted:
  (a) mice: MiceDomain.load_design equals the pre-refactor load_design (frozen copy below, from commit
      5080e28) exactly; the analysis ids equal the original run order (run_nes.py loops, build_explorer
      ORDER); two_sample on the rows of Analysis.select equals genotype_contrast exactly for stages 1-6
      (units, values, dtypes; every window / stat / prefix); size_adjusted_round1 gives the same table
      through the Analysis objects as through the old id parsing; frame_matrix_rows equals frame_matrix.
  (b) ants: the design (dataset/ants/eci/, scripts/eci/ants_prepare.py) has v2 T counts 1: 24, 2: 20 and
      v3 counts t=2: 38, t=6: 34, t=8: 35; annotations.csv has 44 * 3000 + 212 * 3000 = 768000 rows, v2
      rows first, one contiguous block per video, unique ids; the three analyses select those videos;
      v3_2_vs_8 carries its recording-day confound.

Runs standalone (`python tests/eci/test_domain.py`) or under pytest. The mice / ants tests read the real
annotations (a minute each); they are skipped when the files are missing.
"""
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eci import contrasts as C  # noqa: E402
from eci.domain import get_domain  # noqa: E402

# the analysis order of the original runners (run_nes.py / run_nes_bouts.py loops, build_explorer ORDER)
ORDER_ORIG = [f"A_{g}_{t}" for g in ("het", "wt") for t in ("1to2", "2to3", "4to5", "5to6")] + \
             [f"B_stage{s}" for s in range(1, 7)]


def load_design_orig(annotations_csv, experiment_csv):
    """Frozen copy of contrasts.load_design before the domain refactor (commit 5080e28): the reference."""
    a = pd.read_csv(annotations_csv, usecols=["observation_id", "frame_idx"])
    oid = a["observation_id"].values
    brk = np.flatnonzero(oid[1:] != oid[:-1]) + 1
    starts, ends = np.r_[0, brk], np.r_[brk, len(oid)]
    blocks = pd.DataFrame({"observation_id": oid[starts], "row_start": starts, "row_end": ends})
    if blocks["observation_id"].duplicated().any():
        raise ValueError("an observation is split over non-contiguous row blocks")
    fi = a["frame_idx"].values
    for s, e in zip(starts, ends):
        if not (np.diff(fi[s:e]) > 0).all():
            raise ValueError("frames are not in increasing frame_idx order within an observation")
    e = pd.read_csv(experiment_csv)
    e["stage"] = [C.STAGES[(o, p)] for o, p in zip(e["odor"], e["phase"])]
    e["T"] = (e["genotype"] == "het").astype(int)
    d = blocks.merge(e[["observation_id", "pool", "genotype", "T", "stage", "line", "sex"]], on="observation_id",
                     how="left", validate="1:1")
    if d["pool"].isna().any():
        raise ValueError("observations missing from experiment.csv")
    d["n_frames"] = d["row_end"] - d["row_start"]
    d["obs_row"] = np.arange(len(d))
    return d


_cache = {}


def mice_design():
    M = get_domain("mice")
    if not (M.ann_path.exists() and M.experiment_csv.exists()):
        import pytest
        pytest.skip("mice v1 annotations / experiment.csv missing")
    if "mice" not in _cache:
        _cache["mice"] = M.load_design()
    return M, _cache["mice"]


def ants_design():
    A = get_domain("ants")
    if not (A.ann_path.exists() and A.experiment_csv.exists()):
        import pytest
        pytest.skip("run scripts/eci/ants_prepare.py first")
    if "ants" not in _cache:
        _cache["ants"] = A.load_design()
    return A, _cache["ants"]


def fake_summaries(n_obs, m=64, seed=0):
    rng = np.random.default_rng(seed)
    return {(w, s): rng.gamma(0.5, 1.0, (n_obs, m)) * (rng.random((n_obs, m)) < 0.7)
            for w in C.WINDOWS for s in ("mean", "rate")}


# ------------------------------------------------------------------------------------------ mice
def test_mice_design_equals_original():
    M, d = mice_design()
    ref = load_design_orig(M.ann_path, M.experiment_csv)
    pd.testing.assert_frame_equal(d, ref, check_exact=True)
    assert len(d) == 432 and d["pool"].nunique() == 72


def test_mice_analysis_ids():
    M = get_domain("mice")
    assert M.analysis_ids() == ORDER_ORIG
    for an in M.analyses:
        if an.family == "A":
            assert an.stages == C.TRANSITIONS[an.id.split("_")[2]] and an.genotype == an.id.split("_")[1]
            assert an.matched == (an.stages[0] in (1, 4)) and an.directions == ("up", "down")
            assert an.meta == {"genotype": an.genotype, "stage": "", "transition": an.id.split("_")[2]}
        else:
            s = int(an.id[len("B_stage"):])
            assert an.where == {"stage": s} and an.directions == ("het>wt", "het<wt")
            assert an.meta == {"genotype": "het_vs_wt", "stage": s, "transition": ""}


def test_two_sample_equals_genotype_contrast():
    M, d = mice_design()
    summ = fake_summaries(len(d))
    for an in M.analyses:
        if an.family != "B":
            continue
        rows = an.select(d)
        for w in C.WINDOWS:
            for stat in ("mean", "rate"):
                for prefix in (None, 16):
                    u0, Z0, T0 = C.genotype_contrast(summ, d, an.where["stage"], stat, w, prefix)
                    u1, Z1, T1 = C.two_sample(summ, rows, stat, w, prefix, an.unit)
                    assert np.array_equal(u0, u1) and u0.dtype == u1.dtype
                    assert np.array_equal(Z0, Z1) and Z0.dtype == Z1.dtype
                    assert np.array_equal(T0, T1) and T0.dtype == T1.dtype
        assert len(rows) == 72 and rows["T"].sum() == 36


def test_paired_rows_equal_select():
    """Analysis.select of an A analysis = the videos of that genotype at stages a and b (unit_rows)."""
    M, d = mice_design()
    for an in M.analyses:
        if an.family == "A":
            a, b = an.stages
            ref = d[(d["genotype"] == an.genotype) & d["stage"].isin([a, b])]["obs_row"].values
            assert np.array_equal(an.select(d)["obs_row"].values, ref)


def test_size_adjusted_with_analyses_equals_id_parsing():
    M, d = mice_design()
    rng = np.random.default_rng(1)
    vals = {w: rng.gamma(0.5, 1.0, (len(d), 32)) for w in C.WINDOWS}
    nfg = {w: rng.uniform(5, 60, len(d)) for w in C.WINDOWS}
    wmap = {"full": ("full", "full"), "matched": ("last", "full"), "trim30": ("trim", "trim")}
    r1 = pd.DataFrame([{"analysis_id": aid, "prefix": 128, "window": w, "neuron": j, "tau": t, "p": 1e-4,
                        "threshold": 1e-3}
                       for aid in ORDER_ORIG for w in ("full", "trim30") for j, t in ((3, 0.2), (17, -0.1))])
    old = C.size_adjusted_round1(r1, d, vals, nfg, wmap)
    new = C.size_adjusted_round1(r1, d, vals, nfg, wmap, {a.id: a for a in M.analyses})
    pd.testing.assert_frame_equal(old, new, check_exact=True)


def test_frame_matrix_rows_equals_frame_matrix():
    rng = np.random.default_rng(2)
    n = [5, 7, 4, 6, 3, 8]
    starts = np.r_[0, np.cumsum(n)[:-1]]
    d = pd.DataFrame({"row_start": starts, "row_end": starts + n, "stage": [2, 1, 2, 2, 1, 2],
                      "T": [1, 0, 0, 1, 1, 0], "genotype": ["het", "wt", "wt", "het", "het", "wt"]})
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "codes.npy"
        np.save(p, rng.random((sum(n), 10)).astype(np.float16))
        an = get_domain("mice").analysis("B_stage2")
        for prefix in (None, 4):
            a = C.frame_matrix(p, d, 2, prefix)
            b = C.frame_matrix_rows(p, an.select(d), prefix)
            assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b))


# ------------------------------------------------------------------------------------------ ants
def test_ants_design_counts():
    A, d = ants_design()
    assert d["observation_id"].is_unique and len(d) == 44 + 212
    v2 = d[d["experiment"] == "v2"]["T"].value_counts().to_dict()
    v3 = d[d["experiment"] == "v3"]["T"].value_counts().to_dict()
    assert v2 == {1: 24, 2: 20}, v2
    assert (v3[2], v3[6], v3[8]) == (38, 34, 35), v3
    assert list(d.columns[:3]) == ["observation_id", "row_start", "row_end"]
    for c in ("experiment", "T", "batch", "position", "annotator", "recording_date", "nestbox"):
        assert c in d.columns, c
    assert (d["n_frames"] == 3000).all()
    assert d[d["experiment"] == "v2"]["nestbox"].isna().all() and d[d["experiment"] == "v3"]["nestbox"].notna().all()
    # v3 t=8 only on day C, t=2 / t=6 never on day C (the confound flagged on v3_2_vs_8)
    v3d = d[d["experiment"] == "v3"]
    assert set(v3d[v3d["T"] == 8]["recording_date"]) == {"C"}
    assert "C" not in set(v3d[v3d["T"].isin([2, 6])]["recording_date"])


def test_ants_annotations_rows():
    A, d = ants_design()
    a = pd.read_csv(A.ann_path, usecols=["observation_id", "experiment", "T"])
    assert len(a) == 44 * 3000 + 212 * 3000 == 768000
    assert (a["experiment"].values[:132000] == "v2").all() and (a["experiment"].values[132000:] == "v3").all()
    assert list(C.observation_blocks(A.ann_path)["observation_id"]) == list(d["observation_id"])
    assert (a["T"].values == np.repeat(d["T"].values, d["n_frames"].values)).all()


def test_ants_analyses():
    A, d = ants_design()
    assert A.analysis_ids() == ["v2_1_vs_2", "v3_2_vs_6", "v3_2_vs_8"]
    want = {"v2_1_vs_2": (24, 20), "v3_2_vs_6": (38, 34), "v3_2_vs_8": (38, 35)}
    summ = fake_summaries(len(d))
    for an in A.analyses:
        assert an.family == "B" and an.unit == "observation_id"
        rows = an.select(d)
        u, Z, T = C.two_sample(summ, rows, unit=an.unit)
        assert ((T == 0).sum(), (T == 1).sum()) == want[an.id], an.id
        assert len(set(u)) == len(u) and (rows["experiment"] == an.meta["experiment"]).all()
        assert set(rows.loc[rows["T"] == 1, "observation_id"]) == \
            set(d[(d["experiment"] == an.meta["experiment"]) & (d["T"] == an.meta["treatment"])]["observation_id"])
        assert bool(an.meta["confound"]) == (an.id == "v3_2_vs_8")
    assert A.nuisance == "none"
    assert "T" in d and set(d["T"]) == {1, 2, 4, 6, 7, 8, 9}  # select() never overwrites the raw design T


def test_ants_pairs():
    """'pairs' = the 3 core analyses unchanged (first) + every other within-experiment pair (control < treatment);
    confound flag = recording day x arm chi-square p < 0.05, i.e. exactly the v3 pairs with one arm in {8, 9}
    and the other in {2, 4, 6, 7}."""
    A, d = ants_design()
    P = get_domain("ants", "pairs")
    assert [(a.id, a.meta) for a in P.analyses[:3]] == [(a.id, a.meta) for a in A.analyses]
    ids = P.analysis_ids()
    assert len(ids) == len(set(ids)) == 1 + 15
    summ = fake_summaries(len(d))
    for an, row in zip(P.analyses, P.analysis_table()):
        e, c, t = an.meta["experiment"], an.meta["control"], an.meta["treatment"]
        assert an.id == row["analysis_id"] == f"{e}_{c}_vs_{t}" and c < t
        u, Z, T = C.two_sample(summ, an.select(d), unit=an.unit)
        g = d[d["experiment"] == e]
        assert ((T == 0).sum(), (T == 1).sum()) == ((g["T"] == c).sum(), (g["T"] == t).sum()) == \
            (row["n_control"], row["n_treatment"])
        day_split = e == "v3" and (c in (8, 9)) != (t in (8, 9))
        assert row["confounded"] == day_split == bool(an.meta["confound"]), an.id
    try:
        get_domain("mice", "pairs")
        raise AssertionError("mice has no pairs set")
    except ValueError:
        pass


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"  PASS  {t.__name__}  ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
