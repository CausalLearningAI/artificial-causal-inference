"""Tests of the SOMP spatial aggregation (src/eci/somp.py).

What is asserted:
  (a) recovery: frames built from known sparse atom combinations (the same s atoms for every token of a
      frame, random coefficients, small noise, variable token counts, padded batch) -> somp_batched with
      K = s recovers exactly the true atom set of every frame; with K > s the extra atoms get ~0 importance
      when there is no noise (index -1 after the frame is explained).
  (b) the batched GPU/CPU version equals the numpy loop (somp_reference, a transcription of ResiDual's somp)
      on random frames: same atoms in the same order, coefficients equal to 1e-8 (float64).
  (c) padding / batch invariance: a frame gives the same result alone and inside a padded batch.
  (d) real frames: the foreground tokens of 16 stored training frames (mice fg448 and ants stores,
      SAE dictionary in the SAE input space) give the same atoms, order and importances with somp_batched
      (float32 correlations, as in production) as with the float64 loop. Skipped when the data is missing.

Runs standalone (`python tests/eci/test_somp.py`) or under pytest.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eci.somp import sae_dictionary, somp_batched, somp_reference  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _dict(rng, m=512, d=96):
    D = rng.standard_normal((m, d))
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def _pad(frames, d):
    n = max(len(f) for f in frames)
    X = np.zeros((len(frames), n, d))
    M = np.zeros((len(frames), n), bool)
    for i, f in enumerate(frames):
        X[i, :len(f)], M[i, :len(f)] = f, True
    return torch.from_numpy(X), torch.from_numpy(M)


def test_recovery():
    rng = np.random.default_rng(0)
    D = _dict(rng)
    s, frames, truth = 6, [], []
    for _ in range(40):
        S = rng.choice(len(D), s, replace=False)
        n = int(rng.integers(8, 120))
        coef = rng.standard_normal((n, s)) * rng.uniform(0.5, 2.0, s) + rng.choice([-1, 1], s) * 1.0
        frames.append(coef @ D[S] + 0.01 * rng.standard_normal((n, D.shape[1])))
        truth.append(set(S.tolist()))
    X, M = _pad(frames, D.shape[1])
    out = somp_batched(X.to(DEV), M.to(DEV), torch.from_numpy(D).to(DEV), s)
    got = [set(r.tolist()) for r in out["idx"].cpu()]
    n_ok = sum(g == t for g, t in zip(got, truth))
    assert n_ok == len(truth), f"recovered {n_ok}/{len(truth)} frames"
    fve = 1 - out["resid"][:, -1] / out["energy"]
    assert float(fve.min()) > 0.99, float(fve.min())
    # noiseless, K > s: extra atoms are not needed -> importance 0, index -1
    frames0 = [rng.standard_normal((len(f), s)) @ D[sorted(S)] for f, S in zip(frames, truth)]
    X0, M0 = _pad(frames0, D.shape[1])
    out0 = somp_batched(X0.to(DEV, torch.float64), M0.to(DEV), torch.from_numpy(D).to(DEV), s + 4)
    idx0 = out0["idx"].cpu().numpy()
    assert all(set(r[:s].tolist()) == S for r, S in zip(idx0, truth))
    assert (idx0[:, s:] == -1).all() and (out0["importance"][:, s:] == 0).all()


def test_matches_reference_loop():
    rng = np.random.default_rng(1)
    D = _dict(rng, 300, 64)
    frames = [rng.standard_normal((int(rng.integers(5, 60)), 64)) for _ in range(12)]
    X, M = _pad(frames, 64)
    K = 10
    out = somp_batched(X.to(DEV), M.to(DEV), torch.from_numpy(D).to(DEV), K, tol=0)
    for i, f in enumerate(frames):
        ch, W, w, res = somp_reference(f, D, K)
        assert out["idx"][i].tolist() == ch, (i, out["idx"][i].tolist(), ch)
        C = out["coef"][i, :len(f)].cpu().numpy()  # (n, K)
        assert np.abs(C - W.T).max() < 1e-8
        assert np.abs(out["importance"][i].cpu().numpy() - w / np.sqrt(len(f))).max() < 1e-10
        assert np.abs(out["resid"][i].cpu().numpy() - res).max() < 1e-8 * max(1, res[0])
        assert np.abs(C[len(f):]).sum() == 0


def test_padding_invariance():
    rng = np.random.default_rng(2)
    D = torch.from_numpy(_dict(rng, 200, 48)).to(DEV)
    frames = [rng.standard_normal((int(n), 48)) for n in (3, 40, 17)]
    X, M = _pad(frames, 48)
    X = X + 5.0 * (~M)[..., None]  # garbage in the padded rows must not matter
    full = somp_batched(X.to(DEV), M.to(DEV), D, 8)
    for i, f in enumerate(frames):
        Xi, Mi = _pad([f], 48)
        one = somp_batched(Xi.to(DEV), Mi.to(DEV), D, 8)
        assert one["idx"][0].tolist() == full["idx"][i].tolist()
        assert torch.allclose(one["importance"][0], full["importance"][i], atol=1e-10)


def _real(store_dir, sae_path, n_frames=16):
    from src.eci.foreground import FgTokenStore
    from src.eci.sae import load_sae
    if not (Path(store_dir) / "shards").exists() or not Path(sae_path).exists():
        return None
    st = FgTokenStore(store_dir)
    tok, rows = np.asarray(st.tokens(0)), st.row(0)
    sae, norm, _ = load_sae(sae_path, DEV)
    to_x, D = sae_dictionary(sae, norm)
    u = np.unique(rows)[:n_frames]
    frames = [to_x(torch.from_numpy(tok[rows == r]).to(DEV)).double().cpu().numpy() for r in u]
    X, M = _pad(frames, frames[0].shape[1])
    K = 16
    out = somp_batched(X.float().to(DEV), M.to(DEV), D, K)  # production dtype
    n_same, max_rel = 0, 0.0
    for i, f in enumerate(frames):
        ch, W, w, res = somp_reference(f, D.double().cpu().numpy(), K)
        same = out["idx"][i].tolist() == ch
        n_same += same
        if same:
            imp = w / np.sqrt(len(f))
            max_rel = max(max_rel, float(np.abs(out["importance"][i].cpu().numpy() - imp).max() / imp.max()))
    return n_same, len(frames), max_rel, [len(f) for f in frames]


def _check_real(store, sae):
    r = _real(ROOT / store, ROOT / sae)
    if r is None:
        try:
            import pytest
            pytest.skip("data missing")
        except ImportError:
            print("  (skipped: data missing)")
            return
    n_same, n, max_rel, ns = r
    print(f"    {store}: {n_same}/{n} frames same atoms and order (tokens per frame {min(ns)}-{max(ns)}), "
          f"max relative importance difference {max_rel:.2e}")
    assert n_same == n and max_rel < 1e-3


def test_real_frames_mice():
    _check_real("dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448_fps1",
                "dataset/mice/v1/eci/sae/matryoshka_btk_1024_k16_fg448_s0/sae.pt")


def test_real_frames_ants():
    _check_real("dataset/ants/eci/train_tokens/dinov2_base_l-1_antsfg_fps1",
                "dataset/ants/eci/sae/matryoshka_btk_1024_k16_antsfg_s0/sae.pt")


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
