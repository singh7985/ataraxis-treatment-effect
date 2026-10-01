"""The inverse-censoring weights must be exactly zero for patients censored
before the horizon and at least one for everyone else, both in the model
code and in the experiment runner's nuisance function."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from treatment_effect import models as Mo  # noqa: E402
from treatment_effect.metrics import ipcw_outcome  # noqa: E402
from run_candidates import fold_nuisances  # noqa: E402


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    w = (rng.random(n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    T = rng.exponential(8 * np.exp(0.3 * X[:, 1] + 0.4 * w * X[:, 0]))
    C = rng.exponential(14 * np.exp(0.4 * X[:, 2] - 0.3 * w))
    return X, w, np.minimum(T, C), (T <= C).astype(int)


def test_ipcw_outcome_weights():
    time = np.array([1.0, 25.0, 5.0, 30.0])
    event = np.array([0, 0, 1, 1])
    y, wt = ipcw_outcome(time, event, 20.0, G_before=np.array([0.9, 0.5, 0.8, 0.5]))
    assert np.allclose(y, [1.0, 20.0, 5.0, 20.0])
    assert wt[0] == 0.0                      # censored before the horizon: unobserved
    assert np.allclose(wt[1:], [2.0, 1.25, 2.0])  # observed: 1 / G


def test_censoring_model_and_fold_nuisances_agree_on_who_is_observed():
    X, w, t, e = synthetic(1500, 0)
    Xv, wv, tv, ev = synthetic(500, 1)
    H = 20.0
    cens = Mo.CensoringModel().fit(X, w, t, e)
    y, wt = Mo.ipcw_from_model(cens, Xv, wv, tv, ev, H)
    unobserved = (ev == 0) & (tv < H)
    assert np.array_equal(wt == 0, unobserved)
    assert (wt[~unobserved] >= 1.0).all()

    nuis = fold_nuisances(X, w, t, e, Xv, wv, tv, ev, H, seed=0)
    assert np.array_equal(nuis["wt"] == 0, unobserved)
    assert ((nuis["e"] > 0) & (nuis["e"] < 1)).all()
    assert np.isfinite(nuis["phi"]).all()
    assert np.array_equal((nuis["a"] + nuis["b"]) == 0, unobserved)
