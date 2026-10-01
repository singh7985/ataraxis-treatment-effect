"""From a patient's set of 4-d vectors to a fixed-length feature vector.

What I learned in the EDA (see the notebook) and encoded here:

* The order of the vectors carries nothing, so every feature is permutation
  invariant (means, spreads, extremes, fractions).
* Dimensions 2 and 3 are bimodal: a main mode around 0 and a second mode
  around 3.3. The valley between them sits near 2.4. A vector in that second
  mode behaves like a "flag", so I count them and summarise them separately.
* Dimensions 2 and 3 are missing about a third of the time, and how much is
  missing is itself tied to both the treatment decision and the outcome.
  The missing rates are therefore features, not just a nuisance.

The same function is used for training, batch inference and the API, so the
feature definition lives in exactly one place.
"""
from __future__ import annotations

import numpy as np

# Valley between the two modes of dims 2 and 3 (from the pooled histogram).
HIGH_MODE_THRESHOLD = 2.4

FEATURE_NAMES: list[str] = (
    [f"d{d}_{s}" for d in range(4) for s in ("mean", "std", "min", "max")]
    + ["n_vectors", "miss_d2", "miss_d3", "miss_both"]
    + [f"hi{d}_{s}" for d in (2, 3) for s in ("frac", "mean")]
    + ["lo2_mean", "lo3_mean", "hi_any_frac", "hi_both_frac", "d0_d1_corr", "norm01_mean"]
)


def _nan_stats(col: np.ndarray) -> tuple[float, float, float, float]:
    obs = col[~np.isnan(col)]
    if obs.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(obs.mean()), float(obs.std()), float(obs.min()), float(obs.max())


def patient_features(emb: np.ndarray, thr: float = HIGH_MODE_THRESHOLD) -> np.ndarray:
    """emb: (n, 4) float array with NaN for missing entries. Returns (30,) float array."""
    emb = np.asarray(emb, dtype=float)
    n = emb.shape[0]
    out: list[float] = []
    for d in range(4):
        out.extend(_nan_stats(emb[:, d]))

    miss2 = np.isnan(emb[:, 2])
    miss3 = np.isnan(emb[:, 3])
    out += [float(n), float(miss2.mean()), float(miss3.mean()), float((miss2 & miss3).mean())]

    hi_masks = {}
    for d in (2, 3):
        col = emb[:, d]
        obs = col[~np.isnan(col)]
        hi = obs > thr
        hi_masks[d] = np.where(np.isnan(col), False, col > thr)
        out.append(float(hi.mean()) if obs.size else 0.0)
        out.append(float(obs[hi].mean()) if hi.any() else 0.0)
    for d in (2, 3):
        col = emb[:, d]
        obs = col[~np.isnan(col)]
        lo = obs[obs <= thr]
        out.append(float(lo.mean()) if lo.size else 0.0)

    observed_any = ~(miss2 & miss3)
    out.append(float((hi_masks[2] | hi_masks[3])[observed_any].mean()) if observed_any.any() else 0.0)
    observed_both = ~(miss2 | miss3)
    out.append(float((hi_masks[2] & hi_masks[3])[observed_both].mean()) if observed_both.any() else 0.0)

    both01 = ~(np.isnan(emb[:, 0]) | np.isnan(emb[:, 1]))
    a0, a1 = emb[both01, 0], emb[both01, 1]
    if both01.sum() > 2 and a0.std() > 0 and a1.std() > 0:
        out.append(float(np.corrcoef(a0, a1)[0, 1]))
    else:
        out.append(0.0)
    out.append(float(np.sqrt(a0 ** 2 + a1 ** 2).mean()) if both01.any() else 0.0)

    feats = np.array(out, dtype=float)
    assert feats.shape[0] == len(FEATURE_NAMES), (feats.shape, len(FEATURE_NAMES))
    return feats


def feature_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    """Stack ``patient_features`` over a list of patients -> (N, 30)."""
    return np.vstack([patient_features(e) for e in embeddings])


# ---------------------------------------------------------------------------
# Padded tensors for the set neural network (DeepSets-style encoder).
# ---------------------------------------------------------------------------
SET_INPUT_DIM = 6  # 4 values (missing -> 0) + 2 missing indicators for dims 2 and 3


def padded_sets(embeddings: list[np.ndarray], max_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, mask): X is (N, L, 6) float32, mask is (N, L) bool, True where a real vector sits."""
    if max_len is None:
        max_len = max(len(e) for e in embeddings)
    N = len(embeddings)
    X = np.zeros((N, max_len, SET_INPUT_DIM), dtype=np.float32)
    mask = np.zeros((N, max_len), dtype=bool)
    for i, e in enumerate(embeddings):
        e = np.asarray(e, dtype=float)[:max_len]
        n = e.shape[0]
        vals = np.nan_to_num(e, nan=0.0)
        X[i, :n, :4] = vals
        X[i, :n, 4] = np.isnan(e[:, 2])
        X[i, :n, 5] = np.isnan(e[:, 3])
        mask[i, :n] = True
    return X, mask
