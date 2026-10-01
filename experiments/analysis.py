"""Helpers for reading the cross-validation results back into the notebook.

* ``load_results(tag)``           fold-level results + summary for one run
* ``paired_table(res)``            every candidate against the constant baseline, fold by fold
* ``ensemble_scores(names, ...)``  score an average of candidates from their out-of-fold predictions
* ``agreement(names, ...)``        correlation between candidates' out-of-fold predictions
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from treatment_effect import metrics as M  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def _tag(tag: str) -> str:
    return f"_{tag}" if tag else ""


def load_results(tag: str = ""):
    res = pd.read_csv(RESULTS / f"cv_fold_results{_tag(tag)}.csv")
    summary = pd.read_csv(RESULTS / f"cv_summary{_tag(tag)}.csv", index_col=0)
    return res, summary


def paired_table(res: pd.DataFrame, baseline: str = "constant_ate") -> pd.DataFrame:
    """Mean and standard error (over folds) of each score, and of the DR-loss difference to the baseline."""
    base = res[res.candidate == baseline].set_index("fold")["dr_loss"]
    res = res.copy()
    res["dr_loss_vs_constant"] = res["dr_loss"].values - base.loc[res["fold"]].values
    n_folds = res.fold.nunique()
    g = res.groupby("candidate")
    out = pd.DataFrame({
        "dr_loss_vs_constant": g["dr_loss_vs_constant"].mean(),
        "se": g["dr_loss_vs_constant"].std() / np.sqrt(n_folds),
        "folds_better": g["dr_loss_vs_constant"].apply(lambda v: int((v < 0).sum())),
        "autoc": g["autoc_ipw"].mean(),
        "autoc_se": g["autoc_ipw"].std() / np.sqrt(n_folds),
        "qini": g["qini_ipw"].mean(),
        "cindex_control": g["cindex_0"].mean() if "cindex_0" in res else np.nan,
        "cindex_treated": g["cindex_1"].mean() if "cindex_1" in res else np.nan,
        "effect_sd": g["tau_sd"].mean(),
        "effect_mean": g["tau_mean"].mean(),
        "seconds": g["seconds"].mean(),
    })
    return out.sort_values("dr_loss_vs_constant")


def _oof(tag: str = ""):
    oof = pd.read_csv(RESULTS / f"oof_predictions{_tag(tag)}.csv")
    nuis = pd.read_csv(RESULTS / f"oof_nuisance{_tag(tag)}.csv")
    assert (oof.patient_id == nuis.patient_id).all()
    return oof, nuis


def score_predictions(tau: np.ndarray, nuis: pd.DataFrame) -> pd.DataFrame:
    """Per-fold DR-loss and weighted AUTOC for any vector of dev-split predictions."""
    rows = []
    for k, idx in nuis.groupby("fold").indices.items():
        n = nuis.iloc[idx]
        rows.append({"fold": int(k), "dr_loss": M.dr_loss(tau[idx], n.phi.values),
                     "autoc_ipw": M.autoc_weighted(tau[idx], n.y.values, n.a.values, n.b.values)})
    return pd.DataFrame(rows)


def ensemble_scores(names: list[str], tag: str = "", baseline: str = "constant_ate") -> pd.DataFrame:
    """Average the out-of-fold predictions of ``names`` and score the result like any candidate."""
    oof, nuis = _oof(tag)
    tau = oof[names].mean(axis=1).values
    sc = score_predictions(tau, nuis)
    base = score_predictions(oof[baseline].values, nuis)
    diff = sc.dr_loss.values - base.dr_loss.values
    return pd.DataFrame({
        "ensemble": ["+".join(names)],
        "dr_loss_vs_constant": [diff.mean()],
        "se": [diff.std(ddof=1) / np.sqrt(len(diff))],
        "folds_better": [int((diff < 0).sum())],
        "autoc": [sc.autoc_ipw.mean()],
        "autoc_se": [sc.autoc_ipw.std(ddof=1) / np.sqrt(len(sc))],
        "effect_sd": [float(np.std(tau))],
    })


def agreement(names: list[str], tag: str = "") -> pd.DataFrame:
    oof, _ = _oof(tag)
    return oof[names].corr().round(2)
