"""Cross-validate the candidate treatment-effect learners on the development split.

Usage (from the project root, with the virtualenv active):

    python experiments/run_candidates.py                 # all candidates, 5 folds
    python experiments/run_candidates.py --only dr_learner_ridge,t_learner_gbsa
    python experiments/run_candidates.py --seed 1        # different fold assignment

For every fold the same three validation measures are computed for every
candidate, from the same nuisance models, so the numbers are comparable:

    dr_loss     mean squared distance to a doubly-robust pseudo-outcome built
                from nuisance models (propensity, censoring, outcome) fit on
                the training part of the fold. Lower is better. Only
                differences between models mean anything.
    autoc_ipw   rank-weighted average treatment effect on the validation part,
                with inverse-propensity x inverse-censoring weights. Higher is
                better, 0 = useless ranking. (qini_ipw: same, weighted to the top.)
    cindex_0/1  Harrell's C of the arm-specific RMST predictions, where the
                learner produces them. Measures the outcome model, not the
                effect.

Outputs go to experiments/results/: one row per candidate x fold, a summary,
and the out-of-fold predictions on the development split (useful for ensembles).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from treatment_effect import metrics as M  # noqa: E402
from treatment_effect import models as Mo  # noqa: E402
from treatment_effect.data import load_patients  # noqa: E402
from treatment_effect.features import feature_matrix  # noqa: E402
from treatment_effect.splits import cv_folds, dev_test_split, labeled  # noqa: E402

warnings.filterwarnings("ignore")
HORIZON = Mo.DEFAULT_HORIZON


def candidates(seed: int, horizon: float) -> dict[str, object]:
    return {name: Mo.build_learner(name, horizon, seed) for name in Mo.CANDIDATE_NAMES}


def fold_nuisances(X_tr, w_tr, t_tr, e_tr, X_va, w_va, t_va, e_va, horizon, seed):
    """Nuisance pieces for the validation part, all fitted on the training part of the fold.

    e      propensity P(treated | x)
    y, wt  restricted outcome min(T, H) and its inverse-censoring weight (0 if censored before H)
    mu0/1  arm-wise outcome models (weighted GBM on the restricted outcome)
    phi    doubly-robust pseudo-outcome for the effect
    a, b   per-subject arm weights (treated / control) = IPW x IPCW, for weighted subgroup effects
    """
    prop = Mo.make_classifier("gbm", seed).fit(X_tr, w_tr)
    e_hat = np.clip(prop.predict_proba(X_va)[:, 1], 0.02, 0.98)  # propensity; e_va stays the event indicator
    cens = Mo.CensoringModel().fit(X_tr, w_tr, t_tr, e_tr)
    y_tr, wt_tr = Mo.ipcw_from_model(cens, X_tr, w_tr, t_tr, e_tr, horizon)
    y_va, wt_va = Mo.ipcw_from_model(cens, X_va, w_va, t_va, e_va, horizon)
    mu = {}
    for arm in (0, 1):
        m = w_tr == arm
        mu[arm] = Mo.fit_weighted(Mo.make_regressor("gbm", seed), X_tr[m], y_tr[m], wt_tr[m]).predict(X_va)
    mu_w = np.where(w_va == 1, mu[1], mu[0])
    phi = mu[1] - mu[0] + (w_va / e_hat - (1 - w_va) / (1 - e_hat)) * wt_va * (y_va - mu_w)
    a = w_va * wt_va / e_hat
    b = (1 - w_va) * wt_va / (1 - e_hat)
    return {"e": e_hat, "y": y_va, "wt": wt_va, "mu0": mu[0], "mu1": mu[1], "phi": phi, "a": a, "b": b}


def score(tau, nuis, w_va, t_va, e_va, horizon, arms=None):
    out = {
        "dr_loss": M.dr_loss(tau, nuis["phi"]),
        "autoc_ipw": M.autoc_weighted(tau, nuis["y"], nuis["a"], nuis["b"]),
        "qini_ipw": M.autoc_weighted(tau, nuis["y"], nuis["a"], nuis["b"], weighting="qini"),
        "tau_mean": float(np.mean(tau)),
        "tau_sd": float(np.std(tau)),
    }
    if arms is not None:
        mu0, mu1 = arms
        for arm, mu in ((0, mu0), (1, mu1)):
            m = w_va == arm
            out[f"cindex_{arm}"] = M.concordance(t_va[m], e_va[m], -mu[m])
            wt = nuis["wt"][m]
            out[f"outcome_mse_{arm}"] = float(np.sum(wt * (mu[m] - nuis["y"][m]) ** 2) / np.sum(wt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated candidate names")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--horizon", type=float, default=HORIZON)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = labeled(load_patients(ROOT / "data" / "patient_records.json"))
    dev_idx, _ = dev_test_split(data)
    dev = data.subset(np.isin(np.arange(len(data)), dev_idx))
    X = feature_matrix(dev.embeddings)
    w, t, e = dev.treatment.astype(int), dev.duration, dev.event
    folds = cv_folds(w, e, args.folds, args.seed)

    cands = candidates(args.seed, args.horizon)
    if args.only:
        cands = {k: v for k, v in cands.items() if k in args.only.split(",")}
    print(f"dev split: n={len(dev)}, treated={int(w.sum())}, events={int(e.sum())}; {len(folds)} folds; {len(cands)} candidates; H={args.horizon}")

    nuis = []
    nuis_rows = {key: np.full(len(dev), np.nan) for key in ("fold", "e", "y", "wt", "mu0", "mu1", "phi", "a", "b")}
    for k, (tr, va) in enumerate(folds):
        nk = fold_nuisances(X[tr], w[tr], t[tr], e[tr], X[va], w[va], t[va], e[va], args.horizon, args.seed)
        nuis.append(nk)
        nuis_rows["fold"][va] = k
        for key in ("e", "y", "wt", "mu0", "mu1", "phi", "a", "b"):
            nuis_rows[key][va] = nk[key]
    rows, oof = [], {}
    for name, proto in cands.items():
        oof[name] = np.full(len(dev), np.nan)
        for k, (tr, va) in enumerate(folds):
            learner = candidates(args.seed, args.horizon)[name]
            t0 = time.time()
            emb_tr = [dev.embeddings[i] for i in tr]
            emb_va = [dev.embeddings[i] for i in va]
            needs_emb = isinstance(learner, Mo.SetNetLearner)
            if needs_emb:
                learner.fit(X[tr], w[tr], t[tr], e[tr], embeddings=emb_tr)
                tau = learner.predict(X[va], embeddings=emb_va)
            else:
                learner.fit(X[tr], w[tr], t[tr], e[tr])
                tau = learner.predict(X[va])
            arms = None
            if hasattr(learner, "predict_arms"):
                try:
                    arms = learner.predict_arms(X[va], embeddings=emb_va) if needs_emb else learner.predict_arms(X[va])
                except AttributeError:
                    arms = None
            s = score(tau, nuis[k], w[va], t[va], e[va], args.horizon, arms)
            s.update({"candidate": name, "fold": k, "seconds": round(time.time() - t0, 1)})
            rows.append(s)
            oof[name][va] = tau
            print(f"{name:34s} fold {k}  dr_loss={s['dr_loss']:.3f}  autoc={s['autoc_ipw']:+.3f}  qini={s['qini_ipw']:+.3f}  "
                  f"c0={s.get('cindex_0', float('nan')):.3f} c1={s.get('cindex_1', float('nan')):.3f}  tau_sd={s['tau_sd']:.2f}  {s['seconds']}s", flush=True)

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"cv_fold_results{tag}.csv", index=False)
    cols = ["dr_loss", "autoc_ipw", "qini_ipw", "cindex_0", "cindex_1", "outcome_mse_0", "outcome_mse_1", "tau_mean", "tau_sd", "seconds"]
    cols = [c for c in cols if c in res.columns]
    summary = res.groupby("candidate")[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    # paired comparison against the constant baseline on dr_loss
    if "constant_ate" in res.candidate.values:
        base = res[res.candidate == "constant_ate"].set_index("fold")["dr_loss"]
        diff = res.apply(lambda r: r["dr_loss"] - base.loc[r["fold"]], axis=1)
        res["dr_loss_minus_constant"] = diff
        d = res.groupby("candidate")["dr_loss_minus_constant"].agg(["mean", "std"])
        summary["dr_loss_vs_constant_mean"] = d["mean"]
        summary["dr_loss_vs_constant_std"] = d["std"]
    summary = summary.sort_values("dr_loss_mean")
    summary.to_csv(out_dir / f"cv_summary{tag}.csv")
    oof_df = pd.DataFrame(oof)
    oof_df.insert(0, "patient_id", dev.patient_id)
    oof_df.to_csv(out_dir / f"oof_predictions{tag}.csv", index=False)
    nuis_df = pd.DataFrame(nuis_rows)
    nuis_df.insert(0, "patient_id", dev.patient_id)
    nuis_df["treatment"], nuis_df["duration"], nuis_df["event"] = w, t, e
    nuis_df.to_csv(out_dir / f"oof_nuisance{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== summary (sorted by dr_loss) ===")
    show = ["dr_loss_mean", "dr_loss_std", "dr_loss_vs_constant_mean", "autoc_ipw_mean", "autoc_ipw_std", "qini_ipw_mean", "cindex_0_mean", "cindex_1_mean", "tau_sd_mean", "seconds_mean"]
    print(summary[[c for c in show if c in summary.columns]].round(3).to_string())


if __name__ == "__main__":
    main()
