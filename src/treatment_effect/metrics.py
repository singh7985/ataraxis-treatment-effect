"""Survival and treatment-effect metrics used throughout the project.

Everything is built on a small weighted Kaplan-Meier estimator written in
numpy, so the same code works for the plain RCT case (equal weights) and for
the observational case (inverse propensity weights).

The quantity we care about is the restricted mean survival time (RMST) up to a
horizon H: the expected event-free time within the next H time units. A
treatment effect is always "RMST if treated minus RMST if not treated".

Main pieces:

* ``kaplan_meier`` / ``rmst``              survival curve and area under it
* ``rmst_pseudo_values``                   jackknife pseudo-values, so censored
                                           survival times can be regressed on
                                           like ordinary numbers
* ``ate_rmst``                             (weighted) difference in RMST
* ``toc_curve`` / ``autoc``                rank-weighted average treatment effect
                                           (Yadlowsky et al. 2021): does ranking
                                           by predicted benefit find the people
                                           who actually benefit?
* ``calibration_by_bins``                  observed vs predicted effect by
                                           predicted-benefit quantile
* ``cox_interaction_test``                 treatment x score interaction in a Cox
                                           model, the classic clinical check
* ``dr_pseudo_outcome`` / ``dr_loss``      doubly-robust score for selecting
                                           between CATE models on observational
                                           validation folds
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Kaplan-Meier and RMST
# ---------------------------------------------------------------------------
def kaplan_meier(time, event, weights=None):
    """Weighted Kaplan-Meier. Returns (unique_times, survival_at_those_times).

    The survival function is right-continuous: S(t) = surv[k] for
    unique_times[k] <= t < unique_times[k+1] and S(t) = 1 before the first time.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    w = np.ones_like(time) if weights is None else np.asarray(weights, dtype=float)
    if time.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(time, kind="mergesort")
    t, e, w = time[order], event[order], w[order]
    ut, first = np.unique(t, return_index=True)
    at_risk = np.cumsum(w[::-1])[::-1][first]
    d = np.bincount(np.searchsorted(ut, t), weights=w * e, minlength=len(ut))
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.where(at_risk > 0, d / at_risk, 0.0)
    surv = np.cumprod(1.0 - hazard)
    return ut, surv


def survival_at(ut, surv, t):
    """Evaluate a KM step function at time(s) t."""
    t = np.asarray(t, dtype=float)
    if len(ut) == 0:
        return np.ones_like(t)
    idx = np.searchsorted(ut, t, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)


def rmst_from_km(ut, surv, horizon):
    """Area under the KM curve from 0 to horizon."""
    mask = ut < horizon
    knots = np.concatenate([[0.0], ut[mask], [horizon]])
    heights = np.concatenate([[1.0], surv[mask]])
    return float(np.sum(np.diff(knots) * heights))


def rmst(time, event, horizon, weights=None):
    ut, surv = kaplan_meier(time, event, weights)
    return rmst_from_km(ut, surv, horizon)


def rmst_pseudo_values(time, event, horizon, chunk=256):
    """Jackknife pseudo-values of RMST(horizon) for every subject.

    PV_i = n * RMST(all) - (n - 1) * RMST(all but i)

    With independent censoring the mean of PV over any subgroup defined by
    baseline covariates estimates that subgroup's expected min(T, H), so the
    pseudo-values can be fed to any regression model. The leave-one-out curves
    are computed in a vectorised way: removing subject i only changes the risk
    set at times <= T_i (and the event count at T_i itself).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    n = len(time)
    ut, surv = kaplan_meier(time, event)
    theta = rmst_from_km(ut, surv, horizon)

    keep = ut < horizon
    ut_h = ut[keep]
    m = len(ut_h)
    # full-sample risk set and event counts on the retained times
    ts = np.sort(time)
    at_risk = (n - np.searchsorted(ts, ut_h, side="left")).astype(float)
    pos = np.searchsorted(ut_h, time)  # index of each subject's time among retained times
    in_range = time < horizon
    d = np.bincount(pos[in_range], weights=event[in_range].astype(float), minlength=m + 1)[:m]

    widths = np.diff(np.concatenate([[0.0], ut_h, [horizon]]))  # m + 1 segment widths
    pv = np.empty(n)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        T = time[sl][:, None]
        E = event[sl][:, None]
        le = ut_h[None, :] <= T                     # subject was at risk at these times
        eq = (ut_h[None, :] == T) & E               # subject's own event
        nr = at_risk[None, :] - le
        dd = d[None, :] - eq
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(nr > 0, dd / nr, 0.0)
        s = np.cumprod(1.0 - h, axis=1)
        heights = np.concatenate([np.ones((s.shape[0], 1)), s], axis=1)
        loo = heights @ widths
        pv[sl] = n * theta - (n - 1) * loo
    return pv


def censoring_survival(time, event):
    """KM of the censoring distribution G(t) = P(C > t). Returns a callable."""
    ut, surv = kaplan_meier(time, ~np.asarray(event, dtype=bool))
    return lambda t: survival_at(ut, surv, t)


# ---------------------------------------------------------------------------
# Treatment effects on the RMST scale
# ---------------------------------------------------------------------------
def _ipw_weights(w, propensity):
    if propensity is None:
        return np.ones(len(w))
    p = np.clip(np.asarray(propensity, dtype=float), 0.02, 0.98)
    return np.where(w == 1, 1.0 / p, 1.0 / (1.0 - p))


def ate_rmst(time, event, w, horizon, propensity=None):
    """Treated minus control RMST(horizon). Weighted by inverse propensity if given."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    w = np.asarray(w).astype(int)
    wt = _ipw_weights(w, propensity)
    m1, m0 = w == 1, w == 0
    if m1.sum() == 0 or m0.sum() == 0:
        return np.nan
    return rmst(time[m1], event[m1], horizon, wt[m1]) - rmst(time[m0], event[m0], horizon, wt[m0])


def toc_curve(score, time, event, w, horizon, propensity=None, grid=None, min_frac=0.05):
    """Targeting operator characteristic: ATE among the top-q fraction by score, minus overall ATE."""
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    n = len(score)
    if grid is None:
        grid = np.linspace(min_frac, 1.0, 40)
    time, event, w = np.asarray(time), np.asarray(event), np.asarray(w)
    prop = None if propensity is None else np.asarray(propensity)
    overall = ate_rmst(time, event, w, horizon, prop)
    toc = np.empty(len(grid))
    for k, q in enumerate(grid):
        top = order[: max(int(np.ceil(q * n)), 2)]
        toc[k] = ate_rmst(time[top], event[top], w[top], horizon, None if prop is None else prop[top]) - overall
    return grid, toc


def autoc(score, time, event, w, horizon, propensity=None, grid=None, weighting="autoc"):
    """Area under the TOC curve (AUTOC) or Qini (q-weighted). Zero means the ranking is uninformative."""
    grid, toc = toc_curve(score, time, event, w, horizon, propensity, grid)
    toc = np.nan_to_num(toc)
    if weighting == "qini":
        toc = toc * grid
    # trapezoid over q in [grid[0], 1] plus the triangle below grid[0] is ignored (tiny, noisy)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(toc, grid) / (grid[-1] - grid[0]))


def autoc_with_ci(score, time, event, w, horizon, propensity=None, n_boot=200, n_perm=200, seed=0, weighting="autoc"):
    """AUTOC with a patient-level bootstrap CI and a permutation p-value (score shuffled)."""
    rng = np.random.default_rng(seed)
    score, time, event, w = map(np.asarray, (score, time, event, w))
    prop = None if propensity is None else np.asarray(propensity)
    n = len(score)
    point = autoc(score, time, event, w, horizon, prop, weighting=weighting)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = autoc(score[idx], time[idx], event[idx], w[idx], horizon, None if prop is None else prop[idx], weighting=weighting)
    perms = np.empty(n_perm)
    for b in range(n_perm):
        perms[b] = autoc(rng.permutation(score), time, event, w, horizon, prop, weighting=weighting)
    p_value = float((np.sum(perms >= point) + 1) / (n_perm + 1))
    return {
        "autoc": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "boot_sd": float(boots.std()),
        "perm_p_value": p_value,
        "perm_null_sd": float(perms.std()),
    }


def calibration_by_bins(score, time, event, w, horizon, n_bins=5, propensity=None, n_boot=200, seed=0):
    """Observed (KM) vs predicted treatment effect inside predicted-benefit quantile bins.

    Returns a DataFrame with one row per bin (lowest predicted benefit first) and
    the calibration slope: a weighted regression of observed on predicted across
    bins. Slope 1 = predicted magnitudes are right, 0 = no relation.
    """
    rng = np.random.default_rng(seed)
    score, time, event, w = map(np.asarray, (score, time, event, w))
    prop = None if propensity is None else np.asarray(propensity)
    ranks = pd.Series(score).rank(method="first").values
    bins = np.floor((ranks - 1) / len(score) * n_bins).astype(int)
    rows = []
    for b in range(n_bins):
        m = bins == b
        obs = ate_rmst(time[m], event[m], w[m], horizon, None if prop is None else prop[m])
        idx_all = np.flatnonzero(m)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(idx_all, len(idx_all), replace=True)
            boots.append(ate_rmst(time[idx], event[idx], w[idx], horizon, None if prop is None else prop[idx]))
        boots = np.array(boots)
        rows.append({
            "bin": b + 1,
            "n": int(m.sum()),
            "n_treated": int((w[m] == 1).sum()),
            "pred_mean": float(score[m].mean()),
            "obs_effect": float(obs),
            "ci_low": float(np.nanpercentile(boots, 2.5)),
            "ci_high": float(np.nanpercentile(boots, 97.5)),
        })
    table = pd.DataFrame(rows)
    x, y = table["pred_mean"].values, table["obs_effect"].values
    se = (table["ci_high"] - table["ci_low"]).values / 3.92 + 1e-9
    wls = 1.0 / se**2
    xm, ym = np.average(x, weights=wls), np.average(y, weights=wls)
    slope = float(np.sum(wls * (x - xm) * (y - ym)) / max(np.sum(wls * (x - xm) ** 2), 1e-12))
    return table, slope


def cox_interaction_test(score, time, event, w):
    """Cox model: hazard ~ treatment + score + treatment:score (score standardised).

    A negative interaction coefficient means patients with higher predicted
    benefit have a larger hazard reduction under treatment, i.e. the ranking
    carries real heterogeneity. Returns coefficient, hazard ratio, p-value.
    """
    from lifelines import CoxPHFitter

    s = (np.asarray(score, dtype=float) - np.mean(score)) / (np.std(score) + 1e-12)
    df = pd.DataFrame({"T": time, "E": event, "treatment": np.asarray(w).astype(int), "score": s})
    df["treatment_x_score"] = df["treatment"] * df["score"]
    cph = CoxPHFitter(penalizer=0.0).fit(df, "T", "E")
    row = cph.summary.loc["treatment_x_score"]
    return {"coef": float(row["coef"]), "hr": float(np.exp(row["coef"])), "p": float(row["p"]), "se": float(row["se(coef)"])}


# ---------------------------------------------------------------------------
# Model selection on observational data
# ---------------------------------------------------------------------------
def dr_pseudo_outcome(y, w, propensity, mu0, mu1, clip=(0.02, 0.98)):
    """Doubly-robust (AIPW) pseudo-outcome for the CATE. y is the RMST pseudo-value."""
    y, w = np.asarray(y, dtype=float), np.asarray(w).astype(int)
    e = np.clip(np.asarray(propensity, dtype=float), *clip)
    mu0, mu1 = np.asarray(mu0, dtype=float), np.asarray(mu1, dtype=float)
    return mu1 - mu0 + w * (y - mu1) / e - (1 - w) * (y - mu0) / (1 - e)


def dr_loss(tau_hat, phi):
    """Mean squared distance between a CATE prediction and the DR pseudo-outcome.

    The absolute value is dominated by noise in phi, but differences between
    models are meaningful: the model minimising this also (asymptotically)
    minimises the true CATE mean squared error.
    """
    return float(np.mean((np.asarray(tau_hat) - np.asarray(phi)) ** 2))


def concordance(time, event, risk):
    """Harrell's C-index (higher risk should mean earlier event)."""
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(np.asarray(event, dtype=bool), np.asarray(time, dtype=float), np.asarray(risk, dtype=float))[0])


# ---------------------------------------------------------------------------
# Inverse probability of censoring weighting (IPCW)
# ---------------------------------------------------------------------------
def ipcw_outcome(time, event, horizon, G_before):
    """Restricted outcome and its censoring weight for every subject.

    y_i      = min(T_i, H)
    observed = 1 if we know y_i: the event happened by H, or follow-up reached H
    weight   = observed / G(y_i^- | x_i, w_i), capped so one subject cannot dominate

    With G the (covariate-dependent) probability of still being under follow-up,
    E[weight * f(y)] = E[f(min(T, H))] for any f, which is what lets ordinary
    weighted regressions estimate the RMST conditional on x.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    y = np.minimum(time, horizon)
    observed = (event & (time <= horizon)) | (time >= horizon)
    G = np.clip(np.asarray(G_before, dtype=float), 0.05, 1.0)
    weight = np.where(observed, 1.0 / G, 0.0)
    return y, weight


def weighted_ate(y, a, b):
    """Hajek estimate of E[Y(1)] - E[Y(0)] from per-subject arm weights a (treated) and b (control)."""
    a, b, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(y, dtype=float)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    return float(np.sum(a * y) / np.sum(a) - np.sum(b * y) / np.sum(b))


def toc_curve_weighted(score, y, a, b, min_frac=0.05):
    """TOC curve at every rank, from arm weights (fast, via cumulative sums).

    a_i = W_i * w_cens_i / e_i and b_i = (1 - W_i) * w_cens_i / (1 - e_i): inverse
    propensity times inverse censoring weights. In an RCT e_i is 0.5 for all.
    """
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, a, b = np.asarray(y, dtype=float)[order], np.asarray(a, dtype=float)[order], np.asarray(b, dtype=float)[order]
    n = len(score)
    ca, cay, cb, cby = np.cumsum(a), np.cumsum(a * y), np.cumsum(b), np.cumsum(b * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ate_top = cay / ca - cby / cb
    q = np.arange(1, n + 1) / n
    keep = q >= min_frac
    overall = ate_top[-1]
    return q[keep], np.nan_to_num(ate_top[keep] - overall)


def autoc_weighted(score, y, a, b, weighting="autoc", min_frac=0.05):
    q, toc = toc_curve_weighted(score, y, a, b, min_frac)
    if weighting == "qini":
        toc = toc * q
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(toc, q) / (q[-1] - q[0]))


def autoc_weighted_with_ci(score, y, a, b, n_boot=200, n_perm=200, seed=0, weighting="autoc"):
    rng = np.random.default_rng(seed)
    score, y, a, b = map(lambda v: np.asarray(v, dtype=float), (score, y, a, b))
    n = len(score)
    point = autoc_weighted(score, y, a, b, weighting)
    boots = np.array([autoc_weighted(score[i], y[i], a[i], b[i], weighting) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    perms = np.array([autoc_weighted(rng.permutation(score), y, a, b, weighting) for _ in range(n_perm)])
    return {
        "autoc": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "boot_sd": float(boots.std()),
        "perm_p_value": float((np.sum(perms >= point) + 1) / (n_perm + 1)),
        "perm_null_sd": float(perms.std()),
    }


def calibration_by_bins_weighted(score, y, a, b, n_bins=5, n_boot=200, seed=0):
    """Observed (weighted) vs predicted effect inside predicted-benefit quantile bins."""
    rng = np.random.default_rng(seed)
    score, y, a, b = map(lambda v: np.asarray(v, dtype=float), (score, y, a, b))
    ranks = pd.Series(score).rank(method="first").values
    bins = np.floor((ranks - 1) / len(score) * n_bins).astype(int)
    rows = []
    for k in range(n_bins):
        m = np.flatnonzero(bins == k)
        obs = weighted_ate(y[m], a[m], b[m])
        boots = np.array([weighted_ate(y[i], a[i], b[i]) for i in (rng.choice(m, len(m), replace=True) for _ in range(n_boot))])
        rows.append({"bin": k + 1, "n": len(m), "pred_mean": float(score[m].mean()), "obs_effect": obs,
                     "ci_low": float(np.nanpercentile(boots, 2.5)), "ci_high": float(np.nanpercentile(boots, 97.5))})
    table = pd.DataFrame(rows)
    x, yy = table["pred_mean"].values, table["obs_effect"].values
    se = (table["ci_high"] - table["ci_low"]).values / 3.92 + 1e-9
    wls = 1.0 / se**2
    xm, ym = np.average(x, weights=wls), np.average(yy, weights=wls)
    slope = float(np.sum(wls * (x - xm) * (yy - ym)) / max(np.sum(wls * (x - xm) ** 2), 1e-12))
    return table, slope
