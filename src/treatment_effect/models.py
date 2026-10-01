"""The treatment-effect learners that were tried, and the final model wrapper.

Every learner has the same tiny interface::

    learner.fit(X, w, time, event)   # X features, w treatment (0/1), outcome
    learner.predict(X)               # predicted effect = RMST(treated) - RMST(control)

and, when it makes sense, ``learner.predict_arms(X)`` returning the two RMST
predictions separately.

Families
--------
* ``ConstantEffect``        one number for everyone (the adjusted ATE). The
                            "no personalisation" baseline every model must beat.
* ``CoxLearner``            Cox proportional hazards, either treatment as a plain
                            covariate (S-learner) or with treatment x feature
                            interactions. Survival curves -> RMST per arm.
* ``TLearnerSurvival``      one survival model per arm (Cox, gradient boosted
                            survival, or random survival forest).
* ``MetaLearner``           standard CATE meta-learners (S, T, X, DR, R) on the
                            restricted outcome min(T, H). Censoring is handled
                            with inverse probability of censoring weights from a
                            Cox model of the censoring time given features and
                            treatment (the EDA showed censoring is not
                            independent of the features). DR and R also use
                            propensity scores and cross-fitting, so they are the
                            ones that deal with confounding head-on. The older
                            "km_pseudo" option (jackknife pseudo-values, which
                            assume independent censoring) is kept for comparison.
* ``SetNetLearner``         a small DeepSets-style network over the raw vectors
                            with two discrete-time survival heads (one per arm).
* ``EnsembleLearner``       average of several learners.
* ``TreatmentEffectModel``  the deployable wrapper: raw patient json in,
                            predictions out, with save/load.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import PatientData
from .features import FEATURE_NAMES, feature_matrix, padded_sets
from .metrics import ipcw_outcome, rmst_from_km, rmst_pseudo_values

DEFAULT_HORIZON = 20.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def make_regressor(kind: str, seed: int = 0):
    """Regularised regressors for noisy targets."""
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.04, max_iter=250, l2_regularization=1.0,
            min_samples_leaf=40, random_state=seed,
        )
    if kind == "gbm_small":
        return HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.03, max_iter=200, l2_regularization=3.0,
            min_samples_leaf=80, random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=30, max_features=0.5, random_state=seed, n_jobs=-1)
    raise ValueError(kind)


def make_classifier(kind: str, seed: int = 0):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            min_samples_leaf=40, random_state=seed,
        )
    raise ValueError(kind)


def fit_weighted(reg, X, y, sample_weight):
    """Fit a regressor or a (scaler, regressor) pipeline with sample weights."""
    if hasattr(reg, "steps"):
        return reg.fit(X, y, **{f"{reg.steps[-1][0]}__sample_weight": sample_weight})
    return reg.fit(X, y, sample_weight=sample_weight)


def cross_fit_propensity(X, w, kind="gbm", n_splits=5, seed=0):
    """Out-of-fold propensity scores e(x) = P(treated | x)."""
    e = np.zeros(len(w))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, w):
        clf = make_classifier(kind, seed).fit(X[tr], w[tr])
        e[te] = clf.predict_proba(X[te])[:, 1]
    return e


def pseudo_values_by_arm(w, time, event, horizon):
    """Jackknife RMST pseudo-values inside each arm (assumes censoring independent of x)."""
    pv = np.zeros(len(w), dtype=float)
    for arm in (0, 1):
        m = w == arm
        pv[m] = rmst_pseudo_values(time[m], event[m], horizon)
    return pv


def _rmst_from_survival_functions(times, surv_matrix, horizon):
    """surv_matrix: (N, len(times)) step survival curves -> RMST per row."""
    return np.array([rmst_from_km(times, s, horizon) for s in surv_matrix])


# ---------------------------------------------------------------------------
# censoring model and IPCW outcomes
# ---------------------------------------------------------------------------
class CensoringModel:
    """Cox PH model for the censoring time given features and treatment.

    G(t | x, w) = P(still under follow-up at t | x, w). Used to weight the
    patients whose restricted outcome min(T, H) we actually observe.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, w, time, event):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        D = np.c_[X, w.astype(float)]
        self.scaler_ = StandardScaler().fit(D)
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha).fit(
            self.scaler_.transform(D), Surv.from_arrays(~np.asarray(event, dtype=bool), time)
        )
        return self

    def survival_before(self, X, w, t):
        """G(t^- | x, w) for every subject at its own time t."""
        D = self.scaler_.transform(np.c_[X, np.asarray(w, dtype=float)])
        S = self.model_.predict_survival_function(D, return_array=True)
        times = self.model_.unique_times_
        idx = np.searchsorted(times, np.asarray(t, dtype=float), side="left") - 1  # last jump strictly before t
        rows = np.arange(len(t))
        return np.where(idx >= 0, S[rows, np.clip(idx, 0, len(times) - 1)], 1.0)


def ipcw_from_model(cens: CensoringModel, X, w, time, event, horizon):
    """(y, weight): restricted outcome min(T, H) and its IPCW weight (0 if censored before H)."""
    y = np.minimum(np.asarray(time, dtype=float), horizon)
    G = cens.survival_before(X, w, y)
    return ipcw_outcome(time, event, horizon, G)


# ---------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------
class ConstantEffect:
    """Everyone gets the same effect: a doubly-robust (AIPW + IPCW) estimate of the ATE."""

    name = "constant_ate"

    def __init__(self, horizon=DEFAULT_HORIZON, seed=0):
        self.horizon, self.seed = horizon, seed

    def fit(self, X, w, time, event):
        w = w.astype(int)
        cens = CensoringModel().fit(X, w, time, event)
        y, wt = ipcw_from_model(cens, X, w, time, event, self.horizon)
        e = np.clip(cross_fit_propensity(X, w, "gbm", seed=self.seed), 0.02, 0.98)
        mu = {0: np.zeros(len(w)), 1: np.zeros(len(w))}
        for tr, te in KFold(5, shuffle=True, random_state=self.seed).split(X):
            for arm in (0, 1):
                m = tr[w[tr] == arm]
                mu[arm][te] = fit_weighted(make_regressor("gbm", self.seed), X[m], y[m], wt[m]).predict(X[te])
        mu_w = np.where(w == 1, mu[1], mu[0])
        phi = mu[1] - mu[0] + (w / e - (1 - w) / (1 - e)) * wt * (y - mu_w)
        self.ate_ = float(phi.mean())
        self.ate_se_ = float(phi.std() / np.sqrt(len(phi)))
        return self

    def predict(self, X):
        return np.full(len(X), self.ate_)


class CoxLearner:
    """Cox PH with treatment as a covariate, optionally with treatment x feature interactions."""

    def __init__(self, interactions=False, alpha=1.0, horizon=DEFAULT_HORIZON):
        self.interactions, self.alpha, self.horizon = interactions, alpha, horizon
        self.name = "cox_interactions" if interactions else "cox_s_learner"

    def _design(self, X, w):
        cols = [X, w[:, None]]
        if self.interactions:
            cols.append(X * w[:, None])
        return np.hstack(cols)

    def fit(self, X, w, time, event):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        D = self._design(X, w.astype(float))
        self.scaler_ = StandardScaler().fit(D)
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha).fit(self.scaler_.transform(D), Surv.from_arrays(event.astype(bool), time))
        return self

    def predict_arms(self, X):
        out = []
        for arm in (0, 1):
            D = self.scaler_.transform(self._design(X, np.full(len(X), float(arm))))
            S = self.model_.predict_survival_function(D, return_array=True)
            out.append(_rmst_from_survival_functions(self.model_.unique_times_, S, self.horizon))
        return out[0], out[1]

    def predict(self, X):
        mu0, mu1 = self.predict_arms(X)
        return mu1 - mu0


class TLearnerSurvival:
    """One survival model per arm; effect = difference of the two predicted RMSTs.

    Survival models condition on x, so they only need censoring to be
    independent of the event time *given* x and the arm, which is the
    assumption the EDA supports.
    """

    def __init__(self, base="gbsa", horizon=DEFAULT_HORIZON, seed=0):
        self.base, self.horizon, self.seed = base, horizon, seed
        self.name = f"t_learner_{base}"

    def _make(self):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        if self.base == "cox":
            return make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=1.0))
        if self.base == "gbsa":
            return GradientBoostingSurvivalAnalysis(
                n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                min_samples_leaf=30, random_state=self.seed,
            )
        if self.base == "rsf":
            return RandomSurvivalForest(n_estimators=300, min_samples_leaf=25, max_features="sqrt", random_state=self.seed, n_jobs=-1)
        raise ValueError(self.base)

    def fit(self, X, w, time, event):
        from sksurv.util import Surv

        self.models_ = {}
        for arm in (0, 1):
            m = w == arm
            self.models_[arm] = self._make().fit(X[m], Surv.from_arrays(event[m].astype(bool), time[m]))
        return self

    def _surv_times(self, model):
        return model[-1].unique_times_ if hasattr(model, "steps") else model.unique_times_

    def predict_arms(self, X):
        out = []
        for arm in (0, 1):
            model = self.models_[arm]
            S = model.predict_survival_function(X, return_array=True)
            out.append(_rmst_from_survival_functions(self._surv_times(model), S, self.horizon))
        return out[0], out[1]

    def predict(self, X):
        mu0, mu1 = self.predict_arms(X)
        return mu1 - mu0

    def risk_scores(self, X, arm):
        """Risk score of the arm-specific model (for C-index checks)."""
        return self.models_[arm].predict(X)


class MetaLearner:
    """CATE meta-learners on the restricted outcome min(T, H).

    meta       "s", "t", "x", "dr" or "r"
    base       regressor for the outcome models ("ridge", "gbm", "gbm_small", "rf")
    effect_base regressor for the effect itself (DR, R, X second stage); defaults to base
    nuisance   "reg" (weighted regressions on the restricted outcome) or "gbsa"
               (RMST read off boosted survival models) for the DR outcome models
    censoring  "ipcw" (Cox censoring model, weights) or "km_pseudo" (jackknife
               pseudo-values, assumes censoring independent of x)
    """

    def __init__(self, meta="dr", base="gbm", effect_base=None, propensity="gbm", nuisance="reg", censoring="ipcw",
                 interactions=False, horizon=DEFAULT_HORIZON, n_splits=5, seed=0):
        self.meta, self.base, self.propensity, self.nuisance, self.censoring = meta, base, propensity, nuisance, censoring
        self.interactions, self.horizon, self.n_splits, self.seed = interactions, horizon, n_splits, seed
        self.effect_base = effect_base or base
        self.name = f"{meta}_learner_{self.effect_base}" if meta in ("dr", "r") else f"{meta}_learner_{base}"
        if meta == "dr" and nuisance != "reg":
            self.name += f"_{nuisance}nuisance"
        if meta == "s" and interactions:
            self.name += "_interactions"
        if censoring == "km_pseudo":
            self.name += "_kmpv"

    # -- outcomes ---------------------------------------------------------
    def _outcomes(self, X, w, time, event):
        if self.censoring == "km_pseudo":
            return pseudo_values_by_arm(w, time, event, self.horizon), np.ones(len(w))
        self.cens_ = CensoringModel().fit(X, w, time, event)
        return ipcw_from_model(self.cens_, X, w, time, event, self.horizon)

    def _s_design(self, X, w_col):
        cols = [X, w_col[:, None]]
        if self.interactions:
            cols.append(X * w_col[:, None])
        return np.hstack(cols)

    def _reg(self, kind=None):
        return make_regressor(kind or self.base, self.seed)

    # -- fitting ----------------------------------------------------------
    def fit(self, X, w, time, event):
        w = w.astype(int)
        y, wt = self._outcomes(X, w, time, event)
        self.y_, self.weight_ = y, wt
        if self.meta == "s":
            self.model_ = fit_weighted(self._reg(), self._s_design(X, w.astype(float)), y, wt)
        elif self.meta in ("t", "x"):
            self.models_ = {arm: fit_weighted(self._reg(), X[w == arm], y[w == arm], wt[w == arm]) for arm in (0, 1)}
            if self.meta == "x":
                m1, m0 = w == 1, w == 0
                d1 = y[m1] - self.models_[0].predict(X[m1])  # imputed effects for the treated
                d0 = self.models_[1].predict(X[m0]) - y[m0]  # imputed effects for the controls
                self.tau1_ = fit_weighted(self._reg(self.effect_base), X[m1], d1, wt[m1])
                self.tau0_ = fit_weighted(self._reg(self.effect_base), X[m0], d0, wt[m0])
                self.prop_ = make_classifier(self.propensity, self.seed).fit(X, w)
        elif self.meta in ("dr", "r"):
            e = np.clip(cross_fit_propensity(X, w, self.propensity, self.n_splits, self.seed), 0.02, 0.98)
            mu0, mu1, m_all = np.zeros(len(w)), np.zeros(len(w)), np.zeros(len(w))
            for tr, te in KFold(self.n_splits, shuffle=True, random_state=self.seed).split(X):
                if self.meta == "dr" and self.nuisance == "gbsa":
                    t_learner = TLearnerSurvival("gbsa", self.horizon, self.seed).fit(X[tr], w[tr], time[tr], event[tr])
                    mu0[te], mu1[te] = t_learner.predict_arms(X[te])
                elif self.meta == "dr":
                    for arm, store in ((0, mu0), (1, mu1)):
                        idx = tr[w[tr] == arm]
                        store[te] = fit_weighted(self._reg(), X[idx], y[idx], wt[idx]).predict(X[te])
                else:
                    m_all[te] = fit_weighted(self._reg(), X[tr], y[tr], wt[tr]).predict(X[te])
            self.e_ = e
            if self.meta == "dr":
                mu_w = np.where(w == 1, mu1, mu0)
                phi = mu1 - mu0 + (w / e - (1 - w) / (1 - e)) * wt * (y - mu_w)
                self.phi_ = phi
                self.model_ = self._reg(self.effect_base).fit(X, phi)
            else:
                resid_w = w - e
                target = (y - m_all) / resid_w
                self.model_ = fit_weighted(self._reg(self.effect_base), X, target, wt * resid_w**2)
        else:
            raise ValueError(self.meta)
        return self

    # -- prediction -------------------------------------------------------
    def predict_arms(self, X):
        if self.meta == "s":
            return (self.model_.predict(self._s_design(X, np.zeros(len(X)))),
                    self.model_.predict(self._s_design(X, np.ones(len(X)))))
        if self.meta in ("t", "x"):
            return self.models_[0].predict(X), self.models_[1].predict(X)
        raise AttributeError(f"{self.name} predicts effects directly, not arm-wise outcomes")

    def predict(self, X):
        if self.meta in ("s", "t"):
            mu0, mu1 = self.predict_arms(X)
            return mu1 - mu0
        if self.meta == "x":
            e = np.clip(self.prop_.predict_proba(X)[:, 1], 0.02, 0.98)
            return e * self.tau0_.predict(X) + (1 - e) * self.tau1_.predict(X)
        return self.model_.predict(X)


class EnsembleLearner:
    def __init__(self, learners, name="ensemble"):
        self.learners, self.name = learners, name

    def fit(self, X, w, time, event, embeddings=None):
        for l in self.learners:
            if isinstance(l, SetNetLearner):
                l.fit(X, w, time, event, embeddings=embeddings)
            else:
                l.fit(X, w, time, event)
        return self

    def predict(self, X, embeddings=None):
        preds = [l.predict(X, embeddings=embeddings) if isinstance(l, SetNetLearner) else l.predict(X) for l in self.learners]
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# DeepSets + two discrete-time survival heads
# ---------------------------------------------------------------------------
class SetNetLearner:
    """Permutation-invariant encoder over the raw 4-d vectors, shared trunk, one
    discrete-time hazard head per arm (a TARNet-style layout for survival).

    Trained by maximum likelihood on the discrete-time survival model, which
    like the other survival models only needs censoring to be independent of
    the event time given the inputs. Effect = difference of the two RMSTs read
    off the predicted survival curves.
    """

    name = "setnet"

    def __init__(self, horizon=DEFAULT_HORIZON, n_bins=20, hidden=64, epochs=150, lr=1e-3, weight_decay=1e-3,
                 batch_size=256, patience=15, use_features=True, seed=0, verbose=False):
        self.horizon, self.n_bins, self.hidden, self.epochs, self.lr, self.weight_decay = horizon, n_bins, hidden, epochs, lr, weight_decay
        self.batch_size, self.patience, self.use_features, self.seed, self.verbose = batch_size, patience, use_features, seed, verbose
        self.name = "setnet_plus_features" if use_features else "setnet"

    def _build(self, n_feat):
        import torch
        import torch.nn as nn

        H = self.hidden

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.phi = nn.Sequential(nn.Linear(6, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
                s.trunk = nn.Sequential(nn.Linear(2 * H + n_feat, H), nn.ReLU(), nn.Dropout(0.1), nn.Linear(H, H), nn.ReLU())
                s.heads = nn.ModuleList([nn.Linear(H, self.n_bins) for _ in range(2)])

            def forward(s, x, mask, feats):
                h = s.phi(x) * mask.unsqueeze(-1)
                cnt = mask.sum(1, keepdim=True).clamp(min=1)
                mean = h.sum(1) / cnt
                mx = (h + (mask.unsqueeze(-1) - 1) * 1e4).max(1).values
                z = torch.cat([mean, mx, feats], dim=1)
                z = s.trunk(z)
                return torch.stack([head(z) for head in s.heads], dim=1)  # (N, 2, n_bins) logits of hazard

        return Net()

    def _bins(self, time):
        qs = np.quantile(time, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        return np.concatenate([[0.0], qs, [max(time.max(), self.horizon) + 1e-6]])

    def _targets(self, time, event):
        """Discrete-time likelihood targets: index of the bin containing T and the event flag."""
        k = np.clip(np.searchsorted(self.edges_, time, side="right") - 1, 0, self.n_bins - 1)
        return k, event.astype(np.float32)

    def fit(self, X_feats, w, time, event, embeddings=None):
        import torch

        assert embeddings is not None, "SetNetLearner needs the raw embeddings (pass embeddings=...)"
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.edges_ = self._bins(time)
        self.scaler_ = StandardScaler().fit(X_feats)
        F = self.scaler_.transform(X_feats).astype(np.float32) if self.use_features else np.zeros((len(w), 0), np.float32)
        self.max_len_ = max(len(e) for e in embeddings)
        Xs, mask = padded_sets(embeddings, self.max_len_)
        k, ev = self._targets(time, event)
        n = len(w)
        val = rng.random(n) < 0.15
        tr = ~val
        self.net_ = self._build(F.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        to_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
        data = dict(x=to_t(Xs), m=to_t(mask), f=to_t(F), w=to_t(w, torch.long), k=to_t(k, torch.long), e=to_t(ev))

        def nll(idx):
            logits = self.net_(data["x"][idx], data["m"][idx], data["f"][idx])
            logits = logits[torch.arange(len(idx)), data["w"][idx]]  # (B, n_bins) for the observed arm
            kk, ee = data["k"][idx], data["e"][idx]
            logh = torch.nn.functional.logsigmoid(logits)
            log1mh = torch.nn.functional.logsigmoid(-logits)
            ar = torch.arange(self.n_bins).unsqueeze(0)
            before = (ar < kk.unsqueeze(1)).float()         # survived these bins
            at = (ar == kk.unsqueeze(1)).float() * ee.unsqueeze(1)  # event in this bin
            ll = (before * log1mh).sum(1) + (at * logh).sum(1)
            return -ll.mean()

        tr_idx, val_idx = np.flatnonzero(tr), np.flatnonzero(val)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            perm = rng.permutation(tr_idx)
            for s in range(0, len(perm), self.batch_size):
                idx = torch.as_tensor(perm[s : s + self.batch_size])
                opt.zero_grad()
                loss = nll(idx)
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vloss = float(nll(torch.as_tensor(val_idx)))
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                bad += 1
            if self.verbose and epoch % 10 == 0:
                print(f"epoch {epoch} val nll {vloss:.4f}")
            if bad >= self.patience:
                break
        self.net_.load_state_dict(best_state)
        self.epochs_run_ = epoch + 1
        self.best_val_nll_ = best
        return self

    def _survival(self, X_feats, embeddings):
        import torch

        F = self.scaler_.transform(X_feats).astype(np.float32) if self.use_features else np.zeros((len(X_feats), 0), np.float32)
        Xs, mask = padded_sets(embeddings, max(self.max_len_, max(len(e) for e in embeddings)))
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.as_tensor(Xs), torch.as_tensor(mask, dtype=torch.float32), torch.as_tensor(F))
            h = torch.sigmoid(logits).numpy()  # (N, 2, n_bins)
        return np.cumprod(1 - h, axis=2)  # survival at the end of each bin

    def _rmst_trapezoid(self, S_arm):
        """S_arm: (N, n_bins) survival at the end of each bin; linear in between, S(0) = 1."""
        edges = self.edges_
        S_full = np.concatenate([np.ones((S_arm.shape[0], 1)), S_arm], axis=1)
        total = np.zeros(S_arm.shape[0])
        for k in range(len(edges) - 1):
            lo, hi = edges[k], min(edges[k + 1], self.horizon)
            if hi <= lo:
                break
            frac = (hi - lo) / (edges[k + 1] - edges[k])
            s_hi = S_full[:, k] + frac * (S_full[:, k + 1] - S_full[:, k])
            total += 0.5 * (S_full[:, k] + s_hi) * (hi - lo)
        return total

    def predict_arms(self, X_feats, embeddings=None):
        S = self._survival(X_feats, embeddings)
        return self._rmst_trapezoid(S[:, 0, :]), self._rmst_trapezoid(S[:, 1, :])

    def predict(self, X_feats, embeddings=None):
        mu0, mu1 = self.predict_arms(X_feats, embeddings)
        return mu1 - mu0


# ---------------------------------------------------------------------------
# registry: one place that knows how to build every candidate by name
# ---------------------------------------------------------------------------
def build_learner(name: str, horizon: float = DEFAULT_HORIZON, seed: int = 0):
    H = horizon
    registry = {
        "constant_ate": lambda: ConstantEffect(H, seed),
        "cox_s_learner": lambda: CoxLearner(interactions=False, horizon=H),
        "cox_interactions": lambda: CoxLearner(interactions=True, horizon=H),
        "t_learner_cox": lambda: TLearnerSurvival("cox", H, seed),
        "t_learner_gbsa": lambda: TLearnerSurvival("gbsa", H, seed),
        "t_learner_rsf": lambda: TLearnerSurvival("rsf", H, seed),
        "s_learner_gbm": lambda: MetaLearner("s", "gbm", horizon=H, seed=seed),
        "s_learner_ridge_interactions": lambda: MetaLearner("s", "ridge", interactions=True, horizon=H, seed=seed),
        "t_learner_gbm": lambda: MetaLearner("t", "gbm", horizon=H, seed=seed),
        "t_learner_ridge": lambda: MetaLearner("t", "ridge", horizon=H, seed=seed),
        "x_learner_gbm": lambda: MetaLearner("x", "gbm", horizon=H, seed=seed),
        "dr_learner_ridge": lambda: MetaLearner("dr", "gbm", effect_base="ridge", horizon=H, seed=seed),
        "dr_learner_gbm_small": lambda: MetaLearner("dr", "gbm", effect_base="gbm_small", horizon=H, seed=seed),
        "dr_learner_gbm": lambda: MetaLearner("dr", "gbm", effect_base="gbm", horizon=H, seed=seed),
        "dr_learner_rf": lambda: MetaLearner("dr", "gbm", effect_base="rf", horizon=H, seed=seed),
        "dr_learner_ridge_gbsanuisance": lambda: MetaLearner("dr", "gbm", effect_base="ridge", nuisance="gbsa", horizon=H, seed=seed),
        "r_learner_ridge": lambda: MetaLearner("r", "gbm", effect_base="ridge", horizon=H, seed=seed),
        "r_learner_gbm_small": lambda: MetaLearner("r", "gbm", effect_base="gbm_small", horizon=H, seed=seed),
        "dr_learner_ridge_kmpv": lambda: MetaLearner("dr", "gbm", effect_base="ridge", censoring="km_pseudo", horizon=H, seed=seed),
        "t_learner_gbm_kmpv": lambda: MetaLearner("t", "gbm", censoring="km_pseudo", horizon=H, seed=seed),
        "setnet": lambda: SetNetLearner(H, use_features=False, seed=seed),
        "setnet_plus_features": lambda: SetNetLearner(H, use_features=True, seed=seed),
    }
    if name.startswith("ensemble:"):
        parts = name.split(":", 1)[1].split("+")
        return EnsembleLearner([build_learner(p, H, seed) for p in parts], name=name)
    if name not in registry:
        raise KeyError(f"unknown learner {name!r}; known: {sorted(registry)}")
    learner = registry[name]()
    assert learner.name == name, (learner.name, name)
    return learner


CANDIDATE_NAMES = [
    "constant_ate", "cox_s_learner", "cox_interactions", "t_learner_cox", "t_learner_gbsa", "t_learner_rsf",
    "s_learner_gbm", "s_learner_ridge_interactions", "t_learner_gbm", "t_learner_ridge", "x_learner_gbm",
    "dr_learner_ridge", "dr_learner_gbm_small", "dr_learner_gbm", "dr_learner_rf", "dr_learner_ridge_gbsanuisance",
    "r_learner_ridge", "r_learner_gbm_small", "dr_learner_ridge_kmpv", "t_learner_gbm_kmpv",
    "setnet", "setnet_plus_features",
]


# ---------------------------------------------------------------------------
# the deployable wrapper
# ---------------------------------------------------------------------------
@dataclass
class TreatmentEffectModel:
    """Raw patient records in, treatment-effect predictions out."""

    learner: Any
    horizon: float = DEFAULT_HORIZON
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _needs_embeddings(learner) -> bool:
        return isinstance(learner, SetNetLearner) or (
            isinstance(learner, EnsembleLearner) and any(isinstance(l, SetNetLearner) for l in learner.learners)
        )

    def fit(self, data: PatientData) -> "TreatmentEffectModel":
        if not data.has_outcomes:
            raise ValueError("training data needs event and duration")
        keep = ~np.isnan(data.treatment)
        d = data.subset(keep)
        X = feature_matrix(d.embeddings)
        w = d.treatment.astype(int)
        if self._needs_embeddings(self.learner):
            self.learner.fit(X, w, d.duration, d.event, embeddings=d.embeddings)
        else:
            self.learner.fit(X, w, d.duration, d.event)
        self.meta.update({"n_train": int(len(d)), "n_dropped_missing_treatment": int((~keep).sum()), "horizon": self.horizon})
        return self

    def predict(self, data: PatientData) -> pd.DataFrame:
        X = feature_matrix(data.embeddings)
        kwargs = {"embeddings": data.embeddings} if self._needs_embeddings(self.learner) else {}
        effect = self.learner.predict(X, **kwargs)
        out = pd.DataFrame({"patient_id": data.patient_id, "predicted_effect": effect})
        if hasattr(self.learner, "predict_arms"):
            try:
                mu0, mu1 = self.learner.predict_arms(X, **kwargs)
                out["rmst_if_treated"] = mu1
                out["rmst_if_untreated"] = mu0
            except AttributeError:
                pass
        return out

    # attributes that are only needed during fitting; dropping them keeps the
    # artifact small and the inference image free of training-only packages
    _TRAINING_ONLY = ("cens_", "phi_", "y_", "weight_", "e_", "pv_")

    def slim(self) -> "TreatmentEffectModel":
        learners = self.learner.learners if isinstance(self.learner, EnsembleLearner) else [self.learner]
        for l in learners:
            for attr in self._TRAINING_ONLY:
                if hasattr(l, attr):
                    delattr(l, attr)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.slim()
        joblib.dump(self, path)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({"learner": getattr(self.learner, "name", type(self.learner).__name__), "horizon": self.horizon, **self.meta}, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "TreatmentEffectModel":
        return joblib.load(path)
