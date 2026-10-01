"""Data splits used everywhere, so the notebook, the experiments and the
final training all agree on who is in the test set.

* Patients with an unknown treatment are dropped before any modelling (we
  cannot learn an arm-specific outcome without the arm).
* 20% of the remaining patients are set aside as the test split, stratified on
  treatment x event so both arms and the censoring rate look the same on both
  sides. Nothing is tuned on them.
* Model selection is 5-fold CV on the remaining 80% (the development split),
  with the same stratification.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import PatientData

TEST_SEED = 42
CV_SEED = 0


def labeled(data: PatientData) -> PatientData:
    return data.subset(~np.isnan(data.treatment))


def strata(w: np.ndarray, event: np.ndarray) -> np.ndarray:
    return w.astype(int) * 2 + event.astype(int)


def dev_test_split(data: PatientData, test_size: float = 0.2, seed: int = TEST_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (dev_idx, test_idx) into ``data`` (which should already be labeled)."""
    idx = np.arange(len(data))
    dev, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strata(data.treatment, data.event))
    return np.sort(dev), np.sort(test)


def cv_folds(w: np.ndarray, event: np.ndarray, n_splits: int = 5, seed: int = CV_SEED):
    """List of (train_idx, val_idx) pairs, stratified on treatment x event."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(w)), strata(w, event)))
