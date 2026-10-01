"""Quick checks that the pieces fit together. Run with: pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from treatment_effect import metrics as M  # noqa: E402
from treatment_effect.data import load_patients, patients_from_object  # noqa: E402
from treatment_effect.features import FEATURE_NAMES, feature_matrix, patient_features  # noqa: E402
from treatment_effect.models import CoxLearner, TreatmentEffectModel, build_learner  # noqa: E402


def fake_records(n=120, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        k = rng.integers(30, 52)
        emb = rng.normal(size=(k, 4)).tolist()
        for row in emb:  # sprinkle missing values in dims 2 and 3
            for d in (2, 3):
                if rng.random() < 0.3:
                    row[d] = None
        recs.append({
            "patient_id": f"p{i}",
            "embeddings": emb,
            "treatment": int(rng.integers(0, 2)) if rng.random() > 0.05 else None,
            "event": int(rng.random() < 0.6),
            "duration": float(rng.exponential(10)),
        })
    return recs


def test_both_json_layouts_load_the_same(tmp_path):
    recs = fake_records(20)
    cols = {k: {str(i): r[k] for i, r in enumerate(recs)} for k in ["patient_id", "embeddings", "treatment"]}
    a = patients_from_object(recs)
    b = patients_from_object(cols)
    assert list(a.patient_id) == list(b.patient_id)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a.embeddings, b.embeddings))
    assert np.array_equal(a.treatment, b.treatment, equal_nan=True)
    assert a.has_outcomes and not b.has_outcomes
    p = tmp_path / "x.json"
    p.write_text(json.dumps(recs))
    assert len(load_patients(p)) == 20


def test_single_record_and_missing_values():
    rec = {"patient_id": "z", "embeddings": [[0.1, None, 2.0, None], [1.0, 1.0, None, 3.5]], "treatment": None}
    d = patients_from_object(rec)
    assert len(d) == 1 and np.isnan(d.treatment[0])
    f = patient_features(d.embeddings[0])
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()


def test_features_are_permutation_invariant():
    recs = fake_records(5)
    d = patients_from_object(recs)
    X = feature_matrix(d.embeddings)
    assert X.shape == (5, len(FEATURE_NAMES)) and np.isfinite(X).all()
    rng = np.random.default_rng(1)
    shuffled = [e[rng.permutation(len(e))] for e in d.embeddings]
    assert np.allclose(feature_matrix(shuffled), X)


def test_km_and_rmst_against_lifelines():
    lifelines = pytest.importorskip("lifelines")
    from lifelines.utils import restricted_mean_survival_time

    rng = np.random.default_rng(0)
    T, C = rng.exponential(10, 500), rng.exponential(12, 500)
    t, e = np.minimum(T, C), T <= C
    km = lifelines.KaplanMeierFitter().fit(t, e)
    assert abs(M.rmst(t, e, 15) - restricted_mean_survival_time(km, t=15)) < 1e-8
    pv = M.rmst_pseudo_values(t, e, 15)
    assert abs(pv.mean() - M.rmst(t, e, 15)) < 1e-8


def test_autoc_is_positive_for_an_informative_score_and_near_zero_otherwise():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    w = rng.integers(0, 2, n)
    T = rng.exponential(8 * np.exp(0.3 * x * w), n)
    C = rng.exponential(25, n)
    t, e = np.minimum(T, C), T <= C
    good = M.autoc(x, t, e, w, 20)
    junk = np.mean([M.autoc(rng.normal(size=n), t, e, w, 20) for _ in range(5)])
    assert good > 0.3 and abs(junk) < 0.3


def test_model_fit_save_load_predict(tmp_path):
    d = patients_from_object(fake_records(200))
    model = TreatmentEffectModel(learner=CoxLearner(interactions=False), horizon=15.0).fit(d)
    preds = model.predict(d)
    assert list(preds.columns) == ["patient_id", "predicted_effect", "rmst_if_treated", "rmst_if_untreated"]
    assert len(preds) == 200 and np.isfinite(preds.predicted_effect).all()
    path = tmp_path / "m.joblib"
    model.save(path)
    again = TreatmentEffectModel.load(path).predict(d)
    assert np.allclose(again.predicted_effect, preds.predicted_effect)
    assert (path.with_suffix(".json")).exists()


def test_registry_builds_every_candidate():
    from treatment_effect.models import CANDIDATE_NAMES

    for name in CANDIDATE_NAMES:
        assert build_learner(name).name == name
    ens = build_learner("ensemble:cox_s_learner+t_learner_cox")
    assert len(ens.learners) == 2
