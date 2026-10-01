"""Fit the final model on every labeled patient and save it.

    python -m treatment_effect.train --data data/patient_records.json --out artifacts/model.joblib

Options let you swap the learner (any name from ``models.CANDIDATE_NAMES`` or
an ``ensemble:a+b`` spec), the RMST horizon and the seed. The default learner
is the one picked in the notebook.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .data import load_patients
from .models import DEFAULT_HORIZON, TreatmentEffectModel, build_learner

FINAL_LEARNER = "dr_learner_ridge"  # overwritten below once the notebook's selection is final


def train(data_path: str | Path, out_path: str | Path, learner: str = FINAL_LEARNER, horizon: float = DEFAULT_HORIZON, seed: int = 0) -> TreatmentEffectModel:
    data = load_patients(data_path)
    model = TreatmentEffectModel(learner=build_learner(learner, horizon, seed), horizon=horizon)
    t0 = time.time()
    model.fit(data)
    model.meta.update({"learner": learner, "seed": seed, "train_seconds": round(time.time() - t0, 1), "data_file": str(Path(data_path).name)})
    model.save(out_path)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/patient_records.json")
    ap.add_argument("--out", default="artifacts/model.joblib")
    ap.add_argument("--learner", default=FINAL_LEARNER)
    ap.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    model = train(args.data, args.out, args.learner, args.horizon, args.seed)
    print(f"saved {args.out}: learner={args.learner}, horizon={args.horizon}, "
          f"trained on {model.meta['n_train']} patients ({model.meta['n_dropped_missing_treatment']} dropped for unknown treatment), "
          f"{model.meta['train_seconds']}s")


if __name__ == "__main__":
    main()
