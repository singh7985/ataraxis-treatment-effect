"""Score new patients with a saved model.

Batch, from the command line:

    python -m treatment_effect.predict --model artifacts/model.joblib \
        --input data/inference_inputs.json --output predictions.csv

The input can be any of the json layouts in this project (a list of patient
records, the pandas "columns" layout of inference_inputs.json, or a single
record). Outcomes and treatment are not needed for prediction. The output is a
csv (or json with --format json) with one row per patient:

    patient_id, predicted_effect, rmst_if_treated, rmst_if_untreated

``predicted_effect`` is the expected event-free time gained by treating,
within the model's horizon (same time units as ``duration``). Positive means
treatment is expected to help.

As a service:

    python -m treatment_effect.predict serve --model artifacts/model.joblib --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @patients.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import load_patients, patients_from_object
from .models import TreatmentEffectModel


def predict_file(model: TreatmentEffectModel, input_path: str | Path) -> pd.DataFrame:
    data = load_patients(input_path)
    return model.predict(data)


def write_predictions(preds: pd.DataFrame, output_path: str | Path | None, fmt: str = "csv") -> None:
    if fmt == "json":
        payload = preds.to_dict(orient="records")
        if output_path is None:
            json.dump(payload, sys.stdout, indent=2)
        else:
            with open(output_path, "w") as f:
                json.dump(payload, f, indent=2)
    else:
        if output_path is None:
            preds.to_csv(sys.stdout, index=False)
        else:
            preds.to_csv(output_path, index=False)


def make_app(model: TreatmentEffectModel):
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(title="treatment effect model", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "learner": getattr(model.learner, "name", "?"), "horizon": model.horizon}

    @app.post("/predict")
    async def predict(request: Request):
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"body is not valid json: {exc}")
        try:
            data = patients_from_object(payload)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return model.predict(data).to_dict(orient="records")

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="predict", choices=["predict", "serve"])
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--input", help="json file with patients (predict mode)")
    ap.add_argument("--output", help="where to write predictions; stdout if omitted")
    ap.add_argument("--format", default="csv", choices=["csv", "json"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    model = TreatmentEffectModel.load(args.model)
    if args.mode == "serve":
        import uvicorn

        uvicorn.run(make_app(model), host=args.host, port=args.port)
        return
    if not args.input:
        ap.error("--input is required in predict mode")
    preds = predict_file(model, args.input)
    write_predictions(preds, args.output, args.format)
    if args.output:
        print(f"wrote {len(preds)} predictions to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
