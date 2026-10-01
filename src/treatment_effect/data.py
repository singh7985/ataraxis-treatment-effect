"""Loading the patient json files.

Two layouts show up in this project:

* ``patient_records.json`` is a list of records, one dict per patient.
* ``inference_inputs.json`` and ``fake_inference_outcomes.json`` are pandas
  "columns" orient: ``{"patient_id": {"0": "p15000", ...}, "embeddings": {...}}``.

``load_patients`` accepts both, and also a single record or a plain dict of
columns, so the same function serves training, batch inference and the API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

EMBED_DIM = 4


@dataclass
class PatientData:
    """A batch of patients in a convenient shape.

    ``embeddings[i]`` is a float array of shape (n_i, 4) with NaN for missing
    values. ``treatment`` is float with NaN where it is unknown. ``event`` and
    ``duration`` are None for inference inputs that carry no outcomes.
    """

    patient_id: np.ndarray
    embeddings: list[np.ndarray]
    treatment: np.ndarray
    event: np.ndarray | None = None
    duration: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patient_id)

    @property
    def has_outcomes(self) -> bool:
        return self.event is not None and self.duration is not None

    def subset(self, mask: np.ndarray) -> "PatientData":
        idx = np.flatnonzero(mask)
        return PatientData(
            patient_id=self.patient_id[idx],
            embeddings=[self.embeddings[i] for i in idx],
            treatment=self.treatment[idx],
            event=None if self.event is None else self.event[idx],
            duration=None if self.duration is None else self.duration[idx],
        )

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({"patient_id": self.patient_id, "treatment": self.treatment})
        if self.has_outcomes:
            out["event"] = self.event
            out["duration"] = self.duration
        out["n_vectors"] = [len(e) for e in self.embeddings]
        return out


def _to_array(vectors: Iterable[Iterable[float | None]]) -> np.ndarray:
    arr = np.array(
        [[np.nan if v is None else float(v) for v in vec] for vec in vectors],
        dtype=float,
    )
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise ValueError(f"expected an (n, {EMBED_DIM}) list of vectors, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError("a patient needs at least one embedding vector")
    return arr


def _records_from_columns(cols: dict[str, dict]) -> list[dict]:
    """Turn pandas 'columns' orient into a list of record dicts, keeping row order."""
    keys = sorted(cols["patient_id"].keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
    return [{name: cols[name][k] for name in cols} for k in keys]


def _normalise(obj: Any) -> list[dict]:
    """Accept any of the layouts described in the module docstring."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "embeddings" in obj and isinstance(obj["embeddings"], dict):
            return _records_from_columns(obj)
        if "embeddings" in obj and isinstance(obj["embeddings"], list):
            # a single patient
            return [obj]
        if "patients" in obj:
            return _normalise(obj["patients"])
    raise ValueError("unrecognised patient json layout")


def patients_from_object(obj: Any) -> PatientData:
    records = _normalise(obj)
    ids, embs, trt, ev, dur = [], [], [], [], []
    any_outcome = False
    for i, r in enumerate(records):
        ids.append(str(r.get("patient_id", f"row{i}")))
        embs.append(_to_array(r["embeddings"]))
        t = r.get("treatment", None)
        trt.append(np.nan if t is None else float(t))
        if "event" in r and "duration" in r and r["event"] is not None:
            any_outcome = True
            ev.append(int(r["event"]))
            dur.append(float(r["duration"]))
        else:
            ev.append(-1)
            dur.append(np.nan)
    data = PatientData(
        patient_id=np.array(ids, dtype=object),
        embeddings=embs,
        treatment=np.array(trt, dtype=float),
    )
    if any_outcome:
        data.event = np.array(ev, dtype=int)
        data.duration = np.array(dur, dtype=float)
    return data


def load_patients(path: str | Path) -> PatientData:
    with open(path) as f:
        obj = json.load(f)
    return patients_from_object(obj)


def load_outcomes(path: str | Path) -> pd.DataFrame:
    """Outcomes file -> DataFrame with patient_id, event, duration (and treatment if present)."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "patient_id" in obj and isinstance(obj["patient_id"], dict):
        df = pd.DataFrame(_records_from_columns(obj))
    else:
        df = pd.DataFrame(obj)
    df["patient_id"] = df["patient_id"].astype(str)
    return df
