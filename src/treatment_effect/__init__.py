"""Treatment effect model for the Ataraxis take-home.

The package is split by responsibility:

    data.py      load the json files in either layout they come in
    features.py  turn a patient's set of 4-d vectors into a fixed feature vector
    models.py    the CATE learners that were tried, plus the final model wrapper
    metrics.py   survival + treatment-effect metrics (RMST, IPCW, AUTOC, calibration)
    splits.py    the dev/test split and the cross-validation folds
    train.py     command line: fit the final model and save it
    predict.py   command line / HTTP: load a saved model and score new patients
"""

__version__ = "0.1.0"
