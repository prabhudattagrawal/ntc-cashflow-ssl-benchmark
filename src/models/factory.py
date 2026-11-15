from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_lgbm(params, seed):
    params = dict(params)
    # LightGBM grabs every core by default, which fights with torch during
    # pretraining, so stay single-threaded unless the caller says otherwise.
    params.setdefault("n_jobs", 1)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(**params, random_state=seed, verbosity=-1)),
    ])
