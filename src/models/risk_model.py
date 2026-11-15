import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate
from src.models.factory import make_lgbm


def fit_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test,
                     output_dir, lgbm_params, seed):
    # Preprocessing lives inside the pipeline and only ever sees the training
    # rows. Validation and test get scored, they are never fitted on.
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, solver="lbfgs",
                                         random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed)),
        ]),
        "lightgbm": make_lgbm(lgbm_params, seed),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        valid_p = model.predict_proba(X_valid)[:, 1]
        test_p = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "split": "validation",
                     **evaluate(y_valid, valid_p)})
        rows.append({"model": name, "split": "test",
                     **evaluate(y_test, test_p)})
        joblib.dump(model, output_dir / f"{name}.joblib")

    return pd.DataFrame(rows), models
