from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)


class SequenceOnlyPredictor:

    MODELS = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    def __init__(
        self,
        data_file: str,
        target: str,
        model_name: str,
    ):
        self.data_file = Path(data_file)
        self.target = target
        self.model_name = model_name

        self.model = None
        self.feature_columns = None

    def _load_data(self):

        if not self.data_file.exists():
            raise FileNotFoundError(
                self.data_file
            )

        df = pd.read_csv(
            self.data_file
        )

        if self.target not in df.columns:
            raise ValueError(
                f"Target column '{self.target}' "
                f"not found"
            )

        excluded = {
            "pdb_id",
            self.target,
        }

        self.feature_columns = [
            column
            for column in df.columns
            if column not in excluded
        ]

        if not self.feature_columns:
            raise ValueError(
                "No sequence feature columns available"
            )

        X = df[
            self.feature_columns
        ].copy()

        y = df[self.target].copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        X = X.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

        X = X.fillna(
            X.median()
        )

        if y.isna().any():
            raise ValueError(
                "Target contains missing values"
            )

        return X, y

    def fit(self):

        if self.model_name not in self.MODELS:
            raise ValueError(
                f"Unknown sequence-only model: "
                f"{self.model_name}"
            )

        X, y = self._load_data()

        model_class = (
            self.MODELS[
                self.model_name
            ].__class__
        )

        model_params = (
            self.MODELS[
                self.model_name
            ].get_params()
        )

        self.model = model_class(
            **model_params
        )

        self.model.fit(
            X,
            y,
        )

        return {
            "target": self.target,
            "rows": len(X),
            "feature_count": len(
                self.feature_columns
            ),
            "model": self.model_name,
        }

    def predict(
        self,
        features: dict,
    ):

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() first."
            )

        missing = [
            column
            for column in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Missing sequence features: "
                + ", ".join(missing)
            )

        row = pd.DataFrame(
            [
                {
                    column: features[column]
                    for column in self.feature_columns
                }
            ]
        )

        row = row.apply(
            pd.to_numeric,
            errors="coerce",
        )

        prediction = self.model.predict(
            row
        )[0]

        return {
            "target": self.target,
            "prediction": float(
                prediction
            ),
            "model": self.model_name,
        }
