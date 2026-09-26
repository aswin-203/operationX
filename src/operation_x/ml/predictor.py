from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


class OperationXPredictor:

    MODELS = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42,
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
            self.target,
            "query_pdb",
            "similar_pdb",
        }

        feature_columns = [
            column
            for column in df.columns
            if column not in excluded
        ]

        if not feature_columns:
            raise ValueError(
                "No feature columns available"
            )

        X = df[feature_columns].copy()
        y = df[self.target].copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        X = X.fillna(
            X.median()
        )

        if y.isna().any():
            raise ValueError(
                "Target contains missing values"
            )

        self.feature_columns = feature_columns

        return X, y

    def fit(self):

        if self.model_name not in self.MODELS:
            raise ValueError(
                f"Unknown model: "
                f"{self.model_name}"
            )

        X, y = self._load_data()

        self.model = self.MODELS[
            self.model_name
        ]

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
                "Missing features: "
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