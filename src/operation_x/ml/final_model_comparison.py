from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import KFold, cross_val_predict


class FinalModelComparison:

    MODELS = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }

    def __init__(
        self,
        data_file: str,
        target: str,
        n_splits: int = 5,
    ):
        self.data_file = Path(data_file)
        self.target = target
        self.n_splits = n_splits

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
                f"not found in dataset"
            )

        excluded_columns = {
            self.target,
            "query_pdb",
            "similar_pdb",
        }

        feature_columns = [
            column
            for column in df.columns
            if column not in excluded_columns
        ]

        if not feature_columns:
            raise ValueError(
                "No feature columns available"
            )

        X = df[feature_columns].copy()
        y = df[self.target].copy()

        # Ensure all features are numeric.
        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Replace invalid values with column medians.
        X = X.fillna(
            X.median()
        )

        if y.isna().any():
            raise ValueError(
                "Target contains missing values"
            )

        return X, y

    def _metrics(
        self,
        y_true,
        predictions,
    ):

        return {
            "mae": float(
                mean_absolute_error(
                    y_true,
                    predictions,
                )
            ),
            "rmse": float(
                mean_squared_error(
                    y_true,
                    predictions,
                ) ** 0.5
            ),
            "r2": float(
                r2_score(
                    y_true,
                    predictions,
                )
            ),
        }

    def evaluate(self):

        X, y = self._load_data()

        if len(X) < self.n_splits:
            raise ValueError(
                "Number of rows must be at least "
                "n_splits"
            )

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42,
        )

        results = {}

        for name, model in self.MODELS.items():

            predictions = cross_val_predict(
                model,
                X,
                y,
                cv=kfold,
                n_jobs=1,
            )

            results[name] = self._metrics(
                y,
                predictions,
            )

        # Mean baseline
        baseline_predictions = [
            y.mean()
        ] * len(y)

        results["Mean Baseline"] = (
            self._metrics(
                y,
                baseline_predictions,
            )
        )

        return {
            "target": self.target,
            "rows": len(X),
            "feature_count": X.shape[1],
            "results": results,
        }

    def best_model(
        self,
        result=None,
    ):

        if result is None:
            result = self.evaluate()

        candidates = {
            name: metrics
            for name, metrics
            in result["results"].items()
            if name != "Mean Baseline"
        }

        if not candidates:
            raise ValueError(
                "No model results available"
            )

        # Primary criterion: lowest MAE.
        best_name = min(
            candidates,
            key=lambda name:
                candidates[name]["mae"],
        )

        best_metrics = candidates[
            best_name
        ].copy()

        best_metrics["model"] = (
            best_name
        )

        return best_metrics