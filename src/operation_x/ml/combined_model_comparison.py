from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold


class CombinedModelComparison:

    RANDOM_STATE = 42

    def __init__(
        self,
        data_file: str,
        target: str,
    ):
        self.data_file = Path(data_file)
        self.target = target

    def _load_data(self) -> pd.DataFrame:

        if not self.data_file.exists():
            raise FileNotFoundError(
                self.data_file
            )

        df = pd.read_csv(self.data_file)

        if self.target not in df.columns:
            raise ValueError(
                f"Target '{self.target}' "
                f"not found in dataset"
            )

        if "query_pdb" not in df.columns:
            raise ValueError(
                "Dataset must contain query_pdb"
            )

        return df

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ):
        excluded = {
            self.target,
            "query_pdb",
        }

        feature_columns = [
            column
            for column in df.columns
            if column not in excluded
        ]

        X = df[feature_columns].copy()
        y = df[self.target].copy()

        # Convert everything to numeric.
        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if X.isna().any().any():
            raise ValueError(
                "Feature matrix contains "
                "missing or non-numeric values"
            )

        if y.isna().any():
            raise ValueError(
                "Target contains missing values"
            )

        return (
            X,
            y,
            feature_columns,
        )

    def _models(self):

        return {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
            ),

            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=self.RANDOM_STATE,
            ),
        }

    def _evaluate_predictions(
        self,
        y_true,
        y_pred,
    ):

        return {
            "mae": float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),
            "r2": float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),
        }

    def evaluate(
        self,
        n_splits: int = 5,
    ) -> dict:

        df = self._load_data()

        X, y, feature_columns = (
            self._prepare_features(df)
        )

        if len(df) < n_splits:
            raise ValueError(
                "n_splits cannot exceed "
                "number of rows"
            )

        groups = df["query_pdb"]

        unique_groups = groups.nunique()

        if unique_groups < n_splits:
            raise ValueError(
                "n_splits cannot exceed "
                "number of query groups"
            )

        # KFold is intentionally used here because
        # each combined dataset currently has one
        # row per query protein.
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.RANDOM_STATE,
        )

        models = self._models()

        results = {}

        for name in models:
            fold_metrics = []

            for train_idx, validation_idx in splitter.split(
                X
            ):

                X_train = X.iloc[
                    train_idx
                ]

                X_validation = X.iloc[
                    validation_idx
                ]

                y_train = y.iloc[
                    train_idx
                ]

                y_validation = y.iloc[
                    validation_idx
                ]

                model = self._models()[name]

                model.fit(
                    X_train,
                    y_train,
                )

                predictions = model.predict(
                    X_validation
                )

                metrics = (
                    self._evaluate_predictions(
                        y_validation,
                        predictions,
                    )
                )

                fold_metrics.append(
                    metrics
                )

            results[name] = {
                "mae": float(
                    np.mean(
                        [
                            x["mae"]
                            for x in fold_metrics
                        ]
                    )
                ),
                "rmse": float(
                    np.mean(
                        [
                            x["rmse"]
                            for x in fold_metrics
                        ]
                    )
                ),
                "r2": float(
                    np.mean(
                        [
                            x["r2"]
                            for x in fold_metrics
                        ]
                    )
                ),
                "folds": fold_metrics,
            }

        # Mean baseline.
        baseline_predictions = []

        baseline_folds = []

        for train_idx, validation_idx in splitter.split(
            X
        ):

            y_train = y.iloc[
                train_idx
            ]

            y_validation = y.iloc[
                validation_idx
            ]

            mean_value = float(
                y_train.mean()
            )

            predictions = np.full(
                len(y_validation),
                mean_value,
            )

            metrics = (
                self._evaluate_predictions(
                    y_validation,
                    predictions,
                )
            )

            baseline_folds.append(
                metrics
            )

            baseline_predictions.extend(
                predictions
            )

        results["Mean Baseline"] = {
            "mae": float(
                np.mean(
                    [
                        x["mae"]
                        for x in baseline_folds
                    ]
                )
            ),
            "rmse": float(
                np.mean(
                    [
                        x["rmse"]
                        for x in baseline_folds
                    ]
                )
            ),
            "r2": float(
                np.mean(
                    [
                        x["r2"]
                        for x in baseline_folds
                    ]
                )
            ),
            "folds": baseline_folds,
        }

        return {
            "target": self.target,
            "rows": len(df),
            "groups": unique_groups,
            "feature_count": len(
                feature_columns
            ),
            "features": feature_columns,
            "results": results,
        }