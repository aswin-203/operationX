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
from sklearn.model_selection import GroupKFold


class LargeModelComparison:
    """
    Compare regression models for one target in the
    large Operation X dataset.

    Validation is grouped by query PDB so that highly
    similar records from the same query protein do not
    leak between training and validation sets.
    """

    def __init__(
        self,
        features_file: str,
        targets_file: str,
        metadata_file: str,
    ):
        self.features_file = Path(features_file)
        self.targets_file = Path(targets_file)
        self.metadata_file = Path(metadata_file)

        self.features = pd.read_csv(
            self.features_file
        )

        self.targets = pd.read_csv(
            self.targets_file
        )

        self.metadata = pd.read_csv(
            self.metadata_file
        )

        if len(self.features) != len(self.targets):
            raise ValueError(
                "Features and targets must have "
                "the same number of rows"
            )

        if len(self.features) != len(self.metadata):
            raise ValueError(
                "Features and metadata must have "
                "the same number of rows"
            )

        if self.targets.shape[1] != 1:
            raise ValueError(
                "Targets file must contain exactly "
                "one target column"
            )

        if "query_pdb" not in self.metadata.columns:
            raise ValueError(
                "Metadata must contain query_pdb"
            )

        self.target = self.targets.columns[0]

        self.X = self.features.copy()

        self.y = self.targets[
            self.target
        ].astype(float)

        self.groups = self.metadata[
            "query_pdb"
        ]

    @staticmethod
    def _models():
        return {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

            "Extra Trees": ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        }

    @staticmethod
    def _metrics(y_true, y_pred):
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

    def _evaluate_model(
        self,
        model,
        splitter,
    ):
        predictions = np.zeros(
            len(self.y),
            dtype=float,
        )

        folds = []

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            splitter.split(
                self.X,
                self.y,
                self.groups,
            ),
            start=1,
        ):
            X_train = self.X.iloc[
                train_index
            ]

            X_validation = self.X.iloc[
                validation_index
            ]

            y_train = self.y.iloc[
                train_index
            ]

            y_validation = self.y.iloc[
                validation_index
            ]

            model.fit(
                X_train,
                y_train,
            )

            y_pred = model.predict(
                X_validation
            )

            predictions[
                validation_index
            ] = y_pred

            metrics = self._metrics(
                y_validation,
                y_pred,
            )

            validation_groups = (
                self.groups.iloc[
                    validation_index
                ].nunique()
            )

            folds.append(
                {
                    "fold": fold_number,
                    **metrics,
                    "rows": len(
                        validation_index
                    ),
                    "validation_groups": int(
                        validation_groups
                    ),
                }
            )

        overall = self._metrics(
            self.y,
            predictions,
        )

        return {
            **overall,
            "folds": folds,
        }

    def _evaluate_baseline(
        self,
        splitter,
    ):
        predictions = np.zeros(
            len(self.y),
            dtype=float,
        )

        folds = []

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            splitter.split(
                self.X,
                self.y,
                self.groups,
            ),
            start=1,
        ):
            y_train = self.y.iloc[
                train_index
            ]

            y_validation = self.y.iloc[
                validation_index
            ]

            mean_value = y_train.mean()

            y_pred = np.full(
                len(validation_index),
                mean_value,
            )

            predictions[
                validation_index
            ] = y_pred

            metrics = self._metrics(
                y_validation,
                y_pred,
            )

            validation_groups = (
                self.groups.iloc[
                    validation_index
                ].nunique()
            )

            folds.append(
                {
                    "fold": fold_number,
                    **metrics,
                    "rows": len(
                        validation_index
                    ),
                    "validation_groups": int(
                        validation_groups
                    ),
                }
            )

        overall = self._metrics(
            self.y,
            predictions,
        )

        return {
            **overall,
            "folds": folds,
        }

    def evaluate(
        self,
        n_splits: int = 5,
    ):
        unique_groups = (
            self.groups.nunique()
        )

        if n_splits > unique_groups:
            raise ValueError(
                "n_splits cannot be greater "
                "than the number of query groups"
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        results = {}

        for name, model in self._models().items():
            results[name] = (
                self._evaluate_model(
                    model,
                    splitter,
                )
            )

        results["Mean Baseline"] = (
            self._evaluate_baseline(
                splitter
            )
        )

        return {
            "target": self.target,
            "rows": len(self.X),
            "groups": int(unique_groups),
            "results": results,
        }