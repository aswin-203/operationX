from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


class ModelComparison:
    """
    Compare regression models using group-aware cross-validation.

    Groups are query proteins, so rows originating from the same
    query protein are never split between training and validation.
    """

    def __init__(
        self,
        features_file: str,
        targets_file: str,
        metadata_file: str,
        target: str = "solvent_percent",
    ):
        self.features_file = Path(
            features_file
        ).expanduser()

        self.targets_file = Path(
            targets_file
        ).expanduser()

        self.metadata_file = Path(
            metadata_file
        ).expanduser()

        self.target = target

        self.features = pd.read_csv(
            self.features_file
        )

        self.targets = pd.read_csv(
            self.targets_file
        )

        self.metadata = pd.read_csv(
            self.metadata_file
        )

        if self.target not in self.targets.columns:
            raise ValueError(
                f"Unknown target: {self.target}"
            )

        if "query_pdb" not in self.metadata.columns:
            raise ValueError(
                "metadata must contain query_pdb"
            )

        if len(self.features) != len(
            self.targets
        ):
            raise ValueError(
                "Features and targets must "
                "have the same number of rows"
            )

        if len(self.features) != len(
            self.metadata
        ):
            raise ValueError(
                "Features and metadata must "
                "have the same number of rows"
            )

    def _models(self) -> dict:
        """
        Return the regression models to compare.
        """

        return {
            "Linear Regression": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        LinearRegression(),
                    ),
                ]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "Extra Trees": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=300,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "Gradient Boosting": Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        GradientBoostingRegressor(
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

    def evaluate(
        self,
        n_splits: int = 5,
    ) -> dict:

        target = self.targets[
            self.target
        ]

        valid_mask = target.notna()

        X = self.features.loc[
            valid_mask
        ].reset_index(drop=True)

        y = target.loc[
            valid_mask
        ].reset_index(drop=True)

        groups = self.metadata.loc[
            valid_mask,
            "query_pdb",
        ].reset_index(drop=True)

        if groups.nunique() < n_splits:
            raise ValueError(
                "Number of groups must be "
                "greater than or equal to "
                "n_splits"
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        models = self._models()

        results = {}

        # --------------------------------------------------
        # Create the exact same folds once.
        # --------------------------------------------------

        folds = list(
            splitter.split(
                X,
                y,
                groups,
            )
        )

        # --------------------------------------------------
        # Evaluate every ML model.
        # --------------------------------------------------

        for model_name, model in models.items():

            fold_results = []

            all_actual = []
            all_predictions = []

            for fold_number, (
                train_index,
                validation_index,
            ) in enumerate(
                folds,
                start=1,
            ):

                X_train = X.iloc[
                    train_index
                ]

                X_validation = X.iloc[
                    validation_index
                ]

                y_train = y.iloc[
                    train_index
                ]

                y_validation = y.iloc[
                    validation_index
                ]

                model.fit(
                    X_train,
                    y_train,
                )

                predictions = model.predict(
                    X_validation
                )

                mae = mean_absolute_error(
                    y_validation,
                    predictions,
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_validation,
                        predictions,
                    )
                )

                r2 = r2_score(
                    y_validation,
                    predictions,
                )

                validation_groups = (
                    groups.iloc[
                        validation_index
                    ].unique()
                )

                fold_results.append(
                    {
                        "fold": fold_number,
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "r2": float(r2),
                        "validation_rows": len(
                            validation_index
                        ),
                        "validation_groups": len(
                            validation_groups
                        ),
                    }
                )

                all_actual.extend(
                    y_validation.tolist()
                )

                all_predictions.extend(
                    predictions.tolist()
                )

            overall_mae = (
                mean_absolute_error(
                    all_actual,
                    all_predictions,
                )
            )

            overall_rmse = np.sqrt(
                mean_squared_error(
                    all_actual,
                    all_predictions,
                )
            )

            overall_r2 = r2_score(
                all_actual,
                all_predictions,
            )

            results[model_name] = {
                "mae": float(
                    overall_mae
                ),
                "rmse": float(
                    overall_rmse
                ),
                "r2": float(
                    overall_r2
                ),
                "folds": fold_results,
            }

        # --------------------------------------------------
        # Mean baseline using the SAME folds.
        # --------------------------------------------------

        baseline_folds = []

        baseline_actual = []
        baseline_predictions = []

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            folds,
            start=1,
        ):

            y_train = y.iloc[
                train_index
            ]

            y_validation = y.iloc[
                validation_index
            ]

            prediction = float(
                y_train.mean()
            )

            predictions = np.full(
                len(y_validation),
                prediction,
            )

            mae = mean_absolute_error(
                y_validation,
                predictions,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_validation,
                    predictions,
                )
            )

            r2 = r2_score(
                y_validation,
                predictions,
            )

            validation_groups = (
                groups.iloc[
                    validation_index
                ].unique()
            )

            baseline_folds.append(
                {
                    "fold": fold_number,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "validation_rows": len(
                        validation_index
                    ),
                    "validation_groups": len(
                        validation_groups
                    ),
                }
            )

            baseline_actual.extend(
                y_validation.tolist()
            )

            baseline_predictions.extend(
                predictions.tolist()
            )

        baseline_mae = (
            mean_absolute_error(
                baseline_actual,
                baseline_predictions,
            )
        )

        baseline_rmse = np.sqrt(
            mean_squared_error(
                baseline_actual,
                baseline_predictions,
            )
        )

        baseline_r2 = r2_score(
            baseline_actual,
            baseline_predictions,
        )

        results["Mean Baseline"] = {
            "mae": float(
                baseline_mae
            ),
            "rmse": float(
                baseline_rmse
            ),
            "r2": float(
                baseline_r2
            ),
            "folds": baseline_folds,
        }

        return {
            "target": self.target,
            "rows": len(y),
            "groups": groups.nunique(),
            "results": results,
        }