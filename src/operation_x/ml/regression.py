from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


class SolventRegression:
    """
    Predict crystallization solvent percentage
    from sequence, structure, and domain similarity
    features.

    Validation is grouped by query protein to avoid
    leakage between related similarity records.
    """

    TARGET = "solvent_percent"

    def __init__(
        self,
        features_file: str,
        targets_file: str,
        metadata_file: str,
        random_state: int = 42,
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

        self.random_state = random_state

    def load_data(self):
        """Load features, targets, and metadata."""

        features = pd.read_csv(
            self.features_file
        )

        targets = pd.read_csv(
            self.targets_file
        )

        metadata = pd.read_csv(
            self.metadata_file
        )

        if len(features) != len(targets):
            raise ValueError(
                "Features and targets have "
                "different row counts"
            )

        if len(features) != len(metadata):
            raise ValueError(
                "Features and metadata have "
                "different row counts"
            )

        if self.TARGET not in targets.columns:
            raise ValueError(
                f"Missing target: {self.TARGET}"
            )

        return (
            features,
            targets,
            metadata,
        )

    def prepare_data(self):
        """
        Prepare rows where the target is known.

        Missing feature values are filled with the
        median calculated from the available training
        data later during each fold.
        """

        (
            features,
            targets,
            metadata,
        ) = self.load_data()

        target = targets[
            self.TARGET
        ]

        valid = target.notna()

        X = features.loc[
            valid
        ].copy()

        y = target.loc[
            valid
        ].copy()

        groups = metadata.loc[
            valid,
            "query_pdb",
        ].copy()

        return (
            X,
            y,
            groups,
        )

    @staticmethod
    def _fill_missing(
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ):
        """
        Fill missing numeric features using medians
        calculated only from the training fold.

        This prevents validation information from
        leaking into training.
        """

        train = train.copy()
        validation = validation.copy()

        medians = train.median(
            numeric_only=True
        )

        train = train.fillna(
            medians
        )

        validation = validation.fillna(
            medians
        )

        return (
            train,
            validation,
        )

    def evaluate(
        self,
        n_splits: int = 5,
    ) -> dict:
        """
        Evaluate the model using grouped
        cross-validation.
        """

        (
            X,
            y,
            groups,
        ) = self.prepare_data()

        unique_groups = groups.nunique()

        if unique_groups < n_splits:
            raise ValueError(
                "Number of groups is smaller "
                "than n_splits"
            )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        actual = []
        predicted = []

        fold_results = []

        for fold, (
            train_index,
            validation_index,
        ) in enumerate(
            splitter.split(
                X,
                y,
                groups,
            ),
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

            (
                X_train,
                X_validation,
            ) = self._fill_missing(
                X_train,
                X_validation,
            )

            model = RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_validation
            )

            actual.extend(
                y_validation.tolist()
            )

            predicted.extend(
                predictions.tolist()
            )

            fold_mae = (
                mean_absolute_error(
                    y_validation,
                    predictions,
                )
            )

            fold_rmse = (
                np.sqrt(
                    mean_squared_error(
                        y_validation,
                        predictions,
                    )
                )
            )

            fold_r2 = r2_score(
                y_validation,
                predictions,
            )

            fold_results.append(
                {
                    "fold": fold,
                    "mae": fold_mae,
                    "rmse": fold_rmse,
                    "r2": fold_r2,
                    "validation_rows": len(
                        validation_index
                    ),
                    "validation_groups": (
                        groups.iloc[
                            validation_index
                        ]
                        .nunique()
                    ),
                }
            )

        overall_mae = (
            mean_absolute_error(
                actual,
                predicted,
            )
        )

        overall_rmse = (
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        )

        overall_r2 = r2_score(
            actual,
            predicted,
        )

        return {
            "target": self.TARGET,
            "rows": len(y),
            "groups": unique_groups,
            "folds": fold_results,
            "mae": overall_mae,
            "rmse": overall_rmse,
            "r2": overall_r2,
        }
