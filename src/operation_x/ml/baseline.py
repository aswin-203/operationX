from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


class MeanBaseline:
    """
    Group-aware mean baseline for crystallization targets.

    For each validation fold, the prediction is the mean
    target value calculated only from the training groups.
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

    def evaluate(
        self,
        n_splits: int = 5,
    ) -> dict:

        target = self.targets[
            self.target
        ]

        valid_mask = target.notna()

        target = target[
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

        folds = []

        all_actual = []
        all_predictions = []

        for fold_number, (
            train_index,
            validation_index,
        ) in enumerate(
            splitter.split(
                target,
                target,
                groups,
            ),
            start=1,
        ):

            y_train = target.iloc[
                train_index
            ]

            y_validation = target.iloc[
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

            folds.append(
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

        overall_mae = mean_absolute_error(
            all_actual,
            all_predictions,
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

        return {
            "target": self.target,
            "rows": len(target),
            "groups": groups.nunique(),
            "mae": float(overall_mae),
            "rmse": float(overall_rmse),
            "r2": float(overall_r2),
            "folds": folds,
        }