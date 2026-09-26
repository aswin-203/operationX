from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


class FeatureSetEvaluator:

    def __init__(
        self,
        data_file: str,
        target: str,
        feature_counts: list[int] | None = None,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.data_file = Path(data_file)
        self.target = target

        self.feature_counts = (
            feature_counts
            if feature_counts is not None
            else [5, 10, 15, 20, 28]
        )

        self.n_splits = n_splits
        self.random_state = random_state

    def _load_data(self) -> pd.DataFrame:

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

        return df

    def _prepare_data(self, df):

        excluded = {
            "query_pdb",
            "similar_pdb",
            self.target,
        }

        feature_columns = [
            column
            for column in df.columns
            if column not in excluded
        ]

        X = df[feature_columns].select_dtypes(
            include=["number"]
        ).copy()

        y = df[self.target].copy()

        if X.empty:
            raise ValueError(
                "No numeric features available"
            )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return X, y

    def _select_features_from_training_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_features: int,
    ) -> list[str]:

        # Median imputation is fitted ONLY on the
        # training fold.
        medians = X_train.median()

        X_train_filled = X_train.fillna(
            medians
        )

        # Feature ranking is also performed ONLY
        # using the training fold.
        selector = ExtraTreesRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1,
        )

        selector.fit(
            X_train_filled,
            y_train,
        )

        ranking = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance":
                    selector.feature_importances_,
            }
        )

        ranking = ranking.sort_values(
            "importance",
            ascending=False,
        )

        n_features = min(
            n_features,
            len(ranking),
        )

        return ranking.head(
            n_features
        )["feature"].tolist()

    def _evaluate_model(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    ):

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        return {
            "mae": float(
                mean_absolute_error(
                    y_test,
                    predictions,
                )
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )
            ),
            "r2": float(
                r2_score(
                    y_test,
                    predictions,
                )
            ),
        }

    def evaluate(self) -> dict:

        df = self._load_data()

        X, y = self._prepare_data(
            df
        )

        if self.n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2"
            )

        if len(X) < self.n_splits:
            raise ValueError(
                "Dataset contains fewer rows "
                "than n_splits"
            )

        available_features = len(
            X.columns
        )

        feature_counts = sorted(
            set(
                min(
                    count,
                    available_features,
                )
                for count in self.feature_counts
                if count > 0
            )
        )

        if not feature_counts:
            raise ValueError(
                "No valid feature counts supplied"
            )

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        results = {}

        for n_features in feature_counts:

            model_results = {
                "Random Forest": [],
                "Extra Trees": [],
            }

            for train_index, test_index in kfold.split(
                X
            ):

                X_train = X.iloc[
                    train_index
                ].copy()

                X_test = X.iloc[
                    test_index
                ].copy()

                y_train = y.iloc[
                    train_index
                ]

                y_test = y.iloc[
                    test_index
                ]

                selected_features = (
                    self._select_features_from_training_data(
                        X_train,
                        y_train,
                        n_features,
                    )
                )

                # Fit imputation using training data
                # only.
                medians = X_train[
                    selected_features
                ].median()

                X_train_selected = (
                    X_train[
                        selected_features
                    ].fillna(medians)
                )

                X_test_selected = (
                    X_test[
                        selected_features
                    ].fillna(medians)
                )

                rf = RandomForestRegressor(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

                et = ExtraTreesRegressor(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

                rf_metrics = self._evaluate_model(
                    rf,
                    X_train_selected,
                    y_train,
                    X_test_selected,
                    y_test,
                )

                et_metrics = self._evaluate_model(
                    et,
                    X_train_selected,
                    y_train,
                    X_test_selected,
                    y_test,
                )

                model_results[
                    "Random Forest"
                ].append(
                    rf_metrics
                )

                model_results[
                    "Extra Trees"
                ].append(
                    et_metrics
                )

            results[n_features] = {}

            for model_name, folds in (
                model_results.items()
            ):

                results[n_features][
                    model_name
                ] = {
                    "mae": float(
                        np.mean(
                            [
                                fold["mae"]
                                for fold in folds
                            ]
                        )
                    ),
                    "rmse": float(
                        np.mean(
                            [
                                fold["rmse"]
                                for fold in folds
                            ]
                        )
                    ),
                    "r2": float(
                        np.mean(
                            [
                                fold["r2"]
                                for fold in folds
                            ]
                        )
                    ),
                }

        return {
            "target": self.target,
            "rows": len(df),
            "available_features": available_features,
            "feature_counts": feature_counts,
            "results": results,
        }

    def save(
        self,
        output_file: str,
    ) -> dict:

        output_path = Path(
            output_file
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result = self.evaluate()

        rows = []

        for feature_count, models in (
            result["results"].items()
        ):

            for model_name, metrics in (
                models.items()
            ):

                rows.append(
                    {
                        "target": self.target,
                        "feature_count": feature_count,
                        "model": model_name,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                    }
                )

        dataframe = pd.DataFrame(
            rows
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        best_row = dataframe.sort_values(
            "r2",
            ascending=False,
        ).iloc[0]

        return {
            "output": str(output_path),
            "rows": len(dataframe),
            "best_feature_count": int(
                best_row["feature_count"]
            ),
            "best_model": best_row["model"],
            "best_r2": float(
                best_row["r2"]
            ),
        }