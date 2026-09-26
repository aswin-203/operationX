from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class HyperparameterTuner:

    def __init__(
        self,
        data_file: str,
        target: str,
        model: str = "random_forest",
        feature_count: int | None = None,
        n_splits: int = 5,
        n_iter: int = 10,
        random_state: int = 42,
    ):
        self.data_file = Path(data_file)
        self.target = target
        self.model = model
        self.feature_count = feature_count
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.random_state = random_state

        if model != "random_forest":
            raise ValueError(
                f"Unsupported model: {model}"
            )

    def _load_data(self):

        df = pd.read_csv(self.data_file)

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

        if self.feature_count is not None:

            if self.feature_count <= 0:
                raise ValueError(
                    "feature_count must be positive"
                )

            if self.feature_count > len(
                feature_columns
            ):
                raise ValueError(
                    "feature_count exceeds "
                    "available features"
                )

            feature_columns = feature_columns[
                : self.feature_count
            ]

        X = df[feature_columns].copy()
        y = df[self.target].copy()

        return X, y, feature_columns

    def _build_model(self, **params):

        model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            **params,
        )

        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    def _evaluate_model(self, model, X, y):

        cv = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=-1,
        )

        mae = mean_absolute_error(
            y,
            predictions,
        )

        rmse = mean_squared_error(
            y,
            predictions,
        ) ** 0.5

        r2 = r2_score(
            y,
            predictions,
        )

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

    def evaluate(self):

        X, y, feature_columns = (
            self._load_data()
        )

        cv = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # -------------------------------------------------
        # DEFAULT MODEL
        # -------------------------------------------------

        default_model = self._build_model(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
        )

        default_metrics = (
            self._evaluate_model(
                default_model,
                X,
                y,
            )
        )

        # -------------------------------------------------
        # HYPERPARAMETER SEARCH
        # -------------------------------------------------

        search_model = self._build_model()

        parameter_distributions = {
            "model__n_estimators": [
                50,
                100,
                150,
                200,
            ],
            "model__max_depth": [
                None,
                3,
                5,
                8,
                12,
            ],
            "model__min_samples_split": [
                2,
                4,
                6,
                8,
            ],
            "model__min_samples_leaf": [
                1,
                2,
                3,
                4,
            ],
            "model__max_features": [
                0.5,
                0.7,
                1.0,
            ],
        }

        search = RandomizedSearchCV(
            estimator=search_model,
            param_distributions=parameter_distributions,
            n_iter=self.n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X, y)

        best_model = search.best_estimator_

        tuned_metrics = (
            self._evaluate_model(
                best_model,
                X,
                y,
            )
        )

        return {
            "target": self.target,
            "model": self.model,
            "rows": len(X),
            "feature_count": len(
                feature_columns
            ),
            "features": feature_columns,
            "default": default_metrics,
            "tuned": tuned_metrics,
            "best_params": search.best_params_,
            "best_cv_score": float(
                search.best_score_
            ),
        }