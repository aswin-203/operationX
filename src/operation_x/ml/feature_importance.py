from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


class FeatureImportanceAnalyzer:

    def __init__(
        self,
        data_file: str,
        target: str,
        random_state: int = 42,
    ):
        self.data_file = Path(data_file)
        self.target = target
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
                f"not found in dataset"
            )

        if len(df) < 5:
            raise ValueError(
                "Dataset must contain at least 5 rows"
            )

        return df

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ):

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

        if not feature_columns:
            raise ValueError(
                "No feature columns available"
            )

        X = df[feature_columns].copy()
        y = df[self.target].copy()

        # Keep only numeric features.
        X = X.select_dtypes(
            include=["number"]
        )

        if X.empty:
            raise ValueError(
                "No numeric feature columns available"
            )

        # Handle missing numeric values.
        X = X.fillna(
            X.median()
        )

        return X, y

    def analyze(self) -> pd.DataFrame:

        df = self._load_data()

        X, y = self._prepare_features(
            df
        )

        random_forest = RandomForestRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1,
        )

        extra_trees = ExtraTreesRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1,
        )

        random_forest.fit(
            X,
            y,
        )

        extra_trees.fit(
            X,
            y,
        )

        result = pd.DataFrame(
            {
                "feature": X.columns,
                "random_forest_importance":
                    random_forest.feature_importances_,
                "extra_trees_importance":
                    extra_trees.feature_importances_,
            }
        )

        result["mean_importance"] = (
            result[
                [
                    "random_forest_importance",
                    "extra_trees_importance",
                ]
            ].mean(axis=1)
        )

        result = result.sort_values(
            "mean_importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        result["rank"] = (
            result.index + 1
        )

        return result[
            [
                "rank",
                "feature",
                "random_forest_importance",
                "extra_trees_importance",
                "mean_importance",
            ]
        ]

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

        result = self.analyze()

        result.to_csv(
            output_path,
            index=False,
        )

        return {
            "output": str(output_path),
            "rows": len(result),
            "feature_count": len(result),
            "top_feature": result.iloc[0][
                "feature"
            ],
        }