from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


class FeatureSelector:

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
                f"not found"
            )

        return df

    def _prepare_features(self, df):

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
            [float("inf"), float("-inf")],
            pd.NA,
        )

        X = X.fillna(
            X.median()
        )

        return X, y

    def rank_features(self) -> pd.DataFrame:

        df = self._load_data()

        X, y = self._prepare_features(
            df
        )

        model = ExtraTreesRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1,
        )

        model.fit(
            X,
            y,
        )

        ranking = pd.DataFrame(
            {
                "feature": X.columns,
                "importance":
                    model.feature_importances_,
            }
        )

        ranking = ranking.sort_values(
            "importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        ranking["rank"] = (
            ranking.index + 1
        )

        return ranking[
            [
                "rank",
                "feature",
                "importance",
            ]
        ]

    def select(
        self,
        n_features: int,
    ) -> list[str]:

        ranking = self.rank_features()

        if n_features <= 0:
            raise ValueError(
                "n_features must be greater than zero"
            )

        n_features = min(
            n_features,
            len(ranking),
        )

        return ranking.head(
            n_features
        )["feature"].tolist()

    def select_multiple(
        self,
        sizes: list[int],
    ) -> dict[int, list[str]]:

        if not sizes:
            raise ValueError(
                "sizes cannot be empty"
            )

        ranking = self.rank_features()

        result = {}

        for size in sizes:

            if size <= 0:
                raise ValueError(
                    "Feature counts must be "
                    "greater than zero"
                )

            size = min(
                size,
                len(ranking),
            )

            result[size] = (
                ranking.head(size)[
                    "feature"
                ].tolist()
            )

        return result

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

        ranking = self.rank_features()

        ranking.to_csv(
            output_path,
            index=False,
        )

        return {
            "output": str(output_path),
            "feature_count": len(ranking),
            "top_feature": ranking.iloc[0][
                "feature"
            ],
        }