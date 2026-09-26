from pathlib import Path

import numpy as np
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
from sklearn.model_selection import KFold


DATA_DIR = Path("data/ml_sequence_only")

TARGETS = [
    "pH",
    "temperature_kelvin",
    "matthews_coefficient",
    "solvent_percent",
]

MODELS = {
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        random_state=42,
    ),
}


def evaluate_target(target):

    path = (
        DATA_DIR
        / f"{target}_sequence_only.csv"
    )

    df = pd.read_csv(path)

    excluded = {
        "pdb_id",
        target,
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    X = df[feature_columns].copy()
    y = df[target].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print()
    print("=" * 70)
    print(target)
    print("=" * 70)
    print(
        f"Samples : {len(df)}"
    )
    print(
        f"Features: {len(feature_columns)}"
    )

    results = []

    for model_name, model_template in MODELS.items():

        maes = []
        rmses = []
        r2s = []

        for train_index, test_index in kfold.split(X):

            X_train = X.iloc[train_index].copy()
            X_test = X.iloc[test_index].copy()

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            # Fit imputation only on training data.
            medians = X_train.median()

            X_train = X_train.fillna(
                medians
            )

            X_test = X_test.fillna(
                medians
            )

            model = model_template.__class__(
                **model_template.get_params()
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            maes.append(
                mean_absolute_error(
                    y_test,
                    predictions,
                )
            )

            rmses.append(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )
            )

            r2s.append(
                r2_score(
                    y_test,
                    predictions,
                )
            )

        result = {
            "model": model_name,
            "mae": np.mean(maes),
            "rmse": np.mean(rmses),
            "r2": np.mean(r2s),
        }

        results.append(result)

        print(
            f"{model_name:20} "
            f"MAE={result['mae']:.4f}  "
            f"RMSE={result['rmse']:.4f}  "
            f"R²={result['r2']:.4f}"
        )

    return results


def main():

    all_results = []

    for target in TARGETS:

        results = evaluate_target(
            target
        )

        for result in results:
            result["target"] = target
            all_results.append(result)

    results_df = pd.DataFrame(
        all_results
    )

    output = (
        DATA_DIR
        / "evaluation_results.csv"
    )

    results_df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print("SEQUENCE-ONLY EVALUATION COMPLETE")
    print("=" * 70)
    print()
    print(
        results_df.to_string(
            index=False
        )
    )
    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
