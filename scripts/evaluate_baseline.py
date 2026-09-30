from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BASE = Path("data/ml_large/final")


TARGETS = {
    "pH": {
        "train": BASE / "pH_train.csv",
        "validation": BASE / "pH_validation.csv",
        "test": BASE / "pH_test.csv",
    },

    "temperature_kelvin": {
        "train": BASE / "temperature_kelvin_train.csv",
        "validation": BASE / "temperature_kelvin_validation.csv",
        "test": BASE / "temperature_kelvin_test.csv",
    },

    "matthews_coefficient": {
        "train": BASE / "matthews_coefficient_train.csv",
        "validation": BASE / "matthews_coefficient_validation.csv",
        "test": BASE / "matthews_coefficient_test.csv",
    },

    "solvent_percent": {
        "train": BASE / "solvent_percent_train.csv",
        "validation": BASE / "solvent_percent_validation.csv",
        "test": BASE / "solvent_percent_test.csv",
    },
}


def evaluate_target(target, paths):

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])

    y_train = pd.to_numeric(
        train[target],
        errors="coerce",
    ).dropna()

    y_test = pd.to_numeric(
        test[target],
        errors="coerce",
    ).dropna()

    # --------------------------------------------------------
    # Baseline:
    #
    # Always predict the training-set mean.
    # --------------------------------------------------------

    baseline_value = y_train.mean()

    predictions = np.full(
        len(y_test),
        baseline_value,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print(target)
    print("=" * 70)

    print(
        f"Training mean : {baseline_value:.4f}"
    )

    print(
        f"Test MAE      : {mae:.4f}"
    )

    print(
        f"Test RMSE     : {rmse:.4f}"
    )

    print(
        f"Test R2       : {r2:.4f}"
    )


def main():

    print("=" * 70)
    print("OPERATION X — BASELINE EVALUATION")
    print("=" * 70)

    for target, paths in TARGETS.items():

        evaluate_target(
            target,
            paths,
        )

    print()
    print("=" * 70)
    print("BASELINE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()