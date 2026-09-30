from pathlib import Path

import pandas as pd
import numpy as np


BASE = Path("data/ml_large/final")

TARGETS = {
    "pH": BASE / "pH_train.csv",
    "temperature_kelvin": BASE / "temperature_kelvin_train.csv",
    "matthews_coefficient": BASE / "matthews_coefficient_train.csv",
    "solvent_percent": BASE / "solvent_percent_train.csv",
}


def analyze_target(name, path):

    df = pd.read_csv(path)

    y = pd.to_numeric(
        df[name],
        errors="coerce"
    ).dropna()

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Records : {len(y)}")
    print()

    print("Statistics")
    print("-" * 70)

    print(f"Minimum : {y.min():.4f}")
    print(f"Maximum : {y.max():.4f}")
    print(f"Mean    : {y.mean():.4f}")
    print(f"Median  : {y.median():.4f}")
    print(f"Std     : {y.std():.4f}")

    print()

    print("Percentiles")
    print("-" * 70)

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:

        print(
            f"{p:2d}% : {y.quantile(p / 100):.4f}"
        )

    print()

    print("Potential outliers using IQR")
    print("-" * 70)

    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = y[
        (y < lower) |
        (y > upper)
    ]

    print(f"Q1          : {q1:.4f}")
    print(f"Q3          : {q3:.4f}")
    print(f"IQR         : {iqr:.4f}")
    print(f"Lower bound : {lower:.4f}")
    print(f"Upper bound : {upper:.4f}")
    print(f"Outliers    : {len(outliers)}")

    print()

    print("Most common values")
    print("-" * 70)

    print(
        y.round(3)
        .value_counts()
        .head(15)
        .to_string()
    )


def main():

    print("=" * 70)
    print("OPERATION X — TARGET DATA ANALYSIS")
    print("=" * 70)

    for name, path in TARGETS.items():

        analyze_target(
            name,
            path
        )

    print()
    print("=" * 70)
    print("TARGET ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()