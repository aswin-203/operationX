import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

FEATURE_FILE = Path(
    "data/ml_large/combined_features.csv"
)

RECORD_FILE = Path(
    "data/ml_large/records_with_groups.json"
)

OUTPUT_FILE = Path(
    "reports/feature_signal_analysis.json"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "pH",
    "temperature_kelvin",
    "matthews_coefficient",
    "solvent_percent",
]


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Feature dataset: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    return df


# ============================================================
# LOAD TARGETS FROM RECORDS
# ============================================================

def load_targets():

    with RECORD_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        records = json.load(f)

    rows = []

    for key, record in records.items():

        rows.append(
            {
                "pdb_id": str(key).strip().lower(),

                "pH": record.get("pH"),

                "temperature_kelvin":
                    record.get(
                        "temperature_kelvin"
                    ),

                "matthews_coefficient":
                    record.get(
                        "matthews_coefficient"
                    ),

                "solvent_percent":
                    record.get(
                        "solvent_percent"
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(features, targets):

    features = features.copy()
    targets = targets.copy()

    features["pdb_id"] = (
        features["pdb_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    targets["pdb_id"] = (
        targets["pdb_id"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged = features.merge(
        targets,
        on="pdb_id",
        how="left",
        suffixes=("", "_target"),
    )

    return merged


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(df):

    excluded = {
        "pdb_id",
        "status",
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


# ============================================================
# FEATURE SUMMARY
# ============================================================

def analyze_features(
    df,
    feature_columns,
):

    results = []

    for feature in feature_columns:

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        non_null = series.notna().sum()

        unique = series.nunique(
            dropna=True
        )

        missing_fraction = (
            1 -
            (
                non_null /
                len(series)
            )
        )

        if non_null > 0:

            minimum = float(
                series.min()
            )

            maximum = float(
                series.max()
            )

            mean = float(
                series.mean()
            )

            std = float(
                series.std()
            )

        else:

            minimum = None
            maximum = None
            mean = None
            std = None

        results.append(
            {
                "feature": feature,

                "non_null": int(
                    non_null
                ),

                "missing_fraction":
                    float(
                        missing_fraction
                    ),

                "unique_values":
                    int(unique),

                "minimum": minimum,

                "maximum": maximum,

                "mean": mean,

                "std": std,
            }
        )

    return results


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(
    df,
    feature_columns,
):

    correlation_results = {}

    for target in TARGETS:

        target_values = pd.to_numeric(
            df[target],
            errors="coerce",
        )

        rows = []

        for feature in feature_columns:

            feature_values = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            valid = (
                feature_values.notna()
                &
                target_values.notna()
            )

            if valid.sum() < 3:

                continue

            x = feature_values[valid]
            y = target_values[valid]

            # Pearson correlation.
            pearson = x.corr(
                y,
                method="pearson",
            )

            # Spearman correlation.
            spearman = x.corr(
                y,
                method="spearman",
            )

            if pd.isna(pearson):
                pearson = 0.0

            if pd.isna(spearman):
                spearman = 0.0

            rows.append(
                {
                    "feature": feature,

                    "pearson": float(
                        pearson
                    ),

                    "abs_pearson": float(
                        abs(pearson)
                    ),

                    "spearman": float(
                        spearman
                    ),

                    "abs_spearman": float(
                        abs(spearman)
                    ),

                    "n": int(
                        valid.sum()
                    ),
                }
            )

        rows.sort(
            key=lambda x:
                x["abs_spearman"],
            reverse=True,
        )

        correlation_results[target] = rows

    return correlation_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "OPERATION X — FEATURE SIGNAL ANALYSIS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data.
    # --------------------------------------------------------

    print()
    print("Loading feature dataset...")

    features = load_features()

    print()
    print("Loading target data...")

    targets = load_targets()

    print(
        f"Target records: {len(targets)}"
    )

    # --------------------------------------------------------
    # Merge.
    # --------------------------------------------------------

    print()
    print("Merging features and targets...")

    df = prepare_data(
        features,
        targets,
    )

    print(
        f"Merged dataset: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Get feature columns.
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        df
    )

    print()
    print(
        f"Candidate feature count: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Feature statistics.
    # --------------------------------------------------------

    print()
    print(
        "Analyzing feature quality..."
    )

    feature_summary = analyze_features(
        df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Correlations.
    # --------------------------------------------------------

    print()
    print(
        "Analyzing feature-target correlations..."
    )

    correlations = analyze_correlations(
        df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    for target in TARGETS:

        print()
        print("=" * 70)
        print(
            f"TOP FEATURES FOR: {target}"
        )
        print("=" * 70)

        rows = correlations[target]

        for row in rows[:15]:

            print(
                f"{row['feature']:35s} "
                f"Spearman={row['spearman']: .4f} "
                f"Pearson={row['pearson']: .4f} "
                f"N={row['n']}"
            )

    # --------------------------------------------------------
    # Identify suspicious features.
    # --------------------------------------------------------

    constant_features = []

    high_missing_features = []

    for row in feature_summary:

        if row["unique_values"] <= 1:

            constant_features.append(
                row["feature"]
            )

        if row["missing_fraction"] >= 0.50:

            high_missing_features.append(
                row["feature"]
            )

    print()
    print("=" * 70)
    print("FEATURE QUALITY")
    print("=" * 70)

    print()
    print(
        "Constant features:"
    )

    for feature in constant_features:

        print(
            f"  - {feature}"
        )

    print()
    print(
        "Features with >=50% missing:"
    )

    for feature in high_missing_features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Save report.
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset": {
            "rows": int(
                len(df)
            ),

            "columns": int(
                len(df.columns)
            ),

            "feature_count":
                len(feature_columns),
        },

        "feature_summary":
            feature_summary,

        "constant_features":
            constant_features,

        "high_missing_features":
            high_missing_features,

        "correlations":
            correlations,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        f"Report saved: {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print(
        "FEATURE SIGNAL ANALYSIS COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()