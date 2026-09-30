import pandas as pd
from pathlib import Path

BASE = Path("data/ml_large")

PHYSIO = BASE / "physicochemical_features.csv"
BLAST = BASE / "blast_features.csv"
DOMAIN = BASE / "domain_features.csv"
EXPRESSION = BASE / "expression_features.csv"

OUTPUT = BASE / "combined_features.csv"


def load(path):
    df = pd.read_csv(path)
    print(f"{path.name:35s}: {len(df)} rows, {len(df.columns)} columns")
    return df


def main():

    print("=" * 60)
    print("OPERATION X — COMBINE ML FEATURES")
    print("=" * 60)
    print()

    physio = load(PHYSIO)
    blast = load(BLAST)
    domain = load(DOMAIN)
    expression = load(EXPRESSION)

    # Start from the clean ML record list.
    combined = physio.copy()

    # Merge feature tables using pdb_id.
    combined = combined.merge(
        blast,
        on="pdb_id",
        how="left",
        suffixes=("", "_blast"),
    )

    combined = combined.merge(
        domain,
        on="pdb_id",
        how="left",
        suffixes=("", "_domain"),
    )

    combined = combined.merge(
        expression,
        on="pdb_id",
        how="left",
        suffixes=("", "_expression"),
    )

    # Remove accidental duplicate columns if any.
    duplicate_columns = [
        c for c in combined.columns
        if c.endswith("_blast")
        or c.endswith("_domain")
        or c.endswith("_expression")
    ]

    if duplicate_columns:
        combined.drop(columns=duplicate_columns, inplace=True)

    # Identify feature columns.
    feature_columns = [
        c for c in combined.columns
        if c != "pdb_id"
    ]

    # Domain failures mean missing domain information.
    # For count-based domain features, zero is appropriate:
    # zero observed domains rather than an invented domain annotation.
    domain_columns = [
        "domain_count",
        "pfam_count",
        "interpro_count",
    ]

    for column in domain_columns:
        if column in combined.columns:
            combined[column] = combined[column].fillna(0)

    # Preserve BLAST no-hit information explicitly.
    if "best_evalue" in combined.columns:
        combined["blast_no_hit"] = combined["best_evalue"].isna().astype(int)

    # Keep the raw e-value missing for now.
    # We will transform/handle it during ML preprocessing.
    
    combined.to_csv(OUTPUT, index=False)

    print()
    print("COMBINED FEATURE SUMMARY")
    print("-" * 60)
    print(f"Rows              : {len(combined)}")
    print(f"Columns           : {len(combined.columns)}")
    print(f"ML feature count  : {len(feature_columns) + 1}")
    print(f"Output            : {OUTPUT}")

    print()
    print("FEATURE COLUMNS")
    print("-" * 60)

    for i, column in enumerate(combined.columns, start=1):
        print(f"{i:3d} {column}")

    print()
    print("MISSING VALUES")
    print("-" * 60)

    missing = combined.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        for column, count in missing.items():
            print(f"{column:35s}: {count}")

    print()
    print("=" * 60)
    print("COMBINED FEATURE BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()