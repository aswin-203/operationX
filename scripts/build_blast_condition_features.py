"""
Operation X
Build BLAST-derived experimental-condition features.

For each target protein:

    Protein
       ↓
    BLAST homologs
       ↓
    Training-set homologs only
       ↓
    Experimental conditions of those homologs
       ↓
    Similarity-weighted condition features

Outputs:

    data/ml_large/blast_condition_features/
        pH.csv
        temperature_kelvin.csv
        matthews_coefficient.csv
        solvent_percent.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BLAST_FILE = Path(
    "data/blast/large/all_vs_all.tsv"
)

RECORDS_FILE = Path(
    "data/ml_large/records_with_groups.json"
)

TARGET_DIR = Path(
    "data/ml_large/final"
)

OUTPUT_DIR = Path(
    "data/ml_large/blast_condition_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MIN_IDENTITY = 20.0
MIN_COVERAGE = 80.0
TOP_K = 20


TARGETS = {
    "pH": {
        "train_file": TARGET_DIR / "pH_train.csv",
        "target_column": "pH",
    },

    "temperature_kelvin": {
        "train_file": TARGET_DIR / "temperature_kelvin_train.csv",
        "target_column": "temperature_kelvin",
    },

    "matthews_coefficient": {
        "train_file": TARGET_DIR / "matthews_coefficient_train.csv",
        "target_column": "matthews_coefficient",
    },

    "solvent_percent": {
        "train_file": TARGET_DIR / "solvent_percent_train.csv",
        "target_column": "solvent_percent",
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize_pdb_id(value: str) -> str:
    """
    Normalize BLAST/PDB identifiers.

    Examples:

        101M
        101M_1
        pdb|101M|1

    all become:

        101m
    """

    value = str(value).strip()

    if value.startswith("pdb|"):

        parts = value.split("|")

        if len(parts) >= 2:
            value = parts[1]

    if "_" in value:
        value = value.split("_")[0]

    return value.lower()


# ============================================================
# LOAD RECORDS
# ============================================================

def load_records() -> dict[str, dict]:
    """
    Load clean Operation X records.

    Supports these possible JSON structures:

    1. List:

        [
            {
                "pdb_id": "101M",
                ...
            },
            ...
        ]

    2. Dictionary containing records:

        {
            "records": [
                {
                    "pdb_id": "101M",
                    ...
                }
            ]
        }

    3. Dictionary keyed by PDB ID:

        {
            "101m": {
                ...
            },
            "102m": {
                ...
            }
        }
    """

    print()
    print("=" * 70)
    print("LOADING OPERATION X RECORDS")
    print("=" * 70)

    with open(
        RECORDS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Detect JSON structure
    # --------------------------------------------------------

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        # Structure:
        #
        # {
        #     "records": [...]
        # }

        if (
            "records" in data
            and isinstance(data["records"], list)
        ):

            records = data["records"]

        else:

            # Structure:
            #
            # {
            #     "101m": {...},
            #     "102m": {...}
            # }

            records = []

            for key, value in data.items():

                if isinstance(value, dict):

                    record = value.copy()

                    if "pdb_id" not in record:

                        record["pdb_id"] = key

                    records.append(record)

    else:

        raise TypeError(
            "Unsupported JSON structure in "
            f"{RECORDS_FILE}"
        )

    # --------------------------------------------------------
    # Build normalized lookup
    # --------------------------------------------------------

    result = {}

    skipped = 0

    for record in records:

        if not isinstance(record, dict):

            skipped += 1

            continue

        if "pdb_id" not in record:

            skipped += 1

            continue

        pdb_id = normalize_pdb_id(
            record["pdb_id"]
        )

        result[pdb_id] = record

    print(
        f"Records loaded: {len(result):,}"
    )

    if skipped:

        print(
            f"Records skipped: {skipped:,}"
        )

    return result


# ============================================================
# LOAD BLAST
# ============================================================

def load_blast() -> pd.DataFrame:
    """
    Load the all-vs-all BLAST result.

    Expected columns:

        qseqid
        sseqid
        pident
        length
        qlen
        slen
        evalue
        bitscore
    """

    columns = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "qlen",
        "slen",
        "evalue",
        "bitscore",
    ]

    print()
    print("=" * 70)
    print("LOADING BLAST RESULTS")
    print("=" * 70)

    df = pd.read_csv(
        BLAST_FILE,
        sep="\t",
        names=columns,
    )

    print(
        f"BLAST hits loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Normalize IDs
    # --------------------------------------------------------

    df["query_id"] = df["qseqid"].map(
        normalize_pdb_id
    )

    df["subject_id"] = df["sseqid"].map(
        normalize_pdb_id
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "pident",
        "length",
        "qlen",
        "slen",
        "evalue",
        "bitscore",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Query coverage
    # --------------------------------------------------------

    df["coverage"] = (
        df["length"]
        /
        df["qlen"].replace(
            0,
            np.nan
        )
    ) * 100.0

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "query_id",
            "subject_id",
            "pident",
            "coverage",
            "bitscore",
        ]
    )

    # --------------------------------------------------------
    # Remove self hits
    # --------------------------------------------------------

    df = df[
        df["query_id"]
        !=
        df["subject_id"]
    ].copy()

    # --------------------------------------------------------
    # Minimum similarity
    # --------------------------------------------------------

    df = df[
        (df["pident"] >= MIN_IDENTITY)
        &
        (df["coverage"] >= MIN_COVERAGE)
    ].copy()

    print(
        f"Usable BLAST hits: {len(df):,}"
    )

    return df


# ============================================================
# SIMILARITY WEIGHT
# ============================================================

def calculate_weight(
    identity: float,
    coverage: float,
) -> float:
    """
    Similarity weight.

    Example:

        identity = 60%
        coverage = 90%

        weight = 0.60 * 0.90
               = 0.54
    """

    return (
        identity / 100.0
    ) * (
        coverage / 100.0
    )


# ============================================================
# WEIGHTED MEAN
# ============================================================

def weighted_mean(
    values: list[float],
    weights: list[float],
) -> float:

    if not values:
        return np.nan

    values_array = np.asarray(
        values,
        dtype=float
    )

    weights_array = np.asarray(
        weights,
        dtype=float
    )

    valid = (
        np.isfinite(values_array)
        &
        np.isfinite(weights_array)
        &
        (weights_array > 0)
    )

    if not valid.any():

        return np.nan

    values_array = values_array[
        valid
    ]

    weights_array = weights_array[
        valid
    ]

    return float(
        np.average(
            values_array,
            weights=weights_array
        )
    )


# ============================================================
# BUILD FEATURES FOR ONE TARGET
# ============================================================

def build_target_features(
    blast_df: pd.DataFrame,
    records: dict[str, dict],
    target_name: str,
    config: dict,
) -> None:

    train_file = config[
        "train_file"
    ]

    target_column = config[
        "target_column"
    ]

    print()
    print("=" * 70)
    print(
        f"TARGET: {target_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load training IDs
    # --------------------------------------------------------

    train_df = pd.read_csv(
        train_file
    )

    train_ids = set(
        train_df[
            "pdb_id"
        ]
        .astype(str)
        .map(normalize_pdb_id)
    )

    print(
        f"Training proteins: "
        f"{len(train_ids):,}"
    )

    # --------------------------------------------------------
    # Keep only BLAST subjects that are in training set
    # --------------------------------------------------------

    usable_blast = blast_df[
        blast_df[
            "subject_id"
        ].isin(train_ids)
    ].copy()

    print(
        "BLAST hits whose subject "
        "is in training set: "
        f"{len(usable_blast):,}"
    )

    # --------------------------------------------------------
    # Group hits by query
    # --------------------------------------------------------

    grouped = usable_blast.groupby(
        "query_id"
    )

    rows = []

    # --------------------------------------------------------
    # Process every protein
    # --------------------------------------------------------

    all_ids = sorted(
        records.keys()
    )

    total = len(all_ids)

    for index, query_id in enumerate(
        all_ids,
        start=1
    ):

        if index % 500 == 0:

            print(
                f"Processing "
                f"{index:,}/{total:,}"
            )

        # ----------------------------------------------------
        # No BLAST neighbors
        # ----------------------------------------------------

        if query_id not in grouped.groups:

            rows.append({

                "pdb_id":
                    query_id,

                "blast_condition_neighbor_count":
                    0,

                "blast_condition_strong_neighbor_count":
                    0,

                "blast_condition_mean_identity_top5":
                    np.nan,

                "blast_condition_mean_coverage_top5":
                    np.nan,

                "blast_condition_weighted_value":
                    np.nan,

                "blast_condition_top_neighbor_identity":
                    np.nan,

                "blast_condition_top_neighbor_coverage":
                    np.nan,

                "blast_condition_top_neighbor_bitscore":
                    np.nan,
            })

            continue

        # ----------------------------------------------------
        # Get neighbors
        # ----------------------------------------------------

        neighbors = grouped.get_group(
            query_id
        ).copy()

        # ----------------------------------------------------
        # Safety:
        # Never use the query itself.
        # ----------------------------------------------------

        neighbors = neighbors[
            neighbors[
                "subject_id"
            ]
            !=
            query_id
        ].copy()

        # ----------------------------------------------------
        # Extract experimental condition
        # ----------------------------------------------------

        neighbor_records = []

        for _, hit in neighbors.iterrows():

            subject_id = hit[
                "subject_id"
            ]

            # Subject must exist
            # in our records.

            if subject_id not in records:

                continue

            record = records[
                subject_id
            ]

            value = record.get(
                target_column
            )

            # Missing target
            if value is None:

                continue

            try:

                value = float(value)

            except (
                TypeError,
                ValueError,
            ):

                continue

            if not np.isfinite(value):

                continue

            neighbor_records.append({

                "subject_id":
                    subject_id,

                "identity":
                    float(
                        hit["pident"]
                    ),

                "coverage":
                    float(
                        hit["coverage"]
                    ),

                "bitscore":
                    float(
                        hit["bitscore"]
                    ),

                "evalue":
                    float(
                        hit["evalue"]
                    ),

                "value":
                    value,
            })

        # ----------------------------------------------------
        # No usable experimental neighbors
        # ----------------------------------------------------

        if not neighbor_records:

            rows.append({

                "pdb_id":
                    query_id,

                "blast_condition_neighbor_count":
                    0,

                "blast_condition_strong_neighbor_count":
                    0,

                "blast_condition_mean_identity_top5":
                    np.nan,

                "blast_condition_mean_coverage_top5":
                    np.nan,

                "blast_condition_weighted_value":
                    np.nan,

                "blast_condition_top_neighbor_identity":
                    np.nan,

                "blast_condition_top_neighbor_coverage":
                    np.nan,

                "blast_condition_top_neighbor_bitscore":
                    np.nan,
            })

            continue

        neighbors_df = pd.DataFrame(
            neighbor_records
        )

        # ----------------------------------------------------
        # Sort neighbors
        # ----------------------------------------------------

        neighbors_df = neighbors_df.sort_values(
            [
                "identity",
                "coverage",
                "bitscore",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        # ----------------------------------------------------
        # Top K neighbors
        # ----------------------------------------------------

        top_neighbors = (
            neighbors_df
            .head(TOP_K)
            .copy()
        )

        # ----------------------------------------------------
        # Calculate similarity weights
        # ----------------------------------------------------

        top_neighbors[
            "weight"
        ] = top_neighbors.apply(

            lambda row:
                calculate_weight(
                    row["identity"],
                    row["coverage"],
                ),

            axis=1,
        )

        values = (
            top_neighbors[
                "value"
            ]
            .tolist()
        )

        weights = (
            top_neighbors[
                "weight"
            ]
            .tolist()
        )

        weighted_value = weighted_mean(
            values,
            weights
        )

        # ----------------------------------------------------
        # Strong neighbors
        #
        # >= 50% identity
        # >= 80% coverage
        # ----------------------------------------------------

        strong_count = int(
            (
                (
                    neighbors_df[
                        "identity"
                    ]
                    >=
                    50.0
                )
                &
                (
                    neighbors_df[
                        "coverage"
                    ]
                    >=
                    80.0
                )
            ).sum()
        )

        # ----------------------------------------------------
        # Build feature row
        # ----------------------------------------------------

        rows.append({

            "pdb_id":
                query_id,

            "blast_condition_neighbor_count":
                len(neighbor_records),

            "blast_condition_strong_neighbor_count":
                strong_count,

            "blast_condition_mean_identity_top5":
                float(
                    top_neighbors
                    .head(5)[
                        "identity"
                    ]
                    .mean()
                ),

            "blast_condition_mean_coverage_top5":
                float(
                    top_neighbors
                    .head(5)[
                        "coverage"
                    ]
                    .mean()
                ),

            "blast_condition_weighted_value":
                weighted_value,

            "blast_condition_top_neighbor_identity":
                float(
                    top_neighbors.iloc[0][
                        "identity"
                    ]
                ),

            "blast_condition_top_neighbor_coverage":
                float(
                    top_neighbors.iloc[0][
                        "coverage"
                    ]
                ),

            "blast_condition_top_neighbor_bitscore":
                float(
                    top_neighbors.iloc[0][
                        "bitscore"
                    ]
                ),
        })

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        OUTPUT_DIR
        /
        f"{target_name}.csv"
    )

    output_df = pd.DataFrame(
        rows
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    available = (
        output_df[
            "blast_condition_weighted_value"
        ]
        .notna()
    )

    print()
    print("RESULT")
    print("-" * 70)

    print(
        f"Proteins: "
        f"{len(output_df):,}"
    )

    print(
        f"With condition evidence: "
        f"{available.sum():,}"
    )

    print(
        f"Without condition evidence: "
        f"{(~available).sum():,}"
    )

    print(
        f"Coverage: "
        f"{available.mean() * 100:.2f}%"
    )

    print(
        f"Saved: "
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "OPERATION X — "
        "BLAST CONDITION FEATURES"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    records = load_records()

    # --------------------------------------------------------
    # Load BLAST
    # --------------------------------------------------------

    blast_df = load_blast()

    # --------------------------------------------------------
    # Build each target
    # --------------------------------------------------------

    for target_name, config in TARGETS.items():

        build_target_features(
            blast_df=blast_df,
            records=records,
            target_name=target_name,
            config=config,
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BLAST CONDITION FEATURES COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()