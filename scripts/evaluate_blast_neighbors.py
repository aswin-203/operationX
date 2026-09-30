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

REPORT_DIR = Path(
    "reports"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "pH": {
        "test": TARGET_DIR / "pH_test.csv",
    },

    "temperature_kelvin": {
        "test": TARGET_DIR / "temperature_kelvin_test.csv",
    },

    "matthews_coefficient": {
        "test": TARGET_DIR / "matthews_coefficient_test.csv",
    },

    "solvent_percent": {
        "test": TARGET_DIR / "solvent_percent_test.csv",
    },
}


# ============================================================
# NORMALIZE PDB ID
# ============================================================

def normalize_pdb_id(value):
    """
    Convert different BLAST/PDB identifier formats
    into the standard PDB ID used by Operation X.

    Examples:

        101M
            -> 101m

        101M_1
            -> 101m

        pdb|101M|1
            -> 101m

        PDB|1ABS|1
            -> 1abs
    """

    value = str(value).strip()

    if not value:
        return ""

    # --------------------------------------------------------
    # Format:
    #
    # pdb|101M|1
    # --------------------------------------------------------

    if value.lower().startswith("pdb|"):

        parts = value.split("|")

        if len(parts) >= 2:

            return (
                parts[1]
                .strip()
                .lower()
            )

    # --------------------------------------------------------
    # Format:
    #
    # 101M_1
    #
    # The first part is the PDB ID.
    # --------------------------------------------------------

    if "_" in value:

        return (
            value.split("_")[0]
            .strip()
            .lower()
        )

    # --------------------------------------------------------
    # Already a normal PDB ID.
    # --------------------------------------------------------

    return value.lower()


# ============================================================
# LOAD RECORDS
# ============================================================

def load_records():

    with RECORDS_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    return records


# ============================================================
# NORMALIZE RECORD KEYS
# ============================================================

def normalize_records(records):

    normalized = {}

    for key, value in records.items():

        normalized_key = normalize_pdb_id(
            key
        )

        normalized[normalized_key] = value

    return normalized


# ============================================================
# LOAD BLAST
# ============================================================

def load_blast():

    columns = [
        "query",
        "subject",
        "pident",
        "length",
        "qlen",
        "slen",
        "evalue",
        "bitscore",
    ]

    blast = pd.read_csv(
        BLAST_FILE,
        sep="\t",
        names=columns,
    )

    # --------------------------------------------------------
    # Normalize BLAST IDs.
    # --------------------------------------------------------

    blast["query"] = (
        blast["query"]
        .apply(normalize_pdb_id)
    )

    blast["subject"] = (
        blast["subject"]
        .apply(normalize_pdb_id)
    )

    # --------------------------------------------------------
    # Convert numerical columns.
    # --------------------------------------------------------

    numerical_columns = [
        "pident",
        "length",
        "qlen",
        "slen",
        "evalue",
        "bitscore",
    ]

    for column in numerical_columns:

        blast[column] = pd.to_numeric(
            blast[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Calculate query coverage.
    # --------------------------------------------------------

    blast["coverage"] = (
        blast["length"] /
        blast["qlen"]
    )

    # --------------------------------------------------------
    # Remove invalid rows.
    # --------------------------------------------------------

    blast = blast.dropna(
        subset=[
            "query",
            "subject",
            "pident",
            "coverage",
        ]
    )

    return blast


# ============================================================
# BUILD TRAINING TARGET TABLE
# ============================================================

def build_training_targets(
    train_ids,
    records,
):

    rows = []

    # --------------------------------------------------------
    # Normalize record keys.
    # --------------------------------------------------------

    normalized_records = (
        normalize_records(records)
    )

    # --------------------------------------------------------
    # Build one row per training PDB.
    # --------------------------------------------------------

    for pdb_id in train_ids:

        normalized_id = normalize_pdb_id(
            pdb_id
        )

        record = normalized_records.get(
            normalized_id
        )

        if record is None:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # These values are stored directly in the record.
        # There is no "crystallization" dictionary.
        # ----------------------------------------------------

        rows.append(
            {
                "pdb_id": normalized_id,

                "pH": record.get(
                    "pH"
                ),

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
# FIND BEST TRAINING HOMOLOG
# ============================================================

def find_best_neighbor(
    query_id,
    blast,
    train_ids,
):

    query_id = normalize_pdb_id(
        query_id
    )

    # --------------------------------------------------------
    # Get BLAST hits for this query.
    # --------------------------------------------------------

    hits = blast[
        blast["query"] == query_id
    ].copy()

    if hits.empty:

        return None

    # --------------------------------------------------------
    # Only keep training-set proteins.
    # --------------------------------------------------------

    hits = hits[
        hits["subject"].isin(
            train_ids
        )
    ]

    if hits.empty:

        return None

    # --------------------------------------------------------
    # Never use the query itself.
    # --------------------------------------------------------

    hits = hits[
        hits["subject"] != query_id
    ]

    if hits.empty:

        return None

    # --------------------------------------------------------
    # Require at least 80% query coverage.
    # --------------------------------------------------------

    hits = hits[
        hits["coverage"] >= 0.80
    ]

    if hits.empty:

        return None

    # --------------------------------------------------------
    # Rank by:
    #
    # 1. Sequence identity
    # 2. Query coverage
    # 3. Bitscore
    # --------------------------------------------------------

    hits = hits.sort_values(
        [
            "pident",
            "coverage",
            "bitscore",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return hits.iloc[0]


# ============================================================
# EVALUATE ONE TARGET
# ============================================================

def evaluate_target(
    target,
    test_df,
    training_targets,
    blast,
    train_ids,
):

    predictions = []
    actuals = []

    identities = []
    coverages = []

    neighbor_records = []

    # --------------------------------------------------------
    # Process each test protein.
    # --------------------------------------------------------

    for _, row in test_df.iterrows():

        query_id = normalize_pdb_id(
            row["pdb_id"]
        )

        actual = pd.to_numeric(
            row[target],
            errors="coerce",
        )

        if pd.isna(actual):
            continue

        # ----------------------------------------------------
        # Find best training homolog.
        # ----------------------------------------------------

        neighbor = find_best_neighbor(
            query_id,
            blast,
            train_ids,
        )

        if neighbor is None:
            continue

        subject_id = normalize_pdb_id(
            neighbor["subject"]
        )

        # ----------------------------------------------------
        # Find the experimental target value
        # of the training homolog.
        # ----------------------------------------------------

        neighbor_rows = (
            training_targets[
                training_targets["pdb_id"]
                == subject_id
            ]
        )

        if neighbor_rows.empty:
            continue

        prediction = pd.to_numeric(
            neighbor_rows.iloc[0][target],
            errors="coerce",
        )

        if pd.isna(prediction):
            continue

        # ----------------------------------------------------
        # Store prediction.
        # ----------------------------------------------------

        predictions.append(
            float(prediction)
        )

        actuals.append(
            float(actual)
        )

        identities.append(
            float(
                neighbor["pident"]
            )
        )

        coverages.append(
            float(
                neighbor["coverage"]
            )
        )

        neighbor_records.append(
            {
                "query": query_id,

                "neighbor": subject_id,

                "identity": float(
                    neighbor["pident"]
                ),

                "coverage": float(
                    neighbor["coverage"]
                ),

                "actual": float(
                    actual
                ),

                "prediction": float(
                    prediction
                ),
            }
        )

    # --------------------------------------------------------
    # No usable predictions.
    # --------------------------------------------------------

    if not predictions:

        return None

    actuals = np.array(
        actuals,
        dtype=float,
    )

    predictions = np.array(
        predictions,
        dtype=float,
    )

    # --------------------------------------------------------
    # Calculate errors.
    # --------------------------------------------------------

    errors = (
        actuals -
        predictions
    )

    # MAE.
    mae = np.mean(
        np.abs(errors)
    )

    # RMSE.
    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    # --------------------------------------------------------
    # R².
    # --------------------------------------------------------

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            actuals -
            np.mean(actuals)
        ) ** 2
    )

    if ss_tot == 0:

        r2 = float("nan")

    else:

        r2 = (
            1 -
            (
                ss_res /
                ss_tot
            )
        )

    return {
        "target": target,

        "evaluated_records": len(
            predictions
        ),

        "coverage_of_test_set": (
            len(predictions) /
            len(test_df)
        ),

        "MAE": float(mae),

        "RMSE": float(rmse),

        "R2": float(r2),

        "mean_identity": float(
            np.mean(identities)
        ),

        "median_identity": float(
            np.median(identities)
        ),

        "mean_coverage": float(
            np.mean(coverages)
        ),

        "neighbors": neighbor_records,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "OPERATION X — BLAST NEAREST-NEIGHBOR BASELINE"
    )
    print("=" * 70)

    # ========================================================
    # LOAD RECORDS
    # ========================================================

    print()
    print("Loading records...")

    records = load_records()

    print(
        f"Records loaded: {len(records)}"
    )

    records = normalize_records(
        records
    )

    print(
        f"Normalized records: {len(records)}"
    )

    # ========================================================
    # LOAD BLAST
    # ========================================================

    print()
    print("Loading BLAST results...")

    blast = load_blast()

    print(
        f"BLAST hits: {len(blast)}"
    )

    # ========================================================
    # LOAD TRAIN / TEST SPLIT
    # ========================================================

    train_file = (
        TARGET_DIR /
        "pH_train.csv"
    )

    test_file = (
        TARGET_DIR /
        "pH_test.csv"
    )

    train_df = pd.read_csv(
        train_file
    )

    test_df = pd.read_csv(
        test_file
    )

    # Normalize IDs.
    train_ids = set(
        train_df["pdb_id"]
        .apply(normalize_pdb_id)
    )

    test_ids = set(
        test_df["pdb_id"]
        .apply(normalize_pdb_id)
    )

    print()
    print("DATA SPLIT")
    print("-" * 70)

    print(
        f"Training PDB IDs : {len(train_ids)}"
    )

    print(
        f"Test PDB IDs     : {len(test_ids)}"
    )

    overlap = (
        train_ids &
        test_ids
    )

    print(
        f"Overlap           : {len(overlap)}"
    )

    if overlap:

        raise ValueError(
            "Training/test PDB overlap detected."
        )

    # ========================================================
    # CHECK BLAST ID MATCHING
    # ========================================================

    test_blast_queries = (
        set(
            blast["query"]
        )
        & test_ids
    )

    train_blast_subjects = (
        set(
            blast["subject"]
        )
        & train_ids
    )

    print()
    print("BLAST ID MATCHING")
    print("-" * 70)

    print(
        f"Test IDs found as BLAST queries : "
        f"{len(test_blast_queries)}"
    )

    print(
        f"Train IDs found as BLAST subjects: "
        f"{len(train_blast_subjects)}"
    )

    # ========================================================
    # BUILD TRAINING TARGET TABLE
    # ========================================================

    print()
    print(
        "Building training target table..."
    )

    training_targets = (
        build_training_targets(
            train_ids,
            records,
        )
    )

    print(
        f"Training target records: "
        f"{len(training_targets)}"
    )

    if training_targets.empty:

        raise RuntimeError(
            "No training target records were found."
        )

    # ========================================================
    # TARGET AVAILABILITY
    # ========================================================

    print()
    print(
        "Training target availability:"
    )

    for target in [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]:

        count = (
            pd.to_numeric(
                training_targets[target],
                errors="coerce",
            )
            .notna()
            .sum()
        )

        print(
            f"  {target:25s}: {count}"
        )

    # ========================================================
    # EVALUATE ALL TARGETS
    # ========================================================

    all_results = []

    for target, paths in TARGETS.items():

        print()
        print("=" * 70)
        print(
            f"TARGET: {target}"
        )
        print("=" * 70)

        target_test_df = pd.read_csv(
            paths["test"]
        )

        result = evaluate_target(
            target,
            target_test_df,
            training_targets,
            blast,
            train_ids,
        )

        if result is None:

            print()
            print(
                "No usable BLAST neighbors found."
            )

            continue

        all_results.append(
            result
        )

        print()
        print(
            f"Evaluated records : "
            f"{result['evaluated_records']}"
        )

        print(
            f"Test-set coverage : "
            f"{result['coverage_of_test_set']:.2%}"
        )

        print(
            f"Mean identity     : "
            f"{result['mean_identity']:.2f}%"
        )

        print(
            f"Median identity   : "
            f"{result['median_identity']:.2f}%"
        )

        print(
            f"Mean coverage     : "
            f"{result['mean_coverage']:.2%}"
        )

        print()

        print(
            f"MAE  : "
            f"{result['MAE']:.4f}"
        )

        print(
            f"RMSE : "
            f"{result['RMSE']:.4f}"
        )

        print(
            f"R2   : "
            f"{result['R2']:.4f}"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIR /
        "blast_neighbor_baseline.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLAST NEAREST-NEIGHBOR ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print()

    if all_results:

        print(
            f"{'TARGET':25s} "
            f"| {'MAE':10s} "
            f"| {'RMSE':10s} "
            f"| {'R2':10s} "
            f"| {'COVERAGE':10s}"
        )

        print("-" * 75)

        for result in all_results:

            print(
                f"{result['target']:25s} "
                f"| {result['MAE']:<10.4f} "
                f"| {result['RMSE']:<10.4f} "
                f"| {result['R2']:<10.4f} "
                f"| {result['coverage_of_test_set']:<10.2%}"
            )

    else:

        print(
            "No target produced a usable result."
        )

    print()

    print(
        f"Report: {report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()