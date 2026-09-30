import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path("data/ml_large")

FEATURE_FILE = BASE / "combined_features.csv"
RECORD_FILE = BASE / "records_with_groups.json"
SPLIT_DIR = BASE / "splits"
OUTPUT_DIR = BASE / "final"

BLAST_CONDITION_DIR = (
    BASE / "blast_condition_features"
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
# LOAD SPLIT
# ============================================================

def load_split(name):

    path = SPLIT_DIR / f"{name}.json"

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # Support either list-style
    # or dictionary-style split files.

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        records = list(
            data.values()
        )

    else:

        raise ValueError(
            f"Unsupported split format: {path}"
        )

    return records


# ============================================================
# GET PDB ID
# ============================================================

def get_pdb_id(record):

    if isinstance(record, dict):

        return str(
            record.get(
                "pdb_id",
                ""
            )
        ).strip().upper()

    return str(
        record
    ).strip().upper()


# ============================================================
# LOAD BLAST CONDITION FEATURES
# ============================================================

def load_blast_condition_features(
    target
):

    path = (
        BLAST_CONDITION_DIR
        /
        f"{target}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "BLAST condition feature file "
            f"not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    df["pdb_id"] = (
        df["pdb_id"]
        .astype(str)
        .str.upper()
    )

    # Remove accidental duplicate IDs
    # if any exist.

    df = df.drop_duplicates(
        subset=["pdb_id"],
        keep="first"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print(
        "OPERATION X — "
        "BUILD TARGET DATASETS"
    )
    print("=" * 60)
    print()

    # ========================================================
    # LOAD ORIGINAL FEATURES
    # ========================================================

    features = pd.read_csv(
        FEATURE_FILE
    )

    features["pdb_id"] = (
        features["pdb_id"]
        .astype(str)
        .str.upper()
    )

    print(
        f"Original feature rows: "
        f"{len(features)}"
    )

    print(
        f"Original feature columns: "
        f"{len(features.columns)}"
    )

    # ========================================================
    # LOAD RECORDS
    # ========================================================

    with RECORD_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    # Your records file is dictionary-style.
    # Keep the existing structure.

    if not isinstance(
        records,
        dict
    ):

        raise ValueError(
            "Expected records_with_groups.json "
            "to contain a dictionary."
        )

    target_rows = []

    for pdb_id, record in records.items():

        row = {

            "pdb_id":
                str(pdb_id)
                .strip()
                .upper(),

            "pH":
                record.get("pH"),

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

            "sequence_group":
                record.get(
                    "sequence_group"
                ),
        }

        target_rows.append(
            row
        )

    targets = pd.DataFrame(
        target_rows
    )

    targets["pdb_id"] = (
        targets["pdb_id"]
        .str.upper()
    )

    # ========================================================
    # MERGE ORIGINAL FEATURES + TARGETS
    # ========================================================

    combined = features.merge(
        targets,
        on="pdb_id",
        how="left",
    )

    print(
        f"Target rows: "
        f"{len(targets)}"
    )

    print(
        f"Combined rows: "
        f"{len(combined)}"
    )

    print()

    # ========================================================
    # LOAD EXISTING LEAKAGE-SAFE SPLITS
    # ========================================================

    split_ids = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        split_records = load_split(
            split_name
        )

        ids = {
            get_pdb_id(record)
            for record in split_records
        }

        split_ids[
            split_name
        ] = ids

        print(
            f"{split_name.capitalize():12s}: "
            f"{len(ids)} records"
        )

    print()

    # ========================================================
    # VERIFY SPLIT OVERLAP
    # ========================================================

    train_ids = split_ids[
        "train"
    ]

    validation_ids = split_ids[
        "validation"
    ]

    test_ids = split_ids[
        "test"
    ]

    assert train_ids.isdisjoint(
        validation_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert validation_ids.isdisjoint(
        test_ids
    )

    print(
        "Split overlap check: PASS"
    )

    print()

    # ========================================================
    # BUILD EACH TARGET DATASET
    # ========================================================

    for target in TARGETS:

        print("=" * 60)
        print(
            f"TARGET: {target}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Load BLAST condition features
        # ----------------------------------------------------

        blast_features = (
            load_blast_condition_features(
                target
            )
        )

        print(
            "BLAST condition rows: "
            f"{len(blast_features)}"
        )

        # ----------------------------------------------------
        # Identify BLAST feature columns
        # ----------------------------------------------------

        blast_feature_columns = [
            column
            for column in blast_features.columns
            if column != "pdb_id"
        ]

        print(
            "BLAST condition features: "
            f"{len(blast_feature_columns)}"
        )

        # ----------------------------------------------------
        # Merge BLAST condition features
        # ----------------------------------------------------

        target_combined = combined.merge(
            blast_features,
            on="pdb_id",
            how="left",
        )

        print(
            "Rows after BLAST merge: "
            f"{len(target_combined)}"
        )

        # ----------------------------------------------------
        # Check condition evidence
        # ----------------------------------------------------

        weighted_column = (
            "blast_condition_weighted_value"
        )

        if weighted_column in (
            target_combined.columns
        ):

            evidence_count = (
                target_combined[
                    weighted_column
                ]
                .notna()
                .sum()
            )

            print(
                "Rows with BLAST condition "
                f"evidence: {evidence_count}"
            )

        # ----------------------------------------------------
        # Only records with target available
        # ----------------------------------------------------

        available = target_combined[
            target_combined[target]
            .notna()
        ].copy()

        print(
            "Records with target: "
            f"{len(available)}"
        )

        # ====================================================
        # BUILD TRAIN / VALIDATION / TEST
        # ====================================================

        for split_name, ids in (
            split_ids.items()
        ):

            split_df = available[
                available[
                    "pdb_id"
                ].isin(ids)
            ].copy()

            # ------------------------------------------------
            # Original feature columns
            # ------------------------------------------------

            original_feature_columns = [
                column
                for column in features.columns
                if column != "pdb_id"
            ]

            # ------------------------------------------------
            # Final feature columns
            # ------------------------------------------------

            feature_columns = (
                original_feature_columns
                +
                blast_feature_columns
            )

            # Remove duplicates while
            # preserving order.

            feature_columns = list(
                dict.fromkeys(
                    feature_columns
                )
            )

            # ------------------------------------------------
            # Output columns
            # ------------------------------------------------

            output_columns = [
                "pdb_id"
            ]

            output_columns += (
                feature_columns
            )

            output_columns += [
                "sequence_group",
                target,
            ]

            # Keep only columns that
            # actually exist.

            output_columns = [
                column
                for column in output_columns
                if column in split_df.columns
            ]

            split_df = split_df[
                output_columns
            ]

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_file = (
                OUTPUT_DIR
                /
                f"{target}_{split_name}.csv"
            )

            split_df.to_csv(
                output_file,
                index=False,
            )

            print(
                f"{split_name.capitalize():12s}: "
                f"{len(split_df)} records | "
                f"{len(split_df.columns)} columns "
                f"-> {output_file.name}"
            )

        print()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("=" * 60)
    print(
        "TARGET DATASET BUILD COMPLETE"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()