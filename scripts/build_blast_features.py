import csv
import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/blast/large/all_vs_all.tsv"
)

RECORDS_FILE = Path(
    "data/ml_large/records_with_groups.json"
)

OUTPUT_FILE = Path(
    "data/ml_large/blast_features.csv"
)

STRONG_IDENTITY = 90.0
STRONG_COVERAGE = 80.0


def extract_pdb_id(sequence_id):
    """
    Convert different BLAST identifiers to
    the PDB ID.

    Examples:

        101M_1
        -> 101M

        pdb|101M|1
        -> 101M

        101M
        -> 101M
    """

    value = str(
        sequence_id
    ).strip().upper()

    # BLAST format:
    # pdb|101M|1
    if value.startswith("PDB|"):

        parts = value.split("|")

        if len(parts) >= 2:
            return parts[1]

    # Entity format:
    # 101M_1
    if "_" in value:

        return value.split(
            "_",
            1
        )[0]

    return value


def calculate_coverage(
    alignment_length,
    query_length
):

    if query_length <= 0:
        return 0.0

    return (
        alignment_length
        / query_length
        * 100.0
    )


def main():

    print()
    print("=" * 60)
    print(
        "OPERATION X — BLAST FEATURE EXTRACTION"
    )
    print("=" * 60)
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"BLAST file not found: "
            f"{INPUT_FILE}"
        )

    if not RECORDS_FILE.exists():

        raise FileNotFoundError(
            f"Records file not found: "
            f"{RECORDS_FILE}"
        )

    # --------------------------------------------------------
    # Load ML records
    # --------------------------------------------------------

    with RECORDS_FILE.open(
        "r",
        encoding="utf-8"
    ) as handle:

        records = json.load(handle)

    query_pdb_ids = {
        extract_pdb_id(record_id)
        for record_id in records.keys()
    }

    print(
        f"ML records         : {len(records)}"
    )

    print(
        f"Unique PDB IDs     : "
        f"{len(query_pdb_ids)}"
    )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    best_hits = {}

    strong_hit_counts = {}

    total_hits = 0

    usable_hits = 0

    self_hits_removed = 0

    # --------------------------------------------------------
    # Read BLAST results
    # --------------------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as handle:

        reader = csv.reader(
            handle,
            delimiter="\t"
        )

        for row in reader:

            if len(row) < 8:
                continue

            total_hits += 1

            raw_query = row[0]
            raw_subject = row[1]

            query_pdb = extract_pdb_id(
                raw_query
            )

            subject_pdb = extract_pdb_id(
                raw_subject
            )

            # ------------------------------------------------
            # Only process records in our ML dataset.
            # ------------------------------------------------

            if query_pdb not in query_pdb_ids:
                continue

            # ------------------------------------------------
            # Remove self-PDB hits.
            #
            # Example:
            #
            # 101M_1 -> pdb|101M|1
            #
            # Both belong to PDB 101M.
            # ------------------------------------------------

            if query_pdb == subject_pdb:

                self_hits_removed += 1

                continue

            try:

                identity = float(
                    row[2]
                )

                alignment_length = int(
                    row[3]
                )

                query_length = int(
                    row[4]
                )

                evalue = float(
                    row[6]
                )

                bitscore = float(
                    row[7]
                )

            except ValueError:

                continue

            coverage = calculate_coverage(
                alignment_length,
                query_length
            )

            usable_hits += 1

            hit = {
                "best_identity": identity,
                "best_coverage": coverage,
                "best_evalue": evalue,
                "best_bitscore": bitscore,
            }

            # ------------------------------------------------
            # Keep the strongest hit.
            #
            # Bitscore is the primary criterion.
            # ------------------------------------------------

            previous = best_hits.get(
                query_pdb
            )

            if (
                previous is None
                or bitscore
                > previous["best_bitscore"]
            ):

                best_hits[query_pdb] = hit

            # ------------------------------------------------
            # Strong similarity hit
            # ------------------------------------------------

            if (
                identity >= STRONG_IDENTITY
                and coverage >= STRONG_COVERAGE
            ):

                strong_hit_counts[
                    query_pdb
                ] = (
                    strong_hit_counts.get(
                        query_pdb,
                        0
                    )
                    + 1
                )

    # --------------------------------------------------------
    # Build feature rows
    # --------------------------------------------------------

    rows = []

    for record_id in records.keys():

        pdb_id = extract_pdb_id(
            record_id
        )

        best = best_hits.get(
            pdb_id
        )

        if best is None:

            rows.append(
                {
                    "pdb_id": record_id,
                    "best_identity": 0.0,
                    "best_coverage": 0.0,
                    "best_evalue": None,
                    "best_bitscore": 0.0,
                    "sequence_strong_hit_count":
                        0,
                }
            )

        else:

            rows.append(
                {
                    "pdb_id": record_id,

                    "best_identity":
                        best[
                            "best_identity"
                        ],

                    "best_coverage":
                        best[
                            "best_coverage"
                        ],

                    "best_evalue":
                        best[
                            "best_evalue"
                        ],

                    "best_bitscore":
                        best[
                            "best_bitscore"
                        ],

                    "sequence_strong_hit_count":
                        strong_hit_counts.get(
                            pdb_id,
                            0
                        ),
                }
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows,
        columns=[
            "pdb_id",
            "best_identity",
            "best_coverage",
            "best_evalue",
            "best_bitscore",
            "sequence_strong_hit_count",
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    records_with_hits = sum(
        1
        for pdb_id in query_pdb_ids
        if pdb_id in best_hits
    )

    records_without_hits = (
        len(query_pdb_ids)
        - records_with_hits
    )

    print()
    print(
        "BLAST SUMMARY"
    )
    print("-" * 60)

    print(
        f"Total BLAST hits      : "
        f"{total_hits}"
    )

    print(
        f"Self-PDB hits removed : "
        f"{self_hits_removed}"
    )

    print(
        f"Usable hits           : "
        f"{usable_hits}"
    )

    print(
        f"Records with hits     : "
        f"{records_with_hits}"
    )

    print(
        f"Records without hits  : "
        f"{records_without_hits}"
    )

    print(
        f"Output rows           : "
        f"{len(df)}"
    )

    print(
        f"Output file           : "
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 60)
    print(
        "BLAST FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()