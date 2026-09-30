import csv
import json
from collections import Counter
from pathlib import Path


ML_RECORDS_FILE = Path(
    "data/ml_large/records.json"
)

GROUPS_FILE = Path(
    "data/similarity/large/sequence_groups.csv"
)

OUTPUT_FILE = Path(
    "data/ml_large/records_with_groups.json"
)

REPORT_FILE = Path(
    "data/ml_large/group_assignment_report.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def normalize_sequence_id(
    pdb_id,
    entity_id,
):
    return (
        f"{str(pdb_id).strip().upper()}_"
        f"{str(entity_id).strip()}"
    )


def load_group_mapping():
    if not GROUPS_FILE.exists():
        raise FileNotFoundError(
            f"Group file not found: {GROUPS_FILE}"
        )

    mapping = {}

    with GROUPS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        required_columns = {
            "sequence_id",
            "group_id",
        }

        missing = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                "Missing columns in group file: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            sequence_id = (
                row["sequence_id"]
                .strip()
            )

            group_id = (
                row["group_id"]
                .strip()
            )

            if not sequence_id:
                continue

            if not group_id:
                continue

            mapping[sequence_id] = group_id

    return mapping


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(path)


def main():

    print()
    print("=" * 60)
    print(
        "OPERATION X — ATTACH SEQUENCE GROUPS"
    )
    print("=" * 60)
    print()

    records = load_json(
        ML_RECORDS_FILE
    )

    group_mapping = (
        load_group_mapping()
    )

    print(
        f"ML records              : "
        f"{len(records)}"
    )

    print(
        f"Available group mappings: "
        f"{len(group_mapping)}"
    )

    print()

    output_records = {}

    assigned = 0
    isolated = 0
    missing_group = []

    isolated_group_number = 1

    for pdb_id, record in records.items():

        entity_id = record.get(
            "entity_id"
        )

        sequence_id = (
            normalize_sequence_id(
                pdb_id,
                entity_id,
            )
        )

        group_id = group_mapping.get(
            sequence_id
        )

        cleaned_record = dict(record)

        if group_id is not None:

            cleaned_record[
                "sequence_group"
            ] = group_id

            assigned += 1

        else:

            # A sequence with no qualifying
            # BLAST relationship gets its
            # own isolated group.
            group_id = (
                f"ISOLATED_{isolated_group_number:04d}"
            )

            cleaned_record[
                "sequence_group"
            ] = group_id

            isolated += 1
            isolated_group_number += 1

            missing_group.append(
                {
                    "pdb_id": pdb_id,
                    "entity_id": entity_id,
                    "sequence_id": sequence_id,
                    "group_id": group_id,
                }
            )

        output_records[pdb_id] = (
            cleaned_record
        )

    group_counts = Counter(
        record["sequence_group"]
        for record in output_records.values()
    )

    report = {
        "input_records": len(records),
        "group_mappings": len(
            group_mapping
        ),
        "assigned_records": assigned,
        "isolated_records": isolated,
        "total_groups": len(
            group_counts
        ),
        "largest_group_size": (
            max(group_counts.values())
            if group_counts
            else 0
        ),
        "isolated_records_detail": (
            missing_group
        ),
    }

    save_json(
        OUTPUT_FILE,
        output_records,
    )

    save_json(
        REPORT_FILE,
        report,
    )

    print("GROUP ASSIGNMENT")
    print("-" * 60)

    print(
        f"Assigned to BLAST group : "
        f"{assigned}"
    )

    print(
        f"Isolated records        : "
        f"{isolated}"
    )

    print(
        f"Total records           : "
        f"{len(output_records)}"
    )

    print(
        f"Total groups            : "
        f"{len(group_counts)}"
    )

    print(
        f"Largest group           : "
        f"{report['largest_group_size']}"
    )

    print()

    if group_counts:

        print("LARGEST GROUPS")
        print("-" * 60)

        largest_groups = sorted(
            group_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        for group_id, count in (
            largest_groups[:10]
        ):
            print(
                f"{group_id}: "
                f"{count} records"
            )

        print()

    print(
        f"Output                  : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Report                   : "
        f"{REPORT_FILE}"
    )

    print()
    print("=" * 60)
    print(
        "SEQUENCE GROUP ASSIGNMENT COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()