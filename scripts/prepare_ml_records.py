import json
from collections import Counter
from pathlib import Path


INPUT_FILE = Path(
    "data/structures/large/records.json"
)

OUTPUT_FILE = Path(
    "data/ml_large/records.json"
)

REPORT_FILE = Path(
    "data/ml_large/quality_report.json"
)

MIN_SEQUENCE_LENGTH = 20

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWYBXZJUO"
)


def load_records():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        records = json.load(handle)

    if not isinstance(records, dict):
        raise ValueError(
            "records.json must contain a JSON object"
        )

    return records


def normalize_sequence(sequence):
    if sequence is None:
        return ""

    return "".join(
        str(sequence).split()
    ).upper()


def is_valid_sequence(sequence):
    if not sequence:
        return False, "empty"

    if len(sequence) < MIN_SEQUENCE_LENGTH:
        return False, "too_short"

    invalid_characters = (
        set(sequence) - VALID_AMINO_ACIDS
    )

    if invalid_characters:
        return (
            False,
            "invalid_characters",
        )

    return True, None


def prepare_records(records):
    accepted = {}
    excluded = []

    exclusion_reasons = Counter()

    for pdb_id, record in records.items():

        sequence = normalize_sequence(
            record.get("sequence")
        )

        valid, reason = is_valid_sequence(
            sequence
        )

        if not valid:

            exclusion_reasons[reason] += 1

            excluded.append(
                {
                    "pdb_id": pdb_id,
                    "reason": reason,
                    "sequence_length": len(
                        sequence
                    ),
                    "sequence": sequence,
                }
            )

            continue

        # Preserve all original metadata.
        cleaned_record = dict(record)

        # Store the normalized sequence.
        cleaned_record["sequence"] = sequence

        accepted[pdb_id] = cleaned_record

    return (
        accepted,
        excluded,
        exclusion_reasons,
    )


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
        "OPERATION X — PREPARE ML DATASET"
    )
    print("=" * 60)
    print()

    print(
        f"Input file              : "
        f"{INPUT_FILE}"
    )

    print(
        f"Minimum sequence length : "
        f"{MIN_SEQUENCE_LENGTH}"
    )

    print()

    records = load_records()

    (
        accepted,
        excluded,
        exclusion_reasons,
    ) = prepare_records(records)

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "minimum_sequence_length": (
            MIN_SEQUENCE_LENGTH
        ),
        "total_records": len(records),
        "accepted_records": len(accepted),
        "excluded_records": len(excluded),
        "exclusion_reasons": dict(
            exclusion_reasons
        ),
        "excluded_records_detail": excluded,
    }

    save_json(
        OUTPUT_FILE,
        accepted,
    )

    save_json(
        REPORT_FILE,
        report,
    )

    print("DATASET SUMMARY")
    print("-" * 60)

    print(
        f"Input records           : "
        f"{len(records)}"
    )

    print(
        f"Accepted records        : "
        f"{len(accepted)}"
    )

    print(
        f"Excluded records        : "
        f"{len(excluded)}"
    )

    print()

    print("EXCLUSION REASONS")
    print("-" * 60)

    if exclusion_reasons:

        for reason, count in sorted(
            exclusion_reasons.items()
        ):
            print(
                f"{reason:24} : {count}"
            )

    else:
        print("None")

    print()

    print(
        f"ML dataset              : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Quality report          : "
        f"{REPORT_FILE}"
    )

    print()
    print("=" * 60)
    print(
        "ML DATASET PREPARATION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()