import csv
import json
from pathlib import Path


RECORDS_FILE = Path(
    "data/structures/large/records.json"
)

OUTPUT_FILE = Path(
    "data/blast/large/pdb_sequences.csv"
)


def load_records():
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Records file not found: {RECORDS_FILE}"
        )

    with RECORDS_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "records.json must contain a JSON object"
        )

    return data


def prepare_dataset(records):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    skipped = 0

    for pdb_id, record in records.items():

        entity_id = record.get("entity_id")
        sequence = record.get("sequence")

        if not pdb_id:
            skipped += 1
            continue

        if not entity_id:
            skipped += 1
            continue

        if not sequence:
            skipped += 1
            continue

        sequence = "".join(
            sequence.split()
        ).upper()

        if not sequence:
            skipped += 1
            continue

        rows.append(
            {
                "pdb_id": str(pdb_id).strip().upper(),
                "entity_id": str(entity_id).strip(),
                "sequence": sequence,
            }
        )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pdb_id",
                "entity_id",
                "sequence",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return len(rows), skipped


def main():
    print()
    print("=" * 60)
    print("OPERATION X — PREPARE BLAST DATASET")
    print("=" * 60)
    print()

    records = load_records()

    print(f"Input records : {len(records)}")

    written, skipped = prepare_dataset(records)

    print(f"Written       : {written}")
    print(f"Skipped       : {skipped}")
    print(f"Output        : {OUTPUT_FILE}")

    print()
    print("=" * 60)
    print("BLAST DATASET PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()