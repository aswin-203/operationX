import csv
import json
from pathlib import Path


RECORDS_FILE = Path(
    "data/structures/large/records.json"
)

GROUPS_FILE = Path(
    "data/similarity/large/sequence_groups.csv"
)


def normalize_id(pdb_id, entity_id):
    return f"{pdb_id.upper()}_{str(entity_id).strip()}"


def main():
    with RECORDS_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        records = json.load(handle)

    with GROUPS_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        grouped_ids = {
            row["sequence_id"]
            for row in csv.DictReader(handle)
        }

    missing = []

    for pdb_id, record in records.items():
        entity_id = record.get("entity_id")

        sequence_id = normalize_id(
            pdb_id,
            entity_id,
        )

        if sequence_id not in grouped_ids:
            sequence = record.get(
                "sequence",
                "",
            )

            missing.append(
                {
                    "pdb_id": pdb_id,
                    "entity_id": entity_id,
                    "length": len(sequence),
                    "sequence": sequence[:50],
                }
            )

    print()
    print("=" * 60)
    print("OPERATION X — UNGROUPED SEQUENCES")
    print("=" * 60)
    print()

    print(f"Total records : {len(records)}")
    print(
        f"Grouped       : {len(grouped_ids)}"
    )
    print(
        f"Missing       : {len(missing)}"
    )

    print()

    for item in missing:
        print(
            f"{item['pdb_id']}_{item['entity_id']} "
            f"| length={item['length']} "
            f"| {item['sequence']}"
        )


if __name__ == "__main__":
    main()