import json
from collections import Counter
from pathlib import Path


RECORDS_FILE = Path("data/structures/large/records.json")


def load_records():
    with RECORDS_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    records = load_records()

    sequences = [
        (pdb_id, record.get("sequence", "").upper())
        for pdb_id, record in records.items()
    ]

    sequence_values = [sequence for _, sequence in sequences]

    sequence_counter = Counter(sequence_values)

    print("=" * 60)
    print("OPERATION X — DATASET QUALITY AUDIT")
    print("=" * 60)

    print()
    print("DATASET")
    print("-" * 60)

    print(f"Total records             : {len(records)}")
    print(
        f"CIF files                 : "
        f"{len(list(RECORDS_FILE.parent.glob('*.cif')))}"
    )

    print()
    print("CRYSTALLIZATION DATA")
    print("-" * 60)

    print(
        f"Crystallization available : "
        f"{sum(bool(r.get('crystallization_available')) for r in records.values())}"
    )

    print(
        f"pH                        : "
        f"{sum(r.get('pH') is not None for r in records.values())}"
    )

    print(
        f"Temperature               : "
        f"{sum(r.get('temperature_kelvin') is not None for r in records.values())}"
    )

    print(
        f"Matthews coefficient     : "
        f"{sum(r.get('matthews_coefficient') is not None for r in records.values())}"
    )

    print(
        f"Solvent percentage       : "
        f"{sum(r.get('solvent_percent') is not None for r in records.values())}"
    )

    print()
    print("EXPRESSION DATA")
    print("-" * 60)

    print(
        f"Expression host           : "
        f"{sum(bool(r.get('expression_host')) for r in records.values())}"
    )

    print(
        f"Expression system        : "
        f"{sum(bool(r.get('expression_system')) for r in records.values())}"
    )

    print()
    print("SEQUENCE QUALITY")
    print("-" * 60)

    print(
        f"Empty sequences           : "
        f"{sum(not sequence for sequence in sequence_values)}"
    )

    print(
        f"Length < 20 aa            : "
        f"{sum(len(sequence) < 20 for sequence in sequence_values)}"
    )

    print(
        f"Length 20–49 aa           : "
        f"{sum(20 <= len(sequence) < 50 for sequence in sequence_values)}"
    )

    print(
        f"Length >= 50 aa           : "
        f"{sum(len(sequence) >= 50 for sequence in sequence_values)}"
    )

    print(
        f"Sequences containing X    : "
        f"{sum('X' in sequence for sequence in sequence_values)}"
    )

    print()
    print("DUPLICATION")
    print("-" * 60)

    unique_sequences = len(sequence_counter)

    duplicate_groups = sum(
        count > 1
        for count in sequence_counter.values()
    )

    duplicate_records = sum(
        count - 1
        for count in sequence_counter.values()
        if count > 1
    )

    print(f"Unique sequences           : {unique_sequences}")
    print(f"Duplicate sequence groups  : {duplicate_groups}")
    print(f"Duplicate records          : {duplicate_records}")

    print()
    print("TOP DUPLICATE SEQUENCES")
    print("-" * 60)

    duplicate_entries = [
        (sequence, count)
        for sequence, count in sequence_counter.items()
        if count > 1
    ]

    duplicate_entries.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    for index, (sequence, count) in enumerate(
        duplicate_entries[:10],
        start=1,
    ):
        print(
            f"{index:2}. "
            f"{count:3} records | "
            f"{len(sequence):5} aa | "
            f"{sequence[:30]}..."
        )

    print()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()