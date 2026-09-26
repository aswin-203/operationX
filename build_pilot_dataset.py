import csv
from pathlib import Path

from operation_x.dataset.pilot_builder import (
    PilotDatasetBuilder,
)


PDB_IDS = [
    "101M",
    "102M",
    "103M",
    "104M",
    "105M",
    "106M",
    "107M",
    "108M",
    "109M",
    "10AF",
    "10AH",
    "10AI",
    "10AJ",
    "10AK",
]


OUTPUT = Path(
    "data/processed/operation_x_pilot.csv"
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


builder = PilotDatasetBuilder()

records = builder.build(
    PDB_IDS
)


if records:

    fieldnames = list(
        records[0].keys()
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(records)


print()
print("=" * 60)
print("OPERATION X PILOT DATASET")
print("=" * 60)

print(
    f"Records written: {len(records)}"
)

print(
    f"Output: {OUTPUT}"
)
