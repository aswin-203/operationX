import json
import random
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path(
    "data/ml_large/records_with_groups.json"
)

OUTPUT_DIRECTORY = Path(
    "data/ml_large/splits"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def load_records():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True
        )


def create_groups(records):

    groups = defaultdict(list)

    for record_id, record in records.items():

        group_id = record["sequence_group"]

        groups[group_id].append(
            record_id
        )

    return groups


def create_balanced_split(groups):

    rng = random.Random(
        RANDOM_SEED
    )

    group_items = list(
        groups.items()
    )

    # Random tie-breaking while keeping
    # large groups together.
    rng.shuffle(group_items)

    # Largest groups first.
    group_items.sort(
        key=lambda item: len(item[1]),
        reverse=True
    )

    total_records = sum(
        len(records)
        for _, records in group_items
    )

    targets = {
        "train": total_records * TRAIN_RATIO,
        "validation": total_records * VALIDATION_RATIO,
        "test": total_records * TEST_RATIO
    }

    split_groups = {
        "train": [],
        "validation": [],
        "test": []
    }

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    for group_id, record_ids in group_items:

        group_size = len(record_ids)

        # Calculate how much each split
        # is currently below its target.
        deficits = {}

        for split in targets:

            deficits[split] = (
                targets[split]
                - split_counts[split]
            )

        # Prefer the split with the
        # largest relative deficit.
        selected_split = max(
            deficits,
            key=lambda split:
            deficits[split] / targets[split]
        )

        split_groups[selected_split].append(
            group_id
        )

        split_counts[selected_split] += (
            group_size
        )

    return (
        split_groups,
        split_counts,
        total_records
    )


def build_records(
    records,
    groups,
    selected_groups
):

    result = {}

    for group_id in selected_groups:

        for record_id in groups[group_id]:

            result[record_id] = records[
                record_id
            ]

    return result


def validate(
    records,
    split_records,
    split_groups
):

    train_ids = set(
        split_records["train"].keys()
    )

    validation_ids = set(
        split_records["validation"].keys()
    )

    test_ids = set(
        split_records["test"].keys()
    )

    # Record overlap
    assert not (
        train_ids & validation_ids
    )

    assert not (
        train_ids & test_ids
    )

    assert not (
        validation_ids & test_ids
    )

    # Record coverage
    all_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    assert all_ids == set(
        records.keys()
    )

    # Group overlap
    train_groups = set(
        split_groups["train"]
    )

    validation_groups = set(
        split_groups["validation"]
    )

    test_groups = set(
        split_groups["test"]
    )

    assert not (
        train_groups
        & validation_groups
    )

    assert not (
        train_groups
        & test_groups
    )

    assert not (
        validation_groups
        & test_groups
    )


def main():

    print()
    print("=" * 60)
    print("OPERATION X — FAST GROUP-BALANCED SPLIT")
    print("=" * 60)
    print()

    records = load_records()

    print(
        f"Input records     : {len(records)}"
    )

    groups = create_groups(
        records
    )

    print(
        f"Sequence groups   : {len(groups)}"
    )

    (
        split_groups,
        split_counts,
        total_records
    ) = create_balanced_split(
        groups
    )

    split_records = {}

    for split in (
        "train",
        "validation",
        "test"
    ):

        split_records[split] = build_records(
            records,
            groups,
            split_groups[split]
        )

    validate(
        records,
        split_records,
        split_groups
    )

    # Save datasets
    save_json(
        OUTPUT_DIRECTORY / "train.json",
        split_records["train"]
    )

    save_json(
        OUTPUT_DIRECTORY / "validation.json",
        split_records["validation"]
    )

    save_json(
        OUTPUT_DIRECTORY / "test.json",
        split_records["test"]
    )

    # Save group assignments
    assignments = {}

    for split in (
        "train",
        "validation",
        "test"
    ):

        for group_id in split_groups[split]:

            assignments[group_id] = split

    save_json(
        OUTPUT_DIRECTORY
        / "split_assignments.json",
        assignments
    )

    # Summary
    summary = {}

    for split in (
        "train",
        "validation",
        "test"
    ):

        count = len(
            split_records[split]
        )

        summary[split] = {
            "groups": len(
                split_groups[split]
            ),
            "records": count,
            "percentage": round(
                count / total_records * 100,
                2
            )
        }

    save_json(
        OUTPUT_DIRECTORY
        / "split_summary.json",
        summary
    )

    print()
    print("SPLIT RESULT")
    print("-" * 60)

    for split in (
        "train",
        "validation",
        "test"
    ):

        print(
            f"{split.capitalize():11}: "
            f"{summary[split]['groups']} groups / "
            f"{summary[split]['records']} records "
            f"({summary[split]['percentage']}%)"
        )

    print()
    print("VALIDATION")
    print("-" * 60)

    print(
        "Record overlap      : 0"
    )

    print(
        "Group overlap       : 0"
    )

    print(
        f"Total records       : "
        f"{sum(len(v) for v in split_records.values())}"
    )

    print()
    print(
        "Split completed successfully."
    )
    print()


if __name__ == "__main__":
    main()