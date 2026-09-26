import argparse

from operation_x.dataset.large_structure_collector import (
    LargeStructureCollector,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the Operation X large "
            "protein structure dataset."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "Number of PDB entries to collect "
            "(default: 100)"
        ),
    )

    args = parser.parse_args()

    collector = LargeStructureCollector(
        output_directory=(
            "data/structures/large"
        ),
        records_file=(
            "data/structures/large/records.json"
        ),
    )

    collector.collect(
        limit=args.limit
    )


if __name__ == "__main__":
    main()