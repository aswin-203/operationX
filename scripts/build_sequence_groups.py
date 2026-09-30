import csv
from collections import defaultdict
from pathlib import Path


BLAST_RESULTS = Path(
    "data/blast/large/all_vs_all.tsv"
)

OUTPUT_FILE = Path(
    "data/similarity/large/sequence_groups.csv"
)

IDENTITY_THRESHOLD = 90.0
COVERAGE_THRESHOLD = 0.80


class UnionFind:
    """Disjoint-set structure for similarity clustering."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, first, second):
        self.add(first)
        self.add(second)

        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second

        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first

        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def normalize_sequence_id(sequence_id: str) -> str:
    """
    Convert BLAST identifiers into the same format.

    Query:
        101M_1

    Subject:
        pdb|101M|1

    Both become:
        101M_1
    """

    sequence_id = sequence_id.strip()

    # BLAST subject format:
    # pdb|101M|1
    if sequence_id.startswith("pdb|"):
        parts = sequence_id.split("|")

        if len(parts) == 3:
            pdb_id = parts[1].strip().upper()
            entity_id = parts[2].strip()

            return f"{pdb_id}_{entity_id}"

    # Query format:
    # 101M_1
    if "_" in sequence_id:
        pdb_id, entity_id = sequence_id.split(
            "_",
            1,
        )

        return (
            f"{pdb_id.strip().upper()}_"
            f"{entity_id.strip()}"
        )

    return sequence_id.upper()


def parse_blast_results():

    union_find = UnionFind()

    total_hits = 0
    self_hits = 0
    passing_hits = 0

    with BLAST_RESULTS.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            if not line.strip():
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) != 8:
                raise ValueError(
                    "Unexpected BLAST format. "
                    f"Expected 8 columns, got {len(fields)}"
                )

            (
                query_id,
                subject_id,
                identity,
                alignment_length,
                query_length,
                subject_length,
                evalue,
                bitscore,
            ) = fields

            total_hits += 1

            query_id = normalize_sequence_id(
                query_id
            )

            subject_id = normalize_sequence_id(
                subject_id
            )

            union_find.add(query_id)
            union_find.add(subject_id)

            # Ignore self hits.
            if query_id == subject_id:
                self_hits += 1
                continue

            identity = float(identity)
            alignment_length = int(
                alignment_length
            )
            query_length = int(
                query_length
            )

            if query_length <= 0:
                continue

            query_coverage = (
                alignment_length / query_length
            )

            if (
                identity >= IDENTITY_THRESHOLD
                and query_coverage >= COVERAGE_THRESHOLD
            ):
                union_find.union(
                    query_id,
                    subject_id,
                )

                passing_hits += 1

    return (
        union_find,
        total_hits,
        self_hits,
        passing_hits,
    )


def build_groups(union_find):

    groups = defaultdict(list)

    for sequence_id in union_find.parent:

        root = union_find.find(
            sequence_id
        )

        groups[root].append(
            sequence_id
        )

    sorted_groups = sorted(
        groups.values(),
        key=lambda group: (
            -len(group),
            sorted(group)[0],
        ),
    )

    group_mapping = {}

    for index, members in enumerate(
        sorted_groups,
        start=1,
    ):

        group_id = f"GROUP_{index:04d}"

        for member in sorted(members):

            group_mapping[
                member
            ] = group_id

    return group_mapping, sorted_groups


def save_groups(group_mapping):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "sequence_id",
                "group_id",
            ]
        )

        for sequence_id in sorted(
            group_mapping
        ):

            writer.writerow(
                [
                    sequence_id,
                    group_mapping[
                        sequence_id
                    ],
                ]
            )


def main():

    if not BLAST_RESULTS.exists():
        raise FileNotFoundError(
            f"BLAST results not found: "
            f"{BLAST_RESULTS}"
        )

    print()
    print("=" * 60)
    print(
        "OPERATION X — BUILD SEQUENCE "
        "SIMILARITY GROUPS"
    )
    print("=" * 60)
    print()

    print(
        f"BLAST results          : "
        f"{BLAST_RESULTS}"
    )

    print(
        f"Identity threshold     : "
        f"{IDENTITY_THRESHOLD:.1f}%"
    )

    print(
        f"Coverage threshold     : "
        f"{COVERAGE_THRESHOLD:.0%}"
    )

    print()

    (
        union_find,
        total_hits,
        self_hits,
        passing_hits,
    ) = parse_blast_results()

    group_mapping, groups = build_groups(
        union_find
    )

    save_groups(group_mapping)

    print("BLAST ANALYSIS")
    print("-" * 60)

    print(
        f"Total BLAST hits        : "
        f"{total_hits}"
    )

    print(
        f"Self hits removed       : "
        f"{self_hits}"
    )

    print(
        f"Passing similarity hits : "
        f"{passing_hits}"
    )

    print()
    print("GROUPS")
    print("-" * 60)

    print(
        f"Sequences represented    : "
        f"{len(group_mapping)}"
    )

    print(
        f"Similarity groups        : "
        f"{len(groups)}"
    )

    if groups:

        largest_group = max(
            len(group)
            for group in groups
        )

        print(
            f"Largest group            : "
            f"{largest_group} sequences"
        )

    print()
    print("LARGEST GROUPS")
    print("-" * 60)

    for index, group in enumerate(
        groups[:10],
        start=1,
    ):

        print(
            f"GROUP_{index:04d}: "
            f"{len(group)} sequences"
        )

    print()
    print(
        f"Output                  : "
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 60)
    print("SEQUENCE GROUPING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()