from operation_x.similarity.homolog_structure_resolver import (
    HomologStructureResolver,
)


def test_parse_pdb_subject_id():
    resolver = HomologStructureResolver(
        "data/structures/large"
    )

    assert resolver.parse_pdb_id(
        "pdb|101M|1"
    ) == "101M"


def test_resolve_existing_structure():
    resolver = HomologStructureResolver(
        "data/structures/large"
    )

    path = resolver.resolve("pdb|101M|1")

    assert path is not None
    assert path.endswith("101M.cif")


def test_resolve_missing_structure():
    resolver = HomologStructureResolver(
        "data/structures/large"
    )

    assert resolver.resolve(
        "pdb|XXXX|1"
    ) is None


def test_resolve_hits():
    resolver = HomologStructureResolver(
        "data/structures/large"
    )

    hits = [
        {
            "subject_id": "pdb|101M|1",
            "identity": 100.0,
            "coverage": 1.0,
            "evalue": 0.0,
            "bitscore": 300,
        },
        {
            "subject_id": "pdb|109M|1",
            "identity": 99.3,
            "coverage": 1.0,
            "evalue": 0.0,
            "bitscore": 290,
        },
    ]

    results = resolver.resolve_hits(hits)

    assert len(results) == 2
    assert results[0]["pdb_id"] == "101M"
    assert results[1]["pdb_id"] == "109M"
