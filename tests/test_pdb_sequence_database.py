import csv

from operation_x.similarity.pdb_sequence_database import (
    PDBSequenceDatabase,
)


def test_create_fasta(tmp_path):

    csv_path = (
        tmp_path / "pilot.csv"
    )

    rows = [
        {
            "pdb_id": "101M",
            "entity_id": "1",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQ",
        },
        {
            "pdb_id": "10AF",
            "entity_id": "1",
            "sequence": "AAAAAAA",
        },
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pdb_id",
                "entity_id",
                "sequence",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    builder = PDBSequenceDatabase(
        blast_dir=str(
            tmp_path / "blast"
        )
    )

    fasta = builder.create_fasta(
        str(csv_path)
    )

    assert fasta.exists()

    content = fasta.read_text()

    assert ">101M_1" in content
    assert "MKTAYIAKQRQISFVKSHFSRQ" in content

    assert ">10AF_1" in content
    assert "AAAAAAA" in content
