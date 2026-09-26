import pandas as pd

from operation_x.similarity.evidence import (
    SimilarityEvidence,
)


def test_parse_pdb_subject_id():

    evidence = SimilarityEvidence(
        "dummy.csv"
    )

    pdb_id, entity_id = (
        evidence._parse_subject_id(
            "pdb|101M|1"
        )
    )

    assert pdb_id == "101M"
    assert entity_id == "1"


def test_collect_similarity_evidence(
    tmp_path,
):

    dataset = pd.DataFrame(
        [
            {
                "pdb_id": "101M",
                "entity_id": "1",
                "crystallization_method":
                    "VAPOR DIFFUSION",
                "pH": 9.0,
                "temperature_kelvin":
                    None,
                "pressure": None,
                "crystallization_time":
                    None,
                "crystallization_details":
                    "3.0 M AMMONIUM SULFATE",
                "expression_host":
                    "Escherichia coli",
                "expression_system":
                    "bacterial",
            }
        ]
    )

    dataset_path = (
        tmp_path / "dataset.csv"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    evidence = SimilarityEvidence(
        str(dataset_path)
    )

    hits = [
        {
            "subject_id":
                "pdb|101M|1",
            "identity":
                99.5,
            "coverage":
                1.0,
            "evalue":
                1e-100,
            "bitscore":
                300.0,
        }
    ]

    results = evidence.collect(
        hits
    )

    assert len(results) == 1

    result = results[0]

    assert result["pdb_id"] == "101M"
    assert result["entity_id"] == "1"

    assert result["identity"] == 99.5
    assert result["coverage"] == 1.0

    assert result[
        "crystallization_method"
    ] == "VAPOR DIFFUSION"

    assert result["pH"] == 9.0

    assert result[
        "expression_host"
    ] == "Escherichia coli"
