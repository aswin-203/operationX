import json

from operation_x.ml.large_feature_builder import (
    LargeFeatureBuilder,
)


def create_dataset(path):
    record = {
        "pdb_id": "101M",
        "evidence": {
            "sequence_evidence": {
                "best_identity": 100.0,
                "best_coverage": 1.0,
                "best_evalue": 1e-100,
                "best_bitscore": 300.0,
                "strong_hit_count": 9,
            },
            "structure_evidence": {
                "best_query_tm_score": 1.0,
                "best_target_tm_score": 1.0,
                "best_rmsd": 0.0,
                "best_alignment_length": 154,
                "strong_hit_count": 9,
            },
            "domain_evidence": {
                "domain_count": 4,
                "pfam_count": 1,
                "interpro_count": 3,
            },
        },
        "crystallization": {
            "pH": 7.0,
            "temperature_kelvin": 293.0,
            "matthews_coefficient": 2.5,
            "solvent_percent": 50.0,
        },
    }

    path.write_text(
        json.dumps([record])
    )


def test_large_feature_builder_creates_target_datasets(
    tmp_path,
):
    input_file = tmp_path / "similarity.json"

    create_dataset(input_file)

    builder = LargeFeatureBuilder(
        input_file=str(input_file),
        output_directory=str(
            tmp_path / "ml"
        ),
    )

    result = builder.save()

    assert set(result.keys()) == {
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    }

    assert result["pH"]["rows"] == 1
    assert result["temperature_kelvin"]["rows"] == 1
    assert result["matthews_coefficient"]["rows"] == 1
    assert result["solvent_percent"]["rows"] == 1


def test_missing_target_is_not_shared_across_targets(
    tmp_path,
):
    input_file = tmp_path / "similarity.json"

    record = {
        "pdb_id": "101M",
        "evidence": {
            "sequence_evidence": {
                "best_identity": 100.0,
                "best_coverage": 1.0,
                "best_evalue": 1e-100,
                "best_bitscore": 300.0,
                "strong_hit_count": 9,
            },
            "structure_evidence": {
                "best_query_tm_score": 1.0,
                "best_target_tm_score": 1.0,
                "best_rmsd": 0.0,
                "best_alignment_length": 154,
                "strong_hit_count": 9,
            },
            "domain_evidence": {
                "domain_count": 4,
                "pfam_count": 1,
                "interpro_count": 3,
            },
        },
        "crystallization": {
            "pH": 7.0,
            "temperature_kelvin": None,
            "matthews_coefficient": 2.5,
            "solvent_percent": 50.0,
        },
    }

    input_file.write_text(
        json.dumps([record])
    )

    builder = LargeFeatureBuilder(
        input_file=str(input_file),
        output_directory=str(
            tmp_path / "ml"
        ),
    )

    result = builder.save()

    assert result["pH"]["rows"] == 1
    assert result["temperature_kelvin"]["rows"] == 0
    assert result["matthews_coefficient"]["rows"] == 1
    assert result["solvent_percent"]["rows"] == 1
