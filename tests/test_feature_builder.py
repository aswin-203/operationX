import pandas as pd

from operation_x.ml.feature_builder import (
    FeatureBuilder,
)


def test_feature_builder_creates_features_and_targets(
    tmp_path,
):

    input_file = (
        tmp_path / "input.csv"
    )

    df = pd.DataFrame(
        [
            {
                "query_pdb": "101M",
                "similar_pdb": "109M",
                "sequence_identity": 99.3,
                "sequence_coverage": 1.0,
                "sequence_bitscore": 307.0,
                "structure_query_tm_score": 0.997,
                "structure_target_tm_score": 0.997,
                "structure_rmsd": 0.27,
                "structure_alignment_length": 154,
                "structure_evalue": 1e-20,
                "pfam_count": 1,
                "interpro_count": 3,
                "matthews_coefficient": 3.07,
                "solvent_percent": 59.9,
                "pH": 9.0,
                "temperature_kelvin": 290.0,
            }
        ]
    )

    df.to_csv(
        input_file,
        index=False,
    )

    builder = FeatureBuilder(
        input_file=str(input_file),
        output_directory=str(
            tmp_path / "ml"
        ),
    )

    metadata, features, targets = (
        builder.build()
    )

    assert len(metadata) == 1
    assert len(features) == 1
    assert len(targets) == 1

    assert (
        features[
            "sequence_identity"
        ].iloc[0]
        == 99.3
    )

    assert (
        targets["pH"].iloc[0]
        == 9.0
    )


def test_feature_builder_marks_missing_structure(
    tmp_path,
):

    input_file = (
        tmp_path / "input.csv"
    )

    df = pd.DataFrame(
        [
            {
                "query_pdb": "101M",
                "similar_pdb": "10AK",
                "sequence_identity": 42.8,
                "sequence_coverage": 0.09,
                "sequence_bitscore": 18.1,
                "structure_query_tm_score": None,
                "structure_target_tm_score": None,
                "structure_rmsd": None,
                "structure_alignment_length": None,
                "structure_evalue": None,
                "pfam_count": 1,
                "interpro_count": 3,
                "matthews_coefficient": 2.32,
                "solvent_percent": 47.12,
                "pH": None,
                "temperature_kelvin": 290.0,
            }
        ]
    )

    df.to_csv(
        input_file,
        index=False,
    )

    builder = FeatureBuilder(
        input_file=str(input_file),
        output_directory=str(
            tmp_path / "ml"
        ),
    )

    _, features, _ = (
        builder.build()
    )

    assert (
        features[
            "structure_missing"
        ].iloc[0]
        == 1
    )

    assert (
        features[
            "structure_rmsd_missing"
        ].iloc[0]
        == 1
    )
