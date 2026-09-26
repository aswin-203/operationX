import pandas as pd

from operation_x.ml.input_feature_builder import (
    InputFeatureBuilder,
)


def test_build_features_from_sequence():

    builder = InputFeatureBuilder()

    features = builder.build(
        sequence="MVLSEGEWQLVLHVWAKVE"
    )

    assert isinstance(features, dict)

    assert "sequence_length" in features
    assert "molecular_weight" in features
    assert "isoelectric_point" in features
    assert "gravy" in features
    assert "instability_index" in features

    assert features["sequence_length"] == 19


def test_build_returns_required_feature_columns():

    builder = InputFeatureBuilder()

    features = builder.build(
        sequence="MKKLL"
    )

    expected_features = {
        "sequence_length",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "instability_index",
        "aromaticity",
        "charged_residue_fraction",
        "polar_residue_fraction",
        "hydrophobic_residue_fraction",
        "cysteine_fraction",
        "glycine_fraction",
        "proline_fraction",
        "ambiguous_residue_count",
        "ambiguous_residue_fraction",
        "valid_sequence_length",
    }

    assert expected_features.issubset(
        features.keys()
    )


def test_invalid_sequence():

    builder = InputFeatureBuilder()

    try:
        builder.build(
            sequence="12345"
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid sequence should raise ValueError"
        )
