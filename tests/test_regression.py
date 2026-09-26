import pandas as pd

from operation_x.ml.regression import (
    SolventRegression,
)


def create_dataset(tmp_path):

    features = pd.DataFrame(
        [
            {
                "sequence_identity": 90.0,
                "sequence_coverage": 1.0,
                "sequence_bitscore": 200.0,
                "structure_query_tm_score": 0.9,
                "structure_target_tm_score": 0.9,
                "structure_rmsd": 1.0,
                "structure_alignment_length": 100,
                "structure_evalue": 1e-10,
                "pfam_count": 1,
                "interpro_count": 2,
                "structure_missing": 0,
                "structure_rmsd_missing": 0,
                "sequence_missing": 0,
            },
            {
                "sequence_identity": 80.0,
                "sequence_coverage": 0.8,
                "sequence_bitscore": 150.0,
                "structure_query_tm_score": 0.8,
                "structure_target_tm_score": 0.8,
                "structure_rmsd": 2.0,
                "structure_alignment_length": 80,
                "structure_evalue": 1e-5,
                "pfam_count": 1,
                "interpro_count": 1,
                "structure_missing": 0,
                "structure_rmsd_missing": 0,
                "sequence_missing": 0,
            },
            {
                "sequence_identity": 70.0,
                "sequence_coverage": 0.7,
                "sequence_bitscore": 100.0,
                "structure_query_tm_score": None,
                "structure_target_tm_score": None,
                "structure_rmsd": None,
                "structure_alignment_length": None,
                "structure_evalue": None,
                "pfam_count": 1,
                "interpro_count": 1,
                "structure_missing": 1,
                "structure_rmsd_missing": 1,
                "sequence_missing": 0,
            },
        ]
    )

    targets = pd.DataFrame(
        {
            "pH": [9.0, 8.0, 7.0],
            "temperature_kelvin": [
                290.0,
                290.0,
                277.0,
            ],
            "matthews_coefficient": [
                3.0,
                2.5,
                2.0,
            ],
            "solvent_percent": [
                60.0,
                50.0,
                40.0,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "query_pdb": [
                "101M",
                "102M",
                "103M",
            ],
            "similar_pdb": [
                "109M",
                "108M",
                "107M",
            ],
        }
    )

    features_file = (
        tmp_path / "features.csv"
    )

    targets_file = (
        tmp_path / "targets.csv"
    )

    metadata_file = (
        tmp_path / "metadata.csv"
    )

    features.to_csv(
        features_file,
        index=False,
    )

    targets.to_csv(
        targets_file,
        index=False,
    )

    metadata.to_csv(
        metadata_file,
        index=False,
    )

    return (
        features_file,
        targets_file,
        metadata_file,
    )


def test_load_data(tmp_path):

    (
        features_file,
        targets_file,
        metadata_file,
    ) = create_dataset(
        tmp_path
    )

    regression = SolventRegression(
        features_file=str(
            features_file
        ),
        targets_file=str(
            targets_file
        ),
        metadata_file=str(
            metadata_file
        ),
    )

    features, targets, metadata = (
        regression.load_data()
    )

    assert len(features) == 3
    assert len(targets) == 3
    assert len(metadata) == 3


def test_prepare_data(tmp_path):

    (
        features_file,
        targets_file,
        metadata_file,
    ) = create_dataset(
        tmp_path
    )

    regression = SolventRegression(
        features_file=str(
            features_file
        ),
        targets_file=str(
            targets_file
        ),
        metadata_file=str(
            metadata_file
        ),
    )

    X, y, groups = (
        regression.prepare_data()
    )

    assert len(X) == 3
    assert len(y) == 3
    assert len(groups) == 3

    assert (
        y.iloc[0] == 60.0
    )


def test_missing_values_are_filled():

    train = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [10.0, 20.0],
        }
    )

    validation = pd.DataFrame(
        {
            "a": [None],
            "b": [None],
        }
    )

    train_result, validation_result = (
        SolventRegression._fill_missing(
            train,
            validation,
        )
    )

    assert not train_result.isna().any().any()
    assert not validation_result.isna().any().any()

    assert (
        validation_result[
            "a"
        ].iloc[0]
        == 1.5
    )


def test_evaluation_requires_enough_groups(
    tmp_path,
):

    (
        features_file,
        targets_file,
        metadata_file,
    ) = create_dataset(
        tmp_path
    )

    regression = SolventRegression(
        features_file=str(
            features_file
        ),
        targets_file=str(
            targets_file
        ),
        metadata_file=str(
            metadata_file
        ),
    )

    try:
        regression.evaluate(
            n_splits=5
        )
    except ValueError as error:
        assert (
            "groups"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
