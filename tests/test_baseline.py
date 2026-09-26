import pandas as pd

from operation_x.ml.baseline import MeanBaseline


def test_mean_baseline(tmp_path):

    features = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
            ]
        }
    )

    targets = pd.DataFrame(
        {
            "solvent_percent": [
                40.0,
                41.0,
                50.0,
                51.0,
                60.0,
                61.0,
            ]
        }
    )

    metadata = pd.DataFrame(
        {
            "query_pdb": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
            "similar_pdb": [
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
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

    baseline = MeanBaseline(
        features_file=str(features_file),
        targets_file=str(targets_file),
        metadata_file=str(metadata_file),
    )

    result = baseline.evaluate(
        n_splits=3
    )

    assert result["target"] == (
        "solvent_percent"
    )

    assert result["rows"] == 6

    assert result["groups"] == 3

    assert len(
        result["folds"]
    ) == 3