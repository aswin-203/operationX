import pandas as pd

from operation_x.ml.model_comparison import (
    ModelComparison,
)


def test_model_comparison(tmp_path):

    features = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "feature_b": [
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    targets = pd.DataFrame(
        {
            "solvent_percent": [
                40.0,
                41.0,
                42.0,
                50.0,
                51.0,
                52.0,
                60.0,
                61.0,
                62.0,
            ]
        }
    )

    metadata = pd.DataFrame(
        {
            "query_pdb": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
            ],
            "similar_pdb": [
                "X",
                "Y",
                "Z",
                "X",
                "Y",
                "Z",
                "X",
                "Y",
                "Z",
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

    comparison = ModelComparison(
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

    result = comparison.evaluate(
        n_splits=3
    )

    assert result["target"] == (
        "solvent_percent"
    )

    assert result["rows"] == 9

    assert result["groups"] == 3

    assert (
        "Mean Baseline"
        in result["results"]
    )

    assert (
        "Linear Regression"
        in result["results"]
    )

    assert (
        "Random Forest"
        in result["results"]
    )

    assert (
        "Extra Trees"
        in result["results"]
    )

    assert (
        "Gradient Boosting"
        in result["results"]
    )

    for model_result in (
        result["results"].values()
    ):
        assert "mae" in model_result
        assert "rmse" in model_result
        assert "r2" in model_result
        assert len(
            model_result["folds"]
        ) == 3