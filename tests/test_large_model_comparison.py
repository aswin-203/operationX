import pandas as pd

from operation_x.ml.large_model_comparison import (
    LargeModelComparison,
)


def create_dataset(tmp_path):
    features = pd.DataFrame(
        {
            "feature_1": [
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
            ],
            "feature_2": [
                2, 4, 6, 8,
                10, 12, 14, 16,
                18, 20, 22, 24,
            ],
        }
    )

    targets = pd.DataFrame(
        {
            "pH": [
                5.0, 5.5, 6.0, 6.5,
                7.0, 7.5, 8.0, 8.5,
                9.0, 9.5, 10.0, 10.5,
            ]
        }
    )

    metadata = pd.DataFrame(
        {
            "query_pdb": [
                "A", "A",
                "B", "B",
                "C", "C",
                "D", "D",
                "E", "E",
                "F", "F",
            ]
        }
    )

    features_file = tmp_path / "features.csv"
    targets_file = tmp_path / "targets.csv"
    metadata_file = tmp_path / "metadata.csv"

    features.to_csv(features_file, index=False)
    targets.to_csv(targets_file, index=False)
    metadata.to_csv(metadata_file, index=False)

    return (
        features_file,
        targets_file,
        metadata_file,
    )


def test_large_model_comparison(tmp_path):
    (
        features_file,
        targets_file,
        metadata_file,
    ) = create_dataset(tmp_path)

    comparison = LargeModelComparison(
        features_file=str(features_file),
        targets_file=str(targets_file),
        metadata_file=str(metadata_file),
    )

    result = comparison.evaluate(
        n_splits=3,
    )

    assert result["target"] == "pH"
    assert result["rows"] == 12
    assert result["groups"] == 6

    assert set(result["results"].keys()) == {
        "Linear Regression",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "Mean Baseline",
    }


def test_each_model_has_fold_results(tmp_path):
    (
        features_file,
        targets_file,
        metadata_file,
    ) = create_dataset(tmp_path)

    comparison = LargeModelComparison(
        features_file=str(features_file),
        targets_file=str(targets_file),
        metadata_file=str(metadata_file),
    )

    result = comparison.evaluate(
        n_splits=3,
    )

    for metrics in result["results"].values():
        assert len(metrics["folds"]) == 3
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
