import pandas as pd
import pytest

from operation_x.ml.feature_set_evaluation import (
    FeatureSetEvaluator,
)


def create_dataset(path):

    rows = []

    for i in range(40):

        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "similar_pdb": f"SIM{i}",
                "feature_a": float(i),
                "feature_b": float(i * 2),
                "feature_c": float(40 - i),
                "feature_d": float(i % 5),
                "feature_e": float(i % 3),
                "pH": float(5 + i * 0.05),
            }
        )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def test_feature_set_evaluation(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    evaluator = FeatureSetEvaluator(
        data_file=str(data_file),
        target="pH",
        feature_counts=[
            2,
            3,
            5,
        ],
        n_splits=5,
    )

    result = evaluator.evaluate()

    assert result["target"] == "pH"
    assert result["rows"] == 40
    assert result["available_features"] == 5

    assert result["feature_counts"] == [
        2,
        3,
        5,
    ]

    assert set(
        result["results"].keys()
    ) == {
        2,
        3,
        5,
    }

    for feature_count in [
        2,
        3,
        5,
    ]:

        assert (
            "Random Forest"
            in result["results"][
                feature_count
            ]
        )

        assert (
            "Extra Trees"
            in result["results"][
                feature_count
            ]
        )

        for model_name in [
            "Random Forest",
            "Extra Trees",
        ]:

            metrics = result[
                "results"
            ][feature_count][
                model_name
            ]

            assert "mae" in metrics
            assert "rmse" in metrics
            assert "r2" in metrics


def test_feature_count_larger_than_available(
    tmp_path,
):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    evaluator = FeatureSetEvaluator(
        data_file=str(data_file),
        target="pH",
        feature_counts=[
            100,
        ],
    )

    result = evaluator.evaluate()

    assert result[
        "feature_counts"
    ] == [5]


def test_invalid_feature_count(
    tmp_path,
):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    evaluator = FeatureSetEvaluator(
        data_file=str(data_file),
        target="pH",
        feature_counts=[
            0,
        ],
    )

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate()


def test_save(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    output_file = (
        tmp_path / "feature_sets.csv"
    )

    evaluator = FeatureSetEvaluator(
        data_file=str(data_file),
        target="pH",
        feature_counts=[
            2,
            4,
        ],
    )

    result = evaluator.save(
        str(output_file)
    )

    assert output_file.exists()

    assert result["rows"] == 4

    assert result[
        "best_feature_count"
    ] in {
        2,
        4,
    }

    assert result[
        "best_model"
    ] in {
        "Random Forest",
        "Extra Trees",
    }

    dataframe = pd.read_csv(
        output_file
    )

    assert len(dataframe) == 4