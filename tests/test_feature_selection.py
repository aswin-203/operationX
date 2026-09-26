import pandas as pd
import pytest

from operation_x.ml.feature_selection import (
    FeatureSelector,
)


def create_dataset(path):

    rows = []

    for i in range(30):

        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "similar_pdb": f"SIM{i}",
                "feature_a": float(i),
                "feature_b": float(i * 2),
                "feature_c": float(30 - i),
                "feature_d": float(i % 3),
                "pH": float(5 + i * 0.1),
            }
        )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def test_rank_features(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    selector = FeatureSelector(
        data_file=str(data_file),
        target="pH",
    )

    ranking = (
        selector.rank_features()
    )

    assert len(ranking) == 4

    assert "rank" in ranking.columns
    assert "feature" in ranking.columns
    assert "importance" in ranking.columns

    assert ranking.iloc[0]["rank"] == 1

    assert (
        ranking["importance"]
        .notna()
        .all()
    )


def test_select(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    selector = FeatureSelector(
        data_file=str(data_file),
        target="pH",
    )

    selected = selector.select(
        2
    )

    assert len(selected) == 2

    assert all(
        feature in {
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        }
        for feature in selected
    )


def test_select_multiple(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    selector = FeatureSelector(
        data_file=str(data_file),
        target="pH",
    )

    result = selector.select_multiple(
        [2, 3, 4]
    )

    assert set(result.keys()) == {
        2,
        3,
        4,
    }

    assert len(result[2]) == 2
    assert len(result[3]) == 3
    assert len(result[4]) == 4


def test_invalid_feature_count(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    selector = FeatureSelector(
        data_file=str(data_file),
        target="pH",
    )

    with pytest.raises(
        ValueError
    ):
        selector.select(0)


def test_save(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    create_dataset(
        data_file
    )

    output_file = (
        tmp_path / "ranking.csv"
    )

    selector = FeatureSelector(
        data_file=str(data_file),
        target="pH",
    )

    result = selector.save(
        str(output_file)
    )

    assert output_file.exists()
    assert result["feature_count"] == 4
    assert result["top_feature"] in {
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
    }