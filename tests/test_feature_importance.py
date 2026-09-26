import pandas as pd

from operation_x.ml.feature_importance import (
    FeatureImportanceAnalyzer,
)


def test_feature_importance(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    rows = []

    for i in range(30):
        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "similar_pdb": f"SIM{i}",
                "feature_a": float(i),
                "feature_b": float(i * 2),
                "feature_c": float(30 - i),
                "pH": float(5 + i * 0.1),
            }
        )

    pd.DataFrame(rows).to_csv(
        data_file,
        index=False,
    )

    analyzer = FeatureImportanceAnalyzer(
        data_file=str(data_file),
        target="pH",
    )

    result = analyzer.analyze()

    assert len(result) == 3

    assert "rank" in result.columns
    assert "feature" in result.columns
    assert (
        "random_forest_importance"
        in result.columns
    )
    assert (
        "extra_trees_importance"
        in result.columns
    )
    assert (
        "mean_importance"
        in result.columns
    )

    assert result.iloc[0]["rank"] == 1

    assert (
        result["mean_importance"]
        .notna()
        .all()
    )


def test_save_feature_importance(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    pd.DataFrame(
        {
            "query_pdb": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
            ],
            "similar_pdb": [
                "X",
                "Y",
                "Z",
                "W",
                "V",
                "U",
            ],
            "feature_a": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "feature_b": [
                6,
                5,
                4,
                3,
                2,
                1,
            ],
            "pH": [
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    ).to_csv(
        data_file,
        index=False,
    )

    output_file = (
        tmp_path / "importance.csv"
    )

    analyzer = FeatureImportanceAnalyzer(
        data_file=str(data_file),
        target="pH",
    )

    result = analyzer.save(
        str(output_file)
    )

    assert output_file.exists()
    assert result["rows"] == 2
    assert result["feature_count"] == 2
    assert result["top_feature"] in {
        "feature_a",
        "feature_b",
    }

    saved = pd.read_csv(
        output_file
    )

    assert len(saved) == 2