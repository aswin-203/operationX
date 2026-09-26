import pandas as pd

from operation_x.ml.combined_model_comparison import (
    CombinedModelComparison,
)


def test_combined_model_comparison(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    rows = []

    for i in range(20):
        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "feature_a": float(i),
                "feature_b": float(i * 2),
                "pH": float(5 + i * 0.1),
            }
        )

    pd.DataFrame(rows).to_csv(
        data_file,
        index=False,
    )

    comparison = CombinedModelComparison(
        data_file=str(data_file),
        target="pH",
    )

    result = comparison.evaluate(
        n_splits=5,
    )

    assert result["target"] == "pH"
    assert result["rows"] == 20
    assert result["groups"] == 20
    assert result["feature_count"] == 2

    assert "Linear Regression" in result[
        "results"
    ]

    assert "Random Forest" in result[
        "results"
    ]

    assert "Extra Trees" in result[
        "results"
    ]

    assert "Gradient Boosting" in result[
        "results"
    ]

    assert "Mean Baseline" in result[
        "results"
    ]


def test_missing_target_fails(tmp_path):

    data_file = (
        tmp_path / "dataset.csv"
    )

    pd.DataFrame(
        {
            "query_pdb": ["A", "B"],
            "feature_a": [1.0, 2.0],
        }
    ).to_csv(
        data_file,
        index=False,
    )

    comparison = CombinedModelComparison(
        data_file=str(data_file),
        target="pH",
    )

    try:
        comparison.evaluate()

        assert False, (
            "Expected ValueError"
        )

    except ValueError:
        pass
