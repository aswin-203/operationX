import pandas as pd

from operation_x.ml.final_model_comparison import (
    FinalModelComparison,
)


def test_final_model_comparison(tmp_path):

    data_file = tmp_path / "dataset.csv"

    rows = []

    for i in range(30):
        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "feature_a": float(i),
                "feature_b": float(i % 5),
                "pH": float(5 + (i % 5) * 0.5),
            }
        )

    pd.DataFrame(rows).to_csv(
        data_file,
        index=False,
    )

    comparison = FinalModelComparison(
        data_file=str(data_file),
        target="pH",
        n_splits=5,
    )

    result = comparison.evaluate()

    assert result["target"] == "pH"
    assert result["rows"] == 30
    assert result["feature_count"] == 2

    assert "Random Forest" in result["results"]
    assert "Extra Trees" in result["results"]
    assert "Gradient Boosting" in result["results"]
    assert "Mean Baseline" in result["results"]

    for metrics in result["results"].values():
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics


def test_best_model(tmp_path):

    data_file = tmp_path / "dataset.csv"

    rows = []

    for i in range(30):
        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "feature_a": float(i),
                "feature_b": float(i * 2),
                "pH": float(i),
            }
        )

    pd.DataFrame(rows).to_csv(
        data_file,
        index=False,
    )

    comparison = FinalModelComparison(
        data_file=str(data_file),
        target="pH",
        n_splits=5,
    )

    result = comparison.evaluate()

    best = comparison.best_model(result)

    assert best["model"] in {
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
    }

    assert "mae" in best
    assert "rmse" in best
    assert "r2" in best
