import pandas as pd

from operation_x.ml.predictor import OperationXPredictor


def test_predictor_loads_dataset(tmp_path):

    data_file = tmp_path / "dataset.csv"

    df = pd.DataFrame({
        "query_pdb": [f"PDB{i}" for i in range(20)],
        "feature_a": [float(i) for i in range(20)],
        "feature_b": [float(i * 2) for i in range(20)],
        "pH": [5.0 + i * 0.1 for i in range(20)],
    })

    df.to_csv(data_file, index=False)

    predictor = OperationXPredictor(
        data_file=str(data_file),
        target="pH",
        model_name="gradient_boosting",
    )

    result = predictor.fit()

    assert result["target"] == "pH"
    assert result["rows"] == 20
    assert result["feature_count"] == 2


def test_predictor_returns_prediction(tmp_path):

    data_file = tmp_path / "dataset.csv"

    df = pd.DataFrame({
        "query_pdb": [f"PDB{i}" for i in range(20)],
        "feature_a": [float(i) for i in range(20)],
        "feature_b": [float(i * 2) for i in range(20)],
        "pH": [5.0 + i * 0.1 for i in range(20)],
    })

    df.to_csv(data_file, index=False)

    predictor = OperationXPredictor(
        data_file=str(data_file),
        target="pH",
        model_name="gradient_boosting",
    )

    predictor.fit()

    result = predictor.predict({
        "feature_a": 10.0,
        "feature_b": 20.0,
    })

    assert "prediction" in result
    assert "target" in result
    assert result["target"] == "pH"
    assert isinstance(result["prediction"], float)