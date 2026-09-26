import pandas as pd

from operation_x.ml.prediction_pipeline import (
    OperationXPredictionPipeline,
)


def test_prediction_pipeline_fit(tmp_path):

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    targets = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    for target in targets:

        rows = []

        for i in range(20):
            rows.append(
                {
                    "query_pdb": f"PDB{i}",
                    "feature_a": float(i),
                    "feature_b": float(i * 2),
                    target: float(i) + 1.0,
                }
            )

        pd.DataFrame(rows).to_csv(
            data_directory
            / f"{target}_combined.csv",
            index=False,
        )

    pipeline = OperationXPredictionPipeline(
        data_directory=str(data_directory),
    )

    result = pipeline.fit()

    assert result["targets"] == targets

    for target in targets:
        assert target in result["results"]
        assert result["results"][target]["rows"] == 20


def test_prediction_pipeline_predict(tmp_path):

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    targets = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    for target in targets:

        rows = []

        for i in range(20):
            rows.append(
                {
                    "query_pdb": f"PDB{i}",
                    "feature_a": float(i),
                    "feature_b": float(i * 2),
                    target: float(i) + 1.0,
                }
            )

        pd.DataFrame(rows).to_csv(
            data_directory
            / f"{target}_combined.csv",
            index=False,
        )

    pipeline = OperationXPredictionPipeline(
        data_directory=str(data_directory),
    )

    pipeline.fit()

    result = pipeline.predict(
        {
            "feature_a": 10.0,
            "feature_b": 20.0,
        }
    )

    assert "pH" in result
    assert "temperature_kelvin" in result
    assert "matthews_coefficient" in result
    assert "solvent_percent" in result

    for target in targets:
        assert isinstance(
            result[target],
            float,
        )
