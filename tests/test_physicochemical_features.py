import pandas as pd

from operation_x.ml.physicochemical_features import (
    PhysicochemicalFeatureBuilder,
)


def test_build_physicochemical_features(tmp_path):
    records_file = tmp_path / "records.json"

    records_file.write_text(
        """
        {
            "TEST1": {
                "sequence": "MVLSEGEWQLVLHVWAKVE"
            },
            "TEST2": {
                "sequence": "MKKLL"
            }
        }
        """
    )

    output_file = tmp_path / "features.csv"

    builder = PhysicochemicalFeatureBuilder(
        records_file=str(records_file),
        output_file=str(output_file),
    )

    df = builder.build()

    assert len(df) == 2
    assert "pdb_id" in df.columns
    assert "sequence_length" in df.columns
    assert "molecular_weight" in df.columns
    assert "isoelectric_point" in df.columns
    assert "gravy" in df.columns


def test_save_physicochemical_features(tmp_path):
    records_file = tmp_path / "records.json"

    records_file.write_text(
        """
        {
            "TEST1": {
                "sequence": "MVLSEGEWQLVLHVWAKVE"
            }
        }
        """
    )

    output_file = tmp_path / "features.csv"

    builder = PhysicochemicalFeatureBuilder(
        records_file=str(records_file),
        output_file=str(output_file),
    )

    result = builder.save()

    assert output_file.exists()
    assert result["rows"] == 1
    assert result["feature_count"] > 0

    df = pd.read_csv(output_file)

    assert len(df) == 1
