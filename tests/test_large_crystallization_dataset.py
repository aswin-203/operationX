from operation_x.ml.large_crystallization_dataset import (
    LargeCrystallizationDataset,
)


DATA_FILE = (
    "data/structures/large/records.json"
)


def test_load_large_crystallization_dataset():

    dataset = LargeCrystallizationDataset(
        DATA_FILE
    )

    df = dataset.load()

    assert len(df) > 0

    assert "query_pdb" in df.columns
    assert "sequence" in df.columns
    assert "condition_text" in df.columns
    assert "condition_parse_status" in df.columns


def test_large_dataset_contains_101M():

    dataset = LargeCrystallizationDataset(
        DATA_FILE
    )

    df = dataset.load()

    row = df[
        df["query_pdb"] == "101M"
    ].iloc[0]

    assert (
        row["condition_text"]
        == "3.0 M AMMONIUM SULFATE, 20 MM TRIS, 1MM EDTA, PH 9.0"
    )

    assert row["precipitant"] == "AMMONIUM SULFATE"
    assert row["buffer"] == "TRIS"
    assert row["pH"] == 9.0


def test_large_dataset_preserves_ml_targets():

    dataset = LargeCrystallizationDataset(
        DATA_FILE
    )

    df = dataset.load()

    row = df[
        df["query_pdb"] == "101M"
    ].iloc[0]

    assert row["matthews_coefficient"] == 3.09
    assert row["solvent_percent"] == 60.2


def test_unique_conditions():

    dataset = LargeCrystallizationDataset(
        DATA_FILE
    )

    df = dataset.load()

    unique = dataset.unique_conditions(df)

    assert len(unique) > 0
    assert (
        "condition_text"
        in unique.columns
    )

    assert (
        unique["condition_text"]
        .nunique()
        == len(unique)
    )