from operation_x.ml.large_crystallization_dataset import (
    LargeCrystallizationDataset,
)


def test_load_large_dataset():

    dataset = LargeCrystallizationDataset(
        "data/structures/large/records.json"
    )

    df = dataset.load()

    assert len(df) == 991


def test_temperature_data_is_present():

    dataset = LargeCrystallizationDataset(
        "data/structures/large/records.json"
    )

    df = dataset.load()

    temperatures = df["temperature_kelvin"].notna().sum()

    assert temperatures == 799


def test_required_columns_exist():

    dataset = LargeCrystallizationDataset(
        "data/structures/large/records.json"
    )

    df = dataset.load()

    required = [
        "query_pdb",
        "sequence",
        "condition_text",
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    for column in required:
        assert column in df.columns