from operation_x.ml.crystallization_condition_dataset import (
    CrystallizationConditionDataset,
)


DATA_FILE = (
    "data/processed/"
    "operation_x_similarity_conditions.csv"
)


def test_condition_dataset_loads():

    dataset = CrystallizationConditionDataset(
        DATA_FILE
    )

    df = dataset.load()

    assert len(df) == 111

    assert "condition_id" in df.columns
    assert "condition_text" in df.columns
    assert "condition_parse_status" in df.columns


def test_condition_parse_status():

    dataset = CrystallizationConditionDataset(
        DATA_FILE
    )

    df = dataset.load()

    status_counts = (
        df["condition_parse_status"]
        .value_counts()
        .to_dict()
    )

    assert status_counts["parsed"] == 92
    assert status_counts["unparsed"] == 19


def test_condition_count():

    dataset = CrystallizationConditionDataset(
        DATA_FILE
    )

    df = dataset.load()

    assert df["condition_id"].nunique() == 7


def test_hdac6_condition_is_structured():

    dataset = CrystallizationConditionDataset(
        DATA_FILE
    )

    df = dataset.load()

    hdac6 = df[
        df["condition_text"].str.contains(
            "HDAC6",
            case=False,
            na=False,
        )
    ]

    assert len(hdac6) == 4

    assert (
        hdac6["precipitant"]
        .notna()
        .all()
    )

    assert (
        hdac6["protein_concentration"]
        .eq(10.0)
        .all()
    )


def test_proteros_condition_is_unparsed():

    dataset = CrystallizationConditionDataset(
        DATA_FILE
    )

    df = dataset.load()

    proteros = df[
        df["condition_text"].str.contains(
            "PROTEROS",
            case=False,
            na=False,
        )
    ]

    assert len(proteros) == 19

    assert (
        proteros["condition_parse_status"]
        .eq("unparsed")
        .all()
    )
