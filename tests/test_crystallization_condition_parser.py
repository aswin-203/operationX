import pytest

from operation_x.ml.crystallization_condition_parser import (
    CrystallizationConditionParser,
)


def test_parse_ammonium_sulfate_condition():

    parser = CrystallizationConditionParser()

    result = parser.parse(
        "3.0 M AMMONIUM SULFATE, "
        "20 MM TRIS, "
        "1MM EDTA, "
        "PH 9.0"
    )

    assert result["precipitant"] == "AMMONIUM SULFATE"
    assert result["precipitant_concentration"] == 3.0
    assert result["precipitant_unit"] == "M"

    assert result["buffer"] == "TRIS"
    assert result["buffer_concentration"] == 20.0
    assert result["buffer_unit"].lower() == "mm"

    assert result["pH"] == 9.0

    assert len(result["additives"]) == 1
    assert result["additives"][0]["name"] == "EDTA"
    assert result["additives"][0]["concentration"] == 1.0


def test_parse_unbuffered_condition():

    parser = CrystallizationConditionParser()

    result = parser.parse(
        "3.0 M AMMONIUM SULFATE, UNBUFFERED, pH 7.0"
    )

    assert result["precipitant"] == "AMMONIUM SULFATE"
    assert result["precipitant_concentration"] == 3.0
    assert result["pH"] == 7.0
    assert result["buffer"] is None


def test_empty_condition_rejected():

    parser = CrystallizationConditionParser()

    with pytest.raises(ValueError):
        parser.parse("")


def test_non_string_condition_rejected():

    parser = CrystallizationConditionParser()

    with pytest.raises(TypeError):
        parser.parse(None)
