from operation_x.clients.pdb_client import PDBClient
from operation_x.parsers.mmcif_parser import MMCIFParser


def test_parse_crystallization_10af():

    client = PDBClient()
    parser = MMCIFParser()

    cif = client.download_mmcif("10AF")

    result = parser.parse_crystallization(cif)

    assert result.available is True

    assert result.method == (
        "VAPOR DIFFUSION, SITTING DROP"
    )

    assert result.pH == 8.5

    assert result.temperature_kelvin == 291

    assert result.details is not None

    assert "Ammonium sulfate" in result.details

    assert result.raw_conditions


def test_parse_crystallization_101m():

    client = PDBClient()
    parser = MMCIFParser()

    cif = client.download_mmcif("101M")

    result = parser.parse_crystallization(cif)

    assert result.available is True

    assert result.pH == 9.0

    assert result.temperature_kelvin is None

    assert result.details is not None

    assert "AMMONIUM SULFATE" in result.details