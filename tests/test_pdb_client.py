from operation_x.clients.pdb_client import PDBClient


def test_get_entry():
    client = PDBClient()

    data = client.get_entry("1CBS")

    assert isinstance(data, dict)
    assert data["rcsb_id"] == "1CBS"


def test_get_polymer_entity():
    client = PDBClient()

    data = client.get_polymer_entity("1CBS", "1")

    assert isinstance(data, dict)
    assert data["rcsb_id"] == "1CBS_1"
    assert data["entity_poly"]["type"] == "polypeptide(L)"


def test_download_mmcif():
    client = PDBClient()

    cif = client.download_mmcif("1CBS")

    assert isinstance(cif, str)
    assert len(cif) > 1000
    assert cif.startswith("data_1CBS")