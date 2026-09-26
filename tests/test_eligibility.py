from operation_x.clients.pdb_client import PDBClient
from operation_x.dataset.eligibility import DatasetEligibility


def test_1cbs_has_no_growth_conditions():

    client = PDBClient()

    cif = client.download_mmcif("1CBS")

    eligibility = DatasetEligibility()

    result = eligibility.has_crystallization_conditions(
        cif
    )

    assert result is False


def test_1cbs_crystallization_fields():

    client = PDBClient()

    cif = client.download_mmcif("1CBS")

    eligibility = DatasetEligibility()

    result = eligibility.get_crystallization_fields(
        cif
    )

    assert result["available"] is False
    assert result["fields"] == {}
