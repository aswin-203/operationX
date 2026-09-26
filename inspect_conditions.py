from operation_x.clients.pdb_client import PDBClient
from operation_x.dataset.eligibility import DatasetEligibility


client = PDBClient()
eligibility = DatasetEligibility()

pdb_ids = [
    "101M", "102M", "103M", "104M",
    "105M", "106M", "107M", "108M",
    "109M", "10AF", "10AH", "10AI",
    "10AJ", "10AK",
]

for pdb_id in pdb_ids:

    print("\n" + "=" * 60)
    print(pdb_id)
    print("=" * 60)

    cif = client.download_mmcif(pdb_id)

    result = eligibility.get_crystallization_fields(cif)

    print("Available:", result["available"])

    fields = result["fields"]

    for field, values in fields.items():
        print(f"{field}: {values}")
