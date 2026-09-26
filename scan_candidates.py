from operation_x.clients.pdb_client import PDBClient
from operation_x.dataset.eligibility import DatasetEligibility


client = PDBClient()
eligibility = DatasetEligibility()

pdb_ids = [
    "101M", "102L", "102M", "103L", "103M",
    "104L", "104M", "105M", "106M", "107L",
    "107M", "108L", "108M", "109L", "109M",
    "10AF", "10AH", "10AI", "10AJ", "10AK",
]

eligible = []
missing = []
errors = []

for pdb_id in pdb_ids:
    try:
        cif = client.download_mmcif(pdb_id)

        has_conditions = (
            eligibility.has_crystallization_conditions(cif)
        )

        if has_conditions:
            eligible.append(pdb_id)
            status = "ELIGIBLE"
        else:
            missing.append(pdb_id)
            status = "NO CONDITIONS"

        print(f"{pdb_id} -> {status}")

    except Exception as error:
        errors.append(pdb_id)
        print(f"{pdb_id} -> ERROR: {error}")


print("\n==============================")
print("DATASET DISCOVERY SUMMARY")
print("==============================")
print(f"Total candidates : {len(pdb_ids)}")
print(f"Eligible         : {len(eligible)}")
print(f"No conditions    : {len(missing)}")
print(f"Errors           : {len(errors)}")

print("\nEligible PDB IDs:")
print(eligible)

print("\nMissing crystallization conditions:")
print(missing)

print("\nErrors:")
print(errors)
