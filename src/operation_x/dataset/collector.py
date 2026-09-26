from operation_x.clients.pdb_client import PDBClient
from operation_x.dataset.eligibility import DatasetEligibility


class DatasetCollector:

    def __init__(self):
        self.client = PDBClient()
        self.eligibility = DatasetEligibility()

    def collect(self, pdb_ids: list[str]) -> dict:

        records = []
        rejected = []
        errors = []

        for pdb_id in pdb_ids:

            try:
                cif = self.client.download_mmcif(pdb_id)

                if self.eligibility.has_crystallization_conditions(
                    cif
                ):
                    records.append({
                        "pdb_id": pdb_id,
                        "cif": cif,
                    })
                else:
                    rejected.append(pdb_id)

            except Exception as error:

                errors.append({
                    "pdb_id": pdb_id,
                    "error": str(error),
                })

        return {
            "records": records,
            "rejected": rejected,
            "errors": errors,
        }
