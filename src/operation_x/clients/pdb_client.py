import requests


class PDBClient:
    BASE_URL = "https://data.rcsb.org/rest/v1/core"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _get(self, endpoint: str) -> dict:
        response = requests.get(
            f"{self.BASE_URL}/{endpoint}",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def get_entry(self, pdb_id: str) -> dict:
        pdb_id = pdb_id.upper().strip()

        if not pdb_id:
            raise ValueError("PDB ID cannot be empty")

        return self._get(f"entry/{pdb_id}")

    def get_polymer_entity(
        self,
        pdb_id: str,
        entity_id: str,
    ) -> dict:
        pdb_id = pdb_id.upper().strip()
        entity_id = str(entity_id).strip()

        if not pdb_id:
            raise ValueError("PDB ID cannot be empty")

        if not entity_id:
            raise ValueError("Entity ID cannot be empty")

        return self._get(
            f"polymer_entity/{pdb_id}/{entity_id}"
        )

    def download_mmcif(self, pdb_id: str) -> str:
        pdb_id = pdb_id.upper().strip()

        if not pdb_id:
            raise ValueError("PDB ID cannot be empty")

        url = f"https://files.rcsb.org/download/{pdb_id}.cif"

        response = requests.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.text
