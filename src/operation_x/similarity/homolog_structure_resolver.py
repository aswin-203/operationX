from pathlib import Path


class HomologStructureResolver:

    def __init__(self, structure_directory):
        self.structure_directory = Path(structure_directory)

    @staticmethod
    def parse_pdb_id(subject_id):
        if not isinstance(subject_id, str):
            raise TypeError("subject_id must be a string")

        subject_id = subject_id.strip()

        if subject_id.startswith("pdb|"):
            parts = subject_id.split("|")

            if len(parts) >= 2:
                return parts[1].upper()

        if "_" in subject_id:
            return subject_id.split("_", 1)[0].upper()

        return subject_id.upper()

    def resolve(self, subject_id):
        pdb_id = self.parse_pdb_id(subject_id)

        structure_path = (
            self.structure_directory / f"{pdb_id}.cif"
        )

        if not structure_path.exists():
            return None

        return str(structure_path)

    def resolve_hits(self, hits):
        results = []

        for hit in hits:
            subject_id = hit.get("subject_id")

            if not subject_id:
                continue

            structure_path = self.resolve(subject_id)

            if structure_path is None:
                continue

            results.append({
                "subject_id": subject_id,
                "pdb_id": self.parse_pdb_id(subject_id),
                "structure_path": structure_path,
                "identity": hit.get("identity"),
                "coverage": hit.get("coverage"),
                "evalue": hit.get("evalue"),
                "bitscore": hit.get("bitscore"),
            })

        return results
