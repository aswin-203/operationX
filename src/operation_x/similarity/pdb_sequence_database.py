import csv
import subprocess
from pathlib import Path


class PDBSequenceDatabase:
    """Build a BLAST protein database from Operation X records."""

    def __init__(
        self,
        blast_dir: str = "data/blast/pilot",
    ):
        self.blast_dir = Path(blast_dir)

    def create_fasta(
        self,
        csv_path: str,
    ) -> Path:

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        self.blast_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fasta_path = (
            self.blast_dir / "pdb_sequences.fasta"
        )

        records = 0

        with (
            csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file,
            fasta_path.open(
                "w",
                encoding="utf-8",
            ) as fasta_file,
        ):

            reader = csv.DictReader(csv_file)

            required_columns = {
                "pdb_id",
                "entity_id",
                "sequence",
            }

            missing = (
                required_columns
                - set(reader.fieldnames or [])
            )

            if missing:
                raise ValueError(
                    "Missing required columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:

                pdb_id = (
                    row["pdb_id"]
                    .strip()
                    .upper()
                )

                entity_id = (
                    row["entity_id"]
                    .strip()
                )

                sequence = (
                    row["sequence"]
                    .strip()
                    .replace(" ", "")
                    .replace("\n", "")
                )

                if not pdb_id:
                    continue

                if not entity_id:
                    continue

                if not sequence:
                    continue

                subject_id = (
                    f"{pdb_id}_{entity_id}"
                )

                fasta_file.write(
                    f">{subject_id}\n"
                )

                # Wrap sequence at 80 characters.
                for start in range(
                    0,
                    len(sequence),
                    80,
                ):
                    fasta_file.write(
                        sequence[
                            start:start + 80
                        ]
                        + "\n"
                    )

                records += 1

        if records == 0:
            raise ValueError(
                "No valid protein sequences found"
            )

        return fasta_path

    def build_database(
        self,
        fasta_path: Path,
    ) -> Path:

        fasta_path = Path(fasta_path)

        if not fasta_path.exists():
            raise FileNotFoundError(
                f"FASTA not found: {fasta_path}"
            )

        database_prefix = (
            self.blast_dir
            / "pdb_sequences"
        )

        command = [
            "makeblastdb",
            "-in",
            str(fasta_path),
            "-dbtype",
            "prot",
            "-parse_seqids",
            "-out",
            str(database_prefix),
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        return database_prefix

    def build(
        self,
        csv_path: str,
    ) -> Path:

        fasta_path = self.create_fasta(
            csv_path
        )

        return self.build_database(
            fasta_path
        )
