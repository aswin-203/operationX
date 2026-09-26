import subprocess
from pathlib import Path


class StructureSimilarity:
    """Run Foldseek structural similarity searches."""

    def __init__(self, executable: str):
        self.executable = str(
            Path(executable).expanduser()
        )

    def version(self) -> str:
        result = subprocess.run(
            [
                self.executable,
                "version",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    def search(
        self,
        query_structure: str,
        target_database: str,
        output_file: str,
        max_hits: int = 10,
    ) -> list[dict]:

        query = Path(
            query_structure
        ).expanduser()

        database = Path(
            target_database
        ).expanduser()

        output = Path(
            output_file
        ).expanduser()

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp_dir = output.parent / (
            f".foldseek_tmp_{query.stem}"
        )

        tmp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            self.executable,
            "easy-search",
            str(query),
            str(database),
            str(output),
            str(tmp_dir),
            "--format-output",
            (
                "query,target,"
                "alntmscore,qtmscore,ttmscore,"
                "prob,rmsd,alnlen,evalue,bits"
            ),
            "--max-seqs",
            str(max_hits),
        ]

        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        if not output.exists():
            return []

        results = []

        for line in output.read_text().splitlines():

            if not line.strip():
                continue

            fields = line.split("\t")

            if len(fields) != 10:
                raise ValueError(
                    "Unexpected Foldseek output format"
                )

            (
                query_id,
                target_id,
                alignment_tm_score,
                query_tm_score,
                target_tm_score,
                probability,
                rmsd,
                alignment_length,
                evalue,
                bits,
            ) = fields

            results.append(
                {
                    "query": query_id,

                    "target": target_id,

                    # Raw alignment TM-score.
                    "alignment_tm_score": float(
                        alignment_tm_score
                    ),

                    # TM-score normalized by query.
                    "query_tm_score": float(
                        query_tm_score
                    ),

                    # TM-score normalized by target.
                    "target_tm_score": float(
                        target_tm_score
                    ),

                    "probability": float(
                        probability
                    ),

                    "rmsd": float(
                        rmsd
                    ),

                    "alignment_length": int(
                        alignment_length
                    ),

                    "evalue": float(
                        evalue
                    ),

                    "bits": float(
                        bits
                    ),
                }
            )

        return results