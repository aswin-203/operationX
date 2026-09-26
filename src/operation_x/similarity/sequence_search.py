import subprocess
import tempfile
from pathlib import Path


class SequenceSimilarity:
    """Run BLASTP sequence similarity searches."""

    def __init__(
        self,
        database: str,
        blastp_path: str = "blastp",
    ):
        self.database = database
        self.blastp_path = blastp_path

    def search(
        self,
        sequence: str,
        max_hits: int = 10,
    ) -> list[dict]:

        if not isinstance(sequence, str):
            raise TypeError(
                "Sequence must be a string"
            )

        sequence = "".join(
            sequence.split()
        ).upper()

        if not sequence:
            raise ValueError(
                "Protein sequence cannot be empty"
            )

        if max_hits <= 0:
            raise ValueError(
                "max_hits must be greater than zero"
            )

        with tempfile.TemporaryDirectory() as temp_dir:

            query_file = (
                Path(temp_dir) / "query.fasta"
            )

            query_file.write_text(
                f">operation_x_query\n"
                f"{sequence}\n"
            )

            command = [
                self.blastp_path,
                "-query",
                str(query_file),
                "-db",
                self.database,
                "-max_target_seqs",
                str(max_hits),
                "-outfmt",
                (
                    "6 qseqid sseqid pident length "
                    "mismatch gapopen qstart qend "
                    "sstart send evalue bitscore"
                ),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )

            return self._parse_results(
                result.stdout,
                len(sequence),
            )

    @staticmethod
    def _parse_results(
        output: str,
        query_length: int,
    ) -> list[dict]:

        if query_length <= 0:
            raise ValueError(
                "query_length must be greater than zero"
            )

        hits = []

        for line in output.splitlines():

            if not line.strip():
                continue

            fields = line.split("\t")

            if len(fields) != 12:
                raise ValueError(
                    "Unexpected BLAST output format"
                )

            (
                qseqid,
                sseqid,
                pident,
                length,
                mismatch,
                gapopen,
                qstart,
                qend,
                sstart,
                send,
                evalue,
                bitscore,
            ) = fields

            alignment_length = int(length)

            coverage = (
                alignment_length / query_length
            )

            hits.append(
                {
                    "query_id": qseqid,
                    "subject_id": sseqid,
                    "identity": float(pident),
                    "alignment_length": alignment_length,
                    "coverage": coverage,
                    "evalue": float(evalue),
                    "bitscore": float(bitscore),
                    "qstart": int(qstart),
                    "qend": int(qend),
                    "sstart": int(sstart),
                    "send": int(send),
                }
            )

        return hits
