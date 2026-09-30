import json
import time
from pathlib import Path

from operation_x.clients.pdb_client import PDBClient
from operation_x.parsers.expression_mmcif_parser import (
    ExpressionMMCIFParser,
)
from operation_x.text_mining.article_fetcher import ArticleFetcher
from operation_x.text_mining.expression_extractor import (
    ExpressionExtractor,
)


class ExpressionEnricher:
    """Enrich existing PDB records with expression information."""

    def __init__(
        self,
        delay: float = 0.2,
    ):
        self.client = PDBClient()
        self.expression_mmcif_parser = (
            ExpressionMMCIFParser()
        )
        self.article_fetcher = ArticleFetcher()
        self.expression_extractor = ExpressionExtractor()
        self.delay = delay

    def enrich_record(self, record: dict) -> dict:
        """Add expression information to one PDB record."""

        pdb_id = record.get("pdb_id")

        enriched = dict(record)

        expression = {
            "expression_host": None,
            "expression_strain": None,
            "expression_system": None,
            "inducer": None,
            "expression_evidence": [],
        }

        # --------------------------------
        # 1. Get expression information
        # from mmCIF
        # --------------------------------

        try:
            cif = self.client.download_mmcif(
                pdb_id
            )

            mmcif_expression = (
                self.expression_mmcif_parser.parse(
                    cif
                )
            )

            expression["expression_host"] = (
                mmcif_expression.host
            )

            expression["expression_strain"] = (
                mmcif_expression.strain
            )

            expression["expression_system"] = (
                mmcif_expression.system
            )

            expression["expression_evidence"].extend(
                mmcif_expression.evidence
            )

        except Exception as error:
            print(
                f"{pdb_id} -> mmCIF expression failed: "
                f"{error}"
            )

        # --------------------------------
        # 2. Literature expression mining
        # --------------------------------

        pubmed_id = record.get("pubmed_id")

        if pubmed_id not in (
            None,
            -1,
            "-1",
            "",
        ):

            try:
                article_text = (
                    self.article_fetcher.fetch_abstract(
                        pubmed_id
                    )
                )

                text_expression = (
                    self.expression_extractor.extract(
                        article_text
                    )
                )

                # Literature fills missing metadata.
                if not expression["expression_host"]:
                    expression["expression_host"] = (
                        text_expression[
                            "expression_host"
                        ]
                    )

                if not expression["expression_strain"]:
                    expression["expression_strain"] = (
                        text_expression[
                            "expression_strain"
                        ]
                    )

                if not expression["expression_system"]:
                    expression["expression_system"] = (
                        text_expression[
                            "expression_system"
                        ]
                    )

                if text_expression["inducer"]:
                    expression["inducer"] = (
                        text_expression["inducer"]
                    )

                expression[
                    "expression_evidence"
                ].extend(
                    text_expression["evidence"]
                )

            except Exception as error:
                print(
                    f"{pdb_id} -> PubMed expression failed: "
                    f"{error}"
                )

        # --------------------------------
        # 3. Remove duplicate evidence
        # --------------------------------

        expression[
            "expression_evidence"
        ] = list(
            dict.fromkeys(
                expression["expression_evidence"]
            )
        )

        # --------------------------------
        # 4. Add expression fields
        # --------------------------------

        enriched.update(expression)

        return enriched

    def enrich_dataset(
        self,
        input_path: str,
        output_path: str,
        limit: int | None = None,
    ):
        """Enrich the PDB dataset with expression information."""

        input_file = Path(input_path)
        output_file = Path(output_path)

        with input_file.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        records = list(data.values())

        if limit is not None:
            records = records[:limit]

        enriched_records = {}

        total = len(records)

        for index, record in enumerate(
            records,
            start=1,
        ):

            pdb_id = record.get(
                "pdb_id"
            )

            print(
                f"[{index}/{total}] "
                f"Processing {pdb_id}"
            )

            try:
                enriched = self.enrich_record(
                    record
                )

                enriched_records[pdb_id] = (
                    enriched
                )

            except Exception as error:
                print(
                    f"{pdb_id} -> ERROR: "
                    f"{error}"
                )

                enriched_records[pdb_id] = record

            if self.delay > 0:
                time.sleep(self.delay)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                enriched_records,
                handle,
                indent=2,
            )

        print()
        print(
            f"Saved {len(enriched_records)} "
            f"records to {output_file}"
        )


if __name__ == "__main__":

    enricher = ExpressionEnricher(
        delay=0.2
    )

    enricher.enrich_dataset(
        input_path=(
            "data/structures/large/records.json"
        ),
        output_path=(
            "data/structures/large/"
            "records_with_expression.json"
        ),
        limit=20,
    )