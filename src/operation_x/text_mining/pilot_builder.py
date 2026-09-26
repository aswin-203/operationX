
from operation_x.clients.pdb_client import PDBClient
from operation_x.parsers.expression_mmcif_parser import (
    ExpressionMMCIFParser,
)
from operation_x.parsers.mmcif_parser import MMCIFParser
from operation_x.text_mining.article_fetcher import ArticleFetcher
from operation_x.text_mining.expression_extractor import (
    ExpressionExtractor,
)


class PilotDatasetBuilder:

    def __init__(self):
        self.client = PDBClient()
        self.mmcif_parser = MMCIFParser()
        self.expression_mmcif_parser = (
            ExpressionMMCIFParser()
        )
        self.article_fetcher = ArticleFetcher()
        self.expression_extractor = ExpressionExtractor()

    def build_record(self, pdb_id: str) -> dict:

        # --------------------------------
        # 1. Get PDB entry
        # --------------------------------

        entry = self.client.get_entry(pdb_id)

        identifiers = entry.get(
            "rcsb_entry_container_identifiers",
            {},
        )

        polymer_ids = identifiers.get(
            "polymer_entity_ids",
            [],
        )

        if not polymer_ids:
            raise ValueError(
                f"No polymer entity found for {pdb_id}"
            )

        entity_id = polymer_ids[0]

        # --------------------------------
        # 2. Get polymer entity
        # --------------------------------

        entity = self.client.get_polymer_entity(
            pdb_id,
            entity_id,
        )

        entity_poly = entity.get(
            "entity_poly",
            {},
        )

        sequence = (
            entity_poly.get(
                "pdbx_seq_one_letter_code_can"
            )
            or entity_poly.get(
                "pdbx_seq_one_letter_code"
            )
            or ""
        )

        container = entity.get(
            "rcsb_polymer_entity_container_identifiers",
            {},
        )

        # --------------------------------
        # 3. Source organism
        # --------------------------------

        source = entity.get(
            "rcsb_entity_source_organism",
            [],
        )

        source_organism = None

        if source:
            source_organism = source[0].get(
                "scientific_name"
            )

        # --------------------------------
        # 4. Download mmCIF
        # --------------------------------

        cif = self.client.download_mmcif(
            pdb_id
        )

        crystallization = (
            self.mmcif_parser.parse_crystallization(
                cif
            )
        )

        expression_mmcif = (
            self.expression_mmcif_parser.parse(
                cif
            )
        )

        # --------------------------------
        # 5. RCSB entry information
        # --------------------------------

        info = entry.get(
            "rcsb_entry_info",
            {},
        )

        # --------------------------------
        # 6. Primary citation
        # --------------------------------

        citation = entry.get(
            "citation",
            [],
        )

        primary_citation = None

        for item in citation:

            if item.get(
                "rcsb_is_primary"
            ) == "Y":

                primary_citation = item
                break

        if primary_citation is None and citation:
            primary_citation = citation[0]

        primary_citation = (
            primary_citation or {}
        )

        # --------------------------------
        # 7. Expression text mining
        # --------------------------------

        pubmed_id = primary_citation.get(
            "pdbx_database_id_PubMed"
        )

        expression_text = {
            "expression_host": None,
            "expression_strain": None,
            "expression_system": None,
            "inducer": None,
            "evidence": [],
        }

        if pubmed_id:

            try:

                article_text = (
                    self.article_fetcher.fetch_abstract(
                        pubmed_id
                    )
                )

                expression_text = (
                    self.expression_extractor.extract(
                        article_text
                    )
                )

            except Exception as error:

                print(
                    f"{pdb_id} -> "
                    f"expression text extraction failed: "
                    f"{error}"
                )

        # --------------------------------
        # 8. Merge expression sources
        # --------------------------------

        expression_host = (
            expression_mmcif.host
            or expression_text[
                "expression_host"
            ]
        )

        expression_strain = (
            expression_mmcif.strain
            or expression_text[
                "expression_strain"
            ]
        )

        expression_system = (
            expression_mmcif.system
            or expression_text[
                "expression_system"
            ]
        )

        inducer = (
            expression_text["inducer"]
        )

        expression_evidence = []

        expression_evidence.extend(
            expression_mmcif.evidence
        )

        expression_evidence.extend(
            expression_text["evidence"]
        )

        # --------------------------------
        # 9. Build final dataset record
        # --------------------------------

        return {

            "pdb_id": pdb_id,

            "entity_id": entity_id,

            "sequence": sequence,

            "sequence_length": len(
                sequence
            ),

            "uniprot_id": (
                container.get(
                    "uniprot_ids",
                    [None],
                )[0]
            ),

            "source_organism": (
                source_organism
            ),

            "experimental_method": (
                info.get(
                    "experimental_method"
                )
            ),

            "molecular_weight": (
                info.get(
                    "molecular_weight"
                )
            ),

            "resolution": (
                info.get(
                    "resolution_combined"
                )
            ),

            "pubmed_id": pubmed_id,

            "doi": (
                primary_citation.get(
                    "pdbx_database_id_DOI"
                )
            ),

            # -------------------------
            # Crystallization
            # -------------------------

            "crystallization_available": (
                crystallization.available
            ),

            "crystallization_method": (
                crystallization.method
            ),

            "pH": (
                crystallization.pH
            ),

            "temperature_kelvin": (
                crystallization.temperature_kelvin
            ),

            "pressure": (
                crystallization.pressure
            ),

            "crystallization_time": (
                crystallization.time
            ),

            "crystallization_details": (
                crystallization.details
            ),

            "matthews_coefficient": (
                crystallization.matthews_coefficient
            ),

            "solvent_percent": (
                crystallization.solvent_percent
            ),

            # -------------------------
            # Expression
            # -------------------------

            "expression_host": (
                expression_host
            ),

            "expression_strain": (
                expression_strain
            ),

            "expression_system": (
                expression_system
            ),

            "inducer": (
                inducer
            ),

            "expression_evidence": (
                " | ".join(
                    expression_evidence
                )
            ),
        }

    def build(
        self,
        pdb_ids: list[str],
    ) -> list[dict]:

        records = []

        for pdb_id in pdb_ids:

            try:

                record = (
                    self.build_record(
                        pdb_id
                    )
                )

                records.append(record)

                print(
                    f"{pdb_id} -> SUCCESS"
                )

            except Exception as error:

                print(
                    f"{pdb_id} -> ERROR: "
                    f"{error}"
                )

        return records

