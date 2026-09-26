import re
from typing import Any

from operation_x.models.protein import ProteinRecord


class PDBParser:
    """Convert RCSB PDB responses into Operation X protein records."""

    def parse(
        self,
        entry_data: dict[str, Any],
        entity_data: dict[str, Any],
    ) -> ProteinRecord:

        pdb_id = entry_data["rcsb_id"]

        entity_id = entity_data["rcsb_polymer_entity"][
            "pdbx_description"
        ] if "pdbx_description" in entity_data.get(
            "rcsb_polymer_entity", {}
        ) else entity_data[
            "rcsb_polymer_entity_container_identifiers"
        ]["entity_id"]

        raw_sequence = entity_data["entity_poly"][
            "pdbx_seq_one_letter_code_can"
        ]

        sequence = self._clean_sequence(raw_sequence)

        entry_info = entry_data.get(
            "rcsb_entry_info",
            {},
        )

        identifiers = entity_data.get(
            "rcsb_polymer_entity_container_identifiers",
            {},
        )

        source_organism = self._get_organism(
            entity_data.get("rcsb_entity_source_organism", [])
        )

        expression_host = self._get_organism(
            entity_data.get("rcsb_entity_host_organism", [])
        )

        citation = entry_data.get("citation", [])

        primary_citation = next(
            (
                item
                for item in citation
                if item.get("id") == "primary"
            ),
            {},
        )

        return ProteinRecord(
            pdb_id=pdb_id,
            entity_id=str(
                identifiers.get("entity_id", "")
            ),
            title=entry_data.get("struct", {}).get("title"),
            sequence=sequence,
            sequence_length=len(sequence),
            molecular_weight=entry_info.get(
                "molecular_weight"
            ),
            experimental_method=entry_info.get(
                "experimental_method"
            ),
            chain_ids=identifiers.get(
                "asym_ids",
                [],
            ),
            source_organism=source_organism,
            expression_host=expression_host,
            uniprot_ids=identifiers.get(
                "uniprot_ids",
                [],
            ),
            pubmed_id=primary_citation.get(
                "pdbx_database_id_PubMed"
            ),
            doi=primary_citation.get(
                "pdbx_database_id_DOI"
            ),
        )

    @staticmethod
    def _clean_sequence(sequence: str) -> str:
        return re.sub(
            r"\s+",
            "",
            sequence,
        ).upper()

    @staticmethod
    def _get_organism(data: list[dict[str, Any]]) -> str | None:
        if not data:
            return None

        first = data[0]

        return (
            first.get("ncbi_scientific_name")
            or first.get("organism_scientific_name")
        )
