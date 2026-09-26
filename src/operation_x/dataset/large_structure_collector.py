
import json
import time
from pathlib import Path

from operation_x.clients.pdb_client import PDBClient
from operation_x.clients.rcsb_search_client import (
    RCSBSearchClient,
)
from operation_x.parsers.mmcif_parser import MMCIFParser
from operation_x.parsers.expression_mmcif_parser import (
    ExpressionMMCIFParser,
)


class LargeStructureCollector:
    """
    Collect a larger, resumable protein structure dataset from RCSB PDB.

    Workflow:

        RCSB search
            ↓
        PDB IDs
            ↓
        Entry metadata
            ↓
        Protein polymer entities
            ↓
        mmCIF download
            ↓
        Crystallization metadata
            ↓
        Expression metadata
            ↓
        JSON records

    The collector is intentionally resumable. Existing records are
    skipped on subsequent runs.
    """

    def __init__(
        self,
        output_directory: str = "data/structures/large",
        records_file: str = "data/structures/large/records.json",
        timeout: int = 30,
        delay: float = 0.1,
    ):
        self.output_directory = Path(
            output_directory
        ).expanduser()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records_file = Path(
            records_file
        ).expanduser()

        self.records_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.pdb_client = PDBClient(
            timeout=timeout
        )

        self.search_client = RCSBSearchClient(
            timeout=timeout
        )

        self.mmcif_parser = MMCIFParser()

        self.expression_mmcif_parser = (
            ExpressionMMCIFParser()
        )

        self.delay = delay

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def _load_records(self) -> dict:
        if not self.records_file.exists():
            return {}

        try:
            with self.records_file.open(
                "r",
                encoding="utf-8",
            ) as handle:

                data = json.load(
                    handle
                )

        except (
            json.JSONDecodeError,
            OSError,
        ):
            return {}

        if not isinstance(data, dict):
            return {}

        return data

    def _save_records(
        self,
        records: dict,
    ) -> None:

        temporary_file = (
            self.records_file.with_suffix(
                ".tmp"
            )
        )

        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                records,
                handle,
                indent=2,
                sort_keys=True,
            )

        temporary_file.replace(
            self.records_file
        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def discover_pdb_ids(
        self,
        limit: int = 100,
        batch_size: int = 100,
        start: int = 0,
    ) -> list[str]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        if start < 0:
            raise ValueError(
                "start must be zero or greater"
            )

        identifiers = []

        current_start = start

        while len(identifiers) < limit:

            rows = min(
                batch_size,
                limit - len(identifiers),
            )

            result_set = (
                self.search_client.search_protein_xray(
                    rows=rows,
                    start=current_start,
                )
            )

            if not result_set:
                break

            identifiers.extend(
                result_set
            )

            current_start += len(
                result_set
            )

            if len(result_set) < rows:
                break

        return identifiers[:limit]

    # ---------------------------------------------------------
    # Entity selection
    # ---------------------------------------------------------
    def _find_protein_entity(
        self,
        pdb_id: str,
        entry: dict,
    ) -> str | None:

        identifiers = entry.get(
            "rcsb_entry_container_identifiers",
            {},
        )

        entity_ids = identifiers.get(
            "polymer_entity_ids",
            [],
        )

        if not entity_ids:
            return None

        for entity_id in entity_ids:

            entity = (
                self.pdb_client.get_polymer_entity(
                    pdb_id,
                    entity_id,
                )
            )

            entity_poly = entity.get(
                "entity_poly",
                {},
            )

            polymer_type = entity_poly.get(
                "rcsb_entity_polymer_type"
            )

            if polymer_type == "Protein":
                return str(entity_id)

        return None
    #-------------------------------------------------
    # Record construction
    # ---------------------------------------------------------

    def collect_record(
        self,
        pdb_id: str,
    ) -> dict:

        pdb_id = (
            pdb_id.upper()
            .strip()
        )

        if not pdb_id:
            raise ValueError(
                "PDB ID cannot be empty"
            )

        entry = (
            self.pdb_client.get_entry(
                pdb_id
            )
        )

        entity_id = (
            self._find_protein_entity(
                pdb_id,
                entry,
            )
        )

        if entity_id is None:
            raise ValueError(
                f"No polymer entity found for {pdb_id}"
            )

        entity = (
            self.pdb_client.get_polymer_entity(
                pdb_id,
                entity_id,
            )
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

        sequence = "".join(
            sequence.split()
        )

        if not sequence:
            raise ValueError(
                f"No protein sequence found for {pdb_id}"
            )

        # ---------------------------------------------
        # Source organism
        # ---------------------------------------------

        source = entity.get(
            "rcsb_entity_source_organism",
            [],
        )

        source_organism = None

        if source:
            source_organism = source[0].get(
                "scientific_name"
            )

        # ---------------------------------------------
        # Container identifiers
        # ---------------------------------------------

        container = entity.get(
            "rcsb_polymer_entity_container_identifiers",
            {},
        )

        uniprot_ids = container.get(
            "uniprot_ids",
            [],
        )

        uniprot_id = (
            uniprot_ids[0]
            if uniprot_ids
            else None
        )

        # ---------------------------------------------
        # Experimental information
        # ---------------------------------------------

        entry_info = entry.get(
            "rcsb_entry_info",
            {},
        )

        # ---------------------------------------------
        # Citation
        # ---------------------------------------------

        citations = entry.get(
            "citation",
            [],
        )

        primary_citation = None

        for citation in citations:

            if citation.get(
                "rcsb_is_primary"
            ) == "Y":

                primary_citation = citation
                break

        if (
            primary_citation is None
            and citations
        ):
            primary_citation = citations[0]

        primary_citation = (
            primary_citation or {}
        )

        # ---------------------------------------------
        # mmCIF
        # ---------------------------------------------

        cif = (
            self.pdb_client.download_mmcif(
                pdb_id
            )
        )

        structure_path = (
            self.output_directory
            / f"{pdb_id}.cif"
        )

        structure_path.write_text(
            cif,
            encoding="utf-8",
        )

        # ---------------------------------------------
        # Crystallization
        # ---------------------------------------------

        crystallization = (
            self.mmcif_parser.parse_crystallization(
                cif
            )
        )

        # ---------------------------------------------
        # Expression
        # ---------------------------------------------

        expression = (
            self.expression_mmcif_parser.parse(
                cif
            )
        )

        return {
            "pdb_id": pdb_id,
            "entity_id": entity_id,
            "sequence": sequence,
            "sequence_length": len(
                sequence
            ),
            "uniprot_id": uniprot_id,
            "source_organism": source_organism,
            "experimental_method": (
                entry_info.get(
                    "experimental_method"
                )
            ),
            "molecular_weight": (
                entry_info.get(
                    "molecular_weight"
                )
            ),
            "resolution": (
                entry_info.get(
                    "resolution_combined"
                )
            ),
            "pubmed_id": (
                primary_citation.get(
                    "pdbx_database_id_PubMed"
                )
            ),
            "doi": (
                primary_citation.get(
                    "pdbx_database_id_DOI"
                )
            ),
            "crystallization_available": (
                crystallization.available
            ),
            "crystallization_method": (
                crystallization.method
            ),
            "pH": crystallization.pH,
            "temperature_kelvin": (
                crystallization.temperature_kelvin
            ),
            "pressure": crystallization.pressure,
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
            "expression_host": expression.host,
            "expression_strain": expression.strain,
            "expression_system": expression.system,
            "inducer": expression.inducer,
            "expression_evidence": expression.evidence,
            "structure_path": str(
                structure_path
            ),
        }

    # ---------------------------------------------------------
    # Dataset collection
    # ---------------------------------------------------------

    def collect(
        self,
        limit: int = 100,
        start: int = 0,
    ) -> dict:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        records = self._load_records()

        pdb_ids = self.discover_pdb_ids(
            limit=limit,
            start=start,
        )

        successful = 0
        failed = []

        print(
            "=" * 60
        )
        print(
            "OPERATION X — LARGE STRUCTURE DATASET"
        )
        print(
            "=" * 60
        )
        print()
        print(
            f"Requested: {limit}"
        )
        print(
            f"Discovered: {len(pdb_ids)}"
        )
        print()

        for index, pdb_id in enumerate(
            pdb_ids,
            start=1,
        ):

            print(
                f"[{index}/{len(pdb_ids)}] "
                f"{pdb_id}"
            )

            if pdb_id in records:

                print(
                    f"{pdb_id} -> "
                    "SKIPPED (already collected)"
                )
                print()

                successful += 1
                continue

            try:

                record = (
                    self.collect_record(
                        pdb_id
                    )
                )

                records[pdb_id] = record

                self._save_records(
                    records
                )

                successful += 1

                print(
                    f"{pdb_id} -> SUCCESS"
                )

            except Exception as error:

                failed.append(
                    {
                        "pdb_id": pdb_id,
                        "error": str(error),
                    }
                )

                print(
                    f"{pdb_id} -> ERROR: "
                    f"{error}"
                )

            print()

            time.sleep(
                self.delay
            )

        print(
            "=" * 60
        )
        print(
            "COLLECTION COMPLETE"
        )
        print(
            "=" * 60
        )
        print(
            f"Requested : {limit}"
        )
        print(
            f"Discovered: {len(pdb_ids)}"
        )
        print(
            f"Successful: {successful}"
        )
        print(
            f"Failed    : {len(failed)}"
        )
        print(
            f"Records   : {self.records_file}"
        )
        print(
            f"Structures: {self.output_directory}"
        )

        return {
            "requested": limit,
            "discovered": len(pdb_ids),
            "successful": successful,
            "failed": failed,
            "records_file": str(
                self.records_file
            ),
            "output_directory": str(
                self.output_directory
            ),
        }

