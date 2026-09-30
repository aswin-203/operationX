import json
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
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
    Fast, resumable protein structure dataset collector.

    Workflow:

        RCSB search
            ↓
        PDB IDs
            ↓
        Skip existing records
            ↓
        Parallel record collection
            ↓
        mmCIF download
            ↓
        Crystallization metadata
            ↓
        Expression metadata
            ↓
        Batch save to records.json

    Existing records are skipped automatically.

    Collection uses multiple worker threads so that network-bound
    RCSB API requests and mmCIF downloads can happen concurrently.
    """

    def __init__(
        self,
        output_directory: str = "data/structures/large",
        records_file: str = "data/structures/large/records.json",
        timeout: int = 30,
        delay: float = 0.1,
        max_workers: int = 8,
        save_every: int = 25,
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

        self.timeout = timeout
        self.delay = delay
        self.max_workers = max_workers
        self.save_every = save_every

        # -----------------------------------------------------
        # Clients
        # -----------------------------------------------------

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

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def _load_records(self) -> dict:
        """
        Load existing records from records.json.
        """

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
        """
        Atomically save records.json.
        """

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
        """
        Discover protein X-ray PDB IDs from RCSB.
        """

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
        """
        Find the first protein polymer entity.
        """

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

    # ---------------------------------------------------------
    # Record construction
    # ---------------------------------------------------------

    def collect_record(
        self,
        pdb_id: str,
    ) -> dict:
        """
        Collect one complete PDB record.

        Includes:

        - sequence
        - source organism
        - UniProt
        - experimental method
        - molecular weight
        - resolution
        - publication information
        - crystallization conditions
        - pH
        - temperature
        - pressure
        - Matthews coefficient
        - solvent percentage
        - expression host
        - expression strain
        - expression system
        - inducer
        - expression evidence
        - local CIF path
        """

        pdb_id = (
            pdb_id.upper()
            .strip()
        )

        if not pdb_id:
            raise ValueError(
                "PDB ID cannot be empty"
            )

        # -----------------------------------------------------
        # Entry metadata
        # -----------------------------------------------------

        entry = (
            self.pdb_client.get_entry(
                pdb_id
            )
        )

        # -----------------------------------------------------
        # Protein entity
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Sequence
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Source organism
        # -----------------------------------------------------

        source = entity.get(
            "rcsb_entity_source_organism",
            [],
        )

        source_organism = None

        if source:

            source_organism = (
                source[0].get(
                    "scientific_name"
                )
            )

        # -----------------------------------------------------
        # Container identifiers
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Experimental information
        # -----------------------------------------------------

        entry_info = entry.get(
            "rcsb_entry_info",
            {},
        )

        # -----------------------------------------------------
        # Citation
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Download mmCIF
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Crystallization
        # -----------------------------------------------------

        crystallization = (
            self.mmcif_parser.parse_crystallization(
                cif
            )
        )

        # -----------------------------------------------------
        # Expression
        # -----------------------------------------------------

        expression = (
            self.expression_mmcif_parser.parse(
                cif
            )
        )

        # -----------------------------------------------------
        # Final record
        # -----------------------------------------------------

        return {
            "pdb_id": pdb_id,

            "entity_id": entity_id,

            "sequence": sequence,

            "sequence_length": len(
                sequence
            ),

            "uniprot_id": uniprot_id,

            "source_organism": (
                source_organism
            ),

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

            # ---------------------------------------------
            # Crystallization
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Expression
            # ---------------------------------------------

            "expression_host": (
                expression.host
            ),

            "expression_strain": (
                expression.strain
            ),

            "expression_system": (
                expression.system
            ),

            "inducer": (
                expression.inducer
            ),

            "expression_evidence": (
                expression.evidence
            ),

            # ---------------------------------------------
            # Structure
            # ---------------------------------------------

            "structure_path": str(
                structure_path
            ),
        }

    # ---------------------------------------------------------
    # Worker
    # ---------------------------------------------------------

    def _collect_one(
        self,
        pdb_id: str,
    ) -> tuple[str, dict | None, str | None]:
        """
        Worker function used by ThreadPoolExecutor.

        Returns:

            (pdb_id, record, error)

        """

        try:

            if self.delay > 0:
                time.sleep(
                    self.delay
                )

            record = (
                self.collect_record(
                    pdb_id
                )
            )

            return (
                pdb_id,
                record,
                None,
            )

        except Exception as error:

            return (
                pdb_id,
                None,
                str(error),
            )

    # ---------------------------------------------------------
    # Dataset collection
    # ---------------------------------------------------------

    def collect(
        self,
        limit: int = 100,
        start: int = 0,
    ) -> dict:
        """
        Collect a requested number of NEW records.

        Existing records are skipped.

        Collection is performed in parallel using
        ThreadPoolExecutor.
        """

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        if self.max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than zero"
            )

        if self.save_every <= 0:
            raise ValueError(
                "save_every must be greater than zero"
            )

        records = self._load_records()

        successful = 0
        failed = []

        current_start = start
        discovered_total = 0

        print()
        print(
            "=" * 60
        )
        print(
            "OPERATION X — FAST LARGE DATASET COLLECTION"
        )
        print(
            "=" * 60
        )
        print()
        print(
            f"Requested new records : {limit}"
        )
        print(
            f"Existing records      : {len(records)}"
        )
        print(
            f"Worker threads        : {self.max_workers}"
        )
        print(
            f"Save every            : {self.save_every}"
        )
        print()

        # -----------------------------------------------------
        # Thread pool
        # -----------------------------------------------------

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            while successful < limit:

                # ---------------------------------------------
                # Discover a reasonably large page
                # ---------------------------------------------

                remaining = (
                    limit - successful
                )

                discovery_size = max(
                    100,
                    min(
                        500,
                        remaining * 3,
                    ),
                )

                pdb_ids = (
                    self.discover_pdb_ids(
                        limit=discovery_size,
                        start=current_start,
                    )
                )

                if not pdb_ids:

                    print()
                    print(
                        "No more PDB IDs found."
                    )

                    break

                discovered_total += len(
                    pdb_ids
                )

                # Move pagination forward immediately.
                current_start += len(
                    pdb_ids
                )

                # ---------------------------------------------
                # Remove already collected IDs
                # ---------------------------------------------

                new_ids = []

                for pdb_id in pdb_ids:

                    if pdb_id in records:

                        print(
                            f"{pdb_id} -> "
                            "SKIPPED"
                        )

                        continue

                    new_ids.append(
                        pdb_id
                    )

                    if len(new_ids) >= remaining:
                        break

                if not new_ids:
                    continue

                print()
                print(
                    f"Submitting {len(new_ids)} "
                    "new records..."
                )
                print()

                # ---------------------------------------------
                # Submit parallel jobs
                # ---------------------------------------------

                futures = {
                    executor.submit(
                        self._collect_one,
                        pdb_id,
                    ): pdb_id
                    for pdb_id in new_ids
                }

                batch_success = 0

                # ---------------------------------------------
                # Process completed jobs
                # ---------------------------------------------

                for future in as_completed(
                    futures
                ):

                    pdb_id = futures[
                        future
                    ]

                    try:

                        (
                            result_pdb_id,
                            record,
                            error,
                        ) = future.result()

                    except Exception as error:

                        result_pdb_id = pdb_id
                        record = None
                        error = str(error)

                    if (
                        record is not None
                        and error is None
                    ):

                        records[
                            result_pdb_id
                        ] = record

                        successful += 1
                        batch_success += 1

                        print(
                            f"[{successful}/{limit}] "
                            f"{result_pdb_id} -> SUCCESS"
                        )

                    else:

                        failed.append(
                            {
                                "pdb_id": result_pdb_id,
                                "error": error,
                            }
                        )

                        print(
                            f"{result_pdb_id} -> "
                            f"ERROR: {error}"
                        )

                    # -----------------------------------------
                    # Batch save
                    # -----------------------------------------

                    if (
                        batch_success > 0
                        and (
                            batch_success
                            % self.save_every
                            == 0
                        )
                    ):

                        self._save_records(
                            records
                        )

                        print(
                            f"Saved checkpoint: "
                            f"{len(records)} records"
                        )

                # ---------------------------------------------
                # Save after every submitted batch
                # ---------------------------------------------

                self._save_records(
                    records
                )

                print()
                print(
                    "Batch complete:"
                )
                print(
                    f"  New records: {batch_success}"
                )
                print(
                    f"  Total records: {len(records)}"
                )
                print(
                    f"  Successful: {successful}/{limit}"
                )
                print(
                    f"  Failed: {len(failed)}"
                )
                print()

        # -----------------------------------------------------
        # Final save
        # -----------------------------------------------------

        self._save_records(
            records
        )

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        print()
        print(
            "=" * 60
        )
        print(
            "COLLECTION COMPLETE"
        )
        print(
            "=" * 60
        )
        print()

        print(
            f"Requested new records : {limit}"
        )

        print(
            f"Discovered             : "
            f"{discovered_total}"
        )

        print(
            f"Successful new records : "
            f"{successful}"
        )

        print(
            f"Failed                 : "
            f"{len(failed)}"
        )

        print(
            f"Total records          : "
            f"{len(records)}"
        )

        print(
            f"Records file           : "
            f"{self.records_file}"
        )

        print(
            f"Structures directory   : "
            f"{self.output_directory}"
        )

        print()

        return {
            "requested": limit,
            "discovered": discovered_total,
            "successful": successful,
            "failed": failed,
            "records_file": str(
                self.records_file
            ),
            "output_directory": str(
                self.output_directory
            ),
        }