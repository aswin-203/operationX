from pathlib import Path

from operation_x.similarity.evidence_fusion import (
    EvidenceFusion,
)
from operation_x.similarity.homolog_structure_resolver import (
    HomologStructureResolver,
)


class SimilarityPipeline:
    """
    Sequence-first Operation X similarity workflow.

    Workflow:
        Sequence -> BLAST
        BLAST hits -> known homolog structures
        Homolog structures -> structural evidence
        Entity -> Pfam/InterPro
        Evidence -> Fusion
    """

    def __init__(
        self,
        sequence_search,
        structure_search,
        domain_similarity,
        structure_database: str,
        homolog_structure_directory: str = "data/structures/large",
        output_directory: str = "data/similarity",
    ):
        self.sequence_search = sequence_search
        self.structure_search = structure_search
        self.domain_similarity = domain_similarity

        self.evidence_fusion = EvidenceFusion()

        self.structure_database = str(
            Path(structure_database).expanduser()
        )

        self.homolog_structure_resolver = (
            HomologStructureResolver(
                homolog_structure_directory
            )
        )

        self.output_directory = Path(
            output_directory
        ).expanduser()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _is_same_pdb(
            subject_id: str,
            query_pdb_id: str,
        ) -> bool:

        subject = str(
            subject_id or ""
        ).upper().strip()

        query = str(
            query_pdb_id or ""
        ).upper().strip()

        if "|" in subject:
            parts = subject.split("|")

            if len(parts) > 1:
                subject = parts[1]

        if "_" in subject:
            subject = subject.split(
                "_",
                1,
            )[0]

        return subject == query

    def analyze(
        self,
        pdb_id: str,
        sequence: str,
        entity: dict,
        max_hits: int = 10,
    ) -> dict:

        # ---------------------------------
        # 1. BLAST sequence similarity
        # ---------------------------------

        sequence_hits = (
            self.sequence_search.search(
                sequence,
                max_hits=max_hits,
            )
        )
        sequence_hits = [
            hit
            for hit in sequence_hits
            if not self._is_same_pdb(
                hit.get("subject_id"),
                pdb_id,
            )
        ]

        # ---------------------------------
        # 2. Resolve BLAST hits to known
        #    experimental structures
        # ---------------------------------

        homolog_structures = (
            self.homolog_structure_resolver.resolve_hits(
                sequence_hits
            )
        )

        # ---------------------------------
        # 3. Structural evidence from
        #    known homolog structures
        # ---------------------------------

        structure_hits = []

        for homolog in homolog_structures:

            homolog_pdb = homolog["pdb_id"]
            structure_path = homolog["structure_path"]

            structure_file = (
                self.output_directory
                / f"{pdb_id}_{homolog_pdb}_foldseek.tsv"
            )

            hits = self.structure_search.search(
                query_structure=structure_path,
                target_database=self.structure_database,
                output_file=str(structure_file),
                max_hits=max_hits,
            )

            hits = [
                hit
                for hit in hits
                if not self._is_same_pdb(
                    hit.get("target"),
                    pdb_id,
                )
            ]

            structure_hits.extend(hits)

        # ---------------------------------
        # 4. Domain annotations
        # ---------------------------------

        domains = (
            self.domain_similarity.extract(
                entity
            )
        )

        # ---------------------------------
        # 5. Evidence fusion
        # ---------------------------------

        evidence = (
            self.evidence_fusion.summarize(
                sequence_hits=sequence_hits,
                structure_hits=structure_hits,
                domains=domains,
            )
        )

        # ---------------------------------
        # 6. Final result
        # ---------------------------------

        return {
            "pdb_id": pdb_id,

            "sequence_hits": sequence_hits,

            "homolog_structures": (
                homolog_structures
            ),

            "structure_hits": structure_hits,

            "domains": domains,

            "evidence": evidence,
        }
