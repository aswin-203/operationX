from operation_x.similarity.similarity_pipeline import (
    SimilarityPipeline,
)


def test_similarity_pipeline_combines_evidence():

    class FakeSequenceSearch:

        def search(self, sequence, max_hits=10):
            return [
                {
                    "subject_id": "pdb|101M|1",
                    "identity": 100.0,
                    "coverage": 1.0,
                    "evalue": 1e-100,
                    "bitscore": 300.0,
                },
                {
                    "subject_id": "pdb|102M|1",
                    "identity": 95.0,
                    "coverage": 0.95,
                    "evalue": 1e-50,
                    "bitscore": 250.0,
                },
            ]


    class FakeStructureSearch:

        def search(
            self,
            query_structure,
            target_database,
            output_file,
            max_hits=10,
        ):
            return [
                {
                    "target": "101M",
                    "query_tm_score": 1.0,
                    "target_tm_score": 1.0,
                    "rmsd": 0.0,
                    "alignment_length": 150,
                },
                {
                    "target": "102M",
                    "query_tm_score": 0.95,
                    "target_tm_score": 0.95,
                    "rmsd": 0.5,
                    "alignment_length": 100,
                },
            ]


    class FakeDomainSimilarity:

        def extract(self, entity):
            return [
                {
                    "domain_id": "PF00042",
                    "domain_name": "Globin",
                    "domain_type": "Pfam",
                    "provenance": "Pfam",
                    "start": 27,
                    "end": 143,
                }
            ]


    pipeline = SimilarityPipeline(
        sequence_search=FakeSequenceSearch(),
        structure_search=FakeStructureSearch(),
        domain_similarity=FakeDomainSimilarity(),
        structure_database="data/foldseek/pilot/structures",
    )


    result = pipeline.analyze(
        pdb_id="101M",
        sequence="MVHLTPEEKS",
        entity={},
    )


    assert result["pdb_id"] == "101M"


    # --------------------------------------------------
    # Sequence self-hit protection
    # --------------------------------------------------

    # The self-hit 101M must be removed.
    # The different homolog 102M must remain.

    assert len(result["sequence_hits"]) == 1

    assert (
        result["sequence_hits"][0]["subject_id"]
        == "pdb|102M|1"
    )

    assert (
        result["sequence_hits"][0]["identity"]
        == 95.0
    )

    assert (
        result["sequence_hits"][0]["coverage"]
        == 0.95
    )


    # --------------------------------------------------
    # Homolog structure resolution
    # --------------------------------------------------

    assert len(result["homolog_structures"]) == 1

    assert (
        result["homolog_structures"][0]["pdb_id"]
        == "102M"
    )


    # --------------------------------------------------
    # Structural self-hit protection
    # --------------------------------------------------

    # The self-hit 101M must be removed.
    # The different structural hit 102M must remain.

    assert len(result["structure_hits"]) == 1

    assert (
        result["structure_hits"][0]["target"]
        == "102M"
    )


    # --------------------------------------------------
    # Domain evidence
    # --------------------------------------------------

    assert len(result["domains"]) == 1

    assert (
        result["domains"][0]["domain_id"]
        == "PF00042"
    )


    # --------------------------------------------------
    # Evidence fusion
    # --------------------------------------------------

    assert "evidence" in result

    assert result["evidence"] is not None


def test_similarity_pipeline_handles_empty_sources():

    class EmptySequenceSearch:

        def search(self, sequence, max_hits=10):
            return []


    class EmptyStructureSearch:

        def search(
            self,
            query_structure,
            target_database,
            output_file,
            max_hits=10,
        ):
            return []


    class EmptyDomainSimilarity:

        def extract(self, entity):
            return []


    pipeline = SimilarityPipeline(
        sequence_search=EmptySequenceSearch(),
        structure_search=EmptyStructureSearch(),
        domain_similarity=EmptyDomainSimilarity(),
        structure_database="data/foldseek/pilot/structures",
    )


    result = pipeline.analyze(
        pdb_id="TEST",
        sequence="MVHLTPEEKS",
        entity={},
    )


    assert result["pdb_id"] == "TEST"

    assert result["sequence_hits"] == []

    assert result["homolog_structures"] == []

    assert result["structure_hits"] == []

    assert result["domains"] == []

    assert "evidence" in result

    assert result["evidence"] is not None