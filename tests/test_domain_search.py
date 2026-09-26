from operation_x.similarity.domain_search import (
    DomainSimilarity,
)


def test_extract_pfam_domain():

    entity = {
        "rcsb_polymer_entity_annotation": [
            {
                "annotation_id": "PF00042",
                "name": "Globin (Globin)",
                "provenance_source": "Pfam",
                "type": "Pfam",
            }
        ],
        "rcsb_polymer_entity_feature": [
            {
                "feature_id": "PF00042",
                "name": "Globin (Globin)",
                "provenance_source": "Pfam",
                "type": "Pfam",
                "feature_positions": [
                    {
                        "beg_seq_id": 27,
                        "end_seq_id": 143,
                    }
                ],
            }
        ],
    }

    extractor = DomainSimilarity()

    domains = extractor.extract(entity)

    assert len(domains) == 1

    domain = domains[0]

    assert domain["domain_id"] == "PF00042"
    assert domain["domain_name"] == "Globin (Globin)"
    assert domain["domain_type"] == "Pfam"
    assert domain["provenance"] == "Pfam"
    assert domain["start"] == 27
    assert domain["end"] == 143


def test_extract_interpro_domain():

    entity = {
        "rcsb_polymer_entity_annotation": [
            {
                "annotation_id": "IPR002335",
                "name": "Myoglobin",
                "provenance_source": "UNIPROT",
                "type": "InterPro",
            }
        ],
        "rcsb_polymer_entity_feature": [],
    }

    extractor = DomainSimilarity()

    domains = extractor.extract(entity)

    assert len(domains) == 1

    domain = domains[0]

    assert domain["domain_id"] == "IPR002335"
    assert domain["domain_name"] == "Myoglobin"
    assert domain["domain_type"] == "InterPro"
    assert domain["start"] is None
    assert domain["end"] is None
