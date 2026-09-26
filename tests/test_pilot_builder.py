from operation_x.text_mining.pilot_builder import (
    PilotDatasetBuilder,
)


def test_pilot_builder_merges_mmcif_and_text_expression():
    builder = PilotDatasetBuilder()

    builder.client.get_entry = lambda pdb_id: {
        "rcsb_entry_container_identifiers": {
            "polymer_entity_ids": ["1"],
        },
        "rcsb_entry_info": {
            "experimental_method": "X-RAY DIFFRACTION",
            "molecular_weight": 10000,
            "resolution_combined": [2.0],
        },
        "citation": [
    	   {
       	      "rcsb_is_primary": "Y",
              "pdbx_database_id_PubMed": "12345678",
              "pdbx_database_id_DOI": "10.1234/test",
           }
       ],
    }

    builder.client.get_polymer_entity = (
        lambda pdb_id, entity_id: {
            "entity_poly": {
                "pdbx_seq_one_letter_code_can": (
                    "MKTAYIAK"
                ),
            },
            "rcsb_polymer_entity_container_identifiers": {
                "uniprot_ids": ["P00001"],
            },
            "rcsb_entity_source_organism": [
                {
                    "scientific_name": (
                        "Escherichia coli"
                    )
                }
            ],
        }
    )

    builder.client.download_mmcif = (
        lambda pdb_id: "fake cif"
    )

    class FakeCrystallization:
        available = True
        method = "VAPOR DIFFUSION"
        pH = 7.0
        temperature_kelvin = 293.0
        pressure = None
        time = "24 hours"
        details = "Test condition"
        matthews_coefficient = 2.5
        solvent_percent = 50.0

    class FakeExpressionMMCIF:
        host = "Escherichia coli"
        strain = "BL21(DE3)"
        system = "bacterial"
        evidence = [
            "mmCIF host: Escherichia coli BL21(DE3)"
        ]

    builder.mmcif_parser.parse_crystallization = (
        lambda cif: FakeCrystallization()
    )

    builder.expression_mmcif_parser.parse = (
        lambda cif: FakeExpressionMMCIF()
    )

    builder.article_fetcher.fetch_abstract = (
        lambda pubmed_id: (
            "Cells were induced with IPTG."
        )
    )

    builder.expression_extractor.extract = (
        lambda text: {
            "expression_host": None,
            "expression_strain": None,
            "expression_system": None,
            "inducer": "IPTG",
            "evidence": [
                "Cells were induced with IPTG."
            ],
        }
    )

    result = builder.build_record("10QH")

    assert result["expression_host"] == (
        "Escherichia coli"
    )

    assert result["expression_strain"] == (
        "BL21(DE3)"
    )

    assert result["expression_system"] == (
        "bacterial"
    )

    assert result["inducer"] == "IPTG"

    assert (
        "mmCIF host: Escherichia coli BL21(DE3)"
        in result["expression_evidence"]
    )

    assert (
        "Cells were induced with IPTG."
        in result["expression_evidence"]
    )
