from pathlib import Path
from unittest.mock import Mock, patch

from operation_x.dataset.large_structure_collector import (
    LargeStructureCollector,
)


def test_collector_creates_output_directory(
    tmp_path,
):
    output = tmp_path / "structures"

    records = tmp_path / "records.json"

    collector = LargeStructureCollector(
        output_directory=str(output),
        records_file=str(records),
    )

    assert output.exists()
    assert collector.output_directory == output


def test_collector_loads_missing_records(
    tmp_path,
):
    collector = LargeStructureCollector(
        output_directory=str(
            tmp_path / "structures"
        ),
        records_file=str(
            tmp_path / "records.json"
        ),
    )

    records = collector._load_records()

    assert records == {}


def test_collector_includes_expression_metadata(
    tmp_path,
):
    collector = LargeStructureCollector(
        output_directory=str(
            tmp_path / "structures"
        ),
        records_file=str(
            tmp_path / "records.json"
        ),
    )

    entry = {
        "rcsb_entry_container_identifiers": {
            "polymer_entity_ids": ["1"],
        },
        "rcsb_entry_info": {
            "experimental_method": "X-RAY DIFFRACTION",
            "molecular_weight": 25000,
            "resolution_combined": [2.0],
        },
        "citation": [],
    }

    entity = {
        "entity_poly": {
            "rcsb_entity_polymer_type": "Protein",
            "pdbx_seq_one_letter_code_can": (
                "MKTAYIAKQRQISFVKSHFSRQ"
            ),
        },
        "rcsb_entity_source_organism": [
            {
                "scientific_name": "Escherichia coli",
            }
        ],
        "rcsb_polymer_entity_container_identifiers": {
            "uniprot_ids": ["P12345"],
        },
    }

    cif = """
data_TEST
_entity_src_gen.pdbx_host_org_scientific_name
'Escherichia coli BL21(DE3)'
_entity_src_gen.pdbx_host_org_strain
?
_entity_src_gen.pdbx_host_org_variant
?
_entity_src_gen.pdbx_host_org_cell_line
?
_entity_src_gen.pdbx_host_org_vector_type
plasmid
"""

    crystallization = Mock()

    crystallization.available = True
    crystallization.method = "vapor diffusion"
    crystallization.pH = 7.0
    crystallization.temperature_kelvin = 293.0
    crystallization.pressure = None
    crystallization.time = "24 hours"
    crystallization.details = "PEG 3350"
    crystallization.matthews_coefficient = 2.5
    crystallization.solvent_percent = 50.0

    with patch.object(
        collector.pdb_client,
        "get_entry",
        return_value=entry,
    ), patch.object(
        collector.pdb_client,
        "get_polymer_entity",
        return_value=entity,
    ), patch.object(
        collector.pdb_client,
        "download_mmcif",
        return_value=cif,
    ), patch.object(
        collector.mmcif_parser,
        "parse_crystallization",
        return_value=crystallization,
    ):
        record = collector.collect_record("test")

    assert record["expression_host"] == (
        "Escherichia coli"
    )

    assert record["expression_strain"] == (
        "BL21(DE3)"
    )

    assert record["expression_system"] == (
        "bacterial"
    )


def test_find_protein_entity_selects_protein():
    entry = {
        "rcsb_entry_container_identifiers": {
            "polymer_entity_ids": [
                "1",
                "2",
                "3",
            ],
        }
    }

    collector = LargeStructureCollector()

    collector.pdb_client.get_polymer_entity = (
        lambda pdb_id, entity_id: {
            "entity_poly": {
                "rcsb_entity_polymer_type": (
                    "Protein"
                    if entity_id == "2"
                    else "DNA"
                )
            }
        }
    )

    entity_id = collector._find_protein_entity(
        "TEST",
        entry,
    )

    assert entity_id == "2"
def test_find_protein_entity_returns_none_when_no_protein():
    entry = {
        "rcsb_entry_container_identifiers": {
            "polymer_entity_ids": [
                "1",
                "2",
            ],
        }
    }

    collector = LargeStructureCollector()

    collector.pdb_client.get_polymer_entity = (
        lambda pdb_id, entity_id: {
            "entity_poly": {
                "rcsb_entity_polymer_type": "DNA"
            }
        }
    )

    entity_id = collector._find_protein_entity(
        "TEST",
        entry,
    )

    assert entity_id is None