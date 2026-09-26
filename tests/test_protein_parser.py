from operation_x.clients.pdb_client import PDBClient
from operation_x.parsers.pdb_parser import PDBParser


def test_parse_protein():

    client = PDBClient()

    entry = client.get_entry("1CBS")

    entity = client.get_polymer_entity(
        "1CBS",
        "1",
    )

    parser = PDBParser()

    protein = parser.parse(
        entry,
        entity,
    )

    assert protein.pdb_id == "1CBS"
    assert protein.entity_id == "1"
    assert protein.sequence_length > 0
    assert protein.sequence.isupper()
    assert len(protein.sequence) == protein.sequence_length
