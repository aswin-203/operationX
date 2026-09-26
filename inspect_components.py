import gemmi

from operation_x.clients.pdb_client import PDBClient


client = PDBClient()

for pdb_id in ["101M", "10AF", "10AH"]:

    print("\n" + "=" * 60)
    print(pdb_id)
    print("=" * 60)

    cif = client.download_mmcif(pdb_id)

    doc = gemmi.cif.read_string(cif)
    block = doc.sole_block()

    grow_comp = block.get_mmcif_category(
        "_exptl_crystal_grow_comp."
    )

    print("Crystal growth components:")
    print(grow_comp)

    print("\nCrystal growth fields:")
    print(
        block.get_mmcif_category(
            "_exptl_crystal_grow."
        )
    )

