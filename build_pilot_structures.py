from pathlib import Path

from operation_x.clients.pdb_client import PDBClient


PDB_IDS = [
    "101M",
    "102M",
    "103M",
    "104M",
    "105M",
    "106M",
    "107M",
    "108M",
    "109M",
    "10AF",
    "10AH",
    "10AI",
    "10AJ",
    "10AK",
]

OUTPUT_DIR = Path(
    "data/structures/pilot"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = PDBClient()

    successful = []
    failed = []

    for pdb_id in PDB_IDS:

        output_file = (
            OUTPUT_DIR / f"{pdb_id}.cif"
        )

        try:

            print(
                f"Downloading {pdb_id}..."
            )

            cif = client.download_mmcif(
                pdb_id
            )

            output_file.write_text(
                cif
            )

            successful.append(
                pdb_id
            )

            print(
                f"{pdb_id} -> SUCCESS "
                f"({len(cif)} bytes)"
            )

        except Exception as error:

            failed.append(
                pdb_id
            )

            print(
                f"{pdb_id} -> ERROR: "
                f"{error}"
            )

    print()
    print("=" * 60)
    print("PILOT STRUCTURE DATASET")
    print("=" * 60)

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed:     {len(failed)}"
    )

    print(
        "Output:     "
        f"{OUTPUT_DIR}"
    )

    if failed:
        print(
            "Failed IDs:",
            failed,
        )


if __name__ == "__main__":
    main()
