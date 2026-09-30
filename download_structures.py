import json
import time
from pathlib import Path

from operation_x.clients.pdb_client import PDBClient

RECORDS_FILE = Path("data/structures/large/records.json")
OUTPUT_DIR = Path("data/structures/large")

records = json.loads(RECORDS_FILE.read_text(encoding="utf-8"))
client = PDBClient(timeout=30)

missing = [
    pdb_id
    for pdb_id in records
    if not (OUTPUT_DIR / f"{pdb_id}.cif").exists()
]

print(f"Total records: {len(records)}")
print(f"Already downloaded: {len(records) - len(missing)}")
print(f"Remaining: {len(missing)}")
print()

success = 0
failed = []

for index, pdb_id in enumerate(missing, start=1):
    path = OUTPUT_DIR / f"{pdb_id}.cif"

    try:
        cif = client.download_mmcif(pdb_id)
        path.write_text(cif, encoding="utf-8")

        success += 1
        print(f"[{index}/{len(missing)}] {pdb_id} -> OK")

    except Exception as error:
        failed.append((pdb_id, str(error)))
        print(f"[{index}/{len(missing)}] {pdb_id} -> ERROR: {error}")

    time.sleep(0.2)

print()
print("=" * 50)
print("DOWNLOAD COMPLETE")
print("=" * 50)
print(f"Successful: {success}")
print(f"Failed:     {len(failed)}")

if failed:
    print("\nFailed PDB IDs:")
    for pdb_id, error in failed:
        print(f"{pdb_id}: {error}")
