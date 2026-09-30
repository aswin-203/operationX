import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from operation_x.clients.pdb_client import PDBClient
from operation_x.similarity.domain_search import DomainSimilarity


INPUT_FILE = Path(
    "data/ml_large/records_with_groups.json"
)

OUTPUT_FILE = Path(
    "data/ml_large/domain_features.csv"
)

MAX_WORKERS = 8


def fetch_domain_features(record_id, record):
    """
    Fetch Pfam and InterPro annotations
    directly using the entity_id already
    stored in the dataset.
    """

    pdb_id = str(
        record.get("pdb_id", record_id)
    ).upper().strip()

    entity_id = str(
        record.get("entity_id", "")
    ).strip()

    if not pdb_id:
        raise ValueError(
            f"{record_id}: missing pdb_id"
        )

    if not entity_id:
        raise ValueError(
            f"{record_id}: missing entity_id"
        )

    client = PDBClient(
        timeout=30
    )

    domain_similarity = DomainSimilarity()

    entity = client.get_polymer_entity(
        pdb_id,
        entity_id
    )

    domains = domain_similarity.extract(
        entity
    )

    pfam_count = sum(
        1
        for domain in domains
        if domain.get("domain_type")
        == "Pfam"
    )

    interpro_count = sum(
        1
        for domain in domains
        if domain.get("domain_type")
        == "InterPro"
    )

    return {
        "pdb_id": record_id,
        "domain_count": len(domains),
        "pfam_count": pfam_count,
        "interpro_count": interpro_count,
        "status": "success",
    }


def main():

    print()
    print("=" * 60)
    print(
        "OPERATION X — FAST DOMAIN FEATURE EXTRACTION"
    )
    print("=" * 60)
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as handle:

        records = json.load(handle)

    print(
        f"Records to process : "
        f"{len(records)}"
    )

    print(
        f"Workers             : "
        f"{MAX_WORKERS}"
    )

    results = []

    failures = []

    completed = 0

    # --------------------------------------------------------
    # Parallel RCSB requests
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                fetch_domain_features,
                record_id,
                record
            ): record_id
            for record_id, record
            in records.items()
        }

        for future in as_completed(
            futures
        ):

            record_id = futures[
                future
            ]

            completed += 1

            try:

                result = future.result()

                results.append(
                    result
                )

            except Exception as exc:

                failures.append(
                    {
                        "pdb_id": record_id,
                        "error": str(exc),
                    }
                )

            if (
                completed % 100 == 0
                or completed == len(records)
            ):

                print(
                    f"Progress: "
                    f"{completed}/"
                    f"{len(records)}"
                )

    # --------------------------------------------------------
    # Keep failed records in output
    # --------------------------------------------------------

    successful_ids = {
        row["pdb_id"]
        for row in results
    }

    for failure in failures:

        record_id = failure[
            "pdb_id"
        ]

        if record_id not in successful_ids:

            results.append(
                {
                    "pdb_id": record_id,
                    "domain_count": 0,
                    "pfam_count": 0,
                    "interpro_count": 0,
                    "status": "failed",
                }
            )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda row: row["pdb_id"]
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results,
        columns=[
            "pdb_id",
            "domain_count",
            "pfam_count",
            "interpro_count",
            "status",
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    success_count = int(
        (df["status"] == "success").sum()
    )

    failure_count = int(
        (df["status"] == "failed").sum()
    )

    print()
    print(
        "DOMAIN SUMMARY"
    )
    print("-" * 60)

    print(
        f"Total records       : "
        f"{len(df)}"
    )

    print(
        f"Successful          : "
        f"{success_count}"
    )

    print(
        f"Failed              : "
        f"{failure_count}"
    )

    print(
        f"Total domains       : "
        f"{df['domain_count'].sum()}"
    )

    print(
        f"Total Pfam          : "
        f"{df['pfam_count'].sum()}"
    )

    print(
        f"Total InterPro      : "
        f"{df['interpro_count'].sum()}"
    )

    print(
        f"Output file         : "
        f"{OUTPUT_FILE}"
    )

    if failures:

        print()
        print(
            "FIRST FAILED RECORDS"
        )
        print("-" * 60)

        for failure in failures[:10]:

            print(
                f"{failure['pdb_id']}: "
                f"{failure['error']}"
            )

    print()
    print("=" * 60)
    print(
        "DOMAIN FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()