import json
import re
from pathlib import Path

INPUT = Path("data/ml_large/records_with_groups.json")
OUTPUT = Path("data/ml_large/expression_features.csv")


def normalize(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def classify_host(host):
    h = normalize(host)

    if not h:
        return "unknown"

    if "escherichia coli" in h or "e. coli" in h or "e coli" in h:
        return "ecoli"

    if any(x in h for x in ["insect", "sf9", "sf-9", "sf21", "baculovirus"]):
        return "insect"

    if any(x in h for x in ["yeast", "saccharomyces", "pichia"]):
        return "yeast"

    if any(x in h for x in [
        "mammalian",
        "human",
        "hamster",
        "chinese hamster",
        "cho cell",
        "hek",
        "hela",
    ]):
        return "mammalian"

    if any(x in h for x in ["cell-free", "cell free", "in vitro"]):
        return "cell_free"

    return "other"


def classify_system(system):
    s = normalize(system)

    if not s:
        return "unknown"

    if any(x in s for x in ["bacterial", "bacterium", "bacteria"]):
        return "bacterial"

    if any(x in s for x in ["insect", "baculovirus"]):
        return "insect"

    if "yeast" in s or "fung" in s:
        return "yeast"

    if any(x in s for x in ["mammalian", "human", "cell culture"]):
        return "mammalian"

    if any(x in s for x in ["cell-free", "cell free", "in vitro"]):
        return "cell_free"

    return "other"


def classify_inducer(inducer):
    i = normalize(inducer)

    if not i:
        return "unknown"

    if "iptg" in i:
        return "iptg"

    return "other"


def main():
    import csv

    with INPUT.open("r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []

    for pdb_id, record in records.items():
        host = record.get("expression_host")
        strain = record.get("expression_strain")
        system = record.get("expression_system")
        inducer = record.get("inducer")
        evidence = record.get("expression_evidence") or []

        host_class = classify_host(host)
        system_class = classify_system(system)
        inducer_class = classify_inducer(inducer)

        rows.append({
            "pdb_id": pdb_id,

            "host_ecoli": int(host_class == "ecoli"),
            "host_insect": int(host_class == "insect"),
            "host_yeast": int(host_class == "yeast"),
            "host_mammalian": int(host_class == "mammalian"),
            "host_cell_free": int(host_class == "cell_free"),
            "host_other": int(host_class == "other"),
            "host_unknown": int(host_class == "unknown"),

            "system_bacterial": int(system_class == "bacterial"),
            "system_insect": int(system_class == "insect"),
            "system_yeast": int(system_class == "yeast"),
            "system_mammalian": int(system_class == "mammalian"),
            "system_cell_free": int(system_class == "cell_free"),
            "system_other": int(system_class == "other"),
            "system_unknown": int(system_class == "unknown"),

            "strain_present": int(bool(normalize(strain))),

            "inducer_present": int(bool(normalize(inducer))),
            "inducer_iptg": int(inducer_class == "iptg"),
            "inducer_other": int(inducer_class == "other"),
            "inducer_unknown": int(inducer_class == "unknown"),

            "expression_evidence_count": len(evidence),
        })

    fieldnames = list(rows[0].keys())

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("OPERATION X — EXPRESSION FEATURE EXTRACTION")
    print("=" * 60)
    print()
    print(f"Input records  : {len(records)}")
    print(f"Output rows    : {len(rows)}")
    print(f"Features       : {len(fieldnames) - 1}")
    print(f"Output file    : {OUTPUT}")
    print()
    print("EXPRESSION FEATURES")
    print("-" * 60)

    for field in fieldnames[1:]:
        total = sum(row[field] for row in rows)
        print(f"{field:30s}: {total}")

    print()
    print("=" * 60)
    print("EXPRESSION FEATURE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()