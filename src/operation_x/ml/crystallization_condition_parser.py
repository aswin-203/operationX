import re


class CrystallizationConditionParser:

    def parse(self, text: str) -> dict:

        if not isinstance(text, str):
            raise TypeError(
                "Crystallization details must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Crystallization details cannot be empty"
            )

        result = {
            "parse_status": "unparsed",
            "protein_concentration": None,
            "inhibitor_concentration": None,
            "inhibitor_unit": None,
            "precipitant": None,
            "precipitant_concentration": None,
            "precipitant_unit": None,
            "buffer": None,
            "buffer_concentration": None,
            "buffer_unit": None,
            "additives": [],
            "pH": None,
        }

        # Protocol descriptions are not chemical recipes.
        if "protocol was established" in text.lower():
            return result

        # -------------------------------------------------
        # pH
        # -------------------------------------------------

        ph_matches = re.findall(
            r"\bpH\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )

        if ph_matches:
            result["pH"] = float(ph_matches[-1])

        # -------------------------------------------------
        # Protein concentration
        # -------------------------------------------------

        protein_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*mg/mL\s+"
            r"HDAC6\s+protein",
            text,
            re.IGNORECASE,
        )

        if protein_match:
            result["protein_concentration"] = float(
                protein_match.group(1)
            )

        # -------------------------------------------------
        # Inhibitor
        # -------------------------------------------------

        inhibitor_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*(mM|M|uM)\s+"
            r"inhibitor",
            text,
            re.IGNORECASE,
        )

        if inhibitor_match:
            result["inhibitor_concentration"] = float(
                inhibitor_match.group(1)
            )
            result["inhibitor_unit"] = (
                inhibitor_match.group(2)
            )

        # -------------------------------------------------
        # PEG / polyethylene glycol
        # -------------------------------------------------

        peg_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*"
            r"(w/v|v/v)\s+"
            r"(Polyethylene glycol|PEG)\s*"
            r"([0-9,]+)?",
            text,
            re.IGNORECASE,
        )

        if peg_match:

            concentration = float(
                peg_match.group(1)
            )

            percentage_type = (
                peg_match.group(2)
            )

            name = "Polyethylene glycol"

            molecular_weight = peg_match.group(4)

            if molecular_weight:
                molecular_weight = (
                    molecular_weight.replace(",", "")
                )
                name += f" {molecular_weight}"

            result["precipitant"] = name
            result["precipitant_concentration"] = (
                concentration
            )
            result["precipitant_unit"] = (
                f"% {percentage_type}"
            )

        # -------------------------------------------------
        # Ammonium sulfate
        # -------------------------------------------------

        ammonium_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*(M|mM)\s+"
            r"AMMONIUM\s+SULFATE",
            text,
            re.IGNORECASE,
        )

        if ammonium_match:

            result["precipitant"] = (
                "AMMONIUM SULFATE"
            )

            result["precipitant_concentration"] = (
                float(ammonium_match.group(1))
            )

            result["precipitant_unit"] = (
                ammonium_match.group(2)
            )

        # -------------------------------------------------
        # Tris
        # -------------------------------------------------

        tris_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*(M|mM)\s+"
            r"TRIS\b",
            text,
            re.IGNORECASE,
        )

        if tris_match:

            result["buffer"] = "TRIS"

            result["buffer_concentration"] = (
                float(tris_match.group(1))
            )

            result["buffer_unit"] = (
                tris_match.group(2)
            )

        # -------------------------------------------------
        # BIS-TRIS propane
        # -------------------------------------------------

        bis_tris_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*(M|mM)\s+"
            r"BIS-TRIS\s+propane",
            text,
            re.IGNORECASE,
        )

        if bis_tris_match:

            result["buffer"] = (
                "BIS-TRIS propane"
            )

            result["buffer_concentration"] = (
                float(bis_tris_match.group(1))
            )

            result["buffer_unit"] = (
                bis_tris_match.group(2)
            )

        # -------------------------------------------------
        # Citric acid / citrate / Tacsimate
        # -------------------------------------------------

        additive_patterns = [
            (
                r"([0-9]+(?:\.[0-9]+)?)\s*(M|mM)\s+"
                r"Citric acid",
                "Citric acid",
            ),
            (
                r"([0-9]+(?:\.[0-9]+)?)\s*(M|mM)\s+"
                r"Sodium citrate tribasic dihydrate",
                "Sodium citrate tribasic dihydrate",
            ),
            (
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*"
                r"(w/v|v/v)\s+Tacsimate",
                "Tacsimate",
            ),
            (
                r"([0-9]+(?:\.[0-9]+)?)\s*(mM|M)\s+"
                r"EDTA",
                "EDTA",
            ),
        ]

        for pattern, name in additive_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if not match:
                continue

            concentration = float(
                match.group(1)
            )

            unit = match.group(2)

            if "%" in match.group(0):
                unit = f"% {unit}"

            result["additives"].append(
                {
                    "name": name,
                    "concentration": concentration,
                    "unit": unit,
                }
            )

        # -------------------------------------------------
        # Determine parse status
        # -------------------------------------------------

        recognized = any([
            result["precipitant"] is not None,
            result["buffer"] is not None,
            result["protein_concentration"] is not None,
            result["inhibitor_concentration"] is not None,
            len(result["additives"]) > 0,
        ])

        if recognized:

            # We consider a recipe with at least one
            # recognized chemical component usable.
            result["parse_status"] = "parsed"

        return result
