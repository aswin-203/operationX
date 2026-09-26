import gemmi

from operation_x.models.crystallization import (
    CrystallizationRecord,
)


class MMCIFParser:

    def parse_crystallization(
        self,
        cif_text: str,
    ) -> CrystallizationRecord:

        doc = gemmi.cif.read_string(cif_text)
        block = doc.sole_block()

        grow = block.get_mmcif_category(
            "_exptl_crystal_grow."
        )

        crystal = block.get_mmcif_category(
            "_exptl_crystal."
        )

        if not grow:
            return CrystallizationRecord(
                available=False
            )

        def value(name: str):
            values = grow.get(name)

            if not values:
                return None

            value = values[0]

            if value is None:
                return None

            return value

        def float_value(name: str):
            raw = value(name)

            if raw is None:
                return None

            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        matthews = None
        solvent_percent = None

        if crystal:

            matthews_raw = (
                crystal.get(
                    "density_Matthews",
                    [None],
                )[0]
            )

            solvent_raw = (
                crystal.get(
                    "density_percent_sol",
                    [None],
                )[0]
            )

            try:
                if matthews_raw is not None:
                    matthews = float(
                        matthews_raw
                    )
            except (TypeError, ValueError):
                pass

            try:
                if solvent_raw is not None:
                    solvent_percent = float(
                        solvent_raw
                    )
            except (TypeError, ValueError):
                pass

        pH_range = value(
            "pdbx_pH_range"
        )

        # Gemmi may interpret the mmCIF
        # missing-value marker "." as False.
        if pH_range is False:
            pH_range = None

        return CrystallizationRecord(
            available=True,

            method=value("method"),

            pH=float_value("pH"),
            pH_range=pH_range,

            temperature_kelvin=float_value(
                "temp"
            ),
            temperature_details=value(
                "temp_details"
            ),

            pressure=float_value(
                "pressure"
            ),
            time=value("time"),

            apparatus=value("apparatus"),
            atmosphere=value("atmosphere"),
            seeding=value("seeding"),

            details=(
                value("details")
                or value("pdbx_details")
            ),

            matthews_coefficient=matthews,
            solvent_percent=solvent_percent,

            raw_conditions=grow,
        )