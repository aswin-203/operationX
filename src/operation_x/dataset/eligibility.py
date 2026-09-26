import gemmi


class DatasetEligibility:

    @staticmethod
    def has_crystallization_conditions(
        cif_text: str,
    ) -> bool:

        doc = gemmi.cif.read_string(cif_text)
        block = doc.sole_block()

        grow = block.get_mmcif_category(
            "_exptl_crystal_grow."
        )

        return bool(grow)

    @staticmethod
    def get_crystallization_fields(
        cif_text: str,
    ) -> dict:

        doc = gemmi.cif.read_string(cif_text)
        block = doc.sole_block()

        grow = block.get_mmcif_category(
            "_exptl_crystal_grow."
        )

        if not grow:
            return {
                "available": False,
                "fields": {},
            }

        return {
            "available": True,
            "fields": grow,
        }
