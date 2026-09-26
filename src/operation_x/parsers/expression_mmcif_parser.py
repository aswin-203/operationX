import re

from operation_x.models.expression import ExpressionRecord


class ExpressionMMCIFParser:
    """Parse expression-source metadata from an mmCIF document."""

    def parse(
        self,
        cif_text: str,
    ) -> ExpressionRecord:

        import gemmi

        doc = gemmi.cif.read_string(cif_text)
        block = doc.sole_block()

        category = block.get_mmcif_category(
            "_entity_src_gen."
        )

        if not category:
            return ExpressionRecord()

        host_values = self._valid_values(
            category.get(
                "pdbx_host_org_scientific_name",
                [],
            )
        )

        strain_values = self._valid_values(
            category.get(
                "pdbx_host_org_strain",
                [],
            )
        )

        variant_values = self._valid_values(
            category.get(
                "pdbx_host_org_variant",
                [],
            )
        )

        cell_line_values = self._valid_values(
            category.get(
                "pdbx_host_org_cell_line",
                [],
            )
        )

        vector_type_values = self._valid_values(
            category.get(
                "pdbx_host_org_vector_type",
                [],
            )
        )

        plasmid_values = self._valid_values(
            category.get(
                "plasmid_name",
                [],
            )
        )

        host = self._extract_host(
            host_values,
            cell_line_values,
        )

        strain = self._extract_strain(
            host_values,
            strain_values,
            variant_values,
            cell_line_values,
        )

        system = self._infer_system(
            host,
            cell_line_values,
        )

        evidence = []

        evidence.extend(
            f"mmCIF host: {value}"
            for value in host_values
        )

        evidence.extend(
            f"mmCIF strain: {value}"
            for value in strain_values
        )

        evidence.extend(
            f"mmCIF variant: {value}"
            for value in variant_values
        )

        evidence.extend(
            f"mmCIF cell line: {value}"
            for value in cell_line_values
        )

        evidence.extend(
            f"mmCIF vector type: {value}"
            for value in vector_type_values
        )

        evidence.extend(
            f"mmCIF plasmid: {value}"
            for value in plasmid_values
        )

        return ExpressionRecord(
            host=host,
            strain=strain,
            system=system,
            evidence=evidence,
        )

    @staticmethod
    def _valid_values(values):

        return [
            str(value).strip()
            for value in values
            if value not in (
                None,
                "",
                "?",
                ".",
            )
        ]

    @staticmethod
    def _extract_host(
        host_values,
        cell_line_values,
    ):

        if host_values:
            host = host_values[0]

            if re.search(
                r"(?:Escherichia\s+coli|E\.?\s*coli)",
                host,
                re.IGNORECASE,
            ):
                return "Escherichia coli"

            return host

        if cell_line_values:
            return cell_line_values[0]

        return None

    @staticmethod
    def _extract_strain(
        host_values,
        strain_values,
        variant_values,
        cell_line_values,
    ):

        if strain_values:
            return strain_values[0]

        if host_values:
            match = re.search(
                r"\bBL21(?:\s*\(DE3\))?",
                host_values[0],
                re.IGNORECASE,
            )

            if match:
                return match.group(0)

        if variant_values:
            return variant_values[0]

        if cell_line_values:
            return cell_line_values[0]

        return None

    @staticmethod
    def _infer_system(
        host,
        cell_line_values,
    ):

        if not host:
            return None

        host_lower = host.lower()

        if (
            "coli" in host_lower
            or host_lower.startswith("bl21")
        ):
            return "bacterial"

        if cell_line_values:
            return "cell-based"

        if host_lower in {
            "sf9",
            "insect cells",
        }:
            return "insect"

        if "pichia" in host_lower:
            return "yeast"

        if "saccharomyces" in host_lower:
            return "yeast"

        if "cho" in host_lower:
            return "mammalian"

        if "hek293" in host_lower:
            return "mammalian"

        return "unknown"

