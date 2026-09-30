import re

from .patterns import (
    EXPRESSION_HOST_PATTERNS,
    EXPRESSION_KEYWORDS,
    INDUCER_PATTERNS,
)


class ExpressionExtractor:
    """Extract expression information from scientific text."""

    def extract(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        result = {
            "expression_host": None,
            "expression_strain": None,
            "expression_system": None,
            "inducer": None,
            "evidence": [],
        }

        # --------------------------------
        # Expression host
        # --------------------------------

        host_match = self._find_first(
            text,
            EXPRESSION_HOST_PATTERNS,
        )

        if host_match:
            result["expression_host"] = (
                self._normalize_host(host_match)
            )

            result["expression_system"] = (
                self._infer_expression_system(host_match)
            )

            result["expression_strain"] = (
                self._extract_strain(text)
            )

        # --------------------------------
        # Inducer
        # --------------------------------

        inducer_match = self._find_first(
            text,
            INDUCER_PATTERNS,
        )

        if inducer_match:
            result["inducer"] = inducer_match

        # --------------------------------
        # Evidence
        # --------------------------------

        result["evidence"] = self._find_evidence(text)

        return result

    @staticmethod
    def _find_first(
        text: str,
        patterns: list[str],
    ):
        """Return the first matching expression pattern."""

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(0)

        return None

    @staticmethod
    def _extract_strain(text: str):
        """Extract a known expression strain."""

        match = re.search(
            r"\bBL21(?:\(DE3\))?",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(0)

        return None

    @staticmethod
    def _normalize_host(host: str) -> str:
        """Normalize common expression-host names."""

        if re.search(
            r"E\.?\s*coli",
            host,
            re.IGNORECASE,
        ):
            return "Escherichia coli"

        if host.upper().startswith("BL21"):
            return "Escherichia coli"

        if re.search(
            r"HEK293",
            host,
            re.IGNORECASE,
        ):
            return "HEK293"

        if re.search(
            r"SF9",
            host,
            re.IGNORECASE,
        ):
            return "Sf9"

        if re.search(
            r"Pichia\s+pastoris",
            host,
            re.IGNORECASE,
        ):
            return "Pichia pastoris"

        if re.search(
            r"Saccharomyces\s+cerevisiae",
            host,
            re.IGNORECASE,
        ):
            return "Saccharomyces cerevisiae"

        if re.search(
            r"CHO\s+cells?",
            host,
            re.IGNORECASE,
        ):
            return "CHO cells"

        if re.search(
            r"insect\s+cells?",
            host,
            re.IGNORECASE,
        ):
            return "insect cells"

        return host

    @staticmethod
    def _infer_expression_system(host: str) -> str:
        """Infer the broad expression system from the host."""

        host_lower = host.lower()

        if (
            "coli" in host_lower
            or host.upper().startswith("BL21")
        ):
            return "bacterial"

        if (
            "sf9" in host_lower
            or "insect" in host_lower
        ):
            return "insect"

        if (
            "hek293" in host_lower
            or "cho" in host_lower
        ):
            return "mammalian"

        if (
            "pichia" in host_lower
            or "saccharomyces" in host_lower
        ):
            return "yeast"

        return "unknown"

    @staticmethod
    def _find_evidence(text: str) -> list[str]:
        """Return sentences containing expression-related keywords."""

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        evidence = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if any(
                keyword in sentence_lower
                for keyword in EXPRESSION_KEYWORDS
            ):
                evidence.append(sentence.strip())

        return evidence