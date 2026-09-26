import re

from operation_x.text_mining.article_fetcher import ArticleFetcher

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

        host_match = self._find_first(
            text,
            EXPRESSION_HOST_PATTERNS,
        )

        if host_match:
            result["expression_host"] = self._normalize_host(
                host_match
            )

            result["expression_system"] = (
                self._infer_expression_system(host_match)
            )

            result["expression_strain"] = (
                self._extract_strain(text)
            )

        inducer_match = self._find_first(
            text,
            INDUCER_PATTERNS,
        )

        if inducer_match:
            result["inducer"] = inducer_match

        result["evidence"] = self._find_evidence(text)

        return result

    @staticmethod
    def _find_first(text: str, patterns: list[str]):
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
        if re.search(
            r"E\.?\s*coli",
            host,
            re.IGNORECASE,
        ):
            return "Escherichia coli"

        if host.upper().startswith("BL21"):
            return "Escherichia coli"

        return host

    @staticmethod
    def _infer_expression_system(host: str) -> str:
        if (
            "coli" in host.lower()
            or host.upper().startswith("BL21")
        ):
            return "bacterial"

        if host.lower() in {
            "sf9",
            "insect cells",
        }:
            return "insect"

        return "unknown"

    @staticmethod
    def _find_evidence(text: str) -> list[str]:
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

