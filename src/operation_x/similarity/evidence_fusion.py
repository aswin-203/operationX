class EvidenceFusion:
    """
    Summarize sequence, structure, and domain similarity evidence.
    """

    def summarize(
        self,
        sequence_hits: list[dict],
        structure_hits: list[dict],
        domains: list[dict],
    ) -> dict:

        # -----------------------------
        # Sequence evidence
        # -----------------------------

        if sequence_hits:

            best_sequence = max(
                sequence_hits,
                key=lambda hit: hit["bitscore"],
            )

            strong_sequence_hits = [
                hit
                for hit in sequence_hits
                if (
                    hit["identity"] >= 70.0
                    and hit["coverage"] >= 0.70
                )
            ]

            sequence_evidence = {
                "best_identity": (
                    best_sequence["identity"]
                ),
                "best_coverage": (
                    best_sequence["coverage"]
                ),
                "best_evalue": (
                    best_sequence["evalue"]
                ),
                "best_bitscore": (
                    best_sequence["bitscore"]
                ),
                "strong_hit_count": (
                    len(strong_sequence_hits)
                ),
            }

        else:

            sequence_evidence = {
                "best_identity": None,
                "best_coverage": None,
                "best_evalue": None,
                "best_bitscore": None,
                "strong_hit_count": 0,
            }

        # -----------------------------
        # Structure evidence
        # -----------------------------

        if structure_hits:

            best_structure = max(
                structure_hits,
                key=lambda hit: hit[
                    "query_tm_score"
                ],
            )

            strong_structure_hits = [
                hit
                for hit in structure_hits
                if (
                    hit["query_tm_score"] >= 0.70
                    and hit["alignment_length"] >= 50
                )
            ]

            structure_evidence = {
                "best_query_tm_score": (
                    best_structure[
                        "query_tm_score"
                    ]
                ),
                "best_target_tm_score": (
                    best_structure[
                        "target_tm_score"
                    ]
                ),
                "best_rmsd": (
                    best_structure["rmsd"]
                ),
                "best_alignment_length": (
                    best_structure[
                        "alignment_length"
                    ]
                ),
                "strong_hit_count": (
                    len(strong_structure_hits)
                ),
            }

        else:

            structure_evidence = {
                "best_query_tm_score": None,
                "best_target_tm_score": None,
                "best_rmsd": None,
                "best_alignment_length": None,
                "strong_hit_count": 0,
            }

        # -----------------------------
        # Domain evidence
        # -----------------------------

        pfam_domains = [
            domain
            for domain in domains
            if domain["domain_type"] == "Pfam"
        ]

        interpro_domains = [
            domain
            for domain in domains
            if domain["domain_type"] == "InterPro"
        ]

        domain_evidence = {
            "domain_count": len(domains),

            "pfam_count": len(
                pfam_domains
            ),

            "interpro_count": len(
                interpro_domains
            ),

            "pfam_ids": [
                domain["domain_id"]
                for domain in pfam_domains
            ],

            "interpro_ids": [
                domain["domain_id"]
                for domain in interpro_domains
            ],
        }

        # -----------------------------
        # Combined result
        # -----------------------------

        return {
            "sequence_evidence": (
                sequence_evidence
            ),

            "structure_evidence": (
                structure_evidence
            ),

            "domain_evidence": (
                domain_evidence
            ),
        }
