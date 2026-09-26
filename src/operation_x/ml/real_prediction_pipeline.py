class RealPredictionPipeline:

    TARGETS = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    def __init__(
        self,
        input_feature_builder,
        similarity_pipeline,
        predictors,
        sequence_only_predictors=None,
    ):
        self.input_feature_builder = (
            input_feature_builder
        )

        self.similarity_pipeline = (
            similarity_pipeline
        )

        self.predictors = predictors

        self.sequence_only_predictors = (
            sequence_only_predictors or {}
        )

    def predict(
        self,
        sequence: str,
        pdb_id: str,
        entity: dict,
        max_hits: int = 10,
    ):

        if not isinstance(sequence, str):
            raise TypeError(
                "Sequence must be a string"
            )

        sequence = "".join(
            sequence.split()
        )

        if not sequence:
            raise ValueError(
                "Protein sequence cannot be empty"
            )

        # ---------------------------------
        # 1. Build sequence features
        # ---------------------------------

        physicochemical = (
            self.input_feature_builder.build(
                sequence
            )
        )

        # ---------------------------------
        # 2. Run similarity pipeline
        # ---------------------------------

        similarity_result = (
            self.similarity_pipeline.analyze(
                pdb_id=pdb_id,
                sequence=sequence,
                entity=entity,
                max_hits=max_hits,
            )
        )

        if "evidence" not in similarity_result:
            raise ValueError(
                "Similarity pipeline result does not "
                "contain evidence"
            )

        sequence_evidence = (
            similarity_result["evidence"].get(
                "sequence_evidence",
                {},
            )
        )

        strong_hit_count = (
            sequence_evidence.get(
                "strong_hit_count",
                0,
            )
        )

        # =================================
        # PATH A: SEQUENCE-ONLY FALLBACK
        # =================================

        if strong_hit_count == 0:

            predictions = {}

            for target in self.TARGETS:

                if target not in (
                    self.sequence_only_predictors
                ):
                    raise ValueError(
                        "Missing sequence-only "
                        f"predictor for target: "
                        f"{target}"
                    )

                predictor = (
                    self.sequence_only_predictors[
                        target
                    ]
                )

                if hasattr(
                    predictor,
                    "model",
                ):
                    if predictor.model is None:
                        predictor.fit()

                elif hasattr(
                    predictor,
                    "fit",
                ):
                    predictor.fit()

                result = predictor.predict(
                    physicochemical
                )

                predictions[target] = (
                    result["prediction"]
                )

            return predictions

        # =================================
        # PATH B: SIMILARITY-BASED MODEL
        # =================================

        from operation_x.ml.real_feature_builder import (
            RealFeatureBuilder,
        )

        real_feature_builder = (
            RealFeatureBuilder()
        )

        real_features = (
            real_feature_builder.build_from_evidence(
                similarity_result["evidence"]
            )
        )

        # ---------------------------------
        # 3. Combine into 28 features
        # ---------------------------------

        from operation_x.ml.prediction_feature_assembler import (
            PredictionFeatureAssembler,
        )

        assembler = (
            PredictionFeatureAssembler()
        )

        features = assembler.assemble(
            physicochemical=physicochemical,
            similarity=real_features,
        )

        if len(features) != 28:
            raise ValueError(
                "Prediction input must contain "
                f"28 features, got {len(features)}"
            )

        # ---------------------------------
        # 4. Run similarity-based predictors
        # ---------------------------------

        predictions = {}

        for target in self.TARGETS:

            if target not in self.predictors:
                raise ValueError(
                    f"Missing predictor for target: "
                    f"{target}"
                )

            predictor = self.predictors[target]

            if hasattr(
                predictor,
                "model",
            ):
                if predictor.model is None:
                    predictor.fit()

            elif hasattr(
                predictor,
                "fit",
            ):
                predictor.fit()

            result = predictor.predict(
                features
            )

            predictions[target] = (
                result["prediction"]
            )

        return predictions
