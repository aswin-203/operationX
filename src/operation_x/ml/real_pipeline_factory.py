from pathlib import Path

from operation_x.ml.input_feature_builder import (
    InputFeatureBuilder,
)
from operation_x.ml.real_prediction_pipeline import (
    RealPredictionPipeline,
)
from operation_x.similarity.domain_search import (
    DomainSimilarity,
)
from operation_x.similarity.sequence_search import (
    SequenceSimilarity,
)
from operation_x.similarity.structure_search import (
    StructureSimilarity,
)
from operation_x.similarity.similarity_pipeline import (
    SimilarityPipeline,
)


class RealPipelineFactory:

    TARGETS = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    BEST_MODELS = {
        "pH": "gradient_boosting",
        "temperature_kelvin": "extra_trees",
        "matthews_coefficient": "gradient_boosting",
        "solvent_percent": "random_forest",
    }

    SEQUENCE_ONLY_MODELS = {
        "pH": "extra_trees",
        "temperature_kelvin": "extra_trees",
        "matthews_coefficient": "random_forest",
        "solvent_percent": "random_forest",
    }

    DATA_DIRECTORY = "data/ml_combined"

    SEQUENCE_ONLY_DATA_DIRECTORY = (
        "data/ml_sequence_only"
    )

    def __init__(
        self,
        blast_database: str,
        structure_database: str,
        foldseek_path: str = "foldseek",
        blastp_path: str = "blastp",
        data_directory: str = DATA_DIRECTORY,
        sequence_only_data_directory: str = (
            SEQUENCE_ONLY_DATA_DIRECTORY
        ),
    ):
        self.blast_database = blast_database
        self.structure_database = structure_database
        self.foldseek_path = foldseek_path
        self.blastp_path = blastp_path

        self.data_directory = Path(
            data_directory
        )

        self.sequence_only_data_directory = (
            Path(
                sequence_only_data_directory
            )
        )

    def _build_similarity_pipeline(self):

        sequence_search = SequenceSimilarity(
            database=self.blast_database,
            blastp_path=self.blastp_path,
        )

        structure_search = StructureSimilarity(
            executable=self.foldseek_path,
        )

        domain_similarity = DomainSimilarity()

        return SimilarityPipeline(
            sequence_search=sequence_search,
            structure_search=structure_search,
            domain_similarity=domain_similarity,
            structure_database=self.structure_database,
            homolog_structure_directory=(
                "data/structures/large"
            ),
        )

    def _build_predictors(self):

        predictors = {}

        from operation_x.ml.predictor import (
            OperationXPredictor,
        )

        for target in self.TARGETS:

            data_file = (
                self.data_directory
                / f"{target}_combined.csv"
            )

            if not data_file.exists():
                raise FileNotFoundError(
                    f"Dataset not found for "
                    f"{target}: {data_file}"
                )

            predictors[target] = (
                OperationXPredictor(
                    data_file=str(
                        data_file
                    ),
                    target=target,
                    model_name=(
                        self.BEST_MODELS[target]
                    ),
                )
            )

        return predictors

    def _build_sequence_only_predictors(
        self,
    ):

        predictors = {}

        from operation_x.ml.sequence_only_predictor import (
            SequenceOnlyPredictor,
        )

        for target in self.TARGETS:

            data_file = (
                self.sequence_only_data_directory
                / f"{target}_sequence_only.csv"
            )

            if not data_file.exists():
                raise FileNotFoundError(
                    f"Sequence-only dataset "
                    f"not found for {target}: "
                    f"{data_file}"
                )

            predictors[target] = (
                SequenceOnlyPredictor(
                    data_file=str(
                        data_file
                    ),
                    target=target,
                    model_name=(
                        self.SEQUENCE_ONLY_MODELS[
                            target
                        ]
                    ),
                )
            )

        return predictors

    def create(self):

        input_feature_builder = (
            InputFeatureBuilder()
        )

        similarity_pipeline = (
            self._build_similarity_pipeline()
        )

        predictors = (
            self._build_predictors()
        )

        sequence_only_predictors = (
            self._build_sequence_only_predictors()
        )

        return RealPredictionPipeline(
            input_feature_builder=(
                input_feature_builder
            ),
            similarity_pipeline=(
                similarity_pipeline
            ),
            predictors=predictors,
            sequence_only_predictors=(
                sequence_only_predictors
            ),
        )
