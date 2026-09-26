from pathlib import Path

from operation_x.ml.predictor import OperationXPredictor


class OperationXMultiPredictor:

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

    def __init__(
        self,
        data_directory: str,
        model_names: dict | None = None,
    ):
        self.data_directory = Path(
            data_directory
        )

        self.model_names = (
            model_names
            if model_names is not None
            else self.BEST_MODELS.copy()
        )

        self.predictors = {}
        self.feature_columns = None

    def _create_predictor(self, target: str):

        if target not in self.model_names:
            raise ValueError(
                f"No model configured for target: {target}"
            )

        data_file = (
            self.data_directory
            / f"{target}_combined.csv"
        )

        if not data_file.exists():
            raise FileNotFoundError(data_file)

        return OperationXPredictor(
            data_file=str(data_file),
            target=target,
            model_name=self.model_names[target],
        )

    def fit(self):

        results = {}

        for target in self.TARGETS:

            predictor = self._create_predictor(target)

            result = predictor.fit()

            self.predictors[target] = predictor
            results[target] = result

            if self.feature_columns is None:
                self.feature_columns = (
                    predictor.feature_columns.copy()
                )

            elif (
                predictor.feature_columns
                != self.feature_columns
            ):
                raise ValueError(
                    f"Feature columns differ for "
                    f"target '{target}'"
                )

        return {
            "targets": self.TARGETS.copy(),
            "results": results,
        }

    def predict(self, features: dict):

        if not self.predictors:
            raise RuntimeError(
                "Models have not been fitted. "
                "Call fit() first."
            )

        missing = [
            column
            for column in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Missing features: "
                + ", ".join(missing)
            )

        predictions = {}

        for target in self.TARGETS:

            result = self.predictors[target].predict(
                features
            )

            predictions[target] = float(
                result["prediction"]
            )

        return predictions
    def __init__(
    self,
        data_directory: str,
        model_name: str | None = None,
        model_names: dict | None = None,
    ):
        self.data_directory = Path(
            data_directory
        )

        if model_names is not None:
            self.model_names = model_names.copy()

        elif model_name is not None:
            self.model_names = {
                target: model_name
                for target in self.TARGETS
            }

        else:
            self.model_names = (
                self.BEST_MODELS.copy()
            )

        self.predictors = {}
        self.feature_columns = None
