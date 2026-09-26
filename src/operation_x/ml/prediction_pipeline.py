from operation_x.ml.multi_predictor import (
    OperationXMultiPredictor,
)


class OperationXPredictionPipeline:

    def __init__(
        self,
        data_directory: str,
        model_names: dict | None = None,
    ):
        self.predictor = OperationXMultiPredictor(
            data_directory=data_directory,
            model_names=model_names,
        )

        self.fitted = False

    def fit(self):

        result = self.predictor.fit()

        self.fitted = True

        return result

    def predict(
        self,
        features: dict,
    ):

        if not self.fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. "
                "Call fit() first."
            )

        return self.predictor.predict(
            features
        )