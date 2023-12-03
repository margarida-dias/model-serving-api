import json
import logging
import os
import typing
import uuid

import httpx
import numpy
import numpy as np
import numpy.typing
import orjson
from click import types
from fastapi.responses import PlainTextResponse
from mlserver.codecs import NumpyCodec
from mlserver.errors import InferenceError
from mlserver.handlers import custom_handler
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput, Parameters
from mlserver_sklearn import SKLearnModel


class ServingApi(SKLearnModel):  # type: ignore

    RESCORER_DEFAULT_SCORE_VALUE_FAILURE = -1.0
    RESCORER_EXIT_FAILURE = 1
    RESCORER_MONITORING_METRICS_TIMEOUT = 0.2
    RESCORER_REQUEST_MIN_INPUTS = 1
    RESCORER_SETTINGS_PATH = os.environ["RESCORER_SETTINGS_PATH"]
    RESCORER_LOG_LEVEL = os.environ.get("RESCORER_LOG_LEVEL", "INFO")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def load(self) -> bool:

        self.model = await super().load()

        logging.info(f"[load] Model ready: {self.model}")

        return self.model

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        request = self.decode(payload.inputs[0], default_codec=NumpyCodec)

        predict = self.calculate_rank_scores(request)

        converted_predict = self.convert_key_and_values_to_int(predict)

        response_bytes = json.dumps(converted_predict).encode("UTF-8")

        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[ResponseOutput(
                    name="echo_response",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=Parameters(content_type="str"),
                )
            ],
        )

    def calculate_rank_scores(self, request: typing.List[typing.List[typing.Union[int, float]]]) \
            -> typing.Dict[int, float]:
        """
        Calculate the rank score for each product using their relevance scores
        (from ElasticSearch) and boosts calculated based on their popularity scores.
        """
        # Get only product_ids
        product_ids = [product[0] for product in request]

        # Get only boosts and ES scores for model input
        input_model = [
            [product[1], product[2]]
            for product in request]

        # Use the dummy model to sum the two features.
        output_model = self.model.predict(input_model)  # type: ignore

        return {product_id: rank_score
                for product_id, rank_score in zip(product_ids, output_model)}

    @staticmethod
    def convert_key_and_values_to_int(input_dict):
        new_dict = {}
        for key, value in input_dict.items():
            if isinstance(key, np.int32) and isinstance(value, np.int64):
                key = int(key)
                value = float(value)
            new_dict[key] = value
        return new_dict

    # def _parse(self, request: InferenceRequest) -> PredictContext:
    #     """Parse and validate InferenceRequest.
    #
    #     Parameters
    #     ----------
    #     request: mlserver.types.InferenceRequest
    #         parsed request payload.
    #
    #     Raises
    #     ------
    #     InferenceError:
    #         Provided payload could not be parsed or given parameter is invalid.
    #
    #     Returns
    #     -------
    #     context: list[list[float]]
    #         list of vectors representing user and product features.
    #     """
    #     if len(request.inputs) != self.RESCORER_REQUEST_MIN_INPUTS:
    #         logging.exception("[parse] Incorrect number of inputs.")
    #
    #         raise InferenceError("Incorrect number of inputs. One is required.")
    #
    #     try:
    #
    #         # mlserver.types.dataplane.Parameters allows custom params.
    #         # TODO: move to inputs.
    #         # TODO: consider using NumpyCodec to parse input
    #
    #         country = request.parameters.ffcountrycode
    #         featureset = request.parameters.featureset
    #         userid = request.parameters.ffuserid
    #
    #         input, *_ = request.inputs
    #         shape = input.shape
    #         products = input.data.__root__  # as defined by TensorData
    #
    #     except AttributeError as e:
    #         logging.exception("[parse] Incorrect num of attrs.")
    #
    #         raise InferenceError(f"Incorrect number of parameters: {e}")
    #
    #     if shape != [len(products)]:
    #         raise InferenceError("Shape does not match to the given input length.")
    #
    #     try:
    #
    #         context = PredictContext(
    #             id=request.id or uuid.uuid4().hex,
    #             country=country,
    #             featureset=featureset,
    #             products=products,
    #             userid=userid,
    #         )
    #
    #         return context
    #
    #     except (TypeError, ValueError) as e:
    #         logging.exception("[parse] Error parsing request.")
    #
    #         raise InferenceError(f"Invalid input: {e}")

