"""`servingapi.metrics` module."""

from prometheus_client import Counter, Summary

rescorer_model_predict = Summary(
    "rescorer_predict_duration",
    "Predict duration.",
)

rescorer_parse_request = Summary(
    "rescorer_request_parse_duration",
    "Request parsing and validation time.",
)

rescorer_parse_vectors = Summary(
    "rescorer_parse_vectors_duration",
    "Join user and products vectors.",
)

rescorer_prediction_missing_fill = Summary(
    "rescorer_response_fill_missing_duration",
    "Fill missing_responses with default value.",
)

rescorer_prediction_missing_counter = Counter(
    "rescorer_prediction_missing_counter",
    "Number of products without features.",
)

rescorer_prediction_missing_percentage = Counter(
    "rescorer_prediction_missing_percentage",
    "Percentage of products without features.",
)

rescorer_redis_command = Summary(
    "rescorer_redis_mget_duration",
    "Redis command latency.",
)

__all__ = [
    "rescorer_model_predict",
    "rescorer_parse_request",
    "rescorer_parse_vectors",
    "rescorer_prediction_missing_fill",
    "rescorer_redis_command",
]
