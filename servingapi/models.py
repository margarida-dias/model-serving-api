"""`servingapi.models` module.
"""

from typing import Any, List

from mlserver.errors import status
from pydantic import BaseModel, BaseSettings, Field, validator
from starlette.exceptions import HTTPException


class AppSettings(BaseSettings):
    """Application Environment settings."""

    redis_url: str


class PredictContext(BaseModel):
    """RequestContext model.

    Parses arguments from request.
    """

    country: str
    products: List[int] = Field(default_factory=list)
    userid: str
    featureset: str
    id: str

    @validator("*")
    def not_empty(cls: "PredictContext", value: Any) -> Any:
        """All fields are mandatory."""
        if not value:
            ValueError("Field cannot be None or empty.")

        return value


class NoContentForInference(HTTPException):
    """NoContentForInference response.

    Contract agreement with PAP team client.
    """

    def __init__(self, _: str):
        super().__init__(status_code=status.HTTP_204_NO_CONTENT)
