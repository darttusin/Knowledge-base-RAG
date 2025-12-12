from typing import Any

from pydantic import BaseModel


class StatsResponse(BaseModel):
    average_processing_time_ms: float
    quantiles: dict[str, float]  # mean, 50%, 95%, 99%
    input_characteristics: dict[str, Any]
