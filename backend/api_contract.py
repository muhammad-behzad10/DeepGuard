"""Validated API response types for /docs and the Flutter client."""
from typing import Literal
from pydantic import BaseModel, Field

class VideoInputError(ValueError):
    """The video could not be prepared for analysis."""

class Prediction(BaseModel):
    class_id: Literal[0, 1]
    label: Literal['Real', 'Fake']
    fake_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    real_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    decision_threshold: Literal[0.5]
    threshold_status: Literal['initial_baseline_not_tuned']
    aggregation: Literal['mean_frame_softmax_fake_score']
    frames_scored: int = Field(ge=8, le=16)
    calibrated: Literal[False]

class FrameScore(BaseModel):
    frame_index: int = Field(ge=0)
    fake_score: float = Field(ge=0, le=1, allow_inf_nan=False)

class AnalysisReceipt(BaseModel):
    api_version: Literal['0.6.0']
    status: Literal['analysed', 'insufficient_evidence']
    upload_id: str
    filename: str
    size_bytes: int = Field(gt=0, le=30_000_000)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stored: Literal[False]
    analysis_performed: bool
    prediction: Prediction | None
    reason: str | None
    video: dict
    model: dict
    preprocessing: dict
    frame_scores: list[FrameScore]
    timing_seconds: dict
