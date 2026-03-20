from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Clip(BaseModel):
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    reason: str = ""
    headline: str = "Sem título"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ProcessingMetrics(BaseModel):
    start_time: Optional[float] = None
    transcription_time: float = 0.0
    analysis_time: float = 0.0
    rendering_time: float = 0.0
    total_clips_found: int = 0
    clips_selected: int = 0
    video_duration: float = 0.0
    model_used: str
    prompt_type: str


class ProcessingHistoryEntry(ProcessingMetrics):
    timestamp: float
    video_path: Path


ClipList = List[Clip]

