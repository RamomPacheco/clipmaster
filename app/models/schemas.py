from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class TiktokCaptionStyle:
    """Estilo ASS da legenda «Default» nos clipes (burn-in TikTok)."""

    font_family: str = "Arial"
    font_size: Optional[int] = None
    margin_v: Optional[int] = None
    primary_color_ass: str = "&H00FFFFFF"
    secondary_color_ass: str = "&H0000E5FF"
    outline_color_ass: str = "&H00101010"
    outline: int = 4
    shadow: int = 1
    bold: bool = True
    italic: bool = False
    scale_x: int = 100
    scale_y: int = 100


@dataclass
class SocialCoverStyle:
    """
    Estilo ASS da linha «Cover» na capa JPG (libass).
    Valores coincidem com o padrão histórico quando usados os defaults.
    """

    font_family: str = "Arial"
    font_size: Optional[int] = None  # None ou 0 = automático (função tiktok_subtitle_style_sizes)
    margin_v: Optional[int] = None  # None ou 0 = automático; distância da margem inferior (alinhamento inferior)
    primary_color_ass: str = "&H0000D7FF"  # BGR + &H00
    secondary_color_ass: str = "&H0000E5FF"
    outline_color_ass: str = "&H00000000"
    outline: int = 6
    shadow: int = 3
    bold: bool = True
    italic: bool = False
    scale_x: int = 102
    scale_y: int = 102


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

