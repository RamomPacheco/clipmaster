from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from faster_whisper import WhisperModel

from app.core.logger import logger


def init_whisper_model() -> WhisperModel:
    """
    Inicializa o Whisper otimizado para CPU (int8), exatamente como no código original.
    """
    logger.info("Inicializando Whisper (CPU, int8)...")
    return WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(
    audio_path: Path,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Transcreve um arquivo WAV usando Faster-Whisper, retornando os segmentos e a duração total.
    Mantém o comportamento original com feedback periódico.
    """
    model = init_whisper_model()
    logger.info("Iniciando transcrição profunda com Whisper...")

    segments_generator, info = model.transcribe(
        str(audio_path),
        beam_size=2,
        vad_filter=True,
    )

    segments: List[Dict[str, Any]] = []
    max_duration = float(info.duration)

    for s in segments_generator:
        segments.append(
            {"start": float(s.start), "end": float(s.end), "text": str(s.text)}
        )
        if len(segments) % 15 == 0:
            logger.info(
                "Transcrevendo... %.2fs processados de %.2fs", s.end, max_duration
            )

    return segments, max_duration

