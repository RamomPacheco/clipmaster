from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

from faster_whisper import WhisperModel

from app.core import config
from app.core.logger import logger


def _release_whisper_resources(model: WhisperModel | None) -> None:
    """Liberta VRAM/RAM após o modelo CTranslate2 / Whisper."""
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        pass


def _looks_like_gpu_share_failure(exc: BaseException) -> bool:
    """Erros comuns quando a GPU fica sem VRAM (ex.: Ollama + Whisper na mesma placa)."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "cudnn",
        "cublas",
        "cuda error",
        "resource exhausted",
        "illegal memory access",
        "outofmemoryerror",
    )
    return any(n in msg for n in needles)


def _transcribe_once(
    audio_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
) -> Tuple[List[Dict[str, Any]], float]:
    logger.info(
        "Whisper (passagem): model=%s device=%s compute=%s beam=%s",
        model_size,
        device,
        compute_type,
        config.WHISPER_BEAM_SIZE,
    )
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    try:
        segments_generator, info = model.transcribe(
            str(audio_path),
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=True,
            word_timestamps=True,
            language=config.WHISPER_LANGUAGE,
        )

        segments: List[Dict[str, Any]] = []
        max_duration = float(info.duration)

        for s in segments_generator:
            words: List[Dict[str, Any]] = []
            if word_list := getattr(s, "words", None):
                for w in word_list:
                    words.append(
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "word": str(w.word),
                        }
                    )
            segments.append(
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": str(s.text),
                    "words": words,
                }
            )
            if len(segments) % 15 == 0:
                logger.info("Transcrevendo... %.2fs processados de %.2fs", s.end, max_duration)

        return segments, max_duration
    finally:
        _release_whisper_resources(model)


def transcribe_audio(
    audio_path: Path,
    model_name: str | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Transcreve o áudio com Faster-Whisper, segmentos alinhados e timestamps por palavra.

    Se a GPU falhar por falta de memória (cenário frequente com ``ollama serve`` na mesma
    GPU), repete automaticamente em CPU (int8), mais lento porém estável.
    """
    device = config.WHISPER_DEVICE
    compute = config.WHISPER_COMPUTE_TYPE
    model_size = model_name or config.WHISPER_MODEL

    try:
        return _transcribe_once(audio_path, model_size, device, compute)
    except Exception as e:  # noqa: BLE001
        if device == "cuda" and _looks_like_gpu_share_failure(e):
            logger.warning(
                "Transcrição na GPU falhou (%s). Com Ollama ou outro processo na mesma GPU "
                "isso é comum. Repetindo em CPU (defina WHISPER_SHARED_GPU_SAFE=1 ou "
                "WHISPER_DEVICE=cpu para ir direto à CPU).",
                e,
            )
            _release_whisper_resources(None)
            return _transcribe_once(audio_path, model_size, "cpu", "int8")
        raise
