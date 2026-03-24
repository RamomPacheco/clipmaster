from __future__ import annotations

import multiprocessing
from multiprocessing.context import BaseContext
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Tuple

from app.core import config
from app.core.logger import logger


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


def worker_transcricao(payload: Dict[str, Any], result_queue: multiprocessing.Queue) -> None:
    """
    Corre apenas no processo filho. Instancia WhisperModel aqui e devolve o resultado pela fila.
    Evita crash do processo principal (ex.: 0xC0000409) na limpeza da VRAM do CTranslate2.
    """
    import gc as gc_local
    from pathlib import Path as PathLocal

    from faster_whisper import WhisperModel

    audio_path = PathLocal(payload["audio_path"])
    model_size = str(payload["model_size"])
    device = str(payload["device"]).lower()
    compute = str(payload["compute"]).lower()
    beam_size = int(payload["beam_size"])
    language = payload.get("language")

    def run_pass(dev: str, ctype: str) -> Tuple[List[Dict[str, Any]], float]:
        model = WhisperModel(
            model_size,
            device=dev,
            compute_type=ctype,
        )
        try:
            segments_generator, info = model.transcribe(
                str(audio_path),
                beam_size=beam_size,
                vad_filter=True,
                word_timestamps=True,
                language=language,
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
                    logger.info(
                        "Transcrevendo... %.2fs processados de %.2fs", s.end, max_duration
                    )

            return segments, max_duration
        finally:
            del model
            gc_local.collect()

    try:
        try:
            segments, max_duration = run_pass(device, compute)
        except Exception as e:  # noqa: BLE001
            if device == "cuda" and _looks_like_gpu_share_failure(e):
                logger.warning(
                    "Transcrição na GPU falhou no worker (%s). Repetindo em CPU (int8).",
                    e,
                )
                segments, max_duration = run_pass("cpu", "int8")
            else:
                raise
        result_queue.put(
            {"ok": True, "segments": segments, "duration": max_duration},
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Falha na transcrição (processo filho Whisper).")
        result_queue.put({"ok": False, "error": repr(e)})


# Timeout generoso: vídeos longos em CPU podem demorar horas
_TRANSCRIBE_QUEUE_TIMEOUT_SEC = 6 * 3600
_JOIN_GRACE_SEC = 120


def transcribe_audio(
    audio_path: Path,
    model_name: str | None = None,
    device_override: str | None = None,
    compute_override: str | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Transcreve o áudio num processo isolado (spawn). O WhisperModel não existe no processo pai.

    O filho usa ``vad_filter=True`` e o ``compute_type`` pedido (ex.: int8_float16 em CUDA);
    em falha de VRAM na GPU, o próprio filho repete em CPU (int8).
    """
    device = (device_override or config.WHISPER_DEVICE).strip().lower()
    compute = (compute_override or config.WHISPER_COMPUTE_TYPE).strip().lower()
    model_size = model_name or config.WHISPER_MODEL

    payload: Dict[str, Any] = {
        "audio_path": str(Path(audio_path).resolve()),
        "model_size": model_size,
        "device": device,
        "compute": compute,
        "beam_size": config.WHISPER_BEAM_SIZE,
        "language": config.WHISPER_LANGUAGE,
    }

    ctx: BaseContext = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=worker_transcricao,
        args=(payload, result_queue),
        name="whisper_transcribe_worker",
    )
    proc.start()
    result: Dict[str, Any] | None = None
    try:
        try:
            result = result_queue.get(timeout=_TRANSCRIBE_QUEUE_TIMEOUT_SEC)
        except Empty:
            logger.error("Timeout à espera da transcrição Whisper (fila vazia).")
            proc.terminate()
            raise RuntimeError(
                "Transcrição excedeu o tempo máximo ou o processo filho não respondeu."
            ) from None
    finally:
        proc.join(timeout=_JOIN_GRACE_SEC)
        if proc.is_alive():
            logger.warning("Processo Whisper ainda ativo após join — forçando término.")
            proc.kill()
            proc.join(timeout=30)

    if not isinstance(result, dict):
        raise RuntimeError("Resposta inválida do worker de transcrição.")
    if not result.get("ok"):
        err = result.get("error", "erro desconhecido")
        raise RuntimeError(f"Transcrição falhou no processo isolado: {err}")

    segments = result.get("segments")
    duration = result.get("duration")
    if not isinstance(segments, list):
        raise RuntimeError("Worker devolveu segmentos inválidos.")
    try:
        max_duration = float(duration)
    except (TypeError, ValueError) as e:
        raise RuntimeError("Duração inválida devolvida pelo worker.") from e

    return segments, max_duration
