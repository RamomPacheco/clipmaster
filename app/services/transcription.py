from __future__ import annotations

import multiprocessing
from multiprocessing.context import BaseContext
import json
import os
import sys
from pathlib import Path
from queue import Empty
import tempfile
import time
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
    from pathlib import Path as PathLocal

    def _send_progress(msg: str) -> None:
        try:
            result_queue.put({"type": "progress", "message": str(msg)})
        except Exception:  # noqa: BLE001
            pass

    def _send_fatal(err: str) -> None:
        try:
            result_queue.put({"ok": False, "error": err})
        except Exception:  # noqa: BLE001
            pass

    _send_progress("Worker Whisper: a carregar bibliotecas (faster-whisper)...")

    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # noqa: BLE001
        _send_fatal(
            f"Falha ao importar faster_whisper no processo filho (comum no .exe sem DLLs): {e!r}"
        )
        logger.exception("Import faster_whisper no worker.")
        os._exit(1)  # noqa: SCS108

    audio_path = PathLocal(payload["audio_path"])
    model_size = str(payload["model_size"])
    device = str(payload["device"]).lower()
    compute = str(payload["compute"]).lower()
    beam_size = int(payload["beam_size"])
    language = payload.get("language")
    out_jsonl_path = PathLocal(payload["out_jsonl_path"])

    def run_pass_and_write_jsonl_and_exit(dev: str, ctype: str) -> None:
        """
        Transcreve e grava a saída em JSONL.
        Importante: em CUDA/Windows, o teardown (del/GC) pode travar; por isso devolvemos
        o resultado ao pai antes de qualquer cleanup pesado e encerramos o processo.
        """
        _send_progress(f"Whisper worker iniciado ({dev}/{ctype}).")
        model = WhisperModel(model_size, device=dev, compute_type=ctype)
        segments_generator, info = model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=True,
            language=language,
        )

        max_duration = float(info.duration)
        t0 = time.time()
        seg_count = 0
        word_count = 0
        last_end = 0.0

        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl_path.open("w", encoding="utf-8") as f:
            for idx, s in enumerate(segments_generator, start=1):
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
                word_count += len(words)
                seg = {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": str(s.text),
                    "words": words,
                }
                f.write(json.dumps(seg, ensure_ascii=False))
                f.write("\n")
                seg_count = idx
                last_end = float(getattr(s, "end", 0.0) or 0.0)
                if idx % 15 == 0:
                    msg = f"Transcrevendo... {float(s.end):.2f}s processados de {max_duration:.2f}s"
                    logger.info("%s", msg)
                    _send_progress(msg)

        final_msg = (
            "Transcrição terminou "
            f"(último={last_end:.2f}s / total={max_duration:.2f}s). "
            f"Ficheiro pronto ({out_jsonl_path.name}) com {seg_count} segmento(s) "
            f"e ~{word_count} palavra(s) em {time.time() - t0:.2f}s."
        )
        logger.info("%s", final_msg)
        _send_progress(final_msg)

        # CRÍTICO (Windows/CUDA): não depender de IPC para “ok”.
        # O processo pai já conhece o caminho do JSONL (payload["out_jsonl_path"]).
        # Se o teardown do CUDA travar, o pai ainda pode matar o worker e carregar o ficheiro.
        os._exit(0)  # noqa: SCS108

    try:
        try:
            run_pass_and_write_jsonl_and_exit(device, compute)
        except Exception as e:  # noqa: BLE001
            if device == "cuda" and _looks_like_gpu_share_failure(e):
                logger.warning(
                    "Transcrição na GPU falhou no worker (%s). Repetindo em CPU (int8).",
                    e,
                )
                run_pass_and_write_jsonl_and_exit("cpu", "int8")
            else:
                raise
        # Se chegou aqui, algo impediu o exit — encerra por segurança.
        os._exit(0)  # noqa: SCS108
    except Exception as e:  # noqa: BLE001
        logger.exception("Falha na transcrição (processo filho Whisper).")
        try:
            result_queue.put({"ok": False, "error": repr(e)})
        finally:
            os._exit(1)  # noqa: SCS108


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

    # Evita congelar por pickle/cópia de um objeto gigante via Queue (Windows).
    fd, out_path = tempfile.mkstemp(
        prefix="whisper_segments_",
        suffix=".jsonl",
        dir=str(Path(audio_path).resolve().parent),
        text=True,
    )
    os.close(fd)
    out_jsonl_path = str(Path(out_path).resolve())

    payload: Dict[str, Any] = {
        "audio_path": str(Path(audio_path).resolve()),
        "model_size": model_size,
        "device": device,
        "compute": compute,
        "beam_size": config.WHISPER_BEAM_SIZE,
        "language": config.WHISPER_LANGUAGE,
        "out_jsonl_path": out_jsonl_path,
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

    def _drain_queue() -> None:
        nonlocal result
        while True:
            try:
                msg = result_queue.get_nowait()
            except Empty:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "progress":
                m = msg.get("message")
                if isinstance(m, str) and m.strip():
                    logger.info("%s", m.strip())
                continue
            result = msg
            if msg.get("ok") is False:
                break

    try:
        # Consome progresso do worker; quando o processo morre cedo, a fila ainda tem mensagens.
        t_deadline = time.time() + float(_TRANSCRIBE_QUEUE_TIMEOUT_SEC)
        while proc.is_alive() and time.time() < t_deadline:
            try:
                msg = result_queue.get(timeout=1.0)
            except Empty:
                continue
            if isinstance(msg, dict) and msg.get("type") == "progress":
                m = msg.get("message")
                if isinstance(m, str) and m.strip():
                    logger.info("%s", m.strip())
                continue
            result = msg if isinstance(msg, dict) else None
            if isinstance(result, dict) and result.get("ok") is False:
                break

        if proc.is_alive() and time.time() >= t_deadline:
            logger.warning(
                "Worker Whisper ainda ativo após timeout de transcrição. A forçar término para prosseguir."
            )
            proc.kill()
        proc.join(timeout=_JOIN_GRACE_SEC)
        _drain_queue()
    finally:
        proc.join(timeout=_JOIN_GRACE_SEC)
        if proc.is_alive():
            logger.warning("Processo Whisper ainda ativo após join — forçando término.")
            proc.kill()
            proc.join(timeout=30)
        _drain_queue()

    segments: List[Dict[str, Any]] = []
    if isinstance(result, dict) and result.get("ok") is False:
        err = result.get("error", "erro desconhecido")
        raise RuntimeError(f"Transcrição falhou no processo isolado: {err}")

    p = Path(out_jsonl_path)
    if not p.exists():
        exit_c = proc.exitcode
        frozen = getattr(sys, "frozen", False)
        hint = ""
        if frozen:
            hint = (
                " No executável: confirme que o build inclui ctranslate2/onnxruntime; "
                "na UI escolha «Forçar CPU» para Whisper; verifique ligação à Internet na 1.ª vez (descarga do modelo)."
            )
        raise RuntimeError(
            f"Ficheiro de transcrição não encontrado (jsonl). Código de saída do worker: {exit_c!r}.{hint}"
        )
    logger.info("A carregar transcrição do disco (%s)...", p.name)
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                segments.append(json.loads(raw))
    finally:
        try:
            p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    # Duração: usa o fim do último segmento (robusto mesmo sem mensagem "duration").
    max_duration = 0.0
    if segments:
        try:
            max_duration = float(segments[-1].get("end", 0.0))
        except Exception:  # noqa: BLE001
            max_duration = 0.0

    return segments, max_duration
