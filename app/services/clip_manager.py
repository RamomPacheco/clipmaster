from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from app.core import config
from app.core.logger import logger
from app.models.schemas import Clip, ProcessingHistoryEntry, ProcessingMetrics


def build_overlapping_chapters(
    segments: List[Dict[str, Any]],
    chunk_seconds: float,
    overlap_seconds: float,
) -> List[List[Dict[str, Any]]]:
    """
    Divide a transcrição em blocos de até `chunk_seconds`, com cauda sobreposta
    para o próximo bloco não começar “do zero” no meio de uma ideia.
    """
    if not segments:
        return []
    chapters: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    chunk_start = float(segments[0]["start"])

    for s in segments:
        if current_chunk and s["end"] - chunk_start > chunk_seconds:
            chapters.append(current_chunk)
            overlap_start = max(chunk_start, current_chunk[-1]["end"] - overlap_seconds)
            tail = [seg for seg in current_chunk if seg["end"] > overlap_start]
            current_chunk = tail + [s]
            chunk_start = min(float(x["start"]) for x in current_chunk)
        else:
            current_chunk.append(s)

    if current_chunk:
        chapters.append(current_chunk)
    return chapters


def _flatten_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for s in segments:
        for w in s.get("words") or []:
            words.append(w)
    words.sort(key=lambda w: float(w["start"]))
    return words


def _snap_start_to_words(t: float, words: List[Dict[str, Any]]) -> float:
    if not words:
        return t
    for w in words:
        if w["start"] <= t <= w["end"]:
            return float(w["start"])
    for w in words:
        if t < w["start"]:
            return float(w["start"])
    return float(words[-1]["start"])


def _snap_end_to_words(t: float, words: List[Dict[str, Any]]) -> float:
    if not words:
        return t
    for w in words:
        if w["start"] <= t <= w["end"]:
            return float(w["end"])
    for w in reversed(words):
        if t >= w["end"]:
            return float(w["end"])
    return float(words[-1]["end"])


def _snap_start_to_segments(t: float, segs: List[Dict[str, Any]]) -> float:
    if not segs:
        return max(0.0, t)
    ordered = sorted(segs, key=lambda s: float(s["start"]))
    if t <= ordered[0]["start"]:
        return float(ordered[0]["start"])
    if t >= ordered[-1]["end"]:
        return float(ordered[-1]["start"])
    for s in ordered:
        if s["start"] <= t <= s["end"]:
            return float(s["start"])
    for i in range(len(ordered) - 1):
        if ordered[i]["end"] < t < ordered[i + 1]["start"]:
            return float(ordered[i + 1]["start"])
    return float(t)


def _snap_end_to_segments(t: float, segs: List[Dict[str, Any]]) -> float:
    if not segs:
        return t
    ordered = sorted(segs, key=lambda s: float(s["start"]))
    if t <= ordered[0]["start"]:
        return float(ordered[0]["end"])
    if t >= ordered[-1]["end"]:
        return float(ordered[-1]["end"])
    for s in ordered:
        if s["start"] <= t <= s["end"]:
            return float(s["end"])
    for i in range(len(ordered) - 1):
        if ordered[i]["end"] < t < ordered[i + 1]["start"]:
            return float(ordered[i]["end"])
    return float(t)


def snap_clip_to_transcript(
    clip: Clip,
    segments: List[Dict[str, Any]],
) -> Clip:
    """Alinha início/fim do clipe aos limites de palavra (ou segmento) da transcrição."""
    if not segments:
        return clip

    words = _flatten_words(segments)
    if len(words) >= 2:
        start = _snap_start_to_words(float(clip.start), words)
        end = _snap_end_to_words(float(clip.end), words)
    else:
        start = _snap_start_to_segments(float(clip.start), segments)
        end = _snap_end_to_segments(float(clip.end), segments)

    max_end = float(segments[-1]["end"])
    start = max(0.0, min(start, max_end))
    end = max(start + 0.25, min(end, max_end))

    return clip.model_copy(
        update={
            "start": round(start, 2),
            "end": round(end, 2),
        }
    )


def snap_clips_to_transcript(
    clips: Iterable[Clip],
    segments: List[Dict[str, Any]],
) -> List[Clip]:
    return [snap_clip_to_transcript(c, segments) for c in clips]


def filter_valid_clips(
    clips: Iterable[Clip],
    max_video_duration: float,
    min_duration: float = 1.0,
) -> List[Clip]:
    """Remove clipes inválidos ou fora do vídeo."""
    out: List[Clip] = []
    for c in clips:
        start = float(c.start)
        end = float(c.end)
        if start < 0 or end <= start:
            logger.info("Clipe descartado: intervalo inválido (%.2f–%.2f)", start, end)
            continue
        if start >= max_video_duration:
            logger.info("Clipe descartado: início após o fim do vídeo")
            continue
        end = min(end, max_video_duration)
        if end - start < min_duration:
            logger.info("Clipe descartado: duração muito curta (%.2fs)", end - start)
            continue
        out.append(c.model_copy(update={"start": round(start, 2), "end": round(end, 2)}))
    return out


def enforce_duration_limits(
    clips: Iterable[Clip],
    max_video_duration: float,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> List[Clip]:
    """
    Garante que os clipes fiquem dentro do intervalo [min_seconds, max_seconds],
    reproduzindo a lógica de _enforce_duration_limits do código antigo.
    """
    min_seconds = min_seconds or config.MIN_CLIP_SECONDS
    max_seconds = max_seconds or config.MAX_CLIP_SECONDS

    adjusted: List[Clip] = []
    for i, clip in enumerate(clips, start=1):
        start = float(clip.start)
        end = float(clip.end)
        duration = end - start

        if duration < min_seconds:
            logger.info(
                "Clipe %s curto (%.1fs). Expandindo para %.1fs...", i, duration, min_seconds
            )
            deficit = min_seconds - duration
            new_start = max(0.0, start - (deficit / 2.0))
            new_end = min(max_video_duration, end + (deficit / 2.0))
            if (new_end - new_start) < min_seconds:
                new_end = min(max_video_duration, new_start + min_seconds)
            clip = clip.model_copy(
                update={
                    "start": round(new_start, 2),
                    "end": round(new_end, 2),
                    "reason": clip.reason + " [Nota de Backend: Expandido para 30s]",
                }
            )
        elif duration > max_seconds:
            logger.warning(
                "Clipe %s excedeu o limite do YouTube Shorts (%.1fs). Cortando em %.1fs.",
                i,
                duration,
                max_seconds,
            )
            clip = clip.model_copy(
                update={
                    "end": round(start + max_seconds, 2),
                    "reason": clip.reason
                    + " [Nota de Backend: Final cortado para respeitar teto de 60s]",
                }
            )

        adjusted.append(clip)

    return adjusted


def remove_duplicate_clips(clips: Iterable[Clip]) -> List[Clip]:
    """
    Remove clipes duplicados/sobrepostos em mais de 50%, preservando o com razão mais detalhada.
    """
    clips_list = list(clips)
    if not clips_list:
        return clips_list

    sorted_clips = sorted(clips_list, key=lambda c: c.start)
    filtered: List[Clip] = []

    for clip in sorted_clips:
        is_duplicate = False
        for existing in filtered:
            overlap_start = max(clip.start, existing.start)
            overlap_end = min(clip.end, existing.end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            min_duration = min(clip.duration, existing.duration)
            if min_duration > 0 and (overlap_duration / min_duration) > 0.5:
                is_duplicate = True
                if len(clip.reason) > len(existing.reason):
                    filtered.remove(existing)
                    filtered.append(clip)
                break

        if not is_duplicate:
            filtered.append(clip)

    logger.info(
        "Removidos %s clipes duplicados/sobrepostos", len(clips_list) - len(filtered)
    )
    return filtered


def append_history_entry(
    metrics: ProcessingMetrics,
    video_path: Path,
    history_file: Path | None = None,
) -> None:
    """
    Atualiza o arquivo de histórico de processamento com uma nova entrada.
    Mantém o mesmo formato de JSON do código original.
    """
    target = history_file or config.PROCESSING_HISTORY_FILE
    try:
        if target.exists():
            with target.open("r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        entry = ProcessingHistoryEntry(
            **metrics.model_dump(),
            timestamp=time.time(),
            video_path=video_path,
        )
        history.append(entry.model_dump(mode="json"))

        if len(history) > 50:
            history = history[-50:]

        with target.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        logger.error("Erro ao salvar histórico: %s", e)

