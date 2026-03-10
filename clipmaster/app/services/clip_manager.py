from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List

from app.core import config
from app.core.logger import logger
from app.models.schemas import Clip, ProcessingHistoryEntry, ProcessingMetrics


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

