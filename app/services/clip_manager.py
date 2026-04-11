from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.core import config
from app.core.logger import logger
from app.models.schemas import Clip, ProcessingHistoryEntry, ProcessingMetrics

_SENTENCE_END_RE = re.compile(r"[.!?\u2026]+\s*$")


def build_overlapping_chapters(
    segments: List[Dict[str, Any]],
    chunk_seconds: float,
    overlap_seconds: float,
) -> List[List[Dict[str, Any]]]:
    """
    Divide a transcrição em blocos de até `chunk_seconds`, com cauda sobreposta
    para o próximo bloco não começar "do zero" no meio de uma ideia.
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


# ────────────────────────────────────────────────────────────────────
# Utilitários internos de snapping
# ────────────────────────────────────────────────────────────────────

def _flatten_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for s in segments:
        for w in s.get("words") or []:
            words.append(w)
    words.sort(key=lambda w: float(w["start"]))
    return words


def _flatten_words_with_segment_text(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """Devolve (words_sorted, {word_index: segment_text_do_segmento_pai})."""
    items: List[Tuple[float, Dict[str, Any], str]] = []
    for s in segments:
        seg_text = str(s.get("text", "")).strip()
        for w in s.get("words") or []:
            items.append((float(w["start"]), w, seg_text))
    items.sort(key=lambda x: x[0])
    flat = [w for _, w, _ in items]
    seg_map = {i: t for i, (_, _, t) in enumerate(items)}
    return flat, seg_map


def _word_gap_ms(w_prev: Dict[str, Any], w_next: Dict[str, Any]) -> float:
    """Intervalo (ms) entre o fim de w_prev e o inicio de w_next."""
    return max(0.0, (float(w_next["start"]) - float(w_prev["end"])) * 1000.0)


def _is_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search((text or "").strip()))


def _score_cut_point(
    gap_ms: float,
    is_sentence_boundary: bool,
    distance_sec: float,
    window_sec: float,
) -> float:
    """
    Pontua um candidato a ponto de corte. Quanto MAIS alto, melhor.
    - Pausa longa entre palavras = bom (nao corta no meio da fala).
    - Fronteira de frase = bom (sentido completo).
    - Proximidade ao ponto original da IA = bom (menos desvio).
    """
    min_pause = float(config.SNAP_MIN_PAUSE_MS)
    w_pause = config.SNAP_PAUSE_WEIGHT
    w_sent = config.SNAP_SENTENCE_WEIGHT
    w_prox = config.SNAP_PROXIMITY_WEIGHT

    pause_score = min(gap_ms / max(min_pause, 1.0), 3.0) * w_pause
    sentence_score = (1.0 if is_sentence_boundary else 0.0) * w_sent
    proximity_score = max(0.0, 1.0 - (distance_sec / max(window_sec, 0.01))) * w_prox

    return pause_score + sentence_score + proximity_score


def _best_start_near(
    t: float,
    words: List[Dict[str, Any]],
    seg_map: Dict[int, str],
    window: float,
) -> float:
    """Melhor ponto de INICIO dentro de [t-window, t+window]: depois de pausa/frase."""
    candidates: List[Tuple[float, float]] = []
    lo = t - window
    hi = t + window

    for i, w in enumerate(words):
        ws = float(w["start"])
        if ws < lo:
            continue
        if ws > hi:
            break
        gap = _word_gap_ms(words[i - 1], w) if i > 0 else 999.0
        prev_text = str(words[i - 1].get("word", "")) if i > 0 else "."
        seg_text = seg_map.get(i - 1, "")
        sentence = _is_sentence_end(prev_text) or _is_sentence_end(seg_text)
        score = _score_cut_point(gap, sentence, abs(ws - t), window)
        candidates.append((score, ws))

    if not candidates:
        return t
    candidates.sort(key=lambda x: (-x[0], abs(x[1] - t)))
    return candidates[0][1]


def _best_end_near(
    t: float,
    words: List[Dict[str, Any]],
    seg_map: Dict[int, str],
    window: float,
) -> float:
    """Melhor ponto de FIM dentro de [t-window, t+window]: apos frase/pausa."""
    candidates: List[Tuple[float, float]] = []
    lo = t - window
    hi = t + window

    for i, w in enumerate(words):
        we = float(w["end"])
        if we < lo:
            continue
        if we > hi:
            break
        gap = _word_gap_ms(w, words[i + 1]) if i + 1 < len(words) else 999.0
        word_text = str(w.get("word", ""))
        seg_text = seg_map.get(i, "")
        sentence = _is_sentence_end(word_text) or _is_sentence_end(seg_text)
        score = _score_cut_point(gap, sentence, abs(we - t), window)
        candidates.append((score, we))

    if not candidates:
        return t
    candidates.sort(key=lambda x: (-x[0], abs(x[1] - t)))
    return candidates[0][1]


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


# ────────────────────────────────────────────────────────────────────
# API publica de snapping
# ────────────────────────────────────────────────────────────────────

def snap_clip_to_transcript(
    clip: Clip,
    segments: List[Dict[str, Any]],
) -> Clip:
    """
    Alinha inicio/fim do clipe ao melhor ponto de corte da transcricao,
    preferindo: pausas longas entre palavras > fim de frase > proximidade ao ponto da IA.
    Funciona com qualquer modelo -- a "inteligencia" vem dos timestamps do Whisper.
    """
    if not segments:
        return clip

    words, seg_map = _flatten_words_with_segment_text(segments)
    window = config.SNAP_SEARCH_WINDOW_SEC

    if len(words) >= 2:
        start = _best_start_near(float(clip.start), words, seg_map, window)
        end = _best_end_near(float(clip.end), words, seg_map, window)
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


# ────────────────────────────────────────────────────────────────────
# Score de densidade de fala (ranking independente do LLM)
# ────────────────────────────────────────────────────────────────────

def speech_density_score(clip: Clip, segments: List[Dict[str, Any]]) -> float:
    """
    Retorna palavras-por-segundo dentro do intervalo do clipe.
    Trechos com mais fala densa tendem a ser mais "engajantes".
    Pode ser usado para filtrar/rankear depois da IA.
    """
    duration = max(0.1, float(clip.end) - float(clip.start))
    word_count = 0
    for s in segments:
        s0 = float(s.get("start", 0.0))
        s1 = float(s.get("end", 0.0))
        if s1 <= clip.start or s0 >= clip.end:
            continue
        word_count += len(s.get("words") or [])
    return word_count / duration


def rank_clips_by_density(
    clips: List[Clip],
    segments: List[Dict[str, Any]],
    min_words_per_sec: float = 1.0,
) -> List[Clip]:
    """
    Reordena clipes pelo score de densidade (mais denso primeiro).
    Remove clipes com densidade abaixo de min_words_per_sec (silencio demais).
    """
    scored = [
        (speech_density_score(c, segments), c) for c in clips
    ]
    scored.sort(key=lambda x: -x[0])
    kept = [(s, c) for s, c in scored if s >= min_words_per_sec]
    removed = len(clips) - len(kept)
    if removed > 0:
        logger.info(
            "Removidos %d clipe(s) com densidade de fala inferior a %.1f palavras/s.",
            removed,
            min_words_per_sec,
        )
    return [c for _, c in kept]


# ────────────────────────────────────────────────────────────────────
# Filtro / limites de duracao
# ────────────────────────────────────────────────────────────────────

def filter_valid_clips(
    clips: Iterable[Clip],
    max_video_duration: float,
    min_duration: float = 1.0,
) -> List[Clip]:
    """Remove clipes invalidos ou fora do video."""
    out: List[Clip] = []
    for c in clips:
        start = float(c.start)
        end = float(c.end)
        if start < 0 or end <= start:
            logger.info("Clipe descartado: intervalo invalido (%.2f-%.2f)", start, end)
            continue
        if start >= max_video_duration:
            logger.info("Clipe descartado: inicio apos o fim do video")
            continue
        end = min(end, max_video_duration)
        if end - start < min_duration:
            logger.info("Clipe descartado: duracao muito curta (%.2fs)", end - start)
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
    reproduzindo a logica de _enforce_duration_limits do codigo antigo.
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
                "Clipe %s excedeu o limite maximo configurado (%.1fs). Cortando em %.1fs.",
                i,
                duration,
                max_seconds,
            )
            clip = clip.model_copy(
                update={
                    "end": round(start + max_seconds, 2),
                    "reason": clip.reason
                    + " [Nota de Backend: Final cortado para respeitar teto de 120s]",
                }
            )

        adjusted.append(clip)

    return adjusted


def remove_duplicate_clips(clips: Iterable[Clip]) -> List[Clip]:
    """
    Remove clipes duplicados/sobrepostos em mais de 50%, preservando o com razao mais detalhada.
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
    Atualiza o arquivo de historico de processamento com uma nova entrada.
    Mantem o mesmo formato de JSON do codigo original.
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
        logger.error("Erro ao salvar historico: %s", e)
