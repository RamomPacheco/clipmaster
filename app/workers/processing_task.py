from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from app.core import config
from app.core.logger import logger
from app.models.schemas import Clip, ClipList, ProcessingMetrics
from app.services.clip_manager import (
    append_history_entry,
    build_overlapping_chapters,
    enforce_duration_limits,
    filter_valid_clips,
    remove_duplicate_clips,
    snap_clips_to_transcript,
)
from app.services.llm_analyzer import analyze_viral_potential
from app.services.transcription import transcribe_audio
from app.services.video_engine import extract_safe_audio, render_clips


class VideoProcessorThread(QThread):
    """
    Thread dedicada ao processamento intensivo de vídeo e IA.
    Agora orquestra serviços desacoplados.
    """

    progress_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)
    clips_ready_signal = Signal(list)  # Lista de dicts para compatibilidade com UI

    def __init__(
        self,
        video_path: str,
        model_name: str,
        output_dir: Optional[str] = None,
        prompt_type: str = "Padrão (Equilibrado)",
        whisper_model: str | None = None,
        resolution: str = "1080p",
        bitrate: str = "",
        custom_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.video_path = Path(video_path)
        self.model_name = model_name
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else config.EXPORTS_ROOT / f"{self.video_path.stem}_processed"
        )
        self.prompt_type = prompt_type
        self.whisper_model = whisper_model
        self.selected_clips: Optional[ClipList] = None
        self.resolution = resolution
        self.bitrate = bitrate
        self.custom_prompt = custom_prompt

        self.metrics = ProcessingMetrics(
            model_used=model_name,
            prompt_type=prompt_type,
        )

    # ----------------------
    # Infra
    # ----------------------
    def check_dependencies(self) -> bool:
        if not shutil.which("ffmpeg"):
            self.error_signal.emit(
                "ERRO CRÍTICO: FFmpeg não detectado nas Variáveis de Ambiente (PATH)!"
            )
            return False
        return True

    # ----------------------
    # Pipeline principal
    # ----------------------
    def run(self) -> None:  # type: ignore[override]
        if not self.check_dependencies():
            return

        try:
            self.metrics.start_time = time.time()
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Fase 0 - extração de áudio
            self.progress_signal.emit("Preparando arquivo de áudio (FFmpeg)...")
            temp_audio_path = extract_safe_audio(self.video_path, self.output_dir)

            # Fase 1 - transcrição
            segments, max_video_duration = transcribe_audio(
                temp_audio_path, model_name=self.whisper_model
            )
            self.metrics.transcription_time = time.time() - self.metrics.start_time

            # Remove áudio temporário
            try:
                temp_audio_path.unlink()
            except Exception:  # noqa: BLE001
                pass

            self.progress_signal.emit(
                "Transcrição concluída. A preparar motor de Análise (Ollama)..."
            )

            import gc

            gc.collect()

            # Fase 2 - chunking com sobreposição (evita cortar ideias na junção de blocos)
            analysis_start = time.time()
            chapters = build_overlapping_chapters(
                segments,
                config.CHUNK_SECONDS,
                config.CHUNK_OVERLAP_SECONDS,
            )

            # Fase 3 - análise via LLM
            all_clips: List[Clip] = []
            total_chapters = len(chapters)

            for i, chunk in enumerate(chapters, start=1):
                self.progress_signal.emit(
                    f"IA analisando Parte {i} de {total_chapters} (Contexto de 10 min)..."
                )
                chunk_text = "\n".join(
                    [
                        f"[{s['start']:.2f}s - {s['end']:.2f}s]: {s['text']}"
                        for s in chunk
                    ]
                )
                raw_clips = analyze_viral_potential(
                    text=chunk_text,
                    model_name=self.model_name,
                    prompt_type=self.prompt_type,
                    custom_prompt=self.custom_prompt,
                )
                for c in raw_clips:
                    try:
                        all_clips.append(
                            Clip(
                                start=float(c.get("start", 0.0)),
                                end=float(c.get("end", 0.0)),
                                reason=str(c.get("reason", "")),
                                headline=str(c.get("headline", "Sem título")),
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Clipe descartado por dados inválidos: %s", e)

            self.metrics.analysis_time = time.time() - analysis_start

            # Fase 3.5 - alinhar aos limites da transcrição e ajustar duração
            all_clips = filter_valid_clips(
                all_clips,
                max_video_duration=max_video_duration,
                min_duration=0.5,
            )
            all_clips = snap_clips_to_transcript(all_clips, segments)
            all_clips = filter_valid_clips(
                all_clips,
                max_video_duration=max_video_duration,
                min_duration=0.5,
            )
            all_clips = enforce_duration_limits(
                all_clips,
                max_video_duration=max_video_duration,
                min_seconds=config.MIN_CLIP_SECONDS,
                max_seconds=config.MAX_CLIP_SECONDS,
            )
            all_clips = snap_clips_to_transcript(all_clips, segments)
            all_clips = enforce_duration_limits(
                all_clips,
                max_video_duration=max_video_duration,
                min_seconds=config.MIN_CLIP_SECONDS,
                max_seconds=config.MAX_CLIP_SECONDS,
            )
            all_clips = remove_duplicate_clips(all_clips)

            self.metrics.total_clips_found = len(all_clips)
            self.metrics.video_duration = max_video_duration

            self.progress_signal.emit(
                f"Análise concluída. {len(all_clips)} clipes únicos identificados."
            )

            if not all_clips:
                self.progress_signal.emit(
                    "Aviso: A IA não encontrou nenhum clipe viral forte o suficiente."
                )
                self.finished_signal.emit(
                    "Processamento concluído sem clipes extraídos."
                )
                return

            # Envia para UI (como lista de dicts)
            self.progress_signal.emit(
                f"✓ Análise completa! {len(all_clips)} clipes identificados. Aguardando sua seleção..."
            )
            self.clips_ready_signal.emit([c.to_dict() for c in all_clips])

            # Espera seleção do usuário
            while self.selected_clips is None:
                time.sleep(0.5)

            if not self.selected_clips:
                self.progress_signal.emit("Nenhum clipe foi selecionado. Cancelando...")
                self.finished_signal.emit("Processamento cancelado pelo usuário.")
                return

            # Fase 5 - renderização apenas dos selecionados
            self.progress_signal.emit(
                f"Iniciando renderização de {len(self.selected_clips)} clipes selecionados..."
            )
            rendering_start = time.time()
            render_clips(
                video_path=self.video_path,
                clips=self.selected_clips,
                output_dir=self.output_dir,
                resolution=self.resolution,
                bitrate=self.bitrate or None,
            )

            self.metrics.rendering_time = time.time() - rendering_start
            self.metrics.clips_selected = len(self.selected_clips)

            append_history_entry(self.metrics, self.video_path)

            self.finished_signal.emit(
                f"Sucesso! {len(self.selected_clips)} clipes gerados e salvos em:\n{self.output_dir.absolute()}"
            )

        except Exception as e:  # noqa: BLE001
            logger.exception("Erro inesperado no pipeline.")
            self.error_signal.emit(f"Erro inesperado: {e}")

