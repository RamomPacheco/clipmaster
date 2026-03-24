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
from app.services.llm_analyzer import analyze_viral_potential, generate_social_package
from app.services.transcription import transcribe_audio
from app.services.video_engine import create_social_cover, extract_safe_audio, render_clips


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
        llm_provider: str = "ollama",
        llm_api_key: str | None = None,
        output_dir: Optional[str] = None,
        prompt_type: str = "Padrão (Equilibrado)",
        whisper_model: str | None = None,
        whisper_device_mode: str = "Auto (recomendado)",
        resolution: str = "1080p",
        export_quality: str = "Alta (mais lenta)",
        aspect_ratio: str = "Vertical (9:16) - Redes sociais",
        framing_mode: str = "Manter conteúdo (com bordas)",
        enable_tiktok_captions: bool = False,
        bitrate: str = "",
        llm_max_new_tokens: int | None = None,
        custom_prompt: Optional[str] = None,
        social_use_same_model: bool = True,
        social_model_name: Optional[str] = None,
        generate_social_cover: bool = True,
        enable_social_package: bool = True,
    ) -> None:
        super().__init__()
        self.video_path = Path(video_path)
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else config.EXPORTS_ROOT / f"{self.video_path.stem}_processed"
        )
        self.prompt_type = prompt_type
        self.whisper_model = whisper_model
        self.whisper_device_mode = whisper_device_mode
        self.selected_clips: Optional[ClipList] = None
        self.resolution = resolution
        self.export_quality = export_quality
        self.aspect_ratio = aspect_ratio
        self.framing_mode = framing_mode
        self.enable_tiktok_captions = enable_tiktok_captions
        self.bitrate = bitrate
        self.llm_max_new_tokens = llm_max_new_tokens
        self.custom_prompt = custom_prompt
        self.social_use_same_model = social_use_same_model
        self.social_model_name = (social_model_name or "").strip() or None
        self.generate_social_cover = generate_social_cover
        self.enable_social_package = enable_social_package

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

    def _build_clip_transcript_text(self, clip: Clip, segments: list[dict]) -> str:
        lines: list[str] = []
        for seg in segments:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            if s1 <= clip.start or s0 >= clip.end:
                continue
            txt = str(seg.get("text", "")).strip()
            if not txt:
                continue
            lines.append(f"[{s0:.2f}s - {s1:.2f}s] {txt}")
        return "\n".join(lines)

    def _build_narrative_context_up_to(
        self,
        segments: list[dict],
        until_abs: float,
        max_chars: int = 28000,
    ) -> str:
        """
        Transcrição acumulada do vídeo original desde o início até ``until_abs`` (ex.: fim do clipe).
        Se exceder ``max_chars``, mantém apenas o final (mais próximo do momento do clipe).
        """
        ordered = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
        lines: list[str] = []
        for seg in ordered:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            if s1 <= 0.0:
                continue
            if s0 >= until_abs:
                break
            txt = str(seg.get("text", "")).strip()
            if not txt:
                continue
            lines.append(f"[{s0:.2f}s - {s1:.2f}s] {txt}")
        full = "\n".join(lines)
        if len(full) <= max_chars:
            return full
        return full[-max_chars:]

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
            device_mode = self.whisper_device_mode.lower()
            if "cpu" in device_mode:
                whisper_device = "cpu"
                whisper_compute = "int8"
            elif "gpu" in device_mode or "cuda" in device_mode:
                whisper_device = "cuda"
                # Mais estável que float16 puro em muitas máquinas Windows/CUDA
                whisper_compute = "int8_float16"
            else:
                # Auto: prioriza estabilidade (CPU) quando possível.
                # Se quiser forçar GPU, use a opção explícita "GPU CUDA".
                whisper_device = "cpu"
                whisper_compute = "int8"
            self.progress_signal.emit(
                f"Transcrição com Whisper em {whisper_device.upper()} ({whisper_compute})."
            )
            segments, max_video_duration = transcribe_audio(
                temp_audio_path,
                model_name=self.whisper_model,
                device_override=whisper_device,
                compute_override=whisper_compute,
            )
            self.metrics.transcription_time = time.time() - self.metrics.start_time

            # Remove áudio temporário
            try:
                temp_audio_path.unlink()
            except Exception:  # noqa: BLE001
                pass

            self.progress_signal.emit(
                f"Transcrição concluída. A preparar motor de Análise ({self.llm_provider.title()})..."
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
                    provider=self.llm_provider,
                    api_key=self.llm_api_key,
                    max_new_tokens=self.llm_max_new_tokens,
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
                segments=segments,
                resolution=self.resolution,
                export_quality=self.export_quality,
                aspect_ratio=self.aspect_ratio,
                framing_mode=self.framing_mode,
                enable_tiktok_captions=self.enable_tiktok_captions,
                bitrate=self.bitrate or None,
            )

            self.metrics.rendering_time = time.time() - rendering_start
            self.metrics.clips_selected = len(self.selected_clips)

            # Fase 6 - pacote social (só se ativado na UI; senão pipeline termina após render)
            if self.enable_social_package:
                social_model = (
                    self.model_name
                    if self.social_use_same_model
                    else (self.social_model_name or self.model_name)
                )
                self.progress_signal.emit(
                    "Gerando pacote social dos clipes (modelo: "
                    f"{social_model}"
                    + (" — com capa JPG." if self.generate_social_cover else " — sem capa JPG.")
                )
                for i, clip in enumerate(self.selected_clips, start=1):
                    clip_text = self._build_clip_transcript_text(clip, segments)
                    narrative = self._build_narrative_context_up_to(
                        segments, until_abs=float(clip.end)
                    )
                    social = generate_social_package(
                        narrative_context=narrative,
                        clip_text=clip_text,
                        clip_start=float(clip.start),
                        clip_end=float(clip.end),
                        model_name=social_model,
                        provider=self.llm_provider,
                        api_key=self.llm_api_key,
                        max_new_tokens=self.llm_max_new_tokens,
                    )

                    hook = str(social.get("hook_phrase", "")).strip() or clip.headline
                    description = str(social.get("description", "")).strip() or clip.reason
                    frame_second_rel = float(
                        social.get("frame_second", max(0.0, clip.duration * 0.45))
                    )
                    frame_second_rel = max(
                        0.0, min(frame_second_rel, max(0.0, float(clip.duration) - 0.1))
                    )
                    frame_second_abs = float(clip.start) + frame_second_rel

                    cover_line: str
                    if self.generate_social_cover:
                        cover_path = create_social_cover(
                            video_path=self.video_path,
                            output_dir=self.output_dir,
                            clip_index=i,
                            frame_second_abs=frame_second_abs,
                            hook_phrase=hook,
                            resolution=self.resolution,
                            export_quality=self.export_quality,
                            aspect_ratio=self.aspect_ratio,
                            framing_mode=self.framing_mode,
                            clip=clip,
                        )
                        cover_line = cover_path.name
                        cover_log = cover_path.name
                    else:
                        cover_line = "(capa desativada na aba Pacote Social)"
                        cover_log = "sem capa"

                    social_path = self.output_dir / f"clip_{i}_social.txt"
                    social_path.write_text(
                        (
                            f"Clipe: clip_{i}_viral.mp4\n"
                            f"Capa: {cover_line}\n"
                            f"Frase de impacto: {hook}\n\n"
                            "Descrição para redes:\n"
                            f"{description}\n"
                        ),
                        encoding="utf-8",
                    )
                    self.progress_signal.emit(
                        f"Pacote social do clipe {i} pronto ({cover_log} + {social_path.name})."
                    )
            else:
                self.progress_signal.emit(
                    "Pacote social desativado: sem contexto narrativo extra, capa nem ficheiros social."
                )

            append_history_entry(self.metrics, self.video_path)

            self.finished_signal.emit(
                f"Sucesso! {len(self.selected_clips)} clipes gerados e salvos em:\n{self.output_dir.absolute()}"
            )

        except Exception as e:  # noqa: BLE001
            logger.exception("Erro inesperado no pipeline.")
            self.error_signal.emit(f"Erro inesperado: {e}")

