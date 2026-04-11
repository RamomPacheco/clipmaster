from __future__ import annotations
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional
from PySide6.QtCore import QThread, Signal
from app.core import config
from app.core.config import LLMParams
from app.core.ffmpeg_bin import ffmpeg_available, get_ffprobe_path
from app.core.logger import ForwardingHandler, logger
from app.models.schemas import (
    Clip,
    ClipList,
    ProcessingMetrics,
    SocialCoverStyle,
    TiktokCaptionStyle,
)
from app.services.clip_manager import (
    append_history_entry,
    build_overlapping_chapters,
    enforce_duration_limits,
    filter_valid_clips,
    rank_clips_by_density,
    remove_duplicate_clips,
    snap_clips_to_transcript,
)
from app.services.engagement_effects_catalog import normalize_engagement_effect_ids
from app.services.llm_analyzer import analyze_viral_potential, generate_social_package
from app.services.transcription import transcribe_audio
from app.services.video_engine import (
    EXPORT_CLIP_SOCIAL_FILENAME,
    EXPORT_CLIP_VIDEO_FILENAME,
    clip_session_subdirectory,
    create_social_cover,
    extract_safe_audio,
    render_clips,
)


def _ffprobe_bin() -> str:
    return get_ffprobe_path() or "ffprobe"


def _ffprobe_format_tags(video_path: Path) -> dict[str, str]:
    try:
        proc = subprocess.run(
            [
                _ffprobe_bin(),
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            return {}
        data = json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}
    fmt = data.get("format") or {}
    tags = fmt.get("tags") or {}
    out: dict[str, str] = {}
    for k, v in tags.items():
        if v is not None and str(v).strip():
            out[str(k)] = str(v)
    return out


def _video_description_folder_label(video_path: Path) -> str:
    """
    Nome lógico do vídeo para a pasta do projeto: metadados (título/descrição)
    ou nome do ficheiro sem extensão.
    """
    tags = _ffprobe_format_tags(video_path)
    for key in (
        "title",
        "TITLE",
        "description",
        "DESCRIPTION",
        "comment",
        "COMMENT",
        "synopsis",
    ):
        raw = tags.get(key)
        if raw:
            t = str(raw).strip().split("\n")[0].strip()
            if t:
                return t
    return video_path.stem


def _sanitize_dir_segment(name: str, max_len: int = 120) -> str:
    name = (name or "").strip()
    if not name:
        name = "projeto"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(". ")
    if not name:
        name = "projeto"
    return name[:max_len]


def _unique_project_subdir(base: Path, video_path: Path) -> Path:
    label = _sanitize_dir_segment(_video_description_folder_label(video_path))
    candidate = base / label
    if not candidate.exists():
        return candidate
    for n in range(1, 1000):
        alt = base / f"{label}_{n}"
        if not alt.exists():
            return alt
    return base / f"{label}_{re.sub(r'[^0-9]', '', str(time.time()))}"


class VideoProcessorThread(QThread):
    """
    Thread dedicada ao processamento intensivo de vídeo e IA.
    Orquestra serviços desacoplados.

    Sinais: ``progress_signal`` (etapas com prefixo > na UI); ``log_signal`` (logging
    espelhado do terminal); ``progress_update`` (barra: valor, máximo; máximo 0 =
    indeterminado).
    """

    progress_signal = Signal(str)
    log_signal = Signal(str)
    progress_update = Signal(int, int)
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
        enable_moviepy_engagement: bool = True,
        bitrate: str = "",
        llm_max_new_tokens: int | None = None,
        custom_prompt: Optional[str] = None,
        social_use_same_model: bool = True,
        social_model_name: Optional[str] = None,
        generate_social_cover: bool = True,
        enable_social_package: bool = True,
        social_cover_style: Optional[SocialCoverStyle] = None,
        tiktok_caption_style: Optional[TiktokCaptionStyle] = None,
        min_clip_seconds: float | None = None,
        max_clip_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.video_path = Path(video_path)
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        base_out = (
            Path(output_dir)
            if output_dir
            else config.EXPORTS_ROOT / f"{self.video_path.stem}_processed"
        )
        self.output_dir = _unique_project_subdir(base_out.resolve(), self.video_path)
        self.prompt_type = prompt_type
        self.whisper_model = whisper_model
        self.whisper_device_mode = whisper_device_mode
        self.selected_clips: Optional[ClipList] = None
        self.resolution = resolution
        self.export_quality = export_quality
        self.aspect_ratio = aspect_ratio
        self.framing_mode = framing_mode
        self.enable_tiktok_captions = enable_tiktok_captions
        self.enable_moviepy_engagement = enable_moviepy_engagement
        self.bitrate = bitrate
        self.llm_max_new_tokens = llm_max_new_tokens
        self.custom_prompt = custom_prompt
        self.social_use_same_model = social_use_same_model
        self.social_model_name = (social_model_name or "").strip() or None
        self.generate_social_cover = generate_social_cover
        self.enable_social_package = enable_social_package
        self.social_cover_style = social_cover_style
        self.tiktok_caption_style = tiktok_caption_style
        self.min_clip_seconds = (
            float(min_clip_seconds)
            if min_clip_seconds is not None
            else float(config.MIN_CLIP_SECONDS)
        )
        self.max_clip_seconds = (
            float(max_clip_seconds)
            if max_clip_seconds is not None
            else float(config.MAX_CLIP_SECONDS)
        )

        self.metrics = ProcessingMetrics(
            model_used=model_name,
            prompt_type=prompt_type,
        )

    # ----------------------
    # Infra
    # ----------------------
    def check_dependencies(self) -> bool:
        if not ffmpeg_available():
            self.error_signal.emit(
                "ERRO CRÍTICO: FFmpeg não encontrado. Coloque ffmpeg em bundled/ffmpeg/<sistema>/, "
                "defina CLIPMASTER_FFMPEG_PATH ou instale no PATH."
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

        root_log = logging.getLogger()
        ui_handler = ForwardingHandler(self.log_signal.emit)
        ui_handler.setLevel(logging.INFO)
        ui_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_log.addHandler(ui_handler)

        try:
            self.progress_update.emit(0, 0)
            self.metrics.start_time = time.time()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.progress_signal.emit(
                f"Pasta deste projeto: «{self.output_dir.name}» (título/descrição do vídeo; destino base: {self.output_dir.parent})."
            )

            # Fase 0 - extração de áudio
            self.progress_signal.emit("Preparando arquivo de áudio (FFmpeg)...")
            self.progress_update.emit(0, 0)
            temp_audio_path = extract_safe_audio(self.video_path, self.output_dir)

            # Fase 1 - transcrição
            device_mode = self.whisper_device_mode.lower()
            if "cpu" in device_mode:
                whisper_device = "cpu"
                whisper_compute = "int8"
                self.progress_signal.emit(
                    f"Transcrição com Whisper em {whisper_device.upper()} ({whisper_compute})."
                )
                self.progress_update.emit(0, 0)
                self.progress_signal.emit(
                    "A iniciar transcrição (1.ª vez pode descarregar o modelo — aguarde)..."
                )
                segments, max_video_duration = transcribe_audio(
                    temp_audio_path,
                    model_name=self.whisper_model,
                    device_override=whisper_device,
                    compute_override=whisper_compute,
                )
            elif "gpu" in device_mode or "cuda" in device_mode:
                whisper_device = "cuda"
                # Mais estável que float16 puro em muitas máquinas Windows/CUDA
                whisper_compute = "int8_float16"
                self.progress_signal.emit(
                    f"Transcrição com Whisper em {whisper_device.upper()} ({whisper_compute})."
                )
                self.progress_update.emit(0, 0)
                self.progress_signal.emit(
                    "A iniciar transcrição (1.ª vez pode descarregar o modelo — aguarde)..."
                )
                segments, max_video_duration = transcribe_audio(
                    temp_audio_path,
                    model_name=self.whisper_model,
                    device_override=whisper_device,
                    compute_override=whisper_compute,
                )
            else:
                # Auto: escolhe CUDA se houver GPU disponível via ctranslate2; caso contrário CPU.
                use_cuda = False
                try:
                    import ctranslate2  # type: ignore[import-not-found]

                    use_cuda = ctranslate2.get_cuda_device_count() > 0
                except Exception:  # noqa: BLE001
                    use_cuda = False

                if use_cuda:
                    whisper_device = "cuda"
                    whisper_compute = "int8_float16"
                else:
                    whisper_device = "cpu"
                    whisper_compute = "int8"

                self.progress_signal.emit(
                    f"Transcrição com Whisper em AUTO: {whisper_device.upper()} ({whisper_compute})."
                )
                self.progress_update.emit(0, 0)
                self.progress_signal.emit(
                    "A iniciar transcrição (1.ª vez pode descarregar o modelo — aguarde)..."
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
            if total_chapters > 0:
                self.progress_update.emit(0, total_chapters)
            else:
                self.progress_update.emit(0, 0)

            for i, chunk in enumerate(chapters, start=1):
                self.progress_signal.emit(
                    f"IA analisando Parte {i} de {total_chapters} "
                    f"(contexto ~{int(config.CHUNK_SECONDS // 60)} min)..."
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
                                engagement_effects=normalize_engagement_effect_ids(
                                    c.get("engagement_effects")
                                ),
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Clipe descartado por dados inválidos: %s", e)

                if total_chapters > 0:
                    self.progress_update.emit(i, total_chapters)

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
                min_seconds=self.min_clip_seconds,
                max_seconds=self.max_clip_seconds,
            )
            all_clips = snap_clips_to_transcript(all_clips, segments)
            all_clips = enforce_duration_limits(
                all_clips,
                max_video_duration=max_video_duration,
                min_seconds=self.min_clip_seconds,
                max_seconds=self.max_clip_seconds,
            )
            all_clips = remove_duplicate_clips(all_clips)
            all_clips = rank_clips_by_density(all_clips, segments, min_words_per_sec=0.8)

            self.metrics.total_clips_found = len(all_clips)
            self.metrics.video_duration = max_video_duration

            self.progress_signal.emit(
                f"Análise concluída. {len(all_clips)} clipes únicos identificados."
            )

            if not all_clips:
                self.progress_signal.emit(
                    "Aviso: A IA não encontrou nenhum clipe viral forte o suficiente."
                )
                logger.warning(
                    "Nenhum clipe encontrado (modelo=%s, num_ctx=%s). "
                    "Se usar Ollama: confirme `ollama serve` e `ollama pull %s`.",
                    self.model_name,
                    LLMParams.NUM_CTX,
                    self.model_name,
                )
                self.finished_signal.emit(
                    "Processamento concluído sem clipes extraídos."
                )
                return

            if total_chapters > 0:
                self.progress_update.emit(total_chapters, total_chapters)

            # Envia para UI (como lista de dicts)
            self.progress_signal.emit(
                f"✓ Análise completa! {len(all_clips)} clipes identificados. Aguardando sua seleção..."
            )
            self.clips_ready_signal.emit([c.to_dict() for c in all_clips])

            # Espera seleção do usuário
            self.progress_update.emit(0, 0)
            while self.selected_clips is None:
                time.sleep(0.5)

            if not self.selected_clips:
                self.progress_signal.emit("Nenhum clipe foi selecionado. Cancelando...")
                self.finished_signal.emit("Processamento cancelado pelo usuário.")
                return

            # Fase 5 - renderização apenas dos selecionados
            n_render = len(self.selected_clips)
            self.progress_signal.emit(
                f"Iniciando renderização de {n_render} clipes selecionados..."
            )
            self.progress_update.emit(0, n_render)
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
                enable_moviepy_engagement=self.enable_moviepy_engagement,
                bitrate=self.bitrate or None,
                tiktok_caption_style=self.tiktok_caption_style,
                on_clip_progress=lambda done, total: self.progress_update.emit(done, total),
            )
            if n_render > 0:
                self.progress_update.emit(n_render, n_render)

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
                n_soc = len(self.selected_clips)
                self.progress_update.emit(0, n_soc)
                for i, clip in enumerate(self.selected_clips, start=1):
                    clip_dir = clip_session_subdirectory(self.output_dir, i)
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
                            output_dir=clip_dir,
                            clip_index=i,
                            frame_second_abs=frame_second_abs,
                            hook_phrase=hook,
                            resolution=self.resolution,
                            export_quality=self.export_quality,
                            aspect_ratio=self.aspect_ratio,
                            framing_mode=self.framing_mode,
                            clip=clip,
                            cover_style=self.social_cover_style,
                        )
                        cover_line = cover_path.name
                        cover_log = cover_path.name
                    else:
                        cover_line = "(capa desativada na aba Pacote Social)"
                        cover_log = "sem capa"

                    social_path = clip_dir / EXPORT_CLIP_SOCIAL_FILENAME
                    social_path.write_text(
                        (
                            f"Clipe: {EXPORT_CLIP_VIDEO_FILENAME}\n"
                            f"Capa: {cover_line}\n"
                            f"Frase de impacto: {hook}\n\n"
                            "Descrição para redes:\n"
                            f"{description}\n"
                        ),
                        encoding="utf-8",
                    )
                    self.progress_signal.emit(
                        f"Pacote social em {clip_dir.name}/ ({cover_log} + {social_path.name})."
                    )
                    self.progress_update.emit(i, n_soc)
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
        finally:
            root_log.removeHandler(ui_handler)

