from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from app.core.logger import logger
from app.services.engagement_effects_catalog import resolve_moviepy_effect_ids
from app.services.video_engine import _ffmpeg_has_encoder

# MoviePy 1.0.3 ainda referencia PIL.Image.ANTIALIAS (removido no Pillow 10+).
try:
    from PIL import Image as _PILImage

    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.LANCZOS  # type: ignore[attr-defined]
except ImportError:
    pass

try:
    from moviepy.editor import CompositeVideoClip, VideoFileClip  # type: ignore[import-untyped]
    import moviepy.video.fx.all as vfx  # type: ignore[import-untyped]

    MOVIEPY_AVAILABLE = True
except ImportError:  # pragma: no cover - ambiente sem moviepy
    MOVIEPY_AVAILABLE = False
    VideoFileClip = Any  # type: ignore[misc,assignment]


def _subtle_center_push(clip: Any, end_scale: float = 1.038) -> Any:
    """Zoom lento e centrado (Ken Burns discreto) ao longo do clipe."""
    w, h = clip.w, clip.h
    duration = max(float(clip.duration), 1e-3)
    z0, z1 = 1.0, float(end_scale)

    def scale_at(t: float) -> float:
        return z0 + (z1 - z0) * min(1.0, max(0.0, float(t) / duration))

    growing = clip.fx(vfx.resize, scale_at)
    return CompositeVideoClip([growing.set_position("center")], size=(w, h)).set_duration(
        clip.duration
    )


def apply_engagement_effects(
    input_path: Path,
    output_path: Path,
    *,
    crf: str = "21",
    preset: str = "medium",
    prefer_nvenc: bool = True,
    engagement_effects: Sequence[str] | None = None,
) -> bool:
    """
    Lê ``input_path`` (H.264/AAC típico do FFmpeg do app) e grava ``output_path``
    com efeitos de retenção no vídeo. Retorna True se gravou com sucesso.

    ``engagement_effects``: IDs pedidos pela IA/UI; ``None`` internamente significa
    “não aplicar MoviePy” (valor ``none`` do catálogo). Lista vazia após resolver
    não deve ocorrer — o chamador usa o default do catálogo antes.

    Com ``prefer_nvenc`` ativo e encoder ``h264_nvenc`` disponível no FFmpeg
    (igual ao primeiro passe em ``video_engine``), usa GPU; caso contrário ``libx264``.
    """
    resolved = resolve_moviepy_effect_ids(engagement_effects)
    if resolved is None:
        return False

    if not MOVIEPY_AVAILABLE:
        logger.warning(
            "MoviePy não está instalado; pule `pip install moviepy` ou desmarque efeitos na exportação."
        )
        return False

    inp = Path(input_path)
    out = Path(output_path)
    if not inp.is_file():
        logger.warning("Arquivo de entrada inexistente para MoviePy: %s", inp)
        return False

    out.parent.mkdir(parents=True, exist_ok=True)

    clip: Any = None
    staged: Any = None
    final: Any = None

    try:
        clip = VideoFileClip(str(inp))
        d = max(float(clip.duration), 1e-3)
        fade_in = min(0.14, max(0.034, d * 0.11))
        fade_out = min(0.22, max(0.045, d * 0.17))

        staged = clip
        if "push_in_subtle" in resolved:
            staged = _subtle_center_push(staged, end_scale=1.038)

        if "fade_video_edges" in resolved:
            final = staged.fx(vfx.fadein, fade_in).fx(vfx.fadeout, fade_out)
        else:
            final = staged

        if clip.audio is not None:
            final = final.set_audio(clip.audio)

        use_nvenc = bool(prefer_nvenc) and _ffmpeg_has_encoder("h264_nvenc")
        cq = str(int(float(str(crf).strip() or "21")))

        if use_nvenc:
            logger.info(
                "MoviePy: segundo passe com NVENC (preset=%s, cq=%s), alinhado ao FFmpeg da exportação.",
                preset,
                cq,
            )
            kwargs = {
                "codec": "h264_nvenc",
                "audio_codec": "aac",
                "temp_audiofile": str(out.with_suffix(".temp-audio.m4a")),
                "remove_temp": True,
                "preset": preset,
                "ffmpeg_params": ["-cq", cq, "-pix_fmt", "yuv420p"],
                "logger": None,
            }
        else:
            if prefer_nvenc:
                logger.info(
                    "MoviePy: NVENC não disponível no FFmpeg; segundo passe com libx264 (CRF %s).",
                    crf,
                )
            kwargs = {
                "codec": "libx264",
                "audio_codec": "aac",
                "temp_audiofile": str(out.with_suffix(".temp-audio.m4a")),
                "remove_temp": True,
                "preset": preset,
                "ffmpeg_params": ["-crf", str(crf), "-pix_fmt", "yuv420p"],
                "logger": None,
            }
        if clip.fps is not None and float(clip.fps) > 0:
            kwargs["fps"] = float(clip.fps)

        try:
            final.write_videofile(str(out), **kwargs)
        except Exception as first_err:  # noqa: BLE001
            if not use_nvenc:
                raise
            logger.warning(
                "MoviePy: falha ao gravar com NVENC (%s); a repetir com libx264.",
                first_err,
            )
            kwargs_fallback = {
                "codec": "libx264",
                "audio_codec": "aac",
                "temp_audiofile": str(out.with_suffix(".temp-audio.m4a")),
                "remove_temp": True,
                "preset": preset,
                "ffmpeg_params": ["-crf", str(crf), "-pix_fmt", "yuv420p"],
                "logger": None,
            }
            if clip.fps is not None and float(clip.fps) > 0:
                kwargs_fallback["fps"] = float(clip.fps)
            if out.exists():
                out.unlink(missing_ok=True)
            final.write_videofile(str(out), **kwargs_fallback)
        return out.is_file()
    except Exception as e:  # noqa: BLE001
        logger.warning("Falha no pós-processamento MoviePy (%s → %s): %s", inp, out, e)
        try:
            if out.exists():
                out.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    finally:
        for c in (final, staged, clip):
            if c is not None:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
