from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.core.logger import logger
from app.models.schemas import Clip


def extract_safe_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extrai um WAV mono leve a 16kHz para o Whisper.
    Equivalente ao bloco FFmpeg que gerava temp_audio_safe.wav.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = output_dir / "temp_audio_safe.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(temp_audio_path),
    ]
    logger.info("Extraindo áudio seguro via FFmpeg...")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return temp_audio_path


def render_clips(
    video_path: Path,
    clips: Iterable[Clip],
    output_dir: Path,
    segments: List[Dict[str, Any]] | None = None,
    resolution: str = "1080p",
    export_quality: str = "Alta (mais lenta)",
    aspect_ratio: str = "Vertical (9:16) - Redes sociais",
    framing_mode: str = "Manter conteúdo (com bordas)",
    enable_tiktok_captions: bool = False,
    bitrate: str | None = None,
) -> None:
    """
    Renderiza uma lista de clipes para MP4 H.264, mantendo a mesma lógica do código original.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_map = {
        "SD (720p)": {"preset": "veryfast", "crf": "24", "height": 720},
        "HD (1080p)": {"preset": "medium", "crf": "21", "height": 1080},
        "2K (1440p)": {"preset": "slow", "crf": "20", "height": 1440},
        "4K (2160p)": {"preset": "slow", "crf": "18", "height": 2160},
    }
    fallback_profile = {"preset": "medium", "crf": "21", "height": 1080}
    profile = profile_map.get(export_quality) or profile_map.get(resolution) or fallback_profile

    is_vertical = "9:16" in aspect_ratio
    base_h = int(profile["height"])
    base_w = int(round(base_h * (9 / 16 if is_vertical else 16 / 9)))
    target_w = base_w - (base_w % 2)
    target_h = base_h - (base_h % 2)
    target_ratio = target_w / target_h

    def _detect_face_center_ratio_for_clip(clip: Clip) -> float | None:
        """
        Detecta o centro horizontal médio de rosto no intervalo do clipe.
        Retorna um ratio entre 0 e 1 (posição x relativa) ou None.
        """
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            logger.warning(
                "OpenCV não disponível. Crop inteligente usará fallback central. "
                "Instale 'opencv-python' para detecção de rosto."
            )
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                return None

            clip_start = float(clip.start)
            clip_end = float(clip.end)
            duration = max(0.1, clip_end - clip_start)
            sample_count = 8
            centers: List[float] = []

            for idx in range(sample_count):
                t = clip_start + (duration * (idx + 0.5) / sample_count)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )
                if len(faces) == 0:
                    continue
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                frame_w = float(frame.shape[1]) if frame.shape[1] else 1.0
                centers.append((x + (w / 2.0)) / frame_w)

            if not centers:
                return None
            return max(0.0, min(1.0, sum(centers) / len(centers)))
        finally:
            cap.release()

    def _crop_filter_with_center_x(center_ratio: float | None) -> str:
        """
        Cria filtro de crop sem distorcer, centralizado no rosto quando possível.
        """
        if center_ratio is None:
            return (
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
        x_expr = f"max(0,min(iw-{target_w},iw*{center_ratio:.6f}-{target_w/2:.2f}))"
        # Em filtergraph do FFmpeg, vírgulas da expressão precisam ser escapadas,
        # senão são interpretadas como separador de filtros.
        x_expr = x_expr.replace(",", r"\,")
        return (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}:{x_expr}:0"
        )

    def sec_to_ass(ts: float) -> str:
        ts = max(0.0, ts)
        total_cs = int(round(ts * 100))
        hours = total_cs // 360000
        rem = total_cs % 360000
        minutes = rem // 6000
        rem = rem % 6000
        seconds = rem // 100
        centiseconds = rem % 100
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def ass_escape(text: str) -> str:
        return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

    def build_tiktok_ass_for_clip(clip: Clip, clip_index: int) -> Path | None:
        if not segments:
            return None
        #TODO: Ajustar a legenda para o TikTok
        # Ajuste fino para evitar adiantamento residual da legenda.
        # Valor positivo atrasa levemente a exibição.
        subtitle_delay_s = 0.20
        words: List[Dict[str, Any]] = []
        for seg in segments:
            for w in seg.get("words") or []:
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", 0.0))
                if we <= clip.start or ws >= clip.end:
                    continue
                words.append(
                    {
                        "start": max(ws, clip.start),
                        "end": min(we, clip.end),
                        "word": str(w.get("word", "")).strip(),
                    }
                )
        words = [w for w in words if w["word"] and w["end"] > w["start"]]

        ass_path = output_dir / f"clip_{clip_index}_captions.ass"
        lines: List[str] = []
        if words:
            groups = [words[idx : idx + 4] for idx in range(0, len(words), 4)]
            for group in groups:
                start = max(0.0, (group[0]["start"] - clip.start) + subtitle_delay_s)
                end = max(start + 0.01, (group[-1]["end"] - clip.start) + subtitle_delay_s)
                karaoke_text = ""
                for idx, item in enumerate(group):
                    current_start = float(item["start"])
                    if idx < len(group) - 1:
                        # Usa o início da próxima palavra para manter pausas naturais
                        # e evitar adiantamento progressivo da legenda.
                        next_start = float(group[idx + 1]["start"])
                        dur_s = max(0.01, next_start - current_start)
                    else:
                        dur_s = max(0.01, float(item["end"]) - current_start)
                    dur_cs = max(1, int(round(dur_s * 100)))
                    karaoke_text += rf"{{\k{dur_cs}}}{ass_escape(item['word'])} "
                lines.append(
                    f"Dialogue: 0,{sec_to_ass(start)},{sec_to_ass(end)},Default,,0,0,0,,{karaoke_text.strip()}"
                )
        else:
            # Fallback: sem timestamps por palavra, usa frases/segmentos para não sair sem legenda.
            for seg in segments:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                if seg_end <= clip.start or seg_start >= clip.end:
                    continue
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                rel_start = max(0.0, max(seg_start, clip.start) - clip.start)
                rel_end = max(rel_start + 0.2, min(seg_end, clip.end) - clip.start)
                rel_start += subtitle_delay_s
                rel_end = max(rel_start + 0.2, rel_end + subtitle_delay_s)
                lines.append(
                    f"Dialogue: 0,{sec_to_ass(rel_start)},{sec_to_ass(rel_end)},Default,,0,0,0,,{ass_escape(text)}"
                )

        if not lines:
            logger.info("Clipe %s sem texto útil para legenda.", clip_index)
            return None
        logger.info("Clipe %s: %s linha(s) de legenda geradas.", clip_index, len(lines))

        style_font_size = max(34, int(round(target_h * 0.06)))
        style_margin_v = max(70, int(round(target_h * 0.09)))
        ass_content = "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                f"PlayResX: {target_w}",
                f"PlayResY: {target_h}",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                f"Style: Default,Arial,{style_font_size},&H00FFFFFF,&H0000E5FF,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,4,1,2,80,80,{style_margin_v},1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                *lines,
            ]
        )
        ass_path.write_text(ass_content, encoding="utf-8")
        return ass_path

    for i, clip in enumerate(clips, start=1):
        output_file = output_dir / f"clip_{i}_viral.mp4"
        logger.info("Renderizando clipe %s em %s", i, output_file)

        ass_path: Path | None = None
        if enable_tiktok_captions:
            ass_path = build_tiktok_ass_for_clip(clip, i)
            if ass_path:
                logger.info("Legenda preparada para clipe %s (%s).", i, ass_path.name)
            else:
                logger.info("Legenda não aplicada no clipe %s (sem conteúdo de legenda).", i)

        # Estratégia em duas etapas:
        # 1) render base (corte/enquadramento) sem legenda;
        # 2) burn-in da legenda no vídeo já renderizado.
        base_output_file = (
            output_dir / f"clip_{i}_base.mp4" if ass_path else output_file
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            str(clip.start),
            "-to",
            str(clip.end),
            "-c:v",
            "libx264",
            "-preset",
            profile["preset"],
            "-crf",
            profile["crf"],
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1",
        ]

        filters: List[str] = ["setpts=PTS-STARTPTS"]
        lower_framing = framing_mode.lower()
        if "inteligente" in lower_framing or "rosto" in lower_framing:
            center_ratio = _detect_face_center_ratio_for_clip(clip)
            filters.append(_crop_filter_with_center_x(center_ratio))
        elif "crop" in lower_framing:
            # Preenche toda a tela 9:16/16:9, cortando o excedente sem distorcer.
            filters.append(
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h}"
            )
        else:
            # Mantém conteúdo completo, adicionando barras quando necessário.
            filters.append(
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
            )

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        if bitrate:
            # Substitui CRF por bitrate fixo
            if "-crf" in cmd:
                idx = cmd.index("-crf")
                cmd[idx] = "-b:v"
                cmd[idx + 1] = f"{bitrate}k"

        cmd.append(str(base_output_file))

        try:
            base_proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=str(output_dir),
            )
        except subprocess.CalledProcessError as e:
            err_tail = "\n".join((e.stderr or "").splitlines()[-25:])
            raise RuntimeError(f"FFmpeg falhou ao renderizar base do clipe {i}.\n{err_tail}") from e

        if ass_path:
            ass_for_ffmpeg = ass_path.name.replace("'", r"\'")
            subtitle_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(base_output_file),
                "-vf",
                f"subtitles=filename='{ass_for_ffmpeg}'",
                "-c:v",
                "libx264",
                "-preset",
                profile["preset"],
                "-crf",
                profile["crf"],
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                str(output_file),
            ]
            try:
                proc = subprocess.run(
                    subtitle_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    cwd=str(output_dir),
                )
            except subprocess.CalledProcessError:
                # Fallback para builds do FFmpeg onde subtitles falha com .ass.
                fallback_cmd = list(subtitle_cmd)
                vf_idx = fallback_cmd.index("-vf")
                fallback_cmd[vf_idx + 1] = f"ass='{ass_for_ffmpeg}'"
                try:
                    proc = subprocess.run(
                        fallback_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                        cwd=str(output_dir),
                    )
                except subprocess.CalledProcessError as e2:
                    err_tail = "\n".join((e2.stderr or "").splitlines()[-25:])
                    raise RuntimeError(
                        f"FFmpeg falhou ao aplicar legenda no clipe {i} (subtitles e ass).\n{err_tail}"
                    ) from e2
            if base_output_file.exists():
                base_output_file.unlink(missing_ok=True)
        else:
            proc = base_proc

        if enable_tiktok_captions and ass_path:
            ffmpeg_log = (proc.stderr or "").strip()
            if ffmpeg_log:
                tail = "\n".join(ffmpeg_log.splitlines()[-12:])
                logger.info("FFmpeg (clipe %s) log final:\n%s", i, tail)
        if ass_path and ass_path.exists():
            ass_path.unlink(missing_ok=True)

