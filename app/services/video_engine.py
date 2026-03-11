from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

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
    resolution: str = "1080p",
    bitrate: str | None = None,
) -> None:
    """
    Renderiza uma lista de clipes para MP4 H.264, mantendo a mesma lógica do código original.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips, start=1):
        output_file = output_dir / f"clip_{i}_viral.mp4"
        logger.info("Renderizando clipe %s em %s", i, output_file)

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
            "slow",
            "-crf",
            "18",
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

        if resolution != "1080p":
            scales = {"720p": "1280:720", "480p": "854:480"}
            if resolution in scales:
                cmd.extend(["-vf", f"scale={scales[resolution]}"])

        if bitrate:
            # Substitui CRF por bitrate fixo
            if "-crf" in cmd:
                idx = cmd.index("-crf")
                cmd[idx] = "-b:v"
                cmd[idx + 1] = f"{bitrate}k"

        cmd.append(str(output_file))

        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

