"""
Resolução de executáveis FFmpeg / ffprobe: empacotados com a app, depois PATH.

Ordem:
1. ``CLIPMASTER_FFMPEG_PATH`` / ``CLIPMASTER_FFPROBE_PATH`` (caminho completo)
2. PyInstaller: ``sys._MEIPASS/ffmpeg-bin/``
3. Projeto: ``bundled/ffmpeg/<plataforma>/``
4. ``shutil.which`` (instalação manual no sistema)

Chamar ``ensure_ffmpeg_runtime()`` no arranque da app para o MoviePy respeitar o binário empacotado.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.core.logger import logger


def _platform_dir() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _exe_suffix() -> str:
    return ".exe" if sys.platform.startswith("win") else ""


def _meipass_bundle_dir() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / "ffmpeg-bin"
        return p if p.is_dir() else None
    return None


def _dev_bundle_dir() -> Path | None:
    p = PROJECT_ROOT / "bundled" / "ffmpeg" / _platform_dir()
    return p if p.is_dir() else None


def get_ffmpeg_path() -> str | None:
    env = (os.environ.get("CLIPMASTER_FFMPEG_PATH") or "").strip()
    if env:
        ep = Path(env)
        if ep.is_file():
            return str(ep.resolve())

    name = f"ffmpeg{_exe_suffix()}"
    for base in (_meipass_bundle_dir(), _dev_bundle_dir()):
        if base is None:
            continue
        cand = base / name
        if cand.is_file():
            return str(cand.resolve())

    return shutil.which("ffmpeg")


def get_ffprobe_path() -> str | None:
    env = (os.environ.get("CLIPMASTER_FFPROBE_PATH") or "").strip()
    if env:
        ep = Path(env)
        if ep.is_file():
            return str(ep.resolve())

    name = f"ffprobe{_exe_suffix()}"
    for base in (_meipass_bundle_dir(), _dev_bundle_dir()):
        if base is None:
            continue
        cand = base / name
        if cand.is_file():
            return str(cand.resolve())

    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    p = get_ffmpeg_path()
    return bool(p and Path(p).is_file())


def ffprobe_available() -> bool:
    p = get_ffprobe_path()
    return bool(p and Path(p).is_file())


def ensure_ffmpeg_runtime() -> None:
    """
    Define ``FFMPEG_BINARY`` para o MoviePy e atualiza config se o pacote já foi importado.
    Deve ser chamado o mais cedo possível em ``main()`` (antes do primeiro ``import moviepy``).
    """
    ff = get_ffmpeg_path()
    if ff:
        os.environ["FFMPEG_BINARY"] = ff
        logger.debug("FFmpeg resolvido para: %s", ff)
    if "moviepy.config" in sys.modules:
        try:
            import moviepy.config as mpc

            if ff:
                mpc.change_settings({"FFMPEG_BINARY": ff})
        except Exception as e:  # noqa: BLE001
            logger.warning("Não foi possível atualizar FFMPEG_BINARY do MoviePy: %s", e)
