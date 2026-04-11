# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 6 — pasta dist/ClipMaster (onedir), com FFmpeg em ffmpeg-bin/ se existir em bundled/ffmpeg/<plataforma>/
# Build:  python -m PyInstaller ClipMaster.spec --noconfirm

import sys
from pathlib import Path

block_cipher = None

try:
    ROOT = Path(SPECPATH).resolve()
except NameError:
    ROOT = Path(".").resolve()

ffmpeg_plat = (
    "windows"
    if sys.platform.startswith("win")
    else ("macos" if sys.platform == "darwin" else "linux")
)
ff_src = ROOT / "bundled" / "ffmpeg" / ffmpeg_plat
ffmpeg_datas: list[tuple[str, str]] = []
if ff_src.is_dir():
    try:
        if any(ff_src.iterdir()):
            ffmpeg_datas = [(str(ff_src), "ffmpeg-bin")]
    except OSError:
        pass

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

# Evitar collect_all(PySide6): incluiria Qt3D/Charts/WebEngine e explodia tempo/tamanho.
# Os hooks hook-PySide6 seguem os imports reais (QtWidgets/QtGui/QtCore).

binaries_extra: list[tuple[str, str]] = []
for pkg in ("torch", "ctranslate2", "onnxruntime", "av"):
    try:
        binaries_extra += collect_dynamic_libs(pkg)
    except Exception:
        pass

# Wheels NVIDIA (CUDA no Windows): DLLs em site-packages\nvidia\<lib>\bin
if sys.platform.startswith("win"):
    try:
        import site

        for sp in site.getsitepackages():
            nv = Path(sp) / "nvidia"
            if not nv.is_dir():
                continue
            for cat in nv.iterdir():
                bin_dir = cat / "bin"
                if not bin_dir.is_dir():
                    continue
                for dll in bin_dir.glob("*.dll"):
                    binaries_extra.append((str(dll), "nvidia"))
    except Exception:
        pass

datas_meta: list[tuple[str, str]] = []
for pkg in ("torch", "ctranslate2", "faster_whisper", "transformers", "google-generativeai"):
    try:
        datas_meta += copy_metadata(pkg)
    except Exception:
        pass

hiddenimports = [
    "app.main",
    "app.ui.main_window",
    "app.workers.processing_task",
    "app.services.transcription",
    "app.services.video_engine",
    "app.services.moviepy_engagement",
    "multiprocessing.spawn",
    "multiprocessing.popen_spawn_win32",
    "app.core.ffmpeg_bin",
    "app.core.cuda_setup",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "google.generativeai",
    "google.api_core",
    "moviepy",
    "moviepy.config",
    "PIL",
    "yaml",
    "dotenv",
    "pydantic",
    "ollama",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries_extra,
    datas=ffmpeg_datas + datas_meta,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClipMaster",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ClipMaster",
)
