"""
Arranque assistido do servidor Ollama quando a API local não responde.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Optional

from app.core.logger import logger

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

_serve_proc: Optional[subprocess.Popen] = None


def is_ollama_cli_on_path() -> bool:
    """True se o executável ``ollama`` estiver encontrável no PATH."""
    return shutil.which("ollama") is not None


def try_start_ollama_serve() -> tuple[bool, str]:
    """
    Executa ``ollama serve`` em segundo plano (sem janela de consola no Windows).
    Se já existe um processo lançado por esta sessão e ainda corre, não duplica.
    """
    global _serve_proc

    exe = shutil.which("ollama")
    if not exe:
        return False, (
            "Comando 'ollama' não encontrado no PATH (Ollama não instalado ou não no PATH). "
            f"Download: {OLLAMA_DOWNLOAD_URL}"
        )

    if _serve_proc is not None and _serve_proc.poll() is None:
        return True, "Servidor Ollama (iniciado por esta app) já está em execução."

    kwargs: dict = {}
    if os.name == "nt":
        # Sem janela de CMD; o servidor fica em background
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    try:
        _serve_proc = subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    except OSError as e:
        logger.warning("Não foi possível iniciar ollama serve: %s", e)
        return False, str(e)

    logger.info("ollama serve lançado em segundo plano (pid=%s).", _serve_proc.pid)
    return True, "ollama serve foi iniciado em segundo plano."


def wait_until_ollama_list_ok(timeout_sec: float = 25.0, interval_sec: float = 0.6) -> bool:
    """Repete ``ollama.list()`` até funcionar ou esgotar o tempo."""
    try:
        import ollama
    except ImportError:
        return False

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            ollama.list()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(interval_sec)
    return False
