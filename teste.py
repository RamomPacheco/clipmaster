import sys
import os
import shutil
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QProgressBar,
    QTextEdit,
)
from PySide6.QtCore import Qt, QThread, Signal
import ollama


# ==========================================
# BOOTSTRAP: Resolução Dinâmica de CUDA DLLs
# ==========================================
def _inject_cuda_environment():
    """
    Força o Python a encontrar as DLLs do CUDA (cublas64_12.dll) contornando
    as limitações de PATH do Windows e do Python 3.8+.
    """
    # Caminhos padrão onde a NVIDIA instala o CUDA 12.x
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

    if os.path.exists(cuda_base):
        # Procura a pasta da versão instalada (ex: v12.1, v12.2)
        for folder in os.listdir(cuda_base):
            if folder.startswith("v12"):
                bin_path = os.path.join(cuda_base, folder, "bin")

                # 1. Adiciona ao PATH da sessão atual
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

                # 2. Registra o diretório nativamente no Python (Crítico para Windows)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_path)
                        logging.info(f"Diretório CUDA injetado com sucesso: {bin_path}")
                    except Exception as e:
                        logging.warning(
                            f"Falha ao injetar DLL via add_dll_directory: {e}"
                        )
                break


# Executa a injeção ANTES de importar a biblioteca C++ do Whisper_inject_cuda_environment()

# Agora sim, importamos o Whisper com segurança
from faster_whisper import WhisperModel
