import os
from pathlib import Path


def inject_cuda_environment(project_root: Path | None = None) -> None:
    """
    Força o Python a encontrar as DLLs do CUDA na raiz do projeto.
    Extraído do main legado para um módulo dedicado.
    """
    try:
        root = project_root or Path.cwd()

        os.environ["PATH"] = str(root) + os.pathsep + os.environ.get("PATH", "")

        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(root))
                print(f"Diretório CUDA injetado: {root}")
            except Exception as e:  # noqa: BLE001
                print(f"Falha ao injetar DLL directory: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"Falha inesperada ao configurar ambiente CUDA: {e}")


# Executa automaticamente no import, mantendo o comportamento do código original.
inject_cuda_environment()

