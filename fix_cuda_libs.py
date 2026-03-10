import os
import sys
import shutil
from pathlib import Path


def resolver_libs_nvidia():
    """
    Busca as bibliotecas instaladas via pip (.dll no Win, .so no Linux)
    e as copia para a raiz do projeto para a C++ engine encontrar.
    """
    raiz_projeto = Path(__file__).parent.absolute()

    # Detecta o caminho dinâmico do .venv dependendo do OS
    if sys.platform == "win32":
        venv_nvidia = raiz_projeto / ".venv" / "Lib" / "site-packages" / "nvidia"
        extensao_alvo = ".dll"
    else:
        # Linux usa letras minúsculas e separa por versão do Python
        versao_py = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_nvidia = (
            raiz_projeto / ".venv" / "lib" / versao_py / "site-packages" / "nvidia"
        )
        extensao_alvo = ".so"  # Formato de bibliotecas no Linux

    if not venv_nvidia.exists():
        print("[!] Pasta NVIDIA não encontrada no .venv. Rode o pip install primeiro.")
        return

    libs_copiadas = 0
    print(f"[*] Iniciando extração de bibliotecas CUDA ({extensao_alvo})...")

    for root, _, files in os.walk(venv_nvidia):
        for file in files:
            if file.endswith(extensao_alvo):
                origem = os.path.join(root, file)
                destino = os.path.join(raiz_projeto, file)
                shutil.copy2(origem, destino)
                print(f"  -> Copiada: {file}")
                libs_copiadas += 1

    print(f"\n✅ Sucesso! {libs_copiadas} arquivos movidos para a raiz.")


if __name__ == "__main__":
    resolver_libs_nvidia()
