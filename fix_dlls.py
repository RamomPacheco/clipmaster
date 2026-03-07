import os
import shutil


def resolver_dlls_nvidia():
    """
    Busca as DLLs instaladas via pip no .venv e as copia
    diretamente para a raiz do projeto (onde o main.py está).
    Isso força o Windows C++ a encontrá-las instantaneamente.
    """
    venv_nvidia_path = r"E:\projetos_python\tiktoksele\.venv\Lib\site-packages\nvidia"
    raiz_projeto = r"E:\projetos_python\tiktoksele"

    if not os.path.exists(venv_nvidia_path):
        print(
            "Pasta NVIDIA não encontrada no .venv. Tem certeza que rodou o pip install?"
        )
        return

    dlls_copiadas = 0
    print("Iniciando extração de DLLs do CUDA...")

    for root, dirs, files in os.walk(venv_nvidia_path):
        for file in files:
            if file.endswith(".dll"):
                origem = os.path.join(root, file)
                destino = os.path.join(raiz_projeto, file)

                # Copia a DLL para a raiz do projeto
                shutil.copy2(origem, destino)
                print(f"[+] Copiada: {file}")
                dlls_copiadas += 1

    print(f"\n✅ Sucesso! {dlls_copiadas} DLLs foram movidas para a raiz do projeto.")
    print("O CTranslate2 agora encontrará a cublas64_12.dll nativamente.")


if __name__ == "__main__":
    resolver_dlls_nvidia()
