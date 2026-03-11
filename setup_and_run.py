import os
import sys
import subprocess
from pathlib import Path
import venv

# Nome do diretório do ambiente virtual
VENV_DIR = ".venv"


def get_venv_python():
    """Retorna o caminho do executável Python dentro do ambiente virtual dependendo do SO."""
    if sys.platform == "win32":
        return Path(VENV_DIR) / "Scripts" / "python.exe"
    else:
        return Path(VENV_DIR) / "bin" / "python"


def run_command(command, error_msg="Erro ao executar comando."):
    """Executa um comando no subprocess e lida com erros."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] {error_msg}")
        print(f"Detalhes do erro: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n[!] Comando não encontrado: {command[0]}")
        sys.exit(1)


def setup_environment():
    """Configura o .venv, dependências e otimizações CUDA se for a primeira vez."""
    venv_path = Path(VENV_DIR)

    if not venv_path.exists():
        print(f"[*] Ambiente virtual não encontrado. Criando um novo na pasta '{VENV_DIR}'...")

        # Cria o ambiente virtual com o pip incluído
        venv.create(VENV_DIR, with_pip=True)

        python_exe = get_venv_python()

        # Atualiza o pip para evitar avisos
        print("[*] Atualizando o pip...")
        run_command(
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
            "Falha ao atualizar o pip.",
        )

        # Instala as dependências
        if Path("requirements.txt").exists():
            print(
                "[*] Instalando as dependências (isso pode demorar alguns minutos na primeira vez)..."
            )
            run_command(
                [str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"],
                "Falha ao instalar dependências do requirements.txt.",
            )
        else:
            print(
                "[!] Arquivo requirements.txt não encontrado. Pulando instalação de dependências."
            )

        # Otimiza o CUDA se o script existir
        if Path("fix_cuda_libs.py").exists():
            print("[*] Configurando bibliotecas de aceleração CUDA/GPU...")
            run_command(
                [str(python_exe), "fix_cuda_libs.py"],
                "Falha ao rodar o script de configuração do CUDA.",
            )

        print("\n[*] Instalação inicial concluída com sucesso!\n")
    else:
        print("[*] Ambiente virtual detectado.")


def run_app():
    """Inicia o aplicativo principal (main.py) usando o Python do ambiente virtual."""
    python_exe = get_venv_python()

    if not Path("main.py").exists():
        print("[!] Erro: main.py não encontrado na raiz do projeto.")
        sys.exit(1)

    print("[*] Iniciando o AI Viral Clipper Pro...")
    print("-" * 40)

    # Roda o main.py. O processo atual ficará "preso" aqui até o main.py ser fechado.
    subprocess.run([str(python_exe), "main.py"])


if __name__ == "__main__":
    print("=" * 40)
    print("   AI Viral Clipper Pro - Lançador   ")
    print("=" * 40)

    setup_environment()
    run_app()
