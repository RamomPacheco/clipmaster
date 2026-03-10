"""
Ponto de entrada legado.

O código original completo foi movido para `originproject/main.py`.
Esta casca fina apenas delega para a nova arquitetura em `clipmaster/app/main.py`.
"""

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    clipmaster_root = repo_root / "clipmaster"
    sys.path.insert(0, str(clipmaster_root))

    from app.main import main as app_main  # type: ignore[import-not-found]

    app_main()


if __name__ == "__main__":
    main()
