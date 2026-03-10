from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import ViralApp


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = ViralApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
