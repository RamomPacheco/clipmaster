from __future__ import annotations

import sys


def main() -> None:
    from app.core.ffmpeg_bin import ensure_ffmpeg_runtime

    ensure_ffmpeg_runtime()

    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import ViralApp

    app = QApplication.instance() or QApplication(sys.argv)
    print(app)
    window = ViralApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
