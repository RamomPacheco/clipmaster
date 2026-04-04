from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.core import config
from app.services.youtube_download import download_youtube_video, list_video_heights


class _FetchHeightsThread(QThread):
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url.strip()

    def run(self) -> None:
        try:
            heights = list_video_heights(self._url)
            self.finished_ok.emit(heights)
        except Exception as e:
            self.failed.emit(str(e))


class _DownloadThread(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, height: int, output_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._url = url.strip()
        self._height = height
        self._output_dir = output_dir

    def run(self) -> None:
        try:
            path = download_youtube_video(self._url, self._height, self._output_dir)
            self.finished_ok.emit(str(path.resolve()))
        except Exception as e:
            self.failed.emit(str(e))


class YoutubeDownloadDialog(QDialog):
    """Descarrega um vídeo do YouTube com escolha de qualidade (yt-dlp)."""

    def __init__(self, default_download_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.downloaded_path: Path | None = None
        self._fetch_thread: _FetchHeightsThread | None = None
        self._dl_thread: _DownloadThread | None = None

        self.setWindowTitle("Baixar do YouTube")
        self.setMinimumWidth(480)
        self._default_dir = Path(default_download_dir)

        self.setStyleSheet(
            """
            QDialog { background-color: #121212; }
            QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
            QLineEdit {
                background-color: #1e1e1e; border: 1px solid #3e3e42;
                border-radius: 4px; padding: 6px; color: #ffffff;
            }
            QComboBox {
                background-color: #1e1e1e; border: 1px solid #3e3e42;
                border-radius: 4px; padding: 6px; min-height: 28px; color: #ffffff;
            }
            QPushButton {
                background-color: #0078D7; color: white;
                font-size: 12px; font-weight: bold;
                border-radius: 6px; border: none; padding: 8px 14px;
            }
            QPushButton:hover { background-color: #1084ea; }
            QPushButton:disabled { background-color: #444444; color: #888888; }
            QPushButton#secondary { background-color: #555555; }
            QPushButton#secondary:hover { background-color: #666666; }
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(QLabel("URL do vídeo"))
        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("https://www.youtube.com/watch?v=…")
        layout.addWidget(self.edit_url)

        row_fetch = QHBoxLayout()
        self.btn_fetch = QPushButton("Obter qualidades")
        self.btn_fetch.clicked.connect(self._on_fetch)
        row_fetch.addWidget(self.btn_fetch)
        row_fetch.addStretch()
        layout.addLayout(row_fetch)

        layout.addWidget(QLabel("Qualidade"))
        self.combo_quality = QComboBox()
        self.combo_quality.setEnabled(False)
        self.combo_quality.setMinimumHeight(32)
        layout.addWidget(self.combo_quality)

        layout.addWidget(QLabel("Pasta de destino"))
        dir_row = QHBoxLayout()
        self.edit_dir = QLineEdit()
        self.edit_dir.setText(str(self._default_dir.resolve()))
        btn_browse = QPushButton("Procurar…")
        btn_browse.setObjectName("secondary")
        btn_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.edit_dir, stretch=1)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setObjectName("secondary")
        btn_cancel.clicked.connect(self.reject)
        self.btn_download = QPushButton("Descarregar")
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self._on_download)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_download)
        layout.addLayout(btn_row)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Pasta de destino", self.edit_dir.text() or str(config.EXPORTS_ROOT)
        )
        if d:
            self.edit_dir.setText(d)

    def _set_busy(self, busy: bool) -> None:
        self.btn_fetch.setEnabled(not busy)
        self.btn_download.setEnabled(not busy and self.combo_quality.count() > 0)
        self.edit_url.setReadOnly(busy)

    def _on_fetch(self) -> None:
        url = self.edit_url.text().strip()
        if not url:
            QMessageBox.warning(self, "URL em falta", "Cole o endereço do vídeo do YouTube.")
            return
        if self._fetch_thread and self._fetch_thread.isRunning():
            return

        self.combo_quality.clear()
        self.combo_quality.setEnabled(False)
        self._set_busy(True)

        self._fetch_thread = _FetchHeightsThread(url, self)
        self._fetch_thread.finished_ok.connect(self._on_fetch_ok)
        self._fetch_thread.failed.connect(self._on_fetch_failed)
        self._fetch_thread.finished.connect(lambda: self._set_busy(False))
        self._fetch_thread.start()

    def _on_fetch_ok(self, heights: list) -> None:
        if not heights:
            QMessageBox.warning(
                self, "Sem vídeo",
                "Não foram encontradas resoluções de vídeo para este link.",
            )
            return
        for h in heights:
            self.combo_quality.addItem(f"{h}p", int(h))
        self.combo_quality.setEnabled(True)
        self.btn_download.setEnabled(True)

    def _on_fetch_failed(self, msg: str) -> None:
        QMessageBox.critical(
            self, "Erro",
            "Não foi possível ler o vídeo. Verifique a URL e a ligação à Internet.\n\n" + msg,
        )

    def _on_download(self) -> None:
        url = self.edit_url.text().strip()
        idx = self.combo_quality.currentIndex()
        if idx < 0:
            return
        height = int(self.combo_quality.currentData())
        out = Path(self.edit_dir.text().strip() or str(self._default_dir))
        if self._dl_thread and self._dl_thread.isRunning():
            return

        self._set_busy(True)
        self._dl_thread = _DownloadThread(url, height, out, self)
        self._dl_thread.finished_ok.connect(self._on_dl_ok)
        self._dl_thread.failed.connect(self._on_dl_failed)
        self._dl_thread.finished.connect(lambda: self._set_busy(False))
        self._dl_thread.start()

    def _on_dl_ok(self, path: str) -> None:
        self.downloaded_path = Path(path)
        self.accept()

    def _on_dl_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Download falhou", msg)
