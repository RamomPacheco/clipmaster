from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout


class DropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #555555;
                border-radius: 10px;
                background-color: #1e1e1e;
            }
            QFrame:hover {
                border: 2px dashed #0078D7;
                background-color: #252526;
            }
        """
        )
        layout = QVBoxLayout()
        self.lbl_text = QLabel(
            "Arraste e solte o seu vídeo aqui\nou clique para procurar"
        )
        self.lbl_text.setAlignment(Qt.AlignCenter)
        self.lbl_text.setStyleSheet(
            "color: #aaaaaa; font-size: 14px; border: none; background: transparent;"
        )
        layout.addWidget(self.lbl_text)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "border: 2px dashed #00FF00; background-color: #1e2a1e; border-radius: 10px;"
            )

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.setStyleSheet(
            """
            QFrame { border: 2px dashed #555555; border-radius: 10px; background-color: #1e1e1e; }
        """
        )

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self.dragLeaveEvent(event)
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                self.file_dropped.emit(file_path)
                break

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo", "", "Videos (*.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self.file_dropped.emit(file_path)

