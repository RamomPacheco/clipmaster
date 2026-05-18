from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from app.models.schemas import Clip


class ClipSelectionDialog(QDialog):
    """
    Diálogo para o usuário selecionar quais clipes deseja salvar.
    Recebe e devolve dados na forma de modelos Clip.
    """

    def __init__(self, clips: List[Clip], parent=None) -> None:
        super().__init__(parent)
        self.clips = clips
        self.selected_clips: List[Clip] = []
        self.checkboxes: List[tuple[QCheckBox, Clip]] = []
        self.edits: List[tuple[QLineEdit, QLineEdit]] = []
        self.schedule_widgets: List[tuple[QCheckBox, QDateTimeEdit]] = []

        self.setWindowTitle("Selecione os Clipes para Salvar")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        self.setStyleSheet(
            """
            QDialog { background-color: #121212; }
            QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
            QCheckBox {
                background-color: #1e1e1e; border: 1px solid #3e3e42;
                border-radius: 4px; padding: 10px; margin: 5px;
            }
            QCheckBox:hover {
                background-color: #2d2d30; border: 1px solid #0078D7;
            }
            QPushButton {
                background-color: #0078D7; color: white;
                font-size: 12px; font-weight: bold;
                border-radius: 6px; border: none; padding: 8px 15px;
            }
            QPushButton:hover { background-color: #1084ea; }
            QPushButton#btn_cancel { background-color: #555555; }
            QPushButton#btn_cancel:hover { background-color: #666666; }
            QScrollArea { border: 1px solid #3e3e42; border-radius: 4px; }
        """
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        lbl_title = QLabel(f"Escolha quais dos {len(self.clips)} clipes deseja salvar:")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff00;")
        layout.addWidget(lbl_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)

        for i, clip in enumerate(self.clips, start=1):
            clip_frame = QFrame()
            clip_frame.setStyleSheet(
                """
                QFrame {
                    background-color: #1e1e1e;
                    border: 1px solid #3e3e42;
                    border-radius: 4px;
                    padding: 10px;
                }
                QFrame:hover {
                    border: 1px solid #0078D7;
                    background-color: #252526;
                }
            """
            )
            clip_layout = QVBoxLayout(clip_frame)
            clip_layout.setSpacing(5)

            duration = clip.duration
            checkbox_text = (
                f"Clipe {i} • {clip.headline} • "
                f"[{clip.start:.1f}s - {clip.end:.1f}s] ({duration:.1f}s)"
            )
            checkbox = QCheckBox(checkbox_text)
            checkbox.setStyleSheet(
                "background: transparent; border: none; padding: 0px; margin: 0px;"
            )
            checkbox.setChecked(True)
            checkbox.setMinimumHeight(25)

            self.checkboxes.append((checkbox, clip))
            clip_layout.addWidget(checkbox)

            lbl_reason = QLabel(f"Por quê: {clip.reason}")
            lbl_reason.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            lbl_reason.setWordWrap(True)
            clip_layout.addWidget(lbl_reason)

            lbl_edit_headline = QLabel("Editar Título:")
            edit_headline = QLineEdit(clip.headline)
            clip_layout.addWidget(lbl_edit_headline)
            clip_layout.addWidget(edit_headline)

            lbl_edit_reason = QLabel("Editar Razão:")
            edit_reason = QLineEdit(clip.reason)
            clip_layout.addWidget(lbl_edit_reason)
            clip_layout.addWidget(edit_reason)

            row_sched = QHBoxLayout()
            chk_sched = QCheckBox("Agendar envio ao TikTok (nesta data e hora)")
            chk_sched.setToolTip(
                "O ficheiro exportado só é enviado à TikTok Content Posting API na hora indicada. "
                "A API oficial não documenta agendamento no servidor como na app TikTok; "
                "o ClipMaster adia o upload até ao momento escolhido. "
                "É necessário access token OAuth em Avançado → TikTok."
            )
            chk_sched.setStyleSheet(
                "background: transparent; border: none; padding: 0px; margin: 0px;"
            )
            dt_sched = QDateTimeEdit(QDateTime.currentDateTime().addSecs(3600))
            dt_sched.setCalendarPopup(True)
            dt_sched.setMinimumDateTime(QDateTime.currentDateTime())
            dt_sched.setDisplayFormat("yyyy-MM-dd HH:mm")
            dt_sched.setMinimumHeight(28)
            dt_sched.setEnabled(False)
            chk_sched.toggled.connect(dt_sched.setEnabled)
            row_sched.addWidget(chk_sched)
            row_sched.addWidget(QLabel("Data/hora:"))
            row_sched.addWidget(dt_sched, stretch=1)
            clip_layout.addLayout(row_sched)
            self.schedule_widgets.append((chk_sched, dt_sched))

            btn_preview = QPushButton("Pré-visualizar")
            btn_preview.clicked.connect(lambda checked, c=clip: self.preview_clip(c))
            clip_layout.addWidget(btn_preview)

            scroll_layout.addWidget(clip_frame)
            self.edits.append((edit_headline, edit_reason))

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        button_layout = QHBoxLayout()

        btn_all = QPushButton("✓ Selecionar Todos")
        btn_all.clicked.connect(self._select_all)
        button_layout.addWidget(btn_all)

        btn_none = QPushButton("✗ Desselecionar Todos")
        btn_none.setObjectName("btn_cancel")
        btn_none.clicked.connect(self._select_none)
        button_layout.addWidget(btn_none)

        button_layout.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setObjectName("btn_cancel")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        btn_confirm = QPushButton("Salvar Seleção")
        btn_confirm.setMinimumWidth(150)
        btn_confirm.clicked.connect(self.accept)
        button_layout.addWidget(btn_confirm)

        layout.addLayout(button_layout)

    def accept(self) -> None:  # type: ignore[override]
        margin_sec = 10
        now = int(time.time())
        for i, (checkbox, _clip) in enumerate(self.checkboxes):
            if not checkbox.isChecked():
                continue
            chk_s, dt_e = self.schedule_widgets[i]
            if not chk_s.isChecked():
                continue
            if dt_e.dateTime().toSecsSinceEpoch() <= now + margin_sec:
                QMessageBox.warning(
                    self,
                    "Agendamento TikTok",
                    "Para cada clipe com envio TikTok agendado, escolha uma data e hora "
                    "ligeiramente no futuro.",
                )
                return
        super().accept()

    def _select_all(self) -> None:
        for checkbox, _ in self.checkboxes:
            checkbox.setChecked(True)

    def _select_none(self) -> None:
        for checkbox, _ in self.checkboxes:
            checkbox.setChecked(False)

    def get_selected_clips(self) -> List[Clip]:
        selected: List[Clip] = []
        for i, (checkbox, clip) in enumerate(self.checkboxes):
            if checkbox.isChecked():
                edit_headline, edit_reason = self.edits[i]
                chk_s, dt_e = self.schedule_widgets[i]
                sched_on = chk_s.isChecked()
                sched_iso: str | None = None
                if sched_on:
                    secs = dt_e.dateTime().toSecsSinceEpoch()
                    sched_iso = datetime.fromtimestamp(
                        secs, tz=timezone.utc
                    ).isoformat()
                selected.append(
                    clip.model_copy(
                        update={
                            "headline": edit_headline.text(),
                            "reason": edit_reason.text(),
                            "tiktok_schedule_enabled": sched_on,
                            "tiktok_schedule_at": sched_iso if sched_on else None,
                        }
                    )
                )
        return selected

    def preview_clip(self, clip: Clip) -> None:
        video_path = (
            Path(self.parent().current_video_path) if self.parent() else None  # type: ignore[attr-defined]
        )
        if not video_path:
            QMessageBox.warning(self, "Erro", "Vídeo não encontrado.")
            return

        duration = clip.duration

        try:
            subprocess.Popen(
                [
                    "ffplay",
                    "-ss",
                    str(clip.start),
                    "-t",
                    str(duration),
                    "-i",
                    str(video_path),
                    "-window_title",
                    f"Pré-visualização: {clip.headline}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "Erro",
                "ffplay não encontrado. Certifique-se de que FFmpeg está instalado e no PATH.",
            )

