from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)


class SaveApiKeyDialog(QDialog):
    """Guarda um perfil nomeado de chave API (Gemini ou Groq)."""

    def __init__(
        self,
        parent=None,
        *,
        default_provider: str = "gemini",
        default_label: str = "",
        default_secret: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Guardar chave API")
        self.setMinimumWidth(420)

        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText('Ex.: "Chave API Gemini — conta pessoal"')
        if default_label:
            self.edit_label.setText(default_label)

        self.combo_provider = QComboBox()
        self.combo_provider.addItems(["Gemini", "Groq"])
        prov = default_provider.strip().lower()
        self.combo_provider.setCurrentIndex(1 if prov == "groq" else 0)

        self.edit_secret = QLineEdit()
        self.edit_secret.setEchoMode(QLineEdit.Password)
        self.edit_secret.setPlaceholderText("Cole a chave secreta")
        if default_secret:
            self.edit_secret.setText(default_secret)

        form = QFormLayout()
        form.addRow(QLabel("Nome do perfil (como aparece na lista):"), self.edit_label)
        form.addRow(QLabel("Servidor / API:"), self.combo_provider)
        form.addRow(QLabel("Chave:"), self.edit_secret)

        hint = QLabel(
            "As chaves são guardadas em ficheiro local (texto). Não partilhe a pasta "
            "AppData e use perfis com nomes claros (ex.: Chave API Groq, Chave API Gemini)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888888; font-size: 11px;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def provider_id(self) -> str:
        return "groq" if self.combo_provider.currentIndex() == 1 else "gemini"

    def profile_label(self) -> str:
        return self.edit_label.text().strip()

    def secret(self) -> str:
        return self.edit_secret.text().strip()
