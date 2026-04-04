from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

# id usado em ApiKeyStore
_API_OPTIONS: list[tuple[str, str]] = [
    ("Gemini", "gemini"),
    ("Groq", "groq"),
    ("OpenAI", "openai"),
]


class SaveApiKeyDialog(QDialog):
    """Guarda um perfil nomeado de chave API (Gemini, Groq ou OpenAI)."""

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
        self.edit_label.setPlaceholderText('Ex.: "Chave OpenAI — conta equipa"')
        if default_label:
            self.edit_label.setText(default_label)

        self.combo_provider = QComboBox()
        for label, pid in _API_OPTIONS:
            self.combo_provider.addItem(label, pid)
        prov = default_provider.strip().lower()
        for i in range(self.combo_provider.count()):
            if self.combo_provider.itemData(i) == prov:
                self.combo_provider.setCurrentIndex(i)
                break

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
            "AppData. Para vários fornecedores, crie perfis com nomes claros."
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
        data = self.combo_provider.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip().lower()
        return "gemini"

    def profile_label(self) -> str:
        return self.edit_label.text().strip()

    def secret(self) -> str:
        return self.edit_secret.text().strip()
