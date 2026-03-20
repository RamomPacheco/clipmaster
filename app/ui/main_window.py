from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core import config  # noqa: F401  # garante import de config
from app.core.cuda_setup import inject_cuda_environment  # noqa: F401
from app.core.logger import logger
from app.models.schemas import Clip
from app.ui.components.drop_zone import DropZone
from app.ui.dialogs.clip_dialog import ClipSelectionDialog
from app.workers.processing_task import VideoProcessorThread


class ViralApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Viral Clipper Pro")
        self.setMinimumSize(900, 650)
        self.current_video_path: str | None = None
        self.output_folder_path: str | None = None
        self.worker: VideoProcessorThread | None = None

        self._setup_ui()
        self._apply_dark_theme()

    # ---------------- UI Setup ----------------
    def _get_available_models(self) -> List[str]:
        try:
            import ollama

            models = ollama.list()
            model_list = (
                models.get("models", [])
                if isinstance(models, dict)
                else getattr(models, "models", [])
            )
            model_names: List[str] = []
            for model in model_list:
                if isinstance(model, dict):
                    name = model.get("name") or model.get("model")
                else:
                    name = getattr(model, "name", None) or getattr(model, "model", None)
                if name:
                    model_names.append(name)
                else:
                    logger.warning("Modelo com estrutura inesperada: %s", model)
            return model_names or [config.DEFAULT_LLM_MODEL]
        except Exception as e:  # noqa: BLE001
            logger.warning("Erro ao buscar modelos Ollama: %s", e)
            return [config.DEFAULT_LLM_MODEL]

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        lbl_title = QLabel("AI Viral Clipper Pro")
        lbl_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        lbl_subtitle = QLabel("Motor H.264 CFR • Aceleração CUDA • IA Ollama")
        lbl_subtitle.setStyleSheet("font-size: 12px; color: #0078D7; margin-bottom: 10px;")

        main_layout.addWidget(lbl_title)
        main_layout.addWidget(lbl_subtitle)

        # Configuração de modelo e prompt
        config_layout = QHBoxLayout()
        lbl_model = QLabel("Modelo de IA (Ollama):")
        lbl_model.setStyleSheet("font-size: 14px; font-weight: bold; color: #dddddd;")

        self.combo_model = QComboBox()
        self.combo_model.addItems(self._get_available_models())
        self.combo_model.setToolTip("Selecione o modelo disponível no Ollama.")
        self.combo_model.setMinimumHeight(30)

        lbl_whisper_model = QLabel("Modelo de Transcrição (Whisper):")
        lbl_whisper_model.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #dddddd; margin-left: 20px;"
        )

        self.combo_whisper_model = QComboBox()
        self.combo_whisper_model.addItems(
            ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        )
        self.combo_whisper_model.setCurrentText(config.WHISPER_MODEL)
        self.combo_whisper_model.setToolTip(
            "Escolha o modelo do Faster-Whisper para transcrição."
        )
        self.combo_whisper_model.setMinimumHeight(30)

        lbl_prompt = QLabel("Tipo de Prompt:")
        lbl_prompt.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #dddddd; margin-left: 20px;"
        )

        self.combo_prompt = QComboBox()
        self.combo_prompt.addItems(
            [
                "Padrão (Equilibrado)",
                "Humor & Comédia",
                "Sério & Alto Valor",
                "Storytelling & Emoção",
                "Educacional & Dicas",
            ]
        )
        self.combo_prompt.setToolTip("Selecione o tipo de foco para análise dos clipes.")
        self.combo_prompt.setMinimumHeight(30)
        self.combo_prompt.setCurrentText("Padrão (Equilibrado)")

        config_layout.addWidget(lbl_model)
        config_layout.addWidget(self.combo_model)
        config_layout.addWidget(lbl_whisper_model)
        config_layout.addWidget(self.combo_whisper_model)
        config_layout.addWidget(lbl_prompt)
        config_layout.addWidget(self.combo_prompt)
        config_layout.addStretch()
        main_layout.addLayout(config_layout)

        # Caminhos
        paths_layout = QVBoxLayout()
        paths_layout.setSpacing(8)

        input_path_layout = QHBoxLayout()
        lbl_input = QLabel("Vídeo de Entrada:")
        lbl_input.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #dddddd; min-width: 100px;"
        )

        self.input_path_field = QLineEdit()
        self.input_path_field.setReadOnly(True)
        self.input_path_field.setPlaceholderText("Nenhum vídeo selecionado...")
        self.input_path_field.setMinimumHeight(25)
        self.input_path_field.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3e3e42; border-radius: 4px; padding: 5px;"
        )

        btn_browse_input = QPushButton("📁 Procurar")
        btn_browse_input.setFixedWidth(80)
        btn_browse_input.setFixedHeight(25)
        btn_browse_input.clicked.connect(self.browse_input_video)

        input_path_layout.addWidget(lbl_input)
        input_path_layout.addWidget(self.input_path_field)
        input_path_layout.addWidget(btn_browse_input)
        paths_layout.addLayout(input_path_layout)

        output_path_layout = QHBoxLayout()
        lbl_output = QLabel("Pasta de Saída:")
        lbl_output.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #dddddd; min-width: 100px;"
        )

        self.output_path_field = QLineEdit()
        self.output_path_field.setReadOnly(True)
        self.output_path_field.setPlaceholderText("Padrão: exports/{nome_do_video}_processed")
        self.output_path_field.setMinimumHeight(25)
        self.output_path_field.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3e3e42; border-radius: 4px; padding: 5px;"
        )

        btn_browse_output = QPushButton("📁 Procurar")
        btn_browse_output.setFixedWidth(80)
        btn_browse_output.setFixedHeight(25)
        btn_browse_output.clicked.connect(self.browse_output_folder)

        btn_reset_output = QPushButton("↺ Resetar")
        btn_reset_output.setFixedWidth(80)
        btn_reset_output.setFixedHeight(25)
        btn_reset_output.clicked.connect(self.reset_output_folder)

        output_path_layout.addWidget(lbl_output)
        output_path_layout.addWidget(self.output_path_field)
        output_path_layout.addWidget(btn_browse_output)
        output_path_layout.addWidget(btn_reset_output)
        paths_layout.addLayout(output_path_layout)

        main_layout.addLayout(paths_layout)

        # Configurações avançadas
        advanced_group = QGroupBox("Configurações Avançadas")
        advanced_layout = QVBoxLayout(advanced_group)

        lbl_resolution = QLabel("Resolução de Saída:")
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["1080p", "720p", "480p"])
        self.combo_resolution.setCurrentText("1080p")
        advanced_layout.addWidget(lbl_resolution)
        advanced_layout.addWidget(self.combo_resolution)

        lbl_bitrate = QLabel("Bitrate de Vídeo (kbps, opcional):")
        self.edit_bitrate = QLineEdit()
        self.edit_bitrate.setPlaceholderText("Deixe vazio para CRF 18")
        advanced_layout.addWidget(lbl_bitrate)
        advanced_layout.addWidget(self.edit_bitrate)

        lbl_custom_prompt = QLabel("Prompt Customizado (opcional):")
        self.edit_custom_prompt = QTextEdit()
        self.edit_custom_prompt.setPlaceholderText("Deixe vazio para usar prompts padrão...")
        advanced_layout.addWidget(lbl_custom_prompt)
        advanced_layout.addWidget(self.edit_custom_prompt)

        self.chk_dark_theme = QCheckBox("Tema Escuro")
        self.chk_dark_theme.setChecked(True)
        self.chk_dark_theme.stateChanged.connect(self.toggle_theme)
        advanced_layout.addWidget(self.chk_dark_theme)

        main_layout.addWidget(advanced_group)

        # DropZone
        self.drop_zone = DropZone()
        self.drop_zone.setMinimumHeight(80)
        self.drop_zone.file_dropped.connect(self.on_video_selected)
        main_layout.addWidget(self.drop_zone)

        # Progresso e ações
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)

        self.btn_action = QPushButton("Iniciar Processamento Viral")
        self.btn_action.setFixedHeight(45)
        self.btn_action.setEnabled(False)
        self.btn_action.clicked.connect(self.start_processing)

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.btn_action)

        self.btn_report = QPushButton("Ver Histórico de Processamento")
        self.btn_report.clicked.connect(self.show_processing_history)
        main_layout.addWidget(self.btn_report)

        lbl_log = QLabel("Terminal de Processamento:")
        lbl_log.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))

        main_layout.addWidget(lbl_log)
        main_layout.addWidget(self.log_output)

    # ---------------- Tema ----------------
    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #121212; }
            QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
            
            QLineEdit {
                background-color: #1e1e1e; border: 1px solid #3e3e42;
                border-radius: 4px; padding: 5px 10px; color: #aaaaaa;
            }
            QLineEdit:focus { border: 1px solid #0078D7; }
            
            QComboBox {
                background-color: #2d2d30; border: 1px solid #3e3e42;
                border-radius: 4px; padding: 5px 15px;
            }
            QComboBox::drop-down { border: none; }
            
            QPushButton {
                background-color: #0078D7; color: white;
                font-size: 14px; font-weight: bold;
                border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #1084ea; }
            QPushButton:disabled { background-color: #333333; color: #777777; }
            
            QTextEdit {
                background-color: #0c0c0c; color: #00ff00;
                border: 1px solid #333333; border-radius: 6px; padding: 10px;
            }
        """
        )

    def _apply_light_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #f0f0f0; }
            QWidget { color: #000000; font-family: 'Segoe UI', Arial, sans-serif; }
            QLineEdit {
                background-color: #ffffff; border: 1px solid #cccccc;
                border-radius: 4px; padding: 5px 10px; color: #000000;
            }
            QLineEdit:focus { border: 1px solid #0078D7; }
            QComboBox {
                background-color: #ffffff; border: 1px solid #cccccc;
                border-radius: 4px; padding: 5px 15px;
            }
            QPushButton {
                background-color: #0078D7; color: white;
                font-size: 14px; font-weight: bold;
                border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #1084ea; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
            QProgressBar {
                background-color: #cccccc; border-radius: 4px; border: none;
            }
            QProgressBar::chunk { background-color: #0078D7; border-radius: 4px; }
            QTextEdit {
                background-color: #ffffff; color: #000000;
                border: 1px solid #cccccc; border-radius: 6px; padding: 10px;
            }
            QGroupBox { border: 1px solid #cccccc; border-radius: 4px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            """
        )

    def toggle_theme(self) -> None:
        if self.chk_dark_theme.isChecked():
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

    # ---------------- Handlers ----------------
    def on_video_selected(self, file_path: str) -> None:
        self.current_video_path = file_path
        self.input_path_field.setText(file_path)
        self.log_output.clear()
        self.update_log(f"[*] VÍDEO CARREGADO: {file_path}")
        self.drop_zone.lbl_text.setText(f"🎥 {Path(file_path).name}\n(Clique para trocar)")

        self.btn_action.setText("Iniciar Processamento Viral")
        self.btn_action.setStyleSheet("")
        self.btn_action.setEnabled(True)

    def browse_input_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo de Entrada", "", "Vídeos (*.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self.on_video_selected(file_path)

    def browse_output_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída", "")
        if folder_path:
            self.output_folder_path = folder_path
            self.output_path_field.setText(folder_path)
            self.update_log(f"[*] Pasta de saída definida: {folder_path}")

    def reset_output_folder(self) -> None:
        self.output_folder_path = None
        self.output_path_field.setText("")
        self.output_path_field.setPlaceholderText("Padrão: exports/{nome_do_video}_processed")
        self.update_log("[*] Pasta de saída resetada para o padrão.")

    def start_processing(self) -> None:
        if self.btn_action.text() == "Processar Novo Vídeo":
            self.reset_ui_for_new_video()
            return

        if not self.current_video_path:
            return

        model_selected = self.combo_model.currentText()
        available_models = self._get_available_models()
        if model_selected not in available_models:
            self.update_log(f"[!] ERRO: Modelo '{model_selected}' não encontrado no Ollama.")
            self.update_log(f"[!] Modelos disponíveis: {', '.join(available_models)}")
            self._unlock_ui_after_process()
            self.btn_action.setText("Selecionar Modelo Válido")
            return

        self.update_log(f"[*] Iniciando motor com IA: {model_selected.upper()}")
        self.update_log(
            f"[*] Modelo de transcrição selecionado: {self.combo_whisper_model.currentText()}"
        )
        self.update_log(f"[*] Tipo de prompt selecionado: {self.combo_prompt.currentText()}")

        self.btn_action.setEnabled(False)
        self.btn_action.setText("Processando... Aguarde.")
        self.combo_model.setEnabled(False)
        self.combo_whisper_model.setEnabled(False)
        self.drop_zone.setEnabled(False)

        self.progress_bar.setProperty("state", "normal")
        self.progress_bar.setVisible(True)

        self.worker = VideoProcessorThread(
            self.current_video_path,
            model_name=model_selected,
            output_dir=self.output_folder_path,
            prompt_type=self.combo_prompt.currentText(),
            whisper_model=self.combo_whisper_model.currentText(),
            resolution=self.combo_resolution.currentText(),
            bitrate=self.edit_bitrate.text().strip(),
            custom_prompt=(
                self.edit_custom_prompt.toPlainText().strip()
                if self.edit_custom_prompt.toPlainText().strip()
                else None
            ),
        )
        self.worker.progress_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.clips_ready_signal.connect(self.on_clips_ready)
        self.worker.start()

    def update_log(self, text: str) -> None:
        self.log_output.append(f"> {text}")

    def on_clips_ready(self, clips_dicts: list[dict]) -> None:
        self.update_log(f"[→] Abrindo seletor de clipes com {len(clips_dicts)} opções...")
        clips = [Clip(**c) for c in clips_dicts]
        dialog = ClipSelectionDialog(clips, self)
        result = dialog.exec()

        if result == QDialog.Accepted:  # type: ignore[name-defined]
            selected_clips = dialog.get_selected_clips()
            self.update_log(f"[✓] Usuário selecionou {len(selected_clips)} clipe(s) para salvar.")
            if self.worker:
                self.worker.selected_clips = selected_clips
        else:
            self.update_log("[✗] Seleção cancelada pelo usuário.")
            if self.worker:
                self.worker.selected_clips = []

    def on_error(self, error_msg: str) -> None:
        self.log_output.append(f"\n[!] ERRO CRÍTICO:\n{error_msg}")
        self._unlock_ui_after_process()
        self.btn_action.setText("Tentar Novamente")

    def on_finished(self, msg: str) -> None:
        self.log_output.append(f"\n[+] {msg}")
        self._unlock_ui_after_process()

        self.btn_action.setText("Processar Novo Vídeo")
        self.btn_action.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; border-radius: 6px;"
        )

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar { background-color: #2d2d30; border-radius: 4px; border: none; }
            QProgressBar::chunk { background-color: #28a745; border-radius: 4px; }
        """
        )

    def _unlock_ui_after_process(self) -> None:
        self.btn_action.setEnabled(True)
        self.combo_model.setEnabled(True)
        self.combo_whisper_model.setEnabled(True)
        self.drop_zone.setEnabled(True)

    def reset_ui_for_new_video(self) -> None:
        self.current_video_path = None
        self.output_folder_path = None
        self.log_output.clear()

        self.input_path_field.setText("")
        self.input_path_field.setPlaceholderText("Nenhum vídeo selecionado...")
        self.output_path_field.setText("")
        self.output_path_field.setPlaceholderText("Padrão: exports/{nome_do_video}_processed")

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet("")

        self.drop_zone.lbl_text.setText("Arraste e solte o seu vídeo aqui\nou clique para procurar")

        self.btn_action.setText("Selecione um Vídeo Acima")
        self.btn_action.setStyleSheet(
            "background-color: #0078D7; color: white; font-weight: bold; border-radius: 6px;"
        )
        self.btn_action.setEnabled(False)

        self.update_log("[*] Sistema limpo e pronto para um novo vídeo.")

    def show_processing_history(self) -> None:
        try:
            history_file = config.PROCESSING_HISTORY_FILE
            if not history_file.exists():
                QMessageBox.information(self, "Histórico", "Nenhum histórico encontrado.")
                return
            with history_file.open("r", encoding="utf-8") as f:
                history = json.load(f)
            if not history:
                QMessageBox.information(self, "Histórico", "Histórico vazio.")
                return
            text = "Últimos Processamentos:\n\n"
            for entry in history[-5:]:
                text += (
                    f"Vídeo: {Path(entry.get('video_path', '')).name}\n"
                    f"Duração: {entry.get('video_duration', 0):.1f}s\n"
                    f"Clipes Encontrados: {entry.get('total_clips_found', 0)}\n"
                    f"Clipes Selecionados: {entry.get('clips_selected', 0)}\n"
                    f"Tempo de Transcrição: {entry.get('transcription_time', 0):.1f}s\n"
                    f"Tempo de Análise: {entry.get('analysis_time', 0):.1f}s\n"
                    f"Tempo de Renderização: {entry.get('rendering_time', 0):.1f}s\n"
                    f"Modelo: {entry.get('model_used', '')}\n"
                    f"Prompt: {entry.get('prompt_type', '')}\n"
                    f"Timestamp: {time.ctime(entry.get('timestamp', 0))}\n\n"
                )
            QMessageBox.information(self, "Histórico de Processamento", text)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Erro", f"Erro ao carregar histórico: {e}")


def run() -> None:
    app = QApplication.instance() or QApplication([])
    window = ViralApp()
    window.show()
    app.exec()
