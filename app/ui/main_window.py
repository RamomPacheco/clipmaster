from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import List
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core import config  # noqa: F401  # garante import de config
from app.core.api_key_store import ApiKeyStore
from app.core.cuda_setup import inject_cuda_environment  # noqa: F401
from app.core.logger import logger
from app.models.schemas import Clip
from app.ui.components.drop_zone import DropZone
from app.ui.dialogs.clip_dialog import ClipSelectionDialog
from app.ui.dialogs.save_api_key_dialog import SaveApiKeyDialog
from app.workers.processing_task import VideoProcessorThread


class ViralApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Viral Clipper Pro")
        self.setMinimumSize(920, 680)
        self.current_video_path: str | None = None
        self.output_folder_path: str | None = None
        self.worker: VideoProcessorThread | None = None
        self._api_key_store = ApiKeyStore()

        self._setup_ui()
        self._apply_dark_theme()
        self._refresh_status_bar()

    # ---------------- UI Setup ----------------
    def _get_gemini_models(self) -> List[str]:
        api_key = self.edit_api_key.text().strip() if hasattr(self, "edit_api_key") else ""
        if not api_key:
            return ["gemini-2.5-flash"]
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            discovered: List[str] = []
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "generateContent" not in methods:
                    continue
                name = str(getattr(model, "name", "")).strip()
                if not name:
                    continue
                discovered.append(name.removeprefix("models/"))

            if discovered:
                # Remove duplicados preservando ordem
                return list(dict.fromkeys(discovered))
        except Exception as e:  # noqa: BLE001
            logger.warning("Não foi possível listar modelos Gemini via API: %s", e)

        return ["gemini-2.5-flash"]

    def _fetch_groq_model_list(self) -> tuple[bool, List[str]]:
        """
        Lista modelos via GET /openai/v1/models.
        Retorna (sucesso_api, ids). Sem chave ou erro → lista padrão e sucesso_api=False.
        """
        default = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]
        key = ""
        if hasattr(self, "edit_api_key"):
            key = self.edit_api_key.text().strip()
        if not key:
            key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if not key:
            return False, list(default)

        try:
            import httpx
        except ImportError:
            logger.warning("httpx não instalado; não é possível listar modelos Groq.")
            return False, list(default)

        url = "https://api.groq.com/openai/v1/models"
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                )
                resp.raise_for_status()
                body = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("Não foi possível listar modelos Groq: %s", e)
            return False, list(default)

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return False, list(default)

        ids: List[str] = []
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
        if not ids:
            return False, list(default)

        ids = sorted(set(ids), key=str.lower)
        return True, ids

    def _current_llm_provider(self) -> str:
        text = self.combo_provider.currentText().strip().lower()
        if "gemini" in text:
            return "gemini"
        if "groq" in text:
            return "groq"
        if "transformers" in text or "hugging face" in text:
            return "transformers"
        return "ollama"

    def _uses_cloud_api_key(self) -> bool:
        return self._current_llm_provider() in ("gemini", "groq")

    def _sync_llm_model_options(self) -> None:
        provider = self._current_llm_provider()
        self.combo_model.clear()
        if provider == "gemini":
            self.lbl_model.setText("Modelo de IA (Gemini API):")
            self.combo_model.addItems(self._get_gemini_models())
            self.combo_model.setToolTip("Selecione o modelo para uso via API Gemini.")
            if "gemini-2.5-flash" in self._get_gemini_models():
                self.combo_model.setCurrentText("gemini-2.5-flash")
            self.combo_model.setEditable(False)
        elif provider == "groq":
            self.lbl_model.setText("Modelo de IA (Groq API):")
            live, groq_models = self._fetch_groq_model_list()
            self.combo_model.addItems(groq_models)
            prefer = "llama-3.3-70b-versatile"
            if prefer in groq_models:
                self.combo_model.setCurrentText(prefer)
            elif groq_models:
                self.combo_model.setCurrentIndex(0)
            self.combo_model.setToolTip(
                "Modelos devolvidos pela API Groq para a sua chave."
                if live
                else "Lista padrão — cole a chave Groq e mude de campo ou de provedor para atualizar."
            )
            self.combo_model.setEditable(not live)
        elif provider == "transformers":
            self.lbl_model.setText("Modelo de IA (Transformers Local):")
            self.combo_model.addItems(
                [
                    "zai-org/GLM-4.7",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                ]
            )
            self.combo_model.setToolTip(
                "Informe um model_id do Hugging Face (ex.: zai-org/GLM-4.7)."
            )
            self.combo_model.setEditable(True)
        else:
            self.lbl_model.setText("Modelo de IA (Ollama Local):")
            self.combo_model.addItems(self._get_available_models())
            self.combo_model.setToolTip("Selecione o modelo disponível no Ollama local.")
            self.combo_model.setEditable(False)

        self._sync_social_llm_model_options()
        self._on_social_model_mode_changed()

    def _sync_social_llm_model_options(self) -> None:
        """Preenche o combo do modelo alternativo do pacote social (mesmo provedor da análise)."""
        if not hasattr(self, "combo_social_model"):
            return
        provider = self._current_llm_provider()
        prev = self.combo_social_model.currentText().strip()
        self.combo_social_model.blockSignals(True)
        self.combo_social_model.clear()
        if provider == "gemini":
            models = self._get_gemini_models()
            self.combo_social_model.addItems(models)
            self.combo_social_model.setToolTip("Modelo Gemini só para capa e descrição social.")
            self.combo_social_model.setEditable(False)
            if prev in models:
                self.combo_social_model.setCurrentText(prev)
            elif "gemini-2.5-flash" in models:
                self.combo_social_model.setCurrentText("gemini-2.5-flash")
        elif provider == "groq":
            live, models = self._fetch_groq_model_list()
            self.combo_social_model.addItems(models)
            self.combo_social_model.setToolTip(
                "Modelos Groq (API) para o pacote social."
                if live
                else "Lista padrão Groq — use chave API para listar todos."
            )
            self.combo_social_model.setEditable(not live)
            if prev in models:
                self.combo_social_model.setCurrentText(prev)
            elif "llama-3.3-70b-versatile" in models:
                self.combo_social_model.setCurrentText("llama-3.3-70b-versatile")
            elif models:
                self.combo_social_model.setCurrentIndex(0)
        elif provider == "transformers":
            self.combo_social_model.addItems(
                [
                    "zai-org/GLM-4.7",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                ]
            )
            self.combo_social_model.setToolTip(
                "Model_id Hugging Face para capa/descrição (pode editar o texto)."
            )
            self.combo_social_model.setEditable(True)
            if prev:
                self.combo_social_model.setCurrentText(prev)
        else:
            self.combo_social_model.addItems(self._get_available_models())
            self.combo_social_model.setToolTip(
                "Modelo Ollama só para gerar frase e descrição do pacote social."
            )
            self.combo_social_model.setEditable(False)
            if prev:
                self.combo_social_model.setCurrentText(prev)
        self.combo_social_model.blockSignals(False)

    def _on_social_model_mode_changed(self) -> None:
        if not hasattr(self, "combo_social_model_source"):
            return
        use_other = self.combo_social_model_source.currentIndex() == 1
        self.lbl_social_model.setVisible(use_other)
        self.combo_social_model.setVisible(use_other)
        self.combo_social_model.setEnabled(use_other)

    def _apply_social_package_controls_state(self) -> None:
        """Sub-opções do pacote social só fazem sentido quando o pacote está ativado."""
        if not hasattr(self, "chk_enable_social_package"):
            return
        on = self.chk_enable_social_package.isChecked()
        self.lbl_social_pack.setEnabled(on)
        self.combo_social_model_source.setEnabled(on)
        self.chk_social_cover.setEnabled(on)
        lbl_hint = getattr(self, "lbl_social_hint", None)
        if lbl_hint is not None:
            lbl_hint.setEnabled(on)
        if not on:
            self.lbl_social_model.setEnabled(False)
            self.combo_social_model.setEnabled(False)
        else:
            self._on_social_model_mode_changed()

    def _on_provider_changed(self) -> None:
        use_key = self._uses_cloud_api_key()
        self.edit_api_key.setEnabled(use_key)
        self.combo_api_profile.setEnabled(use_key)
        self.btn_api_key_save.setEnabled(use_key)
        self.btn_api_key_remove.setEnabled(use_key)
        self.lbl_api_profile.setEnabled(use_key)
        self.lbl_api_key_row.setEnabled(use_key)
        prov = self._current_llm_provider()
        if prov == "gemini":
            self.lbl_api_key_row.setText("Chave API (Gemini)")
        elif prov == "groq":
            self.lbl_api_key_row.setText("Chave API (Groq)")
        else:
            self.lbl_api_key_row.setText("Chave API")
        # Carregar perfil/chave antes de pedir listas à API (Gemini / Groq).
        self._refresh_api_profile_combo()
        if self._current_llm_provider() not in ("gemini", "groq"):
            self._sync_llm_model_options()
        self._refresh_status_bar()

    def _refresh_api_profile_combo(self) -> None:
        if not hasattr(self, "combo_api_profile"):
            return
        self.combo_api_profile.blockSignals(True)
        self.combo_api_profile.clear()
        self.combo_api_profile.addItem("(Colar manualmente — sem perfil guardado)", None)
        prov = self._current_llm_provider()
        for p in self._api_key_store.list_for_provider(prov):
            self.combo_api_profile.addItem(p.label, p.id)
        last = self._api_key_store.last_profile_id(prov) if prov in ("gemini", "groq") else None
        sel = 0
        if last:
            for i in range(self.combo_api_profile.count()):
                if self.combo_api_profile.itemData(i) == last:
                    sel = i
                    break
        self.combo_api_profile.setCurrentIndex(sel)
        self.combo_api_profile.blockSignals(False)
        if sel > 0:
            self._on_api_profile_changed(sel)
        elif self._uses_cloud_api_key():
            # Manual com API na nuvem: respeitar limpeza do último perfil
            self._on_api_profile_changed(0)
        # Com Ollama e índice 0, não chamar — não limpa último perfil Gemini/Groq guardado

    def _on_api_profile_changed(self, index: int) -> None:
        if index < 0 or not hasattr(self, "combo_api_profile"):
            return
        pid = self.combo_api_profile.itemData(index)
        prov = self._current_llm_provider()
        if pid is None:
            if prov in ("gemini", "groq"):
                self._api_key_store.set_last_for_provider(prov, None)
            if self._uses_cloud_api_key():
                self._sync_llm_model_options()
            return
        prof = self._api_key_store.get(str(pid))
        if prof:
            self._api_key_store.set_last_for_provider(prof.provider, prof.id)
            self.edit_api_key.blockSignals(True)
            self.edit_api_key.setText(prof.secret)
            self.edit_api_key.blockSignals(False)
            if prof.provider in ("gemini", "groq") and self._current_llm_provider() == prof.provider:
                self._sync_llm_model_options()

    def _open_save_api_key_dialog(self) -> None:
        prov = self._current_llm_provider()
        default_lbl = "Chave API Groq" if prov == "groq" else "Chave API Gemini"
        dlg = SaveApiKeyDialog(
            self,
            default_provider=prov if prov in ("gemini", "groq") else "gemini",
            default_label=default_lbl,
            default_secret=self.edit_api_key.text().strip(),
        )
        if dlg.exec() != QDialog.Accepted:
            return
        label = dlg.profile_label() or default_lbl
        secret = dlg.secret()
        if not secret:
            QMessageBox.warning(self, "Chave vazia", "Informe a chave antes de guardar.")
            return
        try:
            prof = self._api_key_store.add(label, dlg.provider_id(), secret)
        except ValueError as e:
            QMessageBox.warning(self, "Erro", str(e))
            return
        self._refresh_api_profile_combo()
        for i in range(self.combo_api_profile.count()):
            if self.combo_api_profile.itemData(i) == prof.id:
                self.combo_api_profile.setCurrentIndex(i)
                break
        self.update_log(f"[*] Perfil de chave guardado: {prof.label} ({prof.provider})")

    def _remove_selected_api_profile(self) -> None:
        idx = self.combo_api_profile.currentIndex()
        pid = self.combo_api_profile.itemData(idx)
        if pid is None:
            QMessageBox.information(
                self,
                "Remover perfil",
                "Selecione um perfil guardado na lista (não a opção manual).",
            )
            return
        prof = self._api_key_store.get(str(pid))
        name = prof.label if prof else str(pid)
        if (
            QMessageBox.question(
                self,
                "Remover perfil",
                f"Remover o perfil «{name}» do disco?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._api_key_store.remove(str(pid))
        self._refresh_api_profile_combo()
        self.update_log(f"[*] Perfil de chave removido: {name}")

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

    def _get_video_height(self, file_path: str) -> int | None:
        """Retorna a altura do vídeo via ffprobe (ex.: 1080, 2160)."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=height",
                    "-of",
                    "csv=p=0",
                    file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            out = result.stdout.strip()
            return int(out) if out.isdigit() else None
        except Exception as e:  # noqa: BLE001
            logger.warning("Não foi possível detectar resolução do vídeo: %s", e)
            return None

    def _update_resolution_options_for_video(self, file_path: str) -> None:
        """
        Ajusta opções de saída com base na resolução do vídeo importado.
        Ex.: vídeo 1440p mostra até 2K.
        """
        all_options = [
            ("SD (720p)", 720),
            ("HD (1080p)", 1080),
            ("2K (1440p)", 1440),
            ("4K (2160p)", 2160),
        ]
        video_h = self._get_video_height(file_path)
        selected_before = self.combo_resolution.currentText()
        self.combo_resolution.clear()

        if video_h is None:
            self.combo_resolution.addItems([label for label, _ in all_options])
        else:
            allowed = [label for label, h in all_options if h <= video_h]
            if not allowed:
                allowed = [all_options[0][0]]
            self.combo_resolution.addItems(allowed)

        if selected_before in [self.combo_resolution.itemText(i) for i in range(self.combo_resolution.count())]:
            self.combo_resolution.setCurrentText(selected_before)
        else:
            self.combo_resolution.setCurrentIndex(self.combo_resolution.count() - 1)

    def _refresh_status_bar(self) -> None:
        """Atualiza rodapé com hardware e modo de IA (feedback de sistema)."""
        cuda_line = "CUDA: —"
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                cuda_line = f"● GPU: {name}"
            else:
                cuda_line = "○ CUDA: não disponível (CPU)"
        except Exception:  # noqa: BLE001
            cuda_line = "○ CUDA: —"

        cpu_info = platform.processor() or platform.machine() or "—"
        prov = self._current_llm_provider() if hasattr(self, "combo_provider") else "ollama"
        ia_line = {
            "gemini": "IA: Gemini (API)",
            "groq": "IA: Groq (API)",
            "transformers": "IA: Transformers (local)",
        }.get(prov, "IA: Ollama (local)")

        ffmpeg_ok = shutil.which("ffmpeg") is not None
        ff = "● FFmpeg: OK" if ffmpeg_ok else "○ FFmpeg: não encontrado no PATH"
        msg = f"Status  |  {cuda_line}  |  CPU: {cpu_info}  |  {ia_line}  |  {ff}"
        self.statusBar().showMessage(msg)

    def _toggle_log_visibility(self, visible: bool) -> None:
        self.lbl_log.setVisible(visible)
        self.log_output.setVisible(visible)
        if visible:
            self.log_output.setMinimumHeight(160)
        else:
            self.log_output.setMinimumHeight(0)

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(24, 20, 24, 16)

        lbl_title = QLabel("AI Viral Clipper Pro")
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff;")
        lbl_tagline = QLabel("Corte viral em poucos cliques — ajustes técnicos na aba Avançado.")
        lbl_tagline.setStyleSheet("font-size: 12px; color: #888888; margin-bottom: 4px;")
        main_layout.addWidget(lbl_title)
        main_layout.addWidget(lbl_tagline)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMinimumHeight(320)

        # ---------- Aba Essencial ----------
        tab_basic = QWidget()
        basic_layout = QVBoxLayout(tab_basic)
        basic_layout.setSpacing(14)

        self.drop_zone = DropZone()
        self.drop_zone.setMinimumHeight(140)
        self.drop_zone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.drop_zone.file_dropped.connect(self.on_video_selected)
        basic_layout.addWidget(self.drop_zone)

        out_row = QHBoxLayout()
        lbl_output = QLabel("Pasta de saída")
        lbl_output.setStyleSheet("font-weight: bold; color: #cccccc; min-width: 110px;")
        self.output_path_field = QLineEdit()
        self.output_path_field.setReadOnly(True)
        self.output_path_field.setPlaceholderText("Padrão: exports/{nome_do_video}_processed")
        self.output_path_field.setMinimumHeight(32)
        btn_browse_output = QPushButton("Procurar…")
        btn_browse_output.setObjectName("secondaryButton")
        btn_browse_output.setFixedWidth(100)
        btn_browse_output.clicked.connect(self.browse_output_folder)
        btn_reset_output = QPushButton("Usar padrão")
        btn_reset_output.setObjectName("secondaryButton")
        btn_reset_output.setFixedWidth(100)
        btn_reset_output.clicked.connect(self.reset_output_folder)
        out_row.addWidget(lbl_output)
        out_row.addWidget(self.output_path_field, stretch=1)
        out_row.addWidget(btn_browse_output)
        out_row.addWidget(btn_reset_output)
        basic_layout.addLayout(out_row)

        grid_basic = QGridLayout()
        grid_basic.setHorizontalSpacing(16)
        grid_basic.setVerticalSpacing(10)
        lbl_res = QLabel("Qualidade de exportação")
        lbl_res.setStyleSheet("color: #aaaaaa;")
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(
            ["SD (720p)", "HD (1080p)", "2K (1440p)", "4K (2160p)"]
        )
        self.combo_resolution.setCurrentText("HD (1080p)")
        self.combo_resolution.setMinimumHeight(32)
        lbl_aspect = QLabel("Formato do vídeo")
        lbl_aspect.setStyleSheet("color: #aaaaaa;")
        self.combo_aspect_ratio = QComboBox()
        self.combo_aspect_ratio.addItems(
            ["Vertical (9:16) - Redes sociais", "Horizontal (16:9)"]
        )
        self.combo_aspect_ratio.setCurrentText("Vertical (9:16) - Redes sociais")
        self.combo_aspect_ratio.setMinimumHeight(32)
        grid_basic.addWidget(lbl_res, 0, 0)
        grid_basic.addWidget(self.combo_resolution, 1, 0)
        grid_basic.addWidget(lbl_aspect, 0, 1)
        grid_basic.addWidget(self.combo_aspect_ratio, 1, 1)
        basic_layout.addLayout(grid_basic)

        opts_row = QHBoxLayout()
        self.chk_tiktok_captions = QCheckBox("Legendas estilo TikTok")
        self.chk_tiktok_captions.setChecked(False)
        self.chk_tiktok_captions.setToolTip(
            "Legenda dinâmica com destaque por palavra (timestamps do Whisper)."
        )
        self.chk_skip_preview = QCheckBox("Renderizar sem pré-visualização")
        self.chk_skip_preview.setChecked(False)
        self.chk_skip_preview.setToolTip(
            "Exporta todos os clipes encontrados sem abrir o seletor manual."
        )
        opts_row.addWidget(self.chk_tiktok_captions)
        opts_row.addWidget(self.chk_skip_preview)
        opts_row.addStretch()
        basic_layout.addLayout(opts_row)
        basic_layout.addStretch()

        # ---------- Aba Avançado ----------
        tab_adv = QWidget()
        adv_outer = QVBoxLayout(tab_adv)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        adv_layout = QVBoxLayout(scroll_content)
        adv_layout.setSpacing(14)

        gb_ai = QGroupBox("Motores de IA")
        gb_ai.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 8px; }")
        ai_form = QGridLayout()
        ai_form.setColumnStretch(1, 1)
        lbl_provider = QLabel("Provedor")
        lbl_provider.setStyleSheet("color: #aaaaaa;")
        self.combo_provider = QComboBox()
        self.combo_provider.addItems(
            ["Local (Ollama)", "API (Gemini)", "API (Groq)", "Local (Transformers)"]
        )
        self.combo_provider.setCurrentText("Local (Ollama)")
        self.combo_provider.setMinimumHeight(32)
        self.combo_provider.currentTextChanged.connect(self._on_provider_changed)
        self.lbl_model = QLabel("Modelo de IA")
        self.lbl_model.setStyleSheet("color: #aaaaaa;")
        self.combo_model = QComboBox()
        self.combo_model.setMinimumHeight(32)
        self.lbl_api_profile = QLabel("Perfil de chave guardada")
        self.lbl_api_profile.setStyleSheet("color: #aaaaaa;")
        self.combo_api_profile = QComboBox()
        self.combo_api_profile.setMinimumHeight(32)
        self.combo_api_profile.setToolTip(
            "Mostra só os perfis guardados para o provedor selecionado (Gemini ou Groq). "
            "No Windows o ficheiro fica em AppData\\Local\\AI_Viral_Clipper\\api_keys.json."
        )
        self.btn_api_key_save = QPushButton("Guardar…")
        self.btn_api_key_save.setToolTip("Guardar a chave do campo abaixo com um nome na lista.")
        self.btn_api_key_save.clicked.connect(self._open_save_api_key_dialog)
        self.btn_api_key_remove = QPushButton("Remover")
        self.btn_api_key_remove.setToolTip("Apaga o perfil selecionado do disco.")
        self.btn_api_key_remove.clicked.connect(self._remove_selected_api_profile)
        profile_row = QHBoxLayout()
        profile_row.addWidget(self.combo_api_profile, stretch=1)
        profile_row.addWidget(self.btn_api_key_save)
        profile_row.addWidget(self.btn_api_key_remove)
        profile_wrap = QWidget()
        profile_wrap.setLayout(profile_row)

        self.lbl_api_key_row = QLabel("Chave API")
        self.lbl_api_key_row.setStyleSheet("color: #aaaaaa;")
        self.edit_api_key = QLineEdit()
        self.edit_api_key.setEchoMode(QLineEdit.Password)
        self.edit_api_key.setPlaceholderText("Cole a chave ou escolha um perfil guardado")
        self.edit_api_key.setMinimumHeight(32)
        self.edit_api_key.setEnabled(False)
        self.edit_api_key.textChanged.connect(lambda _t: self._sync_llm_model_options())
        self.combo_api_profile.currentIndexChanged.connect(self._on_api_profile_changed)

        ai_form.addWidget(lbl_provider, 0, 0, Qt.AlignRight)
        ai_form.addWidget(self.combo_provider, 0, 1)
        ai_form.addWidget(self.lbl_model, 1, 0, Qt.AlignRight)
        ai_form.addWidget(self.combo_model, 1, 1)
        ai_form.addWidget(self.lbl_api_profile, 2, 0, Qt.AlignRight | Qt.AlignTop)
        ai_form.addWidget(profile_wrap, 2, 1)
        ai_form.addWidget(self.lbl_api_key_row, 3, 0, Qt.AlignRight | Qt.AlignTop)
        ai_form.addWidget(self.edit_api_key, 3, 1)
        gb_ai.setLayout(ai_form)
        adv_layout.addWidget(gb_ai)

        gb_whisper = QGroupBox("Transcrição (Faster-Whisper)")
        whisper_form = QGridLayout()
        lbl_whisper_model = QLabel("Modelo Whisper")
        lbl_whisper_model.setStyleSheet("color: #aaaaaa;")
        self.combo_whisper_model = QComboBox()
        self.combo_whisper_model.addItems(
            ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        )
        self.combo_whisper_model.setCurrentText(config.WHISPER_MODEL)
        self.combo_whisper_model.setMinimumHeight(32)
        lbl_whisper_device = QLabel("Dispositivo")
        lbl_whisper_device.setStyleSheet("color: #aaaaaa;")
        self.combo_whisper_device = QComboBox()
        self.combo_whisper_device.addItems(
            ["Auto (recomendado)", "CPU (estável)", "GPU CUDA (rápido)"]
        )
        self.combo_whisper_device.setCurrentText("CPU (estável)")
        self.combo_whisper_device.setMinimumHeight(32)
        whisper_form.addWidget(lbl_whisper_model, 0, 0, Qt.AlignRight)
        whisper_form.addWidget(self.combo_whisper_model, 0, 1)
        whisper_form.addWidget(lbl_whisper_device, 1, 0, Qt.AlignRight)
        whisper_form.addWidget(self.combo_whisper_device, 1, 1)
        gb_whisper.setLayout(whisper_form)
        adv_layout.addWidget(gb_whisper)

        gb_analysis = QGroupBox("Análise de conteúdo")
        an_form = QVBoxLayout()
        lbl_prompt = QLabel("Tipo de prompt")
        lbl_prompt.setStyleSheet("color: #aaaaaa;")
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
        self.combo_prompt.setCurrentText("Padrão (Equilibrado)")
        self.combo_prompt.setMinimumHeight(32)
        an_form.addWidget(lbl_prompt)
        an_form.addWidget(self.combo_prompt)
        gb_analysis.setLayout(an_form)
        adv_layout.addWidget(gb_analysis)

        gb_export = QGroupBox("Exportação e extras")
        ex_form = QVBoxLayout()
        lbl_framing = QLabel("Enquadramento")
        lbl_framing.setStyleSheet("color: #aaaaaa;")
        self.combo_framing_mode = QComboBox()
        self.combo_framing_mode.addItems(
            [
                "Manter conteúdo (com bordas)",
                "Preencher tela (crop)",
                "Crop inteligente (rosto)",
            ]
        )
        self.combo_framing_mode.setMinimumHeight(32)
        lbl_bitrate = QLabel("Bitrate de vídeo (kbps, opcional)")
        lbl_bitrate.setStyleSheet("color: #aaaaaa;")
        self.edit_bitrate = QLineEdit()
        self.edit_bitrate.setPlaceholderText("Vazio = qualidade por CRF")
        lbl_custom_prompt = QLabel("Prompt customizado (opcional)")
        lbl_custom_prompt.setStyleSheet("color: #aaaaaa;")
        self.edit_custom_prompt = QTextEdit()
        self.edit_custom_prompt.setPlaceholderText("Sobrescreve o prompt padrão da análise…")
        self.edit_custom_prompt.setMaximumHeight(100)
        lbl_max_tokens = QLabel("Transformers: max_new_tokens (opcional)")
        lbl_max_tokens.setStyleSheet("color: #aaaaaa;")
        self.edit_max_new_tokens = QLineEdit()
        self.edit_max_new_tokens.setPlaceholderText("Ex.: 700")
        self.edit_max_new_tokens.setToolTip("Apenas para provedor Local (Transformers).")
        self.chk_dark_theme = QCheckBox("Tema escuro")
        self.chk_dark_theme.setChecked(True)
        self.chk_dark_theme.stateChanged.connect(self.toggle_theme)
        ex_form.addWidget(lbl_framing)
        ex_form.addWidget(self.combo_framing_mode)
        ex_form.addWidget(lbl_bitrate)
        ex_form.addWidget(self.edit_bitrate)
        ex_form.addWidget(lbl_custom_prompt)
        ex_form.addWidget(self.edit_custom_prompt)
        ex_form.addWidget(lbl_max_tokens)
        ex_form.addWidget(self.edit_max_new_tokens)
        ex_form.addWidget(self.chk_dark_theme)
        gb_export.setLayout(ex_form)
        adv_layout.addWidget(gb_export)
        adv_layout.addStretch()

        # ---------- Aba Pacote Social ----------
        tab_social = QWidget()
        social_layout = QVBoxLayout(tab_social)
        social_layout.setSpacing(14)

        gb_social = QGroupBox("Capa + descrição para redes")
        social_form = QVBoxLayout()
        self.chk_enable_social_package = QCheckBox(
            "Gerar pacote social após renderizar os clipes"
        )
        self.chk_enable_social_package.setChecked(True)
        self.chk_enable_social_package.setToolTip(
            "Se desativar, o fluxo termina só com os vídeos dos clipes — sem IA extra, "
            "sem contexto narrativo longo e sem ficheiros clip_N_social / capa."
        )
        self.chk_enable_social_package.toggled.connect(self._apply_social_package_controls_state)

        self.lbl_social_pack = QLabel("Modelo de IA para pacote social")
        self.lbl_social_pack.setStyleSheet("color: #aaaaaa;")
        self.combo_social_model_source = QComboBox()
        self.combo_social_model_source.addItems(
            [
                "Usar o mesmo modelo da análise",
                "Escolher outro modelo (capa e descrição)",
            ]
        )
        self.combo_social_model_source.setMinimumHeight(32)
        self.combo_social_model_source.setToolTip(
            "Define qual modelo gera a frase de impacto e a descrição após os clipes serem renderizados."
        )
        self.combo_social_model_source.currentIndexChanged.connect(
            self._on_social_model_mode_changed
        )

        self.lbl_social_model = QLabel("Modelo alternativo (mesmo provedor)")
        self.lbl_social_model.setStyleSheet("color: #aaaaaa;")
        self.combo_social_model = QComboBox()
        self.combo_social_model.setMinimumHeight(32)

        self.lbl_social_hint = QLabel(
            "Com o pacote ativo: após os clipes, a IA usa o contexto de toda a fala do vídeo "
            "até ao fim de cada clipe para o título e a descrição."
        )
        self.lbl_social_hint.setStyleSheet("color: #888888; font-size: 11px;")
        self.chk_social_cover = QCheckBox("Gerar imagem de capa (JPG)")
        self.chk_social_cover.setChecked(True)
        self.chk_social_cover.setToolTip(
            "Se desmarcar, só são gerados o texto social (clip_N_social.txt), sem clip_N_capa.jpg."
        )

        social_form.addWidget(self.chk_enable_social_package)
        social_form.addWidget(self.lbl_social_pack)
        social_form.addWidget(self.combo_social_model_source)
        social_form.addWidget(self.lbl_social_model)
        social_form.addWidget(self.combo_social_model)
        social_form.addWidget(self.chk_social_cover)
        social_form.addWidget(self.lbl_social_hint)
        gb_social.setLayout(social_form)
        social_layout.addWidget(gb_social)
        social_layout.addStretch()

        scroll.setWidget(scroll_content)
        adv_outer.addWidget(scroll)

        self.tabs.addTab(tab_basic, "Essencial")
        self.tabs.addTab(tab_adv, "Avançado / Ajustes finos")
        self.tabs.addTab(tab_social, "Pacote Social")
        main_layout.addWidget(self.tabs)

        self._apply_social_package_controls_state()
        self._on_provider_changed()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)

        self.btn_action = QPushButton("Iniciar corte viral")
        self.btn_action.setObjectName("primaryAction")
        self.btn_action.setMinimumHeight(52)
        self.btn_action.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_action.setEnabled(False)
        self.btn_action.clicked.connect(self.start_processing)

        actions_row = QHBoxLayout()
        actions_row.addStretch()
        self.btn_report = QPushButton("Histórico de processamento")
        self.btn_report.setObjectName("secondaryButton")
        self.btn_report.setFixedHeight(36)
        self.btn_report.clicked.connect(self.show_processing_history)
        actions_row.addWidget(self.btn_report, alignment=Qt.AlignRight)

        self.chk_show_logs = QCheckBox("Mostrar terminal de logs")
        self.chk_show_logs.setChecked(False)
        self.chk_show_logs.stateChanged.connect(
            lambda s: self._toggle_log_visibility(bool(s))
        )

        self.lbl_log = QLabel("Terminal de processamento")
        self.lbl_log.setStyleSheet("color: #888888; font-size: 11px;")
        self.lbl_log.setVisible(False)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        self.log_output.setVisible(False)
        self.log_output.setMaximumHeight(220)

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.btn_action)
        main_layout.addLayout(actions_row)
        main_layout.addWidget(self.chk_show_logs)
        main_layout.addWidget(self.lbl_log)
        main_layout.addWidget(self.log_output)

        status = self.statusBar()
        status.setStyleSheet(
            "QStatusBar { background: #1e1e1e; color: #aaaaaa; padding: 4px; "
            "border-top: 1px solid #333333; }"
        )

    # ---------------- Tema ----------------
    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #121212; }
            QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }

            QTabWidget::pane { border: 1px solid #3e3e42; border-radius: 6px; top: -1px; }
            QTabBar::tab {
                background: #2d2d30; color: #cccccc; padding: 10px 20px;
                margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px;
            }
            QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; font-weight: bold; }

            QGroupBox {
                font-weight: bold; border: 1px solid #3e3e42; border-radius: 6px;
                margin-top: 12px; padding-top: 12px; background: #1a1a1a;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #dddddd; }

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

            QPushButton#primaryAction {
                background-color: #0078D7; color: white; font-size: 16px;
                min-height: 52px; padding: 12px 24px; border-radius: 8px;
            }
            QPushButton#primaryAction:hover { background-color: #1084ea; }
            QPushButton#primaryAction:disabled { background-color: #333333; color: #777777; }

            QPushButton#secondaryButton {
                background-color: transparent; color: #cccccc;
                font-size: 12px; font-weight: normal;
                border: 1px solid #555555; border-radius: 6px; padding: 6px 12px;
            }
            QPushButton#secondaryButton:hover {
                background-color: #2d2d30; border-color: #0078D7; color: #ffffff;
            }

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
            QTabWidget::pane { border: 1px solid #cccccc; border-radius: 6px; }
            QTabBar::tab {
                background: #e8e8e8; padding: 10px 20px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
            }
            QTabBar::tab:selected { background: #ffffff; font-weight: bold; }
            QGroupBox {
                border: 1px solid #cccccc; border-radius: 6px; margin-top: 12px;
                padding-top: 12px; background: #fafafa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
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
            QPushButton#primaryAction {
                font-size: 16px; min-height: 52px; padding: 12px 24px; border-radius: 8px;
            }
            QPushButton#secondaryButton {
                background-color: #f5f5f5; color: #333333; font-weight: normal;
                border: 1px solid #bbbbbb; font-size: 12px;
            }
            QPushButton#secondaryButton:hover {
                background-color: #e8e8e8; border-color: #0078D7;
            }
            QProgressBar {
                background-color: #cccccc; border-radius: 4px; border: none;
            }
            QProgressBar::chunk { background-color: #0078D7; border-radius: 4px; }
            QTextEdit {
                background-color: #ffffff; color: #000000;
                border: 1px solid #cccccc; border-radius: 6px; padding: 10px;
            }
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
        self.log_output.clear()
        self.update_log(f"[*] VÍDEO CARREGADO: {file_path}")
        self.drop_zone.lbl_text.setText(f"🎥 {Path(file_path).name}\n(Clique para trocar)")
        self._update_resolution_options_for_video(file_path)

        self.btn_action.setText("Iniciar corte viral")
        self.btn_action.setStyleSheet("")
        self.btn_action.setEnabled(True)

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
        if self.btn_action.text() == "Iniciar novo corte":
            self.reset_ui_for_new_video()
            return

        if not self.current_video_path:
            return

        model_selected = self.combo_model.currentText()
        provider_selected = self._current_llm_provider()
        if provider_selected == "ollama":
            available_models = self._get_available_models()
            if model_selected not in available_models:
                self.update_log(f"[!] ERRO: Modelo '{model_selected}' não encontrado no Ollama.")
                self.update_log(f"[!] Modelos disponíveis: {', '.join(available_models)}")
                self._unlock_ui_after_process()
                self.btn_action.setText("Selecionar Modelo Válido")
                return

        social_same = self.combo_social_model_source.currentIndex() == 0
        social_model_pick = self.combo_social_model.currentText().strip()
        enable_social = self.chk_enable_social_package.isChecked()
        if enable_social and provider_selected == "ollama" and not social_same:
            available_models = self._get_available_models()
            if social_model_pick not in available_models:
                self.update_log(
                    f"[!] ERRO: Modelo social '{social_model_pick}' não encontrado no Ollama."
                )
                self.update_log(f"[!] Modelos disponíveis: {', '.join(available_models)}")
                self._unlock_ui_after_process()
                self.btn_action.setText("Selecionar modelo social válido")
                return

        self.update_log(f"[*] Provedor selecionado: {provider_selected.upper()}")
        self.update_log(f"[*] Iniciando motor com IA: {model_selected.upper()}")
        if enable_social:
            if social_same:
                self.update_log("[*] Pacote social: mesmo modelo da análise")
            else:
                self.update_log(
                    f"[*] Pacote social: modelo alternativo — {social_model_pick or '(vazio)'}"
                )
            self.update_log(
                f"[*] Capa JPG (Pacote Social): {'ATIVADA' if self.chk_social_cover.isChecked() else 'DESATIVADA'}"
            )
        else:
            self.update_log(
                "[*] Pacote social: DESATIVADO — só análise, transcrição e render dos clipes."
            )
        if provider_selected in ("gemini", "groq") and not self.edit_api_key.text().strip():
            self.update_log(
                "[!] ERRO: Informe a chave API (Gemini ou Groq) ou escolha um perfil guardado."
            )
            self._unlock_ui_after_process()
            self.btn_action.setText("Informar API Key")
            return
        self.update_log(
            f"[*] Modelo de transcrição selecionado: {self.combo_whisper_model.currentText()}"
        )
        self.update_log(f"[*] Whisper device: {self.combo_whisper_device.currentText()}")
        self.update_log(f"[*] Tipo de prompt selecionado: {self.combo_prompt.currentText()}")
        self.update_log(f"[*] Qualidade (resolução): {self.combo_resolution.currentText()}")
        self.update_log(f"[*] Formato: {self.combo_aspect_ratio.currentText()}")
        self.update_log(f"[*] Enquadramento: {self.combo_framing_mode.currentText()}")
        self.update_log(
            f"[*] Legendas TikTok: {'ATIVADAS' if self.chk_tiktok_captions.isChecked() else 'DESATIVADAS'}"
        )
        self.update_log(
            f"[*] Pré-visualização: {'DESATIVADA (render direto)' if self.chk_skip_preview.isChecked() else 'ATIVADA'}"
        )

        self.btn_action.setEnabled(False)
        self.btn_action.setText("Processando…")
        self.tabs.setEnabled(False)
        self.combo_provider.setEnabled(False)
        self.combo_model.setEnabled(False)
        self.edit_api_key.setEnabled(False)
        self.combo_api_profile.setEnabled(False)
        self.btn_api_key_save.setEnabled(False)
        self.btn_api_key_remove.setEnabled(False)
        self.combo_whisper_model.setEnabled(False)
        self.combo_whisper_device.setEnabled(False)
        self.combo_prompt.setEnabled(False)
        self.drop_zone.setEnabled(False)
        self.chk_enable_social_package.setEnabled(False)
        self.combo_social_model_source.setEnabled(False)
        self.combo_social_model.setEnabled(False)
        self.chk_social_cover.setEnabled(False)

        self.progress_bar.setProperty("state", "normal")
        self.progress_bar.setVisible(True)

        self.worker = VideoProcessorThread(
            self.current_video_path,
            model_name=model_selected,
            llm_provider=provider_selected,
            llm_api_key=self.edit_api_key.text().strip() or None,
            output_dir=self.output_folder_path,
            prompt_type=self.combo_prompt.currentText(),
            whisper_model=self.combo_whisper_model.currentText(),
            whisper_device_mode=self.combo_whisper_device.currentText(),
            resolution=self.combo_resolution.currentText(),
            export_quality=self.combo_resolution.currentText(),
            aspect_ratio=self.combo_aspect_ratio.currentText(),
            framing_mode=self.combo_framing_mode.currentText(),
            enable_tiktok_captions=self.chk_tiktok_captions.isChecked(),
            bitrate=self.edit_bitrate.text().strip(),
            llm_max_new_tokens=(
                int(self.edit_max_new_tokens.text().strip())
                if self.edit_max_new_tokens.text().strip().isdigit()
                else None
            ),
            custom_prompt=(
                self.edit_custom_prompt.toPlainText().strip()
                if self.edit_custom_prompt.toPlainText().strip()
                else None
            ),
            social_use_same_model=social_same,
            social_model_name=social_model_pick if not social_same else None,
            generate_social_cover=self.chk_social_cover.isChecked(),
            enable_social_package=enable_social,
        )
        self.worker.progress_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.clips_ready_signal.connect(self.on_clips_ready)
        self.worker.start()

    def update_log(self, text: str) -> None:
        self.log_output.append(f"> {text}")

    def on_clips_ready(self, clips_dicts: list[dict]) -> None:
        if self.chk_skip_preview.isChecked():
            selected_clips = [Clip(**c) for c in clips_dicts]
            self.update_log(
                f"[→] Pré-visualização desativada. Renderização direta de {len(selected_clips)} clipe(s)."
            )
            if self.worker:
                self.worker.selected_clips = selected_clips
            return

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

        self.btn_action.setText("Iniciar novo corte")
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
        self._open_output_folder()

    def _open_output_folder(self) -> None:
        try:
            if self.output_folder_path:
                target = Path(self.output_folder_path)
            elif self.current_video_path:
                target = config.EXPORTS_ROOT / f"{Path(self.current_video_path).stem}_processed"
            else:
                return
            target.mkdir(parents=True, exist_ok=True)
            os.startfile(str(target))  # type: ignore[attr-defined]
            self.update_log(f"[*] Pasta de saída aberta: {target}")
        except Exception as e:  # noqa: BLE001
            logger.warning("Falha ao abrir pasta de saída: %s", e)

    def _unlock_ui_after_process(self) -> None:
        self.btn_action.setEnabled(True)
        self.tabs.setEnabled(True)
        self.combo_provider.setEnabled(True)
        self.combo_model.setEnabled(True)
        use_key = self._uses_cloud_api_key()
        self.edit_api_key.setEnabled(use_key)
        self.combo_api_profile.setEnabled(use_key)
        self.btn_api_key_save.setEnabled(use_key)
        self.btn_api_key_remove.setEnabled(use_key)
        self.lbl_api_profile.setEnabled(use_key)
        self.lbl_api_key_row.setEnabled(use_key)
        self.combo_whisper_model.setEnabled(True)
        self.combo_whisper_device.setEnabled(True)
        self.combo_prompt.setEnabled(True)
        self.drop_zone.setEnabled(True)
        self.chk_enable_social_package.setEnabled(True)
        self._apply_social_package_controls_state()

    def reset_ui_for_new_video(self) -> None:
        self.current_video_path = None
        self.output_folder_path = None
        self.log_output.clear()

        self.output_path_field.setText("")
        self.output_path_field.setPlaceholderText("Padrão: exports/{nome_do_video}_processed")

        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet("")

        self.drop_zone.lbl_text.setText("Arraste e solte o seu vídeo aqui\nou clique para procurar")

        self.btn_action.setText("Selecione um vídeo na área acima")
        self.btn_action.setStyleSheet("")
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
