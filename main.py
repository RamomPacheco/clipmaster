import sys
import os
import shutil
import json
import logging
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QProgressBar,
    QTextEdit,
    QFrame,
    QComboBox,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, QThread, Signal
import ollama
from faster_whisper import WhisperModel

# ==========================================
# Configuração de Logging Estruturado
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================
# Núcleo de Processamento (Backend / IA)
# ==========================================
class VideoProcessorThread(QThread):
    """
    Thread dedicada ao processamento intensivo de vídeo e IA.
    Garante que a interface gráfica (PySide6) não congele.
    """

    def _enforce_duration_limits(
        self,
        clips: List[Dict[str, Any]],
        max_video_duration: float,
        min_seconds: float = 30.0,
        max_seconds: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """
        Trava de Backend rigorosa. Garante que os clipes fiquem entre 30s e 60s.
        A IA é péssima em matemática, então o Python corrige na força bruta se necessário.
        """
        for i, clip in enumerate(clips):
            start = float(clip.get("start", 0.0))
            end = float(clip.get("end", start + min_seconds))

            duration = end - start

            if duration < min_seconds:
                logger.info(
                    f"Clipe {i+1} curto ({duration:.1f}s). Expandindo para {min_seconds}s..."
                )
                deficit = min_seconds - duration
                new_start = max(0.0, start - (deficit / 2.0))
                new_end = min(max_video_duration, end + (deficit / 2.0))

                # Se bateu no zero e ainda não tem 30s, joga o tempo pro final
                if (new_end - new_start) < min_seconds:
                    new_end = min(max_video_duration, new_start + min_seconds)

                clip["start"] = round(new_start, 2)
                clip["end"] = round(new_end, 2)
                clip["reason"] += " [Nota de Backend: Expandido para 30s]"

            elif duration > max_seconds:
                logger.warning(
                    f"Clipe {i+1} excedeu o limite do YouTube Shorts ({duration:.1f}s). Guilhotina aplicada em 60s."
                )
                # Mantemos o "start" (o gancho inicial é sagrado) e forçamos o "end"
                clip["end"] = round(start + max_seconds, 2)
                clip[
                    "reason"
                ] += " [Nota de Backend: Final cortado para respeitar teto de 60s]"

        return clips

    progress_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, video_path: str, model_name: str):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name
        self.output_dir = Path(f"exports/{Path(self.video_path).stem}_processed")

    def check_dependencies(self) -> bool:
        """Verifica se o FFmpeg está acessível no sistema."""
        if not shutil.which("ffmpeg"):
            self.error_signal.emit(
                "ERRO CRÍTICO: FFmpeg não detectado nas Variáveis de Ambiente (PATH)!"
            )
            return False
        return True

    def _init_whisper_model(self) -> WhisperModel:
        """
        Inicializa o Whisper otimizado para GPU (CUDA) com float16.
        Possui fallback automático para CPU (int8) caso a GPU falhe.
        """
        try:
            self.progress_signal.emit(
                "Inicializando IA de transcrição via GPU (CUDA)..."
            )
            model = WhisperModel("base", device="cuda", compute_type="float16")
            logger.info("Whisper carregado com sucesso na GPU.")
            return model
        except Exception as cuda_error:
            logger.warning(
                f"Erro ao inicializar CUDA: {cuda_error}. Aplicando Fallback para CPU."
            )
            self.progress_signal.emit(
                "Aviso: CUDA indisponível ou incompleto. Usando processador (CPU)..."
            )
            return WhisperModel("base", device="cpu", compute_type="int8")

    def run(self):
        """Orquestrador principal com Chunking para vídeos ilimitados."""
        if not self.check_dependencies():
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # ==========================================
            # FASE 1: TRANSCRIÇÃO E LIMPEZA
            # ==========================================
            model = self._init_whisper_model()
            self.progress_signal.emit(
                "Lendo o vídeo e transcrevendo o áudio completo..."
            )

            segments_generator, _ = model.transcribe(self.video_path, beam_size=2)
            segments = list(segments_generator)

            if not segments:
                raise ValueError("A transcrição não gerou texto útil.")

            max_video_duration = segments[-1].end

            # Limpeza cirúrgica de VRAM
            self.progress_signal.emit(
                "Limpando VRAM para dar potência total à IA de Análise..."
            )
            import gc

            del model  # Destrói o Whisper
            gc.collect()  # Libera a memória da placa de vídeo

            # ==========================================
            # FASE 2: CHUNKING (DIVIDIR PARA CONQUISTAR)
            # ==========================================
            CHUNK_SECONDS = 1800.0  # 30 minutos em segundos
            chapters = []
            current_chunk = []
            chunk_start = 0.0

            for s in segments:
                # Se o segmento atual passar do limite de 30 minutos, fechamos o capítulo
                if s.end - chunk_start > CHUNK_SECONDS and current_chunk:
                    chapters.append(current_chunk)
                    current_chunk = [s]
                    chunk_start = s.start
                else:
                    current_chunk.append(s)

            if current_chunk:
                chapters.append(current_chunk)  # Adiciona o trecho final

            # ==========================================
            # FASE 3: ANÁLISE EM LOTE (BATCH INFERENCE)
            # ==========================================
            all_clips = []
            total_chapters = len(chapters)

            for i, chunk in enumerate(chapters):
                self.progress_signal.emit(
                    f"IA analisando Parte {i+1} de {total_chapters} (Máximo Contexto)..."
                )

                # Monta a transcrição apenas deste capítulo de 30min
                chunk_text = "\n".join(
                    [f"[{s.start:.2f}s - {s.end:.2f}s]: {s.text}" for s in chunk]
                )

                # Extrai os clipes desta parte
                clips = self._analyze_viral_potential(chunk_text)
                all_clips.extend(clips)

            # [NOVO] Proteção matemática para não sobrepor tempos no FFmpeg
            all_clips = self._enforce_duration_limits(
                all_clips, max_video_duration, 30.0, 60.0
            )

            # ==========================================
            # FASE 4: RENDERIZAÇÃO
            # ==========================================
            if not all_clips:
                self.progress_signal.emit(
                    "Aviso: A IA não encontrou nenhum clipe viral forte o suficiente."
                )
                return

            self._process_video_clips(all_clips)

            self.finished_signal.emit(
                f"Sucesso! {len(all_clips)} clipes gerados e salvos em:\n{self.output_dir.absolute()}"
            )

        except subprocess.CalledProcessError as sub_e:
            error_msg = f"Erro no FFmpeg: {sub_e.stderr}"
            logger.error(error_msg)
            self.error_signal.emit(error_msg)
        except Exception as e:
            logger.exception("Erro inesperado no pipeline.")
            self.error_signal.emit(f"Erro inesperado: {str(e)}")

    def _analyze_viral_potential(self, text: str) -> List[Dict[str, Any]]:
        """
        Prompt Nível Sênior: Força a IA a agir como um Diretor de Edição,
        priorizando a integridade da frase e a lógica do contexto.
        """
        import re

        system_prompt = """Você é um Diretor de Edição Sênior especialista em retenção para TikTok e YouTube Shorts.
        Sua ÚNICA função é extrair blocos de tempo. Retorne APENAS um array JSON puro."""

        user_prompt = f"""
        Analise esta fatiada da transcrição e encontre os momentos mais magnéticos.

        REGRAS DE OURO (CRÍTICAS):
        1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
        2. COERÊNCIA: O clipe deve começar no início exato do raciocínio e terminar na conclusão.
        3. FOCO: Retorne apenas clipes geniais. Se não houver nenhum, retorne [].

        --- TRANSCRIÇÃO ---
        {text}
        --- FIM DA TRANSCRIÇÃO ---

        Retorne APENAS o JSON rigoroso: 
        [
            {{"start": 10.5, "end": 55.0, "reason": "Motivo", "headline": "Título"}}
        ]
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={
                    "num_ctx": 16384,  # Seguro para blocos de 30 minutos em placas de 12GB
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            )
            raw_content = response["message"]["content"]

            match = re.search(r"\[.*\]", raw_content, re.DOTALL)
            if not match:
                return []

            return json.loads(match.group(0).strip())

        except Exception as e:
            logger.error(f"Falha ao extrair clipes deste capítulo: {e}")
            return (
                []
            )  # Se falhar num capítulo, retorna vazio para não quebrar o programa inteiro

    def _process_video_clips(self, clips: List[Dict[str, Any]]):
        """Converte e corta os trechos selecionados para H.264 (MP4) com qualidade máxima de frame."""
        video_name = Path(self.video_path).stem
        desc_content = f"Análise de Viralidade para: {video_name}\n\n"

        for i, clip in enumerate(clips):
            # Alterado de .mov para .mp4
            output_file = self.output_dir / f"clip_{i+1}_viral.mp4"
            self.progress_signal.emit(
                f"Renderizando Clipe {i+1} (H.264 Alta Qualidade)..."
            )

            # Comando FFmpeg otimizado para retenção máxima de qualidade de frame
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                self.video_path,
                "-ss",
                str(clip["start"]),
                "-to",
                str(clip["end"]),
                # --- OTIMIZAÇÕES DE VÍDEO (H.264) ---
                "-c:v",
                "libx264",  # Codec de vídeo universal e super estável
                "-preset",
                "slow",  # Gasta um pouco mais de CPU para garantir a melhor qualidade de imagem possível
                "-crf",
                "18",  # Constant Rate Factor: 18 é qualidade "Visually Lossless" (sem perda visível)
                "-pix_fmt",
                "yuv420p",  # Formato de cor universal (evita incompatibilidade no iPhone/Android)
                # --- CORREÇÃO DE TRAVAMENTOS (STUTTERING) ---
                "-r",
                "30",  # Força 30 FPS cravados
                "-fps_mode",
                "cfr",  # Constant Frame Rate (Mata engasgos de vídeos gravados em celular)
                # --- OTIMIZAÇÕES DE ÁUDIO ---
                "-c:a",
                "aac",  # Codec de áudio padrão do MP4
                "-b:a",
                "192k",  # Bitrate de áudio de alta fidelidade
                "-af",
                "aresample=async=1",  # Mantém a sincronia labial perfeita após o corte
                str(output_file),
            ]

            # Executa com check=True e captura saída em caso de falha
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            desc_content += f"--- Clipe {i+1} ---\n"
            desc_content += f"Título: {clip.get('headline', 'Sem título')}\n"
            desc_content += f"Por que é viral: {clip.get('reason', 'N/A')}\n\n"

        # Salva o log analítico na pasta de destino
        with open(
            self.output_dir / "descricao_e_insights.txt", "w", encoding="utf-8"
        ) as f:
            f.write(desc_content)


class DropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
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

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "border: 2px dashed #00FF00; background-color: #1e2a1e; border-radius: 10px;"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            QFrame { border: 2px dashed #555555; border-radius: 10px; background-color: #1e1e1e; }
        """
        )

    def dropEvent(self, event: QDropEvent):
        self.dragLeaveEvent(event)  # Reseta o estilo
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                self.file_dropped.emit(file_path)
                break  # Pega apenas o primeiro vídeo

    def mousePressEvent(self, event):
        # Permite clicar na DropZone para abrir o File Explorer
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo", "", "Videos (*.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self.file_dropped.emit(file_path)


# ==========================================
# Interface Gráfica (Frontend Desktop)
# ==========================================
class ViralApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Viral Clipper Pro")
        self.setMinimumSize(700, 550)
        self.current_video_path = None
        self.worker = None

        self._setup_ui()
        self._apply_dark_theme()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. Header (Título)
        lbl_title = QLabel("AI Viral Clipper Pro")
        lbl_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        lbl_subtitle = QLabel("Motor H.264 CFR • Aceleração CUDA • IA Ollama")
        lbl_subtitle.setStyleSheet(
            "font-size: 12px; color: #0078D7; margin-bottom: 10px;"
        )

        main_layout.addWidget(lbl_title)
        main_layout.addWidget(lbl_subtitle)

        # 2. Configurações (Seleção de Modelo)
        config_layout = QHBoxLayout()
        lbl_model = QLabel("Modelo de IA (Ollama):")
        lbl_model.setStyleSheet("font-size: 14px; font-weight: bold; color: #dddddd;")

        self.combo_model = QComboBox()
        self.combo_model.addItems(
            ["phi4", "llama3.1", "llama3", "qwen2.5:32b", "mistral"]
        )
        self.combo_model.setToolTip("Selecione o modelo que você já baixou no Ollama.")
        self.combo_model.setMinimumHeight(30)

        config_layout.addWidget(lbl_model)
        config_layout.addWidget(self.combo_model)
        config_layout.addStretch()  # Empurra tudo para a esquerda
        main_layout.addLayout(config_layout)

        # 3. Drop Zone (Arrastar e Soltar)
        self.drop_zone = DropZone()
        self.drop_zone.setMinimumHeight(120)
        self.drop_zone.file_dropped.connect(self.on_video_selected)
        main_layout.addWidget(self.drop_zone)

        # 4. Barra de Progresso e Ação
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)

        self.btn_action = QPushButton("Iniciar Processamento Viral")
        self.btn_action.setFixedHeight(45)
        self.btn_action.setEnabled(False)  # Desativado até selecionar vídeo
        self.btn_action.clicked.connect(self.start_processing)

        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.btn_action)

        # 5. Terminal / Logs
        lbl_log = QLabel("Terminal de Processamento:")
        lbl_log.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))

        main_layout.addWidget(lbl_log)
        main_layout.addWidget(self.log_output)

    def _apply_dark_theme(self):
        """Aplica um CSS pro no estilo Adobe/DaVinci."""
        self.setStyleSheet(
            """
            QMainWindow { background-color: #121212; }
            QWidget { color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
            
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
            
            QPushButton#btn_success { background-color: #28a745; }
            QPushButton#btn_success:hover { background-color: #218838; }
            
            QProgressBar {
                background-color: #2d2d30; border-radius: 4px; border: none;
            }
            QProgressBar::chunk { background-color: #0078D7; border-radius: 4px; }
            QProgressBar[state="success"]::chunk { background-color: #28a745; }
            
            QTextEdit {
                background-color: #0c0c0c; color: #00ff00;
                border: 1px solid #333333; border-radius: 6px; padding: 10px;
            }
        """
        )

    def on_video_selected(self, file_path: str):
        """Callback quando o vídeo é dropado ou selecionado."""
        self.current_video_path = file_path
        self.log_output.clear()
        self.update_log(f"[*] VÍDEO CARREGADO: {file_path}")
        self.drop_zone.lbl_text.setText(
            f"🎥 {Path(file_path).name}\n(Clique para trocar)"
        )

        # Habilita o botão Iniciar e garante que ele tem a cor azul padrão
        self.btn_action.setText("Iniciar Processamento Viral")
        self.btn_action.setObjectName("")
        self.btn_action.setStyleSheet("")  # Força atualização do CSS
        self.btn_action.setEnabled(True)

    def start_processing(self):
        """Inicia a Thread ou reseta a UI se já tiver terminado."""
        if self.btn_action.text() == "Processar Novo Vídeo":
            self.reset_ui_for_new_video()
            return

        if not self.current_video_path:
            return

        model_selected = self.combo_model.currentText()
        self.update_log(f"[*] Iniciando motor com IA: {model_selected.upper()}")

        # Trava a interface
        self.btn_action.setEnabled(False)
        self.btn_action.setText("Processando... Aguarde.")
        self.combo_model.setEnabled(False)
        self.drop_zone.setEnabled(False)

        self.progress_bar.setProperty("state", "normal")
        self.progress_bar.style().unpolish(self.progress_bar)
        self.progress_bar.style().polish(self.progress_bar)
        self.progress_bar.setVisible(True)

        # Inicia a Thread
        self.worker = VideoProcessorThread(
            self.current_video_path, model_name=model_selected
        )
        self.worker.progress_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def update_log(self, text: str):
        self.log_output.append(f"> {text}")

    def on_error(self, error_msg: str):
        self.log_output.append(f"\n[!] ERRO CRÍTICO:\n{error_msg}")
        self._unlock_ui_after_process()
        self.btn_action.setText("Tentar Novamente")

    def on_finished(self, msg: str):
        """Finaliza o fluxo visual com sucesso, travando a barra em 100%."""
        self.log_output.append(f"\n[+] {msg}")
        self._unlock_ui_after_process()

        # 1. Muda o botão para verde (Sucesso) indicando que acabou
        self.btn_action.setText("Processar Novo Vídeo")
        self.btn_action.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; border-radius: 6px;"
        )

        # 2. PARTE CRÍTICA: Para a animação "infinita" e enche a barra até 100%
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        # 3. Pinta a barra de progresso de verde para dar o feedback visual correto
        self.progress_bar.setStyleSheet(
            """
            QProgressBar { background-color: #2d2d30; border-radius: 4px; border: none; }
            QProgressBar::chunk { background-color: #28a745; border-radius: 4px; }
        """
        )

    def _unlock_ui_after_process(self):
        """Libera os botões, mas mantém a barra de progresso visível."""
        self.btn_action.setEnabled(True)
        self.combo_model.setEnabled(True)
        self.drop_zone.setEnabled(True)

    def reset_ui_for_new_video(self):
        """Ciclo de Vida: Limpa completamente a tela para um novo arquivo."""
        self.current_video_path = None
        self.log_output.clear()

        # Reseta a barra de progresso para o estado inicial invisível
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(
            0, 0
        )  # Volta ao modo indeterminado para a próxima vez
        self.progress_bar.setStyleSheet("")  # Limpa a cor verde

        self.drop_zone.lbl_text.setText(
            "Arraste e solte o seu vídeo aqui\nou clique para procurar"
        )

        # Devolve a cor azul padrão do botão e o desativa até um vídeo ser inserido
        self.btn_action.setText("Selecione um Vídeo Acima")
        self.btn_action.setStyleSheet(
            "background-color: #0078D7; color: white; font-weight: bold; border-radius: 6px;"
        )
        self.btn_action.setEnabled(False)

        self.update_log("[*] Sistema limpo e pronto para um novo vídeo.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViralApp()
    window.show()
    sys.exit(app.exec())
