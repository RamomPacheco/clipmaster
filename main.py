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

    def _enforce_minimum_duration(
        self,
        clips: List[Dict[str, Any]],
        max_duration: float,
        min_seconds: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Garante que nenhum clipe tenha menos que `min_seconds`.
        Se for menor, expande as bordas dinamicamente mantendo o assunto centralizado.
        """
        for i, clip in enumerate(clips):
            start = float(clip.get("start", 0.0))
            end = float(clip.get("end", start + min_seconds))

            duration = end - start

            if duration < min_seconds:
                logger.info(
                    f"Clipe {i+1} muito curto ({duration:.1f}s). Expandindo contexto para {min_seconds}s..."
                )

                # Calcula quanto tempo falta
                deficit = min_seconds - duration

                # Tenta expandir igualmente para trás e para frente (para manter o gancho no meio)
                new_start = start - (deficit / 2.0)
                new_end = end + (deficit / 2.0)

                # Valida as bordas (não pode ser menor que 0 nem maior que o vídeo original)
                if new_start < 0.0:
                    new_end += abs(new_start)  # Joga o tempo que faltou para o final
                    new_start = 0.0

                if new_end > max_duration:
                    new_start -= (
                        new_end - max_duration
                    )  # Puxa o tempo excedente para o início
                    new_end = max_duration

                # Trava de segurança final para vídeos que no total têm menos de 30s
                new_start = max(0.0, new_start)

                clip["start"] = round(new_start, 2)
                clip["end"] = round(new_end, 2)
                clip[
                    "reason"
                ] += " [Nota: O sistema expandiu este trecho para garantir o contexto completo.]"

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
        """Orquestrador principal do pipeline de processamento."""
        if not self.check_dependencies():
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # 1. Transcrição de Áudio (Whisper)
            model = self._init_whisper_model()
            self.progress_signal.emit("Lendo o vídeo e transcrevendo o áudio...")

            # Convertendo para lista para sabermos o tamanho e o final do vídeo
            segments_generator, _ = model.transcribe(self.video_path, beam_size=5)
            segments = list(segments_generator)

            if not segments:
                raise ValueError(
                    "A transcrição não gerou texto útil. O vídeo tem áudio claro?"
                )

            # Descobre a duração total baseada na última fala detectada
            max_video_duration = segments[-1].end

            full_text = "\n".join(
                [f"[{s.start:.2f}s - {s.end:.2f}s]: {s.text}" for s in segments]
            )

            # 2. Análise de Viralidade (Ollama)
            self.progress_signal.emit(
                f"Analisando métricas de viralidade com {self.model_name}..."
            )
            clips = self._analyze_viral_potential(full_text)

            # [NOVO] 2.5: Força a duração mínima garantindo o contexto
            clips = self._enforce_minimum_duration(
                clips, max_video_duration, min_seconds=30.0
            )

            # 3. Recorte e Renderização (FFmpeg -> ProRes)
            self._process_video_clips(clips)
            self.finished_signal.emit(
                f"Sucesso! Arquivos e insights salvos em:\n{self.output_dir.absolute()}"
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
        Comunica-se com o modelo local do Ollama para extrair TODOS os ganchos virais.
        A IA decide a quantidade baseada na qualidade do conteúdo.
        """
        import re

        prompt = f"""
        Você é um especialista em retenção de audiência e edição para TikTok/Reels/Shorts.
        Analise a transcrição deste vídeo e identifique TODOS os trechos com alto potencial viral.
        
        REGRA DE OURO (CONTEXTO E QUANTIDADE):
        1. NÃO HÁ LIMITE de clipes. Retorne quantos clipes forem realmente excelentes (podem ser 2, 5, 10 ou mais).
        2. PRIORIZE A QUALIDADE. Se o vídeo for monótono, retorne apenas os 1 ou 2 momentos que se salvam. Se for um conteúdo denso e genial, extraia todos os recortes possíveis.
        3. Um bom vídeo viral precisa de contexto completo (início, meio da explicação e fim impactante). Indique trechos de no mínimo 30 a 60 segundos.
        
        Retorne APENAS um array JSON puro. Não adicione NENHUM texto explicativo antes ou depois.
        Formato obrigatório rigoroso: 
        [
            {{"start": 0.0, "end": 45.0, "reason": "Apresenta o problema X e conclui com a solução Y", "headline": "A Verdade sobre X"}},
            {{"start": 120.0, "end": 165.0, "reason": "História completa que prende a atenção", "headline": "História Incrível"}}
        ]
        
        Transcrição do vídeo: 
        {text}
        """
        try:
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            raw_content = response["message"]["content"]

            # Regex poderoso: Busca TUDO que estiver entre os colchetes principais
            match = re.search(r"\[.*\]", raw_content, re.DOTALL)

            if not match:
                logger.error(
                    f"IA falhou ao retornar array JSON. Resposta bruta:\n{raw_content}"
                )
                raise ValueError("Nenhum array JSON encontrado na resposta da IA.")

            json_str = match.group(0).strip()

            # Força o carregamento do JSON sanitizado
            clips = json.loads(json_str)

            logger.info(f"[*] A IA identificou {len(clips)} clipes virais neste vídeo!")
            return clips

        except json.JSONDecodeError as je:
            logger.error(f"Erro de parsing JSON (A IA formatou mal os dados): {je}")
            return [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "reason": "Fallback: IA retornou formato inválido",
                    "headline": "Melhor Momento Automático",
                }
            ]

        except Exception as e:
            logger.warning(
                f"Falha na comunicação com a IA: {e}. Aplicando clips padrão."
            )
            return [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "reason": "Trecho de segurança",
                    "headline": "Primeiros 30s",
                }
            ]

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
