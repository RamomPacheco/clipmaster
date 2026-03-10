import sys
import os
import shutil
import json
import logging
import subprocess
import re
import time
from pathlib import Path
from typing import List, Dict, Any


# ==========================================
# BOOTSTRAP: Injeção de Ambiente CUDA
# ==========================================
def _inject_cuda_environment():
    """
    Força o Python a encontrar as DLLs do CUDA na raiz do projeto.
    """
    project_root = os.getcwd()

    # Adiciona ao PATH
    os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

    # Registra o diretório no Python
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(project_root)
            print(f"Diretório CUDA injetado: {project_root}")
        except Exception as e:
            print(f"Falha ao injetar DLL directory: {e}")


_inject_cuda_environment()

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
    QDialog,
    QCheckBox,
    QScrollArea,
    QLineEdit,
    QGroupBox,
    QMessageBox,
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
        Garante que os clipes fiquem entre 30s e 60s.
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

    def _remove_duplicate_clips(
        self, clips: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove clipes duplicados ou sobrepostos em mais de 50%.
        Mantém o clipe com a melhor razão (mais detalhada).
        """
        if not clips:
            return clips

        # Ordena por start time
        clips_sorted = sorted(clips, key=lambda x: x.get("start", 0))

        filtered_clips = []
        for clip in clips_sorted:
            is_duplicate = False
            clip_start = clip.get("start", 0)
            clip_end = clip.get("end", 0)
            clip_duration = clip_end - clip_start

            for existing in filtered_clips:
                existing_start = existing.get("start", 0)
                existing_end = existing.get("end", 0)
                existing_duration = existing_end - existing_start

                # Calcula sobreposição
                overlap_start = max(clip_start, existing_start)
                overlap_end = min(clip_end, existing_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Se sobreposição > 50% da duração do menor clipe
                min_duration = min(clip_duration, existing_duration)
                if min_duration > 0 and (overlap_duration / min_duration) > 0.5:
                    is_duplicate = True
                    # Mantém o clipe com razão mais detalhada
                    if len(clip.get("reason", "")) > len(existing.get("reason", "")):
                        # Substitui o existente pelo atual
                        filtered_clips.remove(existing)
                        filtered_clips.append(clip)
                    break

            if not is_duplicate:
                filtered_clips.append(clip)

        logger.info(
            f"Removidos {len(clips) - len(filtered_clips)} clipes duplicados/sobrepostos"
        )
        return filtered_clips

    progress_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)
    clips_ready_signal = Signal(list)  # Sinal para clips prontos de seleção

    def __init__(
        self,
        video_path: str,
        model_name: str,
        output_dir: str = None,
        prompt_type: str = "Padrão (Equilibrado)",
        resolution: str = "1080p",
        bitrate: str = "",
        custom_prompt: str = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name

        # Se o usuário definir um output_dir customizado, usa-o. Caso contrário, usa o padrão
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"exports/{Path(self.video_path).stem}_processed")

        self.prompt_type = prompt_type  # Tipo de prompt selecionado
        self.selected_clips = None  # Armazena os clips selecionados pelo usuário

        # Métricas de performance
        self.performance_metrics = {
            "start_time": None,
            "transcription_time": 0,
            "analysis_time": 0,
            "rendering_time": 0,
            "total_clips_found": 0,
            "clips_selected": 0,
            "video_duration": 0,
            "model_used": model_name,
            "prompt_type": prompt_type,
        }

        # Configurações avançadas
        self.resolution = resolution
        self.bitrate = bitrate
        self.custom_prompt = custom_prompt

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
        Inicializa o Whisper otimizado para CPU (int8) para evitar problemas com CUDA.
        """
        self.progress_signal.emit(
            "Inicializando IA de transcrição via CPU (mais lento, mas estável)..."
        )
        return WhisperModel("base", device="cpu", compute_type="int8")

    def _transcribe_safely(self, audio_path: str) -> tuple[List[Dict[str, Any]], float]:
        """
        Isola o Whisper numa bolha. Ao retornar, o Python limpa a VRAM
        naturalmente sem causar Segmentation Fault no motor C++.
        """
        model = self._init_whisper_model()
        self.progress_signal.emit("Iniciando transcrição profunda com Whisper...")

        # Podemos usar vad_filter=True com segurança aqui, pois o áudio já é um .wav puro
        segments_generator, info = model.transcribe(
            audio_path, beam_size=2, vad_filter=True
        )

        segments = []
        max_duration = float(info.duration)

        for s in segments_generator:
            segments.append(
                {"start": float(s.start), "end": float(s.end), "text": str(s.text)}
            )
            if len(segments) % 15 == 0:
                self.progress_signal.emit(
                    f"Transcrevendo... {s.end:.2f}s processados de {max_duration:.2f}s"
                )

        # Quando a função chega no 'return', o modelo e o gerador são destruídos suavemente pelo Python
        return segments, max_duration

    def run(self):
        """Orquestrador principal com Chunking para vídeos ilimitados e trava de 60s."""
        if not self.check_dependencies():
            return

        try:
            self.performance_metrics["start_time"] = time.time()
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # ==========================================
            # FASE 0: PREPARAÇÃO DO ÁUDIO (Blindagem)
            # ==========================================
            self.progress_signal.emit("Preparando arquivo de áudio leve (FFmpeg)...")
            temp_audio_path = self.output_dir / "temp_audio_safe.wav"

            cmd_audio = [
                "ffmpeg",
                "-y",
                "-i",
                self.video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(temp_audio_path),
            ]
            subprocess.run(
                cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )

            # ==========================================
            # FASE 1: TRANSCRIÇÃO (Escopo Isolado e Seguro)
            # ==========================================
            # Chama a função bolha. Quando ela retorna, o Whisper já foi destruído em segurança.
            segments, max_video_duration = self._transcribe_safely(str(temp_audio_path))

            self.performance_metrics["transcription_time"] = (
                time.time() - self.performance_metrics["start_time"]
            )

            self.progress_signal.emit(
                "Transcrição concluída. A preparar motor de Análise (Ollama)..."
            )

            # Limpa o arquivo de áudio temporário do HD
            if temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except Exception:
                    pass

            import gc

            gc.collect()  # Agora a coleta de lixo é inofensiva e garante os 12GB pro Ollama.

            analysis_start = time.time()

            # ==========================================
            # FASE 2: CHUNKING MICRO (Fatias de 10 Minutos)
            # ==========================================
            # Mantenha 600.0 (10 minutos). É o limite perfeito para o Ollama com 8192 tokens.
            CHUNK_SECONDS = 600.0
            chapters = []
            current_chunk = []
            chunk_start = 0.0

            for s in segments:
                if s["end"] - chunk_start > CHUNK_SECONDS and current_chunk:
                    chapters.append(current_chunk)
                    current_chunk = [s]
                    chunk_start = s["start"]
                else:
                    current_chunk.append(s)

            if current_chunk:
                chapters.append(current_chunk)

            # Daqui para baixo, o código da FASE 3 e FASE 4 continua INTACTO...

            # ==========================================
            # FASE 3: ANÁLISE EM LOTE E TRAVA MICRO (30s a 60s)
            # ==========================================
            all_clips = []
            total_chapters = len(chapters)

            for i, chunk in enumerate(chapters):
                self.progress_signal.emit(
                    f"IA analisando Parte {i+1} de {total_chapters} (Contexto de 10 min)..."
                )

                # Monta o texto apenas deste pedaço
                chunk_text = "\n".join(
                    [
                        f"[{s['start']:.2f}s - {s['end']:.2f}s]: {s['text']}"
                        for s in chunk
                    ]
                )

                # A IA escolhe os melhores momentos
                clips = self._analyze_viral_potential(chunk_text)
                all_clips.extend(clips)

            self.performance_metrics["analysis_time"] = time.time() - analysis_start

            # [A GUILHOTINA MATEMÁTICA]
            # Força o mínimo de 30s e o teto absoluto de 60s para o YouTube Shorts
            all_clips = self._enforce_duration_limits(
                all_clips, max_video_duration, min_seconds=30.0, max_seconds=60.0
            )

            # ==========================================
            # FASE 3.5: FILTRAGEM DE DUPLICATAS
            # ==========================================
            all_clips = self._remove_duplicate_clips(all_clips)
            self.performance_metrics["total_clips_found"] = len(all_clips)
            self.performance_metrics["video_duration"] = max_video_duration
            self.progress_signal.emit(
                f"Análise concluída. {len(all_clips)} clipes únicos identificados."
            )

            # ==========================================
            # FASE 4: SELEÇÃO DE CLIPES PELO USUÁRIO
            # ==========================================
            if not all_clips:
                self.progress_signal.emit(
                    "Aviso: A IA não encontrou nenhum clipe viral forte o suficiente."
                )
                self.finished_signal.emit(
                    "Processamento concluído sem clipes extraídos."
                )
                return

            # Emite o sinal com os clipes prontos para o usuário selecionar
            self.progress_signal.emit(
                f"✓ Análise completa! {len(all_clips)} clipes identificados. Aguardando sua seleção..."
            )
            self.clips_ready_signal.emit(all_clips)

            # Aguarda até que o usuário selecione os clips
            while self.selected_clips is None:
                time.sleep(0.5)

            # FASE 5: RENDERIZAÇÃO APENAS DOS CLIPES SELECIONADOS
            if not self.selected_clips:
                self.progress_signal.emit("Nenhum clipe foi selecionado. Cancelando...")
                self.finished_signal.emit("Processamento cancelado pelo usuário.")
                return

            self.progress_signal.emit(
                f"Iniciando renderização de {len(self.selected_clips)} clipes selecionados..."
            )
            rendering_start = time.time()
            self._process_video_clips(self.selected_clips)

            self.performance_metrics["rendering_time"] = time.time() - rendering_start
            self.performance_metrics["clips_selected"] = len(self.selected_clips)

            self._save_processing_history()

            self.finished_signal.emit(
                f"Sucesso! {len(self.selected_clips)} clipes gerados e salvos em:\n{self.output_dir.absolute()}"
            )

        except subprocess.CalledProcessError as sub_e:
            error_msg = f"Erro no FFmpeg: {sub_e.stderr}"
            logger.error(error_msg)
            self.error_signal.emit(error_msg)
        except Exception as e:
            logger.exception("Erro inesperado no pipeline.")
            self.error_signal.emit(f"Erro inesperado: {str(e)}")

    def _get_prompt_config(self, text: str) -> tuple[str, str]:
        """
        Retorna o system_prompt e user_prompt baseado no tipo selecionado.
        """
        base_system = "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e YouTube Shorts. Sua ÚNICA função é extrair blocos de tempo. Retorne APENAS um array JSON puro."

        if self.custom_prompt:
            return base_system, self.custom_prompt

        base_user = f"""
        Analise esta fatiada da transcrição e encontre os momentos mais magnéticos.

        REGRAS DE OURO (CRÍTICAS):
        1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
        2. COERÊNCIA: O clipe deve começar no início exato do raciocínio e terminar na conclusão.
        3. FOCO: Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
        4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.

        --- TRANSCRIÇÃO ---
        {text}
        --- FIM DA TRANSCRIÇÃO ---

        Retorne APENAS o JSON rigoroso: 
        [
            {{"start": 10.5, "end": 55.0, "reason": "Motivo", "headline": "Título"}}
        ]
        """

        if self.prompt_type == "Padrão (Equilibrado)":
            return base_system, base_user

        elif self.prompt_type == "Humor & Comédia":
            system = "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e YouTube Shorts, focado em conteúdo humorístico e engraçado. Sua ÚNICA função é extrair blocos de tempo. Retorne APENAS um array JSON puro."
            user = f"""
            Analise esta fatiada da transcrição e encontre os momentos mais engraçados e humorísticos.

            REGRAS DE OURO (CRÍTICAS):
            1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
            2. COERÊNCIA: O clipe deve começar no início exato da piada ou situação engraçada e terminar na conclusão.
            3. FOCO: Priorize momentos que gerem risadas, situações cômicas, ironia ou humor leve. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
            4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.

            --- TRANSCRIÇÃO ---
            {text}
            --- FIM DA TRANSCRIÇÃO ---

            Retorne APENAS o JSON rigoroso: 
            [
                {{"start": 10.5, "end": 55.0, "reason": "Motivo engraçado", "headline": "Título humorístico"}}
            ]
            """
            return system, user

        elif self.prompt_type == "Sério & Alto Valor":
            system = "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e YouTube Shorts, focado em conteúdo sério e de alto valor. Sua ÚNICA função é extrair blocos de tempo. Retorne APENAS um array JSON puro."
            user = f"""
            Analise esta fatiada da transcrição e encontre os momentos mais sérios e valiosos.

            REGRAS DE OURO (CRÍTICAS):
            1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
            2. COERÊNCIA: O clipe deve começar no início exato do raciocínio sério e terminar na conclusão valiosa.
            3. FOCO: Priorize momentos que transmitam conhecimento profundo, insights valiosos, conselhos sérios ou conteúdo impactante. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
            4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.

            --- TRANSCRIÇÃO ---
            {text}
            --- FIM DA TRANSCRIÇÃO ---

            Retorne APENAS o JSON rigoroso: 
            [
                {{"start": 10.5, "end": 55.0, "reason": "Motivo sério e valioso", "headline": "Título impactante"}}
            ]
            """
            return system, user

        elif self.prompt_type == "Storytelling & Emoção":
            system = "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e YouTube Shorts, focado em storytelling emocional. Sua ÚNICA função é extrair blocos de tempo. Retorne APENAS um array JSON puro."
            user = f"""
            Analise esta fatiada da transcrição e encontre os momentos mais emocionantes e narrativos.

            REGRAS DE OURO (CRÍTICAS):
            1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
            2. COERÊNCIA: O clipe deve começar no início exato da história ou emoção e terminar na conclusão emocional.
            3. FOCO: Priorize momentos que contem histórias, gerem emoção, inspiração ou conexão emocional. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
            4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.

            --- TRANSCRIÇÃO ---
            {text}
            --- FIM DA TRANSCRIÇÃO ---

            Retorne APENAS o JSON rigoroso: 
            [
                {{"start": 10.5, "end": 55.0, "reason": "Motivo emocional", "headline": "Título inspirador"}}
            ]
            """
            return system, user

        elif self.prompt_type == "Educacional & Dicas":
            system = "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e YouTube Shorts, focado em conteúdo educacional. Sua ÚNICA função é extrair blocos de tempo. Retorne APENAS um array JSON puro."
            user = f"""
            Analise esta fatiada da transcrição e encontre os momentos mais educacionais e com dicas práticas.

            REGRAS DE OURO (CRÍTICAS):
            1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
            2. COERÊNCIA: O clipe deve começar no início exato da explicação ou dica e terminar na conclusão prática.
            3. FOCO: Priorize momentos que ensinem algo novo, deem dicas práticas, expliquem conceitos ou forneçam conhecimento útil. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
            4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.

            --- TRANSCRIÇÃO ---
            {text}
            --- FIM DA TRANSCRIÇÃO ---

            Retorne APENAS o JSON rigoroso: 
            [
                {{"start": 10.5, "end": 55.0, "reason": "Motivo educacional", "headline": "Título instrutivo"}}
            ]
            """
            return system, user

        else:
            return base_system, base_user  # Fallback para padrão

    def _analyze_viral_potential(self, text: str) -> List[Dict[str, Any]]:
        """
        Prompt Nível Sênior: Força a IA a agir como um Diretor de Edição,
        priorizando a integridade da frase e a lógica do contexto.
        """
        import re

        system_prompt, user_prompt = self._get_prompt_config(text)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={
                    "num_ctx": 8192,  # Seguro para blocos de 15 minutos em placas de 12GB
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            )
            raw_content = response["message"]["content"]

            match = re.search(r"\[.*\]", raw_content, re.DOTALL)
            if not match:
                return []

            return json.loads(match.group(0).strip())

        except ollama.ResponseError as e:
            logger.error(f"Erro na resposta do Ollama: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON: {e}")
            return []
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
            ]

            # Aplicar resolução se não for 1080p
            if self.resolution != "1080p":
                scales = {"720p": "1280:720", "480p": "854:480"}
                if self.resolution in scales:
                    cmd.extend(["-vf", f"scale={scales[self.resolution]}"])

            # Aplicar bitrate customizado se fornecido
            if self.bitrate:
                if "-crf" in cmd:
                    idx = cmd.index("-crf")
                    cmd[idx] = "-b:v"
                    cmd[idx + 1] = f"{self.bitrate}k"

            cmd.append(str(output_file))

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

    def _save_processing_history(self):
        """Salva o histórico de processamento em um arquivo JSON."""
        history_file = Path("processing_history.json")
        try:
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []

            # Adiciona a entrada atual
            entry = self.performance_metrics.copy()
            entry["timestamp"] = time.time()
            entry["video_path"] = self.video_path
            history.append(entry)

            # Mantém apenas as últimas 50 entradas
            if len(history) > 50:
                history = history[-50:]

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")


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
# Diálogo de Seleção de Clipes
# ==========================================
class ClipSelectionDialog(QDialog):
    """Diálogo para o usuário selecionar quais clipes deseja salvar."""

    def __init__(self, clips: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.clips = clips
        self.selected_clips = []
        self.checkboxes = []
        self.edits = []

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

    def _setup_ui(self):
        """Configura a interface do diálogo."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Título
        lbl_title = QLabel(f"Escolha quais dos {len(self.clips)} clipes deseja salvar:")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff00;")
        layout.addWidget(lbl_title)

        # Área rolável com checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)

        for i, clip in enumerate(self.clips):
            # Cria um frame para cada clipe
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

            # Checkbox com informações básicas
            start = clip.get("start", 0)
            end = clip.get("end", 0)
            duration = end - start
            headline = clip.get("headline", "Sem título")

            checkbox_text = f"Clipe {i+1} • {headline} • [{start:.1f}s - {end:.1f}s] ({duration:.1f}s)"
            checkbox = QCheckBox(checkbox_text)
            checkbox.setStyleSheet(
                "background: transparent; border: none; padding: 0px; margin: 0px;"
            )
            checkbox.setChecked(True)  # Todos selecionados por padrão
            checkbox.setMinimumHeight(25)

            self.checkboxes.append((checkbox, clip))
            clip_layout.addWidget(checkbox)

            # Motivo (reason)
            reason = clip.get("reason", "N/A")
            lbl_reason = QLabel(f"Por quê: {reason}")
            lbl_reason.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            lbl_reason.setWordWrap(True)
            clip_layout.addWidget(lbl_reason)

            # Campos de edição
            lbl_edit_headline = QLabel("Editar Título:")
            edit_headline = QLineEdit(headline)
            clip_layout.addWidget(lbl_edit_headline)
            clip_layout.addWidget(edit_headline)

            lbl_edit_reason = QLabel("Editar Razão:")
            edit_reason = QLineEdit(reason)
            clip_layout.addWidget(lbl_edit_reason)
            clip_layout.addWidget(edit_reason)

            # Botão de pré-visualização
            btn_preview = QPushButton("Pré-visualizar")
            btn_preview.clicked.connect(lambda checked, c=clip: self.preview_clip(c))
            clip_layout.addWidget(btn_preview)

            scroll_layout.addWidget(clip_frame)

            self.edits.append((edit_headline, edit_reason))

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Botões de ação
        button_layout = QHBoxLayout()

        # Botão "Selecionar Todos"
        btn_all = QPushButton("✓ Selecionar Todos")
        btn_all.clicked.connect(self._select_all)
        button_layout.addWidget(btn_all)

        # Botão "Desselecionar Todos"
        btn_none = QPushButton("✗ Desselecionar Todos")
        btn_none.setObjectName("btn_cancel")
        btn_none.clicked.connect(self._select_none)
        button_layout.addWidget(btn_none)

        button_layout.addStretch()

        # Botão "Cancelar"
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setObjectName("btn_cancel")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)

        # Botão "Confirmar"
        btn_confirm = QPushButton("Salvar Seleção")
        btn_confirm.setMinimumWidth(150)
        btn_confirm.clicked.connect(self.accept)
        button_layout.addWidget(btn_confirm)

        layout.addLayout(button_layout)

    def _select_all(self):
        """Marca todos os checkboxes."""
        for checkbox, _ in self.checkboxes:
            checkbox.setChecked(True)

    def _select_none(self):
        """Desmarca todos os checkboxes."""
        for checkbox, _ in self.checkboxes:
            checkbox.setChecked(False)

    def get_selected_clips(self) -> List[Dict[str, Any]]:
        """Retorna os clipes selecionados com edições."""
        selected = []
        for i, (checkbox, clip) in enumerate(self.checkboxes):
            if checkbox.isChecked():
                edit_headline, edit_reason = self.edits[i]
                clip_copy = clip.copy()
                clip_copy["headline"] = edit_headline.text()
                clip_copy["reason"] = edit_reason.text()
                selected.append(clip_copy)
        return selected

    def preview_clip(self, clip):
        """Mostra uma pré-visualização do clipe tocando o trecho selecionado."""
        video_path = self.parent().current_video_path if self.parent() else None
        if not video_path:
            QMessageBox.warning(self, "Erro", "Vídeo não encontrado.")
            return

        start = clip.get("start", 0)
        end = clip.get("end", 0)
        duration = end - start

        # Usa ffplay para tocar apenas o trecho do clipe
        try:
            subprocess.Popen(
                [
                    "ffplay",
                    "-ss",
                    str(start),
                    "-t",
                    str(duration),  # -t para duração
                    "-i",
                    video_path,
                    "-window_title",
                    f"Pré-visualização: {clip.get('headline', 'Clipe')}",
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


# ==========================================
# Interface Gráfica (Frontend Desktop)
# ==========================================
class ViralApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Viral Clipper Pro")
        self.setMinimumSize(900, 650)
        self.current_video_path = None
        self.output_folder_path = None  # Pasta de saída customizada
        self.worker = None

        self._setup_ui()
        self._apply_dark_theme()

    def _get_available_models(self):
        """Busca os modelos disponíveis no Ollama."""
        try:
            import ollama

            models = ollama.list()
            model_list = (
                models.get("models", [])
                if isinstance(models, dict)
                else getattr(models, "models", [])
            )
            model_names = []
            for model in model_list:
                if isinstance(model, dict):
                    name = model.get("name") or model.get("model")
                else:
                    name = getattr(model, "name", None) or getattr(model, "model", None)
                if name:
                    model_names.append(name)
                else:
                    logger.warning(f"Modelo com estrutura inesperada: {model}")
            return model_names if model_names else ["llama3.2:3b"]
        except Exception as e:
            logger.warning(f"Erro ao buscar modelos Ollama: {e}")
            return ["llama3.2:3b"]  # Fallback

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

        # 2. Configurações (Seleção de Modelo e Prompt)
        config_layout = QHBoxLayout()
        lbl_model = QLabel("Modelo de IA (Ollama):")
        lbl_model.setStyleSheet("font-size: 14px; font-weight: bold; color: #dddddd;")

        self.combo_model = QComboBox()
        available_models = self._get_available_models()
        self.combo_model.addItems(available_models)
        self.combo_model.setToolTip("Selecione o modelo disponível no Ollama.")
        self.combo_model.setMinimumHeight(30)

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
        self.combo_prompt.setToolTip(
            "Selecione o tipo de foco para análise dos clipes."
        )
        self.combo_prompt.setMinimumHeight(30)
        self.combo_prompt.setCurrentText("Padrão (Equilibrado)")  # Padrão selecionado

        config_layout.addWidget(lbl_model)
        config_layout.addWidget(self.combo_model)
        config_layout.addWidget(lbl_prompt)
        config_layout.addWidget(self.combo_prompt)
        config_layout.addStretch()  # Empurra tudo para a esquerda
        main_layout.addLayout(config_layout)

        # 3. Configurações de Caminhos (Input/Output)
        paths_layout = QVBoxLayout()
        paths_layout.setSpacing(8)

        # Input Path
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

        # Output Path
        output_path_layout = QHBoxLayout()
        lbl_output = QLabel("Pasta de Saída:")
        lbl_output.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #dddddd; min-width: 100px;"
        )

        self.output_path_field = QLineEdit()
        self.output_path_field.setReadOnly(True)
        self.output_path_field.setPlaceholderText(
            "Padrão: exports/{nome_do_video}_processed"
        )
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

        # Configurações Avançadas
        advanced_group = QGroupBox("Configurações Avançadas")
        advanced_layout = QVBoxLayout(advanced_group)

        # Resolução
        lbl_resolution = QLabel("Resolução de Saída:")
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["1080p", "720p", "480p"])
        self.combo_resolution.setCurrentText("1080p")
        advanced_layout.addWidget(lbl_resolution)
        advanced_layout.addWidget(self.combo_resolution)

        # Bitrate
        lbl_bitrate = QLabel("Bitrate de Vídeo (kbps, opcional):")
        self.edit_bitrate = QLineEdit()
        self.edit_bitrate.setPlaceholderText("Deixe vazio para CRF 18")
        advanced_layout.addWidget(lbl_bitrate)
        advanced_layout.addWidget(self.edit_bitrate)

        # Prompt Customizado
        lbl_custom_prompt = QLabel("Prompt Customizado (opcional):")
        self.edit_custom_prompt = QTextEdit()
        self.edit_custom_prompt.setPlaceholderText(
            "Deixe vazio para usar prompts padrão..."
        )
        advanced_layout.addWidget(lbl_custom_prompt)
        advanced_layout.addWidget(self.edit_custom_prompt)

        # Tema
        self.chk_dark_theme = QCheckBox("Tema Escuro")
        self.chk_dark_theme.setChecked(True)
        self.chk_dark_theme.stateChanged.connect(self.toggle_theme)
        advanced_layout.addWidget(self.chk_dark_theme)

        main_layout.addWidget(advanced_group)

        # 4. Drop Zone (Arrastar e Soltar)
        self.drop_zone = DropZone()
        self.drop_zone.setMinimumHeight(80)
        self.drop_zone.file_dropped.connect(self.on_video_selected)
        main_layout.addWidget(self.drop_zone)

        # 5. Barra de Progresso e Ação
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

        # Botão de Relatório
        self.btn_report = QPushButton("Ver Histórico de Processamento")
        self.btn_report.clicked.connect(self.show_processing_history)
        main_layout.addWidget(self.btn_report)

        # 6. Terminal / Logs
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
        self.input_path_field.setText(file_path)
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

    def browse_input_video(self):
        """Abre o diálogo para selecionar um vídeo de entrada."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Vídeo de Entrada", "", "Vídeos (*.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self.on_video_selected(file_path)

    def browse_output_folder(self):
        """Abre o diálogo para selecionar a pasta de saída."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Selecionar Pasta de Saída", ""
        )
        if folder_path:
            self.output_folder_path = folder_path
            self.output_path_field.setText(folder_path)
            self.update_log(f"[*] Pasta de saída definida: {folder_path}")

    def reset_output_folder(self):
        """Reseta a pasta de saída para o padrão."""
        self.output_folder_path = None
        self.output_path_field.setText("")
        self.output_path_field.setPlaceholderText(
            "Padrão: exports/{nome_do_video}_processed"
        )
        self.update_log("[*] Pasta de saída resetada para o padrão.")

    def start_processing(self):
        """Inicia a Thread ou reseta a UI se já tiver terminado."""
        if self.btn_action.text() == "Processar Novo Vídeo":
            self.reset_ui_for_new_video()
            return

        if not self.current_video_path:
            return

        model_selected = self.combo_model.currentText()

        # Verifica se o modelo está disponível
        available_models = self._get_available_models()
        if model_selected not in available_models:
            self.update_log(
                f"[!] ERRO: Modelo '{model_selected}' não encontrado no Ollama."
            )
            self.update_log(f"[!] Modelos disponíveis: {', '.join(available_models)}")
            self._unlock_ui_after_process()
            self.btn_action.setText("Selecionar Modelo Válido")
            return

        self.update_log(f"[*] Iniciando motor com IA: {model_selected.upper()}")
        self.update_log(
            f"[*] Tipo de prompt selecionado: {self.combo_prompt.currentText()}"
        )

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
            self.current_video_path,
            model_name=model_selected,
            output_dir=self.output_folder_path,  # Passa a pasta de saída customizada
            prompt_type=self.combo_prompt.currentText(),  # Passa o tipo de prompt selecionado
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

    def update_log(self, text: str):
        self.log_output.append(f"> {text}")

    def on_clips_ready(self, clips: List[Dict[str, Any]]):
        """Mostra o diálogo de seleção de clipes quando estão prontos."""
        self.update_log(f"[→] Abrindo seletor de clipes com {len(clips)} opções...")

        # Cria e mostra o diálogo
        dialog = ClipSelectionDialog(clips, self)
        result = dialog.exec()

        if result == QDialog.Accepted:
            selected_clips = dialog.get_selected_clips()
            self.update_log(
                f"[✓] Usuário selecionou {len(selected_clips)} clipe(s) para salvar."
            )
            self.worker.selected_clips = selected_clips
        else:
            self.update_log("[✗] Seleção cancelada pelo usuário.")
            self.worker.selected_clips = []

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
        self.output_folder_path = None
        self.log_output.clear()

        # Reseta os campos de entrada e saída
        self.input_path_field.setText("")
        self.input_path_field.setPlaceholderText("Nenhum vídeo selecionado...")
        self.output_path_field.setText("")
        self.output_path_field.setPlaceholderText(
            "Padrão: exports/{nome_do_video}_processed"
        )

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

    def toggle_theme(self):
        if self.chk_dark_theme.isChecked():
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

    def _apply_light_theme(self):
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

    def show_processing_history(self):
        try:
            history_file = Path("processing_history.json")
            if not history_file.exists():
                QMessageBox.information(
                    self, "Histórico", "Nenhum histórico encontrado."
                )
                return
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not history:
                QMessageBox.information(self, "Histórico", "Histórico vazio.")
                return
            # Mostra as últimas 5 entradas
            text = "Últimos Processamentos:\n\n"
            for entry in history[-5:]:
                text += f"Vídeo: {Path(entry.get('video_path', '')).name}\n"
                text += f"Duração: {entry.get('video_duration', 0):.1f}s\n"
                text += f"Clipes Encontrados: {entry.get('total_clips_found', 0)}\n"
                text += f"Clipes Selecionados: {entry.get('clips_selected', 0)}\n"
                text += (
                    f"Tempo de Transcrição: {entry.get('transcription_time', 0):.1f}s\n"
                )
                text += f"Tempo de Análise: {entry.get('analysis_time', 0):.1f}s\n"
                text += (
                    f"Tempo de Renderização: {entry.get('rendering_time', 0):.1f}s\n"
                )
                text += f"Modelo: {entry.get('model_used', '')}\n"
                text += f"Prompt: {entry.get('prompt_type', '')}\n"
                text += f"Timestamp: {time.ctime(entry.get('timestamp', 0))}\n\n"
            QMessageBox.information(self, "Histórico de Processamento", text)
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Erro ao carregar histórico: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViralApp()
    window.show()
    sys.exit(app.exec())

