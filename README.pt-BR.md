[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![pt-br](https://img.shields.io/badge/lang-Português-green.svg)](README.pt-BR.md)

---
# ClipMaster - Guia Completo

## 📋 O que é?

**ClipMaster** é uma aplicação desktop Python que automatiza a geração de clipes virais a partir de vídeos longos. Funciona através de:

1. **Transcrição com Faster-Whisper** - Extrai o áudio e converte em texto
2. **Análise com Ollama (IA Local)** - Lê o texto e identifica os melhores momentos
3. **Renderização com FFmpeg** - Corta e exporta os clipes em MP4 H.264

---
### 💙 [Se gostou, faça uma doação ☕](https://livepix.gg/ramompacheco)
## 🔧 INSTALAÇÃO

### Pré-requisitos do Sistema

1. **Python 3.10+**
```powershell
python --version  # Verificar se é 3.10+
```

2. **FFmpeg** (obrigatório)
   - Download: https://ffmpeg.org/download.html
   - Ou por Chocolatey (Windows):
     ```powershell
     choco install ffmpeg
     ```
   - Verificar:
     ```powershell
     ffmpeg -version
     ```

3. **Ollama** (para a IA local)
   - Download: https://ollama.ai
   - Após instalar, execute no terminal:
     ```powershell
     ollama pull phi4       # Modelo com melhor resposta 
     ollama serve           # Rodar o servidor (em outro terminal)
     ```

### Instalação do Projeto

1. **Clonar/Entrar no projeto**
```powershell
git clone https://github.com/RamomPacheco/clipmaster.git
cd clipmaster
```

2. **Criar ambiente virtual (primeira vez)**
```powershell
python -m venv .venv
```

3. **Ativar ambiente virtual**
```# 3. Ative o ambiente virtual
Windows (PowerShell/CMD):
.venv\Scripts\activate
Linux/Mac:
source .venv/bin/activate
```

4. **Instalar dependências**
```powershell
pip install -r requirements.txt
```

---

## ▶️ COMO USAR

### Executar a Aplicação

```powershell
# A partir da raiz do projeto
python main.py
```

A interface gráfica abrirá. Siga os passos:

1. **Selecionar vídeo** - Clique em "Selecionar Vídeo de Entrada"
2. **Escolher modelo de IA** - Dropdown de modelos Ollama disponíveis
3. **Escolher modelo de transcrição (Whisper)** - Dropdown com `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`
4. **Selecionar tipo de análise**:
   - Padrão (Equilibrado) - Para conteúdo geral
   - Humor & Comédia - Prioriza momentos engraçados
   - Sério & Alto Valor - Foca em conteúdo sério/educativo
5. **Clique em "Iniciar Motor"** - Começa o processamento
6. **Revisar e selecionar clipes** - Uma janela mostra os clipes encontrados
7. **Salvar clipes** - Exporte para MP4s em alta qualidade

### Arquivos de Saída

Os clipes gerados vão para:
```
pasta de saída de sua escolha no app
```

---

## 📁 ESTRUTURA DO CÓDIGO

```
clipmaster/
├── pyproject.toml          # Configuração do projeto
├── app/
│   ├── main.py             # Ponto de entrada - cria QApplication
│   ├── core/
│   │   ├── config.py       # Constantes e caminhos
│   │   ├── logger.py       # Sistema de logs
│   │   └── cuda_setup.py   # Setup de CUDA/DLLs
│   ├── models/
│   │   └── schemas.py      # Tipos de dados (Clip, Metrics, etc)
│   ├── services/
│   │   ├── transcription.py    # Whisper - converte áudio em texto
│   │   ├── llm_analyzer.py     # Ollama - análise de conteúdo
│   │   ├── video_engine.py     # FFmpeg - corte de vídeos
│   │   └── clip_manager.py     # Gerenciamento de clipes
│   ├── ui/
│   │   ├── main_window.py      # Interface principal
│   │   ├── components/
│   │   │   └── drop_zone.py    # Zona de arraste para vídeos
│   │   └── dialogs/
│   │       └── clip_dialog.py  # Diálogo de seleção de clipes
│   └── workers/
│       └── processing_task.py  # Thread de processamento pesado
```

---

## 🔍 EXPLICAÇÃO DAS FUNÇÕES PRINCIPAIS

### 1. **core/config.py** - Configurações Globais
```python
PROJECT_ROOT          # Raiz do projeto
EXPORTS_ROOT          # Pasta para exportar clipes
MIN_CLIP_SECONDS      # 30s (duração mínima de um clipe)
MAX_CLIP_SECONDS      # 60s (máximo para TikTok/Shorts)
CHUNK_SECONDS         # 600s = 10 minutos (divide vídeo em pedaços)
DEFAULT_LLM_MODEL     # "llama3.2:3b" (IA padrão)
```

### 2. **core/logger.py** - Sistema de Logs
```python
configure_logging()   # Inicializa logger
logger.info()         # Mensagens informativas
logger.warning()      # Avisos
logger.error()        # Erros
```

### 3. **models/schemas.py** - Estruturas de Dados

#### `Clip`
Representa um clipe de vídeo:
```python
class Clip:
    start: float          # Segundo inicial (ex: 10.5)
    end: float            # Segundo final (ex: 55.0)
    reason: str           # Por que é viral? (ex: "Momento engraçado")
    headline: str         # Título curto do clipe
    
    @property
    def duration:         # Calcula end - start
```

#### `ProcessingMetrics`
Métricas do processamento:
```python
start_time               # Quando começou
transcription_time       # Quanto levou para transcrever
analysis_time            # Quanto levou a IA analisar
rendering_time           # Quanto levou para renderizar
total_clips_found        # Quantos clipes encontrados
clips_selected           # Quantos o usuário selecionou
video_duration           # Duração do vídeo original
model_used               # Ex: "llama3.2:3b"
prompt_type              # Ex: "Humor & Comédia"
```

### 4. **services/transcription.py** - Converter Áudio em Texto

#### `transcribe_audio(audio_path, model_name=None) → (segments, duration)`
Transcreve um arquivo WAV:
- **Entrada**: Caminho para arquivo WAV
- **Parâmetro opcional**: `model_name` (vem do seletor da interface)
- **Saída**: Lista de segmentos + duração total
```python
segments = [
    {"start": 0.5, "end": 2.3, "text": "Olá, como você está?"},
    {"start": 2.3, "end": 4.1, "text": "Estou bem, obrigado!"},
    ...
]
```

### 5. **services/llm_analyzer.py** - Análise com IA

#### `build_prompts(prompt_type, text, custom_prompt)`
Constrói os prompts para a IA:
- **Padrão**: Equilibrado, para qualquer conteúdo
- **Humor & Comédia**: Procura por momentos engraçados
- **Sério & Alto Valor**: Procura por conteúdo educativo/valioso

#### `analyze_viral_potential(text, model, prompt_type)`
Envia o texto para Ollama e recebe clipes:
```python
# Input: Transcrição de 10 minutos
# Output: Lista de clipes com tempos e razões
[
    {"start": 15.0, "end": 45.0, "reason": "Piada hilária", "headline": "O melhor momento"},
    {"start": 120.0, "end": 155.0, "reason": "Conselho valioso", "headline": "Dica importante"},
]
```

### 6. **services/video_engine.py** - Corte e Renderização

#### `extract_safe_audio(video_path, output_dir) → audio_path`
Extrai áudio do vídeo com FFmpeg:
```python
# Input: video.mp4
# Output: temp_audio_safe.wav (mono, 16kHz)
# Usado para Whisper transcrever
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

#### `render_clips(video_path, clips, output_dir, resolution, bitrate)`
Renderiza todos os clipes com H.264:
```python
# Para cada clipe:
# ffmpeg -i video.mp4 -ss 15.0 -to 45.0 -c:v libx264 -crf 18 clip_1.mp4
# 
# Parâmetros:
# -crf 18 → Qualidade (0=máximo, 51=mínimo, 18=ótimo)
# -preset slow → Mais tempo, melhor compressão
# -r 30 → 30 fps (compatível com TikTok/Shorts)
# -c:a aac → Áudio em AAC (padrão)
```

### 7. **services/clip_manager.py** - Gerenciamento de Clipes

#### `enforce_duration_limits(clips, max_video_duration, min, max)`
Garante que clipes respeitem limites (30-60s):
```python
# Se clipe < 30s → Expande (adiciona segundos antes/depois)
# Se clipe > 60s → Corta em 60s exatos
```

#### `remove_duplicate_clips(clips)`
Remove clipes que se sobrepõem > 50%:
```python
# Se dois clipes ocupam >50% do mesmo espaço → Remove o com razão menor
```

#### `append_history_entry(metrics, video_path, history_file)`
Salva histórico de processamentos em JSON:
```json
[
  {
    "timestamp": 1234567890.0,
    "video_path": "/path/to/video.mp4",
    "transcription_time": 45.2,
    "analysis_time": 120.5,
    "rendering_time": 200.0,
    "total_clips_found": 8,
    "clips_selected": 5
  }
]
```

### 8. **workers/processing_task.py** - Orquestração

#### `VideoProcessorThread` (herança de QThread)
Thread que executa o pipeline completo:

**Inicialização:**
```python
thread = VideoProcessorThread(
    video_path="/path/video.mp4",
    model_name="llama3.2:3b",
    whisper_model="small",
    output_dir="/exports/meu_video_processed",
    prompt_type="Humor & Comédia",
    resolution="1080p",
    bitrate="",           # Vazio = usar CRF 18
    custom_prompt=None    # None = usar prompt padrão
)
```

**Sinais emitidos:**
```python
progress_signal.emit("Mensagem de progresso")    # Atualiza UI
finished_signal.emit("Processamento concluído") # Fim
error_signal.emit("ERRO!")                      # Falha
clips_ready_signal.emit(lista_de_clips)         # Clipes prontos
```

**Pipeline (função `run`):**
1. ✅ Verifica se FFmpeg existe
2. 📦 Extrai áudio do vídeo (FFmpeg)
3. 📝 Transcreve com Whisper
4. 🔪 Divide em chunks de 10 minutos
5. 🤖 Analisa cada chunk com Ollama
6. 🧹 Remove duplicatas
7. 📐 Força limites de duração (30-60s)
8. 💾 Salva histórico
9. 🎬 Renderiza todos os clipes (FFmpeg)

### 9. **ui/main_window.py** - Interface Gráfica

#### `ViralApp` (herança de QMainWindow)

**Métodos principais:**

```python
_setup_ui()                    # Constrói a interface
_apply_dark_theme()           # Aplica tema escuro
_on_start_engine_clicked()    # Inicia processamento
_on_save_clips_clicked()      # Salva clipes selecionados
update_log(message)           # Atualiza a caixa de log
_get_available_models()       # Lista modelos Ollama instalados
_load_processing_history()    # Carrega histórico anterior
```

**Sinais conectados:**
```python
# Quando thread emite, UI se atualiza:
thread.progress_signal → update_log()
thread.clips_ready_signal → show ClipSelectionDialog
thread.error_signal → show error message
```

### 10. **ui/components/drop_zone.py** - Zona de Arraste

```python
class DropZone(QLabel):
    file_dropped = Signal(str)  # Emite quando vídeo é arrastado
    
    # Permite arrastar vídeo diretamente no widget
```

### 11. **ui/dialogs/clip_dialog.py** - Seleção de Clipes

```python
class ClipSelectionDialog(QDialog):
    # Mostra preview dos clipes encontrados
    # Usuário marca quais deseja salvar
    # Retorna lista de clipes selectados
```

---

## 🎯 FLUXO COMPLETO

```
1️⃣ Usuário abre main.py
        ↓
2️⃣ Interface aparece (ViralApp)
        ↓
3️⃣ Usuário seleciona vídeo + modelo + tipo
        ↓
4️⃣ Clica "Iniciar Motor"
        ↓
5️⃣ VideoProcessorThread começa:
        ├─ Extrai áudio (FFmpeg) → temp_audio.wav
        ├─ Transcreve (Whisper) → texto com timestamps
        ├─ Divide em 10 min chunks
        ├─ Envia para Ollama (IA)
        ├─ Recebe sugestões de clipes
        ├─ Remove duplicatas
        └─ Força limites de duração
        ↓
6️⃣ Interface mostra clipes (ClipSelectionDialog)
        ↓
7️⃣ Usuário marca quais quer salvar
        ↓
8️⃣ Renderiza todos marcados (FFmpeg)
        ↓
9️⃣ Salva em exports/VIDEO_processed/
        ↓
🔟 Sucesso! Clipes em MP4 prontos
```

---

## ⚙️ CONFIGURAÇÕES IMPORTANTES

### Limites de Duração (config.py)
```python
MIN_CLIP_SECONDS = 30.0  # TikTok/Shorts mínimo
MAX_CLIP_SECONDS = 60.0  # YouTube Shorts máximo
```

### Chunking (config.py)
```python
CHUNK_SECONDS = 600.0  # 10 minutos por chunk
# Vídeo de 1 hora = 6 requisições para Ollama
# Vídeo de 2 horas = 12 requisições
# (Mais chunks = mais análise, mas melhor qualidade)
```

### Modelo Default
```python
DEFAULT_LLM_MODEL = "llama3.2:3b"
# Pode alterar para outros modelos Ollama:
# - llama2
# - neural-chat
# - orca-mini
# etc
```

### Whisper no processador (CPU por padrão)
```python
WHISPER_DEVICE = "cpu"  # padrão atual para evitar disputa com Ollama na GPU
```
Para usar GPU manualmente:
```powershell
$env:WHISPER_DEVICE="cuda"
python main.py
```

---

## 🐛 SOLUÇÃO DE PROBLEMAS

### Erro: "ModuleNotFoundError: No module named 'app'"
**Solução**: Execute sempre da **raiz do projeto**:
```powershell
cd e:\projetos_python\tiktoksele
python main.py
```

### Erro: "ffmpeg not found"
**Solução**: Instale FFmpeg e adicione ao PATH:
```powershell
choco install ffmpeg
```

### Erro: Ollama não conecta
**Solução**: Certifique-se que Ollama servidor está rodando:
```powershell
# Em outro terminal:
ollama serve
```

### Transcrição trava ou falha no fim do vídeo (GPU / CUDA)
Com **Ollama** e **Faster-Whisper** na **mesma GPU**, a VRAM pode acabar no último trecho do áudio (OOM). O app tenta repetir em **CPU** automaticamente; para evitar a falha desde o início:

```powershell
# Antes de abrir o app — só Whisper em CPU (Ollama continua na GPU):
$env:WHISPER_SHARED_GPU_SAFE="1"
python main.py
```

Outras opções: `WHISPER_DEVICE=cpu`, ou `WHISPER_CUDA_LOW_VRAM=1` (mantém CUDA com menos VRAM).  
`WHISPER_COMPUTE_TYPE` continua podendo ser definido manualmente (ex.: `int8_float16`).

### Whisper muito lento
**Motivo**: Usando CPU
**Solução**: Instale CUDA se tiver GPU NVIDIA

### Clipes muito ruins
**Solução**: Teste outros tipos de prompt:
- Mude em `prompt_type` na UI
- Ou crie prompt customizado

---

## 📊 EXEMPLO DE SAÍDA

```
exports/
└── meu_video_processed/
    ├── clip_1_viral.mp4         # 45 segundos
    ├── clip_2_viral.mp4         # 38 segundos
    ├── clip_3_viral.mp4         # 52 segundos
    └── descricao_e_insights.txt # Resumo gerado
```

---

## 🚀 DICAS PRÁTICAS

1. **Para melhor qualidade**: Use `-crf 15-18` (mais tempo, melhor)
2. **Para mais clipes**: Ative "Sério & Alto Valor" (busca mais conteúdo)
3. **Para vídeos longos**: Use chunks de 10 min para melhor análise
4. **Histório**: Verifica `processing_history.json` para estatísticas
5. **Customizar**: Edite os prompts em `llm_analyzer.py` para seu estilo

---
### 💙 [Se gostou, faça uma doação ☕](https://livepix.gg/ramompacheco)
