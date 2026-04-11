[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![pt-br](https://img.shields.io/badge/lang-Português-green.svg)](README.pt-BR.md)

---
# ClipMaster - Guia Completo

## 📋 O que é?

**ClipMaster** (título da janela: *AI Viral Clipper Pro*) é uma aplicação desktop em Python que gera clipes curtos estilo “viral” a partir de vídeos longos. Funciona assim:

1. **Transcrição (Faster-Whisper)** — extrai o áudio e produz texto com marcas de tempo  
2. **Análise por IA** — sugere os melhores trechos; suporta **Ollama** (local), **Gemini** e **Groq** (chave API na app)  
3. **Renderização (FFmpeg)** — corta e exporta MP4 em **H.264** (opcional **NVENC** se disponível), com legendas estilo TikTok opcionais  
4. **Pacote social opcional** — por clipe: texto para publicar e imagem de **capa JPG**  

Durante o processamento, a interface mostra uma **barra de progresso** (por fase: análise em blocos, renderização, pacote social) e um **painel de logs** opcional (*Mostrar terminal de logs*) que espelha o output de **logging** como no terminal (transcrição, FFmpeg, cliente HTTP, etc.).

---
### 💙 [Se gostou, faça uma doação ☕](https://livepix.gg/ramompacheco)
## 🔧 INSTALAÇÃO

### Pré-requisitos do Sistema

1. **Python 3.10+**
```powershell
python --version  # Verificar se é 3.10+
```

2. **FFmpeg** (obrigatório para desenvolvimento, a menos que use binários empacotados)
   - **Opção A — empacotado com o projeto:** copie `ffmpeg.exe` e `ffprobe.exe` para `bundled/ffmpeg/windows/` (ou execute `scripts\fetch_ffmpeg_windows.ps1`).
   - **Opção B — sistema:** https://ffmpeg.org/download.html ou `choco install ffmpeg`; verificar com `ffmpeg -version`.

3. **Ollama** (opcional — para LLM local)
   - Download: https://ollama.ai
   - Após instalar:
     ```powershell
     ollama pull phi4       # ou outro modelo que escolher na app
     ollama serve           # Servidor noutro terminal
     ```  
   Para **Gemini** ou **Groq**, use a chave API na aplicação.

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

### Empacotamento Windows (executável + instalador)

1. Ambiente virtual ativo e `pip install -r requirements.txt`.
2. (Recomendado) Incluir FFmpeg no build:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\fetch_ffmpeg_windows.ps1
   ```
3. Gerar `dist\ClipMaster\` e, se tiver [Inno Setup 6](https://jrsoftware.org/isdl.php), o ficheiro `dist_installer\ClipMaster_Setup_*.exe`:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\build_windows_release.ps1
   ```
   Modo não interativo / CI: acrescente `-SkipFFmpegCheck -Force`. Só PyInstaller: `-SkipInno`.

**Utilizador final:** com o instalador, a app vai para `%LocalAppData%\Programs\ClipMaster\`; exports e histórico ficam em `%LocalAppData%\AI_Viral_Clipper\`.

A interface gráfica abrirá. Siga os passos:

1. **Selecionar vídeo** — arrastar e soltar ou procurar ficheiro  
2. **Pasta de saída** (opcional) — por defeito: `exports/{nome_do_video}_processed` na raiz do projeto  
3. **Provedor de LLM** — Ollama, Gemini ou Groq (+ modelos conforme a UI)  
4. **Modelo Whisper** — ex.: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`  
5. **Tipo de análise / prompt** — equilibrado, humor, educativo, etc.  
6. **Iniciar processamento** — botão principal de ação  
7. **Revisar e selecionar clipes** (salvo se “pular pré-visualização” estiver ativo) — diálogo com sugestões da IA  
8. **Exportação** — MP4 em alta qualidade + ficheiros sociais opcionais (ver abaixo)  
9. **Logs** — ative *Mostrar terminal de logs* para ver o mesmo detalhe que no consola; acompanhe a **barra de progresso**  

### Estrutura das pastas de saída

Tudo é gravado **na pasta base que escolheu** (ou na pasta de export por defeito). A app cria **uma subpasta de projeto** com nome derivado dos **metadados do vídeo** (título, descrição ou comentário nas tags do ficheiro; se não existir, usa o **nome do ficheiro sem extensão**). Se já existir pasta com esse nome, acrescenta sufixo numérico (`_1`, `_2`, …).

Dentro dessa pasta, **cada clipe exportado** tem a sua subpasta `clip_01`, `clip_02`, …:

```
{sua_pasta_ou_exports/...}/{Nome Do Projeto Pelos Metadados}/
├── temp_audio_safe.wav          # só durante a execução; normalmente apagado após transcrição
├── clip_01/
│   ├── clipe.mp4                # vídeo final exportado
│   ├── capa.jpg                 # opcional — capa para redes
│   └── descricao_redes.txt      # opcional — frase de impacto + descrição
├── clip_02/
│   ├── clipe.mp4
│   ...
└── ...
```

Sem o pacote social, mantém-se `clip_XX/clipe.mp4`; a capa e o `descricao_redes.txt` só aparecem quando essa opção está ativa.

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
│   │   ├── llm_analyzer.py     # Análise LLM (Ollama / Gemini / Groq / …)
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

### 2. **core/logger.py** — logs
```python
configure_logging()           # mensagens para consola
logger / ForwardingHandler    # durante um job, o worker pode anexar um handler para espelhar logs INFO+ no painel da UI
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
4️⃣ Inicia processamento (botão principal)
        ↓
5️⃣ VideoProcessorThread executa:
        ├─ Extrai áudio (FFmpeg) → WAV temporário na pasta do projeto
        ├─ Transcreve (Whisper) → segmentos com timestamps
        ├─ Divide a transcrição em blocos (ex.: 10 min) → uma chamada LLM por bloco
        ├─ LLM (Ollama / Gemini / Groq) devolve intervalos candidatos
        ├─ Valida, alinha à transcrição, remove duplicados, aplica limites de duração
        ↓
6️⃣ Interface mostra clipes (ClipSelectionDialog), salvo modo “pular pré-visualização”
        ↓
7️⃣ Utilizador escolhe quais exportar
        ↓
8️⃣ Renderiza os selecionados (FFmpeg → `clip_XX/clipe.mp4`)
        ↓
9️⃣ Opcional: pacote social (texto + capa JPG por pasta)
        ↓
🔟 Concluído — pasta do projeto dentro do destino escolhido
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
└── meu_video_processed/                 # base por defeito se não escolher pasta
    └── Entrevista com Convidado/        # nome a partir do título nos metadados (ou nome do ficheiro)
        ├── clip_01/
        │   ├── clipe.mp4
        │   ├── capa.jpg
        │   └── descricao_redes.txt
        ├── clip_02/
        │   ├── clipe.mp4
        │   └── ...
        └── ...
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
