[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![pt-br](https://img.shields.io/badge/lang-Português-green.svg)](README.pt-BR.md)
# ClipMaster -Complete Guide

## 📋 What is it?

**ClipMaster** (window title: *AI Viral Clipper Pro*) is a Python desktop application that automates viral-style short clips from long videos. It works through:

1. **Transcription (Faster-Whisper)** — extracts audio and converts it to timestamped text  
2. **AI analysis** — finds strong clip candidates; supports **Ollama** (local), **Gemini**, and **Groq** (API key in the app)  
3. **Rendering (FFmpeg)** — cuts and exports **H.264** MP4 (optional **NVENC** when available), with optional TikTok-style burned-in captions  
4. **Optional social package** — per-clip description text and **JPG cover** image for social posts  

During processing, the UI shows a **progress bar** (per phase: analysis chunks, render, social) and an optional **log panel** that mirrors **terminal-style** `logging` output (transcription, FFmpeg messages, HTTP client logs, etc.).

---
### 💙 [If you liked it, make a donation ☕](https://livepix.gg/ramompacheco)
## 🔧 INSTALLATION

### System Prerequisites

1. **Python 3.10+**
```powershell
python --version # Check if it is 3.10+
```

2. **FFmpeg** (required)
   - Download: https://ffmpeg.org/download.html
   - Or by Chocolatey (Windows):
     ```powershell
     choco install ffmpeg
     ```
   - Check:
```powershell
     ffmpeg -version
     ```

3. **Ollama** (optional — for local LLM analysis)
   - Download: https://ollama.ai  
   - After installing:
     ```powershell
     ollama pull phi4   # or another model you select in the app
     ollama serve       # run the server (separate terminal)
     ```  
   For **Gemini** or **Groq**, configure the API key in the application instead.

### Project Installation

1. **Clone / enter the project**
```powershell
git clone https://github.com/RamomPacheco/clipmaster.git
cd clipmaster
```

2. **Create virtual environment (first time)**
```powershell
python -m venv .venv
```

3. **Activate virtual environment**
```# 3. Activate the virtual environment
Windows (PowerShell/CMD):
.venv\Scripts\activate
Linux/Mac:
source .venv/bin/activate
```

4. **Install dependencies**
```powershell
pip install -r requirements.txt
```

---

## ▶️ HOW TO USE

### Run the Application

```powershell
# From the root of the project
python main.py
```

The graphical interface will open. Follow the steps:

1. **Select video** — drag-and-drop or browse  
2. **Output folder** (optional) — default: `exports/{video_stem}_processed` under the project  
3. **LLM provider** — Ollama, Gemini, or Groq (+ model names as shown in the UI)  
4. **Whisper model** — e.g. `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`  
5. **Prompt / analysis type** — balanced, humor, educational, etc.  
6. **Start processing** — primary action button on the main screen  
7. **Review and select clips** (unless “skip preview” is enabled) — dialog lists AI suggestions  
8. **Exports** — high-quality MP4s plus optional social files (see below)  
9. **Logs** — enable *Show logs terminal* to see the same detail as the console; watch the **progress bar** during transcription, LLM chunks, rendering, and social generation  

### Output folder layout

Everything is written under **the folder you chose** (or the default export base). The app creates **one project subfolder** named from **video metadata** (title / description / comment in the file tags when present; otherwise the **filename without extension**). If that name already exists, a numeric suffix is added (`_1`, `_2`, …).

Inside that project folder, each exported clip has **its own subdirectory** `clip_01`, `clip_02`, …:

```
{your_output_or_exports/...}/{Project Name From Metadata}/
├── temp_audio_safe.wav          # present only during run; usually deleted after transcription
├── clip_01/
│   ├── clipe.mp4                # final vertical/exported clip
│   ├── capa.jpg                 # optional — social cover
│   └── descricao_redes.txt      # optional — hook + description for posts
├── clip_02/
│   ├── clipe.mp4
│   ...
└── ...
```

Without the social package, you still get `clip_XX/clipe.mp4`; cover and `descricao_redes.txt` are omitted when that feature is disabled.

---

## 📁 CODE STRUCTURE

```
clipmaster/
├── pyproject.toml # Project configuration
├──app/
│ ├── main.py # Entry point -creates QApplication
│ ├── core/
│ │ ├── config.py # Constants and paths
│ │ ├── logger.py # Logging system
│ │ └── cuda_setup.py # CUDA/DLL setup
│ ├── models/
│ │ └── schemas.py # Data types (Clip, Metrics, etc)
│ ├── services/
│ │ ├── transcription.py # Whisper -convert audio to text
│ │ ├── llm_analyzer.py # LLM analysis (Ollama / Gemini / Groq / …)
│ │ ├── video_engine.py # FFmpeg -cropping videos
│ │ └── clip_manager.py # Clip management
│ ├── ui/
│ │ ├── main_window.py # Main interface
│ │ ├── components/
│ │ │ └── drop_zone.py # Drag zone for videos
│ │ └── dialogs/
│ │ └── clip_dialog.py # Clip selection dialog
│ └── workers/
│ └── processing_task.py # Heavy processing thread
```

---

## 🔍 EXPLANATION OF MAIN FUNCTIONS

### 1. **core/config.py**-Global Settings
```python
PROJECT_ROOT # Project root
EXPORTS_ROOT # Folder to export clips
MIN_CLIP_SECONDS #30s (minimum duration of a clip)
MAX_CLIP_SECONDS #60s (maximum for TikTok/Shorts)
CHUNK_SECONDS #600s = 10 minutes (splits video into chunks)
DEFAULT_LLM_MODEL # "llama3.2:3b" (default AI)
```

### 2. **core/logger.py** — logging
```python
configure_logging()           # root-style messages to the console
logger / ForwardingHandler    # during a job, the worker can attach a handler so INFO+ logs also appear in the UI log panel
```

### 3. **models/schemas.py**-Data Structures

#### `Clip`
Represents a video clip:
```python
classClip:
    start: float # Start second (ex: 10.5)
end: float # End second (ex: 55.0)
    reason: str # Why is it viral? (ex: "Funny moment")
    headline: str # Short title of the clip
    
    @property
    def duration: # Calculate end -start
```

#### `ProcessingMetrics`
Processing metrics:
```python
start_time # When it started
transcription_time # How long it took to transcribe
analysis_time # How long it took the AI to analyze
rendering_time # How long it took to render
total_clips_found # How many clips found
clips_selected # How many the user selected
video_duration # Duration of the original video
model used # In: "llama 3.2:3b"
prompt type # Ex: "Humor and Comedy"
```

### 4. **services/transcription.py**-Convert Audio to Text

#### `transcribe_audio(audio_path, model_name=None) → (segments, duration)`
Transcribes a WAV file:
- **Entry**: Path to WAV file
- **Optional parameter**: `model_name` (provided by the UI selector)
- **Exit**: List of segments + total duration
```python
segments = [
{"start": 0.5, "end": 2.3, "text": "Hello, how are you?"},
    {"start": 2.3, "end": 4.1, "text": "I'm fine, thanks!"},
    ...
]
```

### 5. **services/llm_analyzer.py**-AI Analysis

#### `build_prompts(prompt_type, text, custom_prompt)`
Builds the prompts for the AI:
- **Standard**: Balanced, for any content
- **Humor & Comedy**: Search for funny moments
- **Serious & High Value**: Search for educational/valuable content

#### `analyze_viral_potential(text, model, prompt_type)`
Send text to Ollama and receive clips:
```python
# Input: 10 minute transcript
# Output: List of clips with times and ratios
[
{"start": 15.0, "end": 45.0, "reason": "Hilarious joke", "headline": "The best moment"},
    {"start": 120.0, "end": 155.0, "reason": "Valuable advice", "headline": "Important tip"},
]
```

### 6. **services/video_engine.py**-Trimming and Rendering

#### `extract_safe_audio(video_path, output_dir) → audio_path`
Extract audio from video with FFmpeg:
```python
# Input: video.mp4
# Output: temp_audio_safe.wav (mono, 16kHz)
# Used for Whisper to transcribe
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

#### `render_clips(video_path, clips, output_dir, resolution, bitrate)`
Renders all clips with H.264:
```python
# For each clip:
# ffmpeg -i video.mp4 -ss 15.0 -to 45.0 -c:v libx264 -crf 18 clip_1.mp4
# 
# Parameters:
# -crf 18 → Quality (0=maximum, 51=minimum, 18=optimal)
# -preset slow → More time, better compression
# -r 30 → 30 fps (compatible with TikTok/Shorts)
# -c:a aac → Audio in AAC (default)
```

### 7. **services/clip_manager.py**-Clip Management

#### `enforce_duration_limits(clips, max_video_duration, min, max)`
Ensures clips respect limits (30-60s):
```python
# If clip < 30s → Expand (adds seconds before/after)
# If clip > 60s → Cut to exact 60s
```

#### `remove_duplicate_clips(clips)`
Removes clips that overlap > 50%:
```python
# If two clips occupy >50% of the same space → Remove the one with the smaller ratio
```

#### `append_history_entry(metrics, video_path, history_file)`
Saves processing history in JSON:
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

### 8. **workers/processing_task.py**-Orchestration

#### `VideoProcessorThread` (inheritance from QThread)
Thread that runs the complete pipeline:

**Initialization:**
```python
thread = VideoProcessorThread(
    video_path="/path/video.mp4",
    model_name="llama3.2:3b",
    whisper_model="small",
    output_dir="/exports/meu_video_processed",
prompt type="Humor and Comedy", resolution="1080p", bitrate="", # Empty = use CRF 18 custom_prompt=None # None = use default prompt
)



```

**Signals emitted:**
```python
progress_signal.emit("Progress message") # Update UI
finished_signal.emit("Processing completed") # End
error_signal.emit("ERROR!") # Failed
clips_ready_signal.emit(clips_list) # Ready clips
```

**Pipeline (`run` function):**
1. ✅ Checks if FFmpeg exists
2. 📦 Extract audio from video (FFmpeg)
3. 📝 Transcribe with Whisper
4. 🔪 Divide into 10-minute chunks
5. 🤖 Analyze each chunk with Ollama
6. 🧹 Remove duplicates
7. 📐 Strength duration limits (30-60s)
8. 💾 Saves history
9. 🎬 Renders all clips (FFmpeg)

### 9. **ui/main_window.py**-Graphical Interface

#### `ViralApp` (inheritance from QMainWindow)

**Main methods:**

```python
_setup_ui() # Build the interface
_apply_dark_theme() # Apply dark theme
_on_start_engine_clicked() # Start processing
_on_save_clips_clicked() # Saves selected clips
update_log(message) # Update the log box
_get_available_models() # List installed Ollama models
_load_processing_history() # Load previous history
```

**Connected signals:**
```python
# When thread emits, UI updates itself:
thread.progress_signal → update_log()
thread.clips_ready_signal → show ClipSelectionDialog
thread.error_signal → show error message
```

### 10. **ui/components/drop_zone.py**-Zona de Arraste

```python
class DropZone(QLabel):
file_dropped = Signal(str) # Emits when video is dropped
    
    # Allows you to drag video directly into the widget
```

### 11. **ui/dialogs/clip_dialog.py**-Clip Selection

```python
class ClipSelectionDialog(QDialog):
    # Show preview of found clips
    # User selects which ones they want to save
    # Returns list of selected clips
```

---

## 🎯 FULL FLOW

```
1️⃣ User opens main.py
        ↓
2️⃣ Interface appears (ViralApp)
↓
3️⃣ User selects video + model + type
        ↓
4️⃣ Start processing (primary button)
        ↓
5️⃣ VideoProcessorThread runs:
        ├─ Extract audio (FFmpeg) → temporary WAV in the project folder
        ├─ Transcribe (Whisper) → segments with timestamps
        ├─ Chunk transcript (e.g. 10 min) → one LLM call per chunk
        ├─ LLM (Ollama / Gemini / Groq) returns candidate ranges
        ├─ Validate, snap to transcript, dedupe, enforce duration limits
        ↓
6️⃣ Interface shows clips (ClipSelectionDialog), unless preview skip is on
        ↓
7️⃣ User selects which clips to export
        ↓
8️⃣ Renders selected clips (FFmpeg, per `clip_XX/clipe.mp4`)
        ↓
9️⃣ Optional: social package (text + JPG cover per folder)
        ↓
🔟 Done — project folder under your chosen output path
```

---

## ⚙️ IMPORTANT SETTINGS

### Duration Limits (config.py)
```python
MIN_CLIP_SECONDS = 30.0 # Minimum TikTok/Shorts
MAX_CLIP_SECONDS = 60.0 # YouTube Maximum Shorts
```

### Chunking (config.py)
```python
CHUNK_SECONDS = 600.0 # 10 minutes per chunk
# 1 hour video = 6 requests to Ollama
# 2 hour video = 12 requests
# (More chunks = more analysis, but better quality)
```

### Default model
```python
DEFAULT_LLM_MODEL = "llama3.2:3b"
# You can change to other Ollama models:
# -llama2
# -neural-chat
# -orca-mini
# etc.
```

### Whisper on CPU by default
```python
WHISPER_DEVICE = "cpu"  # default to avoid GPU contention with Ollama
```
To force GPU manually:
```powershell
$env:WHISPER_DEVICE="cuda"
python main.py
```
```python
BEST_LLM-MODEL = "phi4"
# for RTX 3060 12gb, ryzen 5 5500
```

---

## 🐛 TROUBLESHOOTING

### Error: "ModuleNotFoundError: No module named 'app'"
**Solution**: Always perform **project root**:
```powershell
cd e:\python_projects\tiktoksele
python main.py
```

### Error: "ffmpeg not found"
**Solution**: Install FFmpeg and add to PATH:
```powershell
choco install ffmpeg
```

### Error: Ollama does not connect
**Solution**: Make sure the Ollama server is running:
```powershell
# In another terminal:
ollama serve
```

### Whisper very slow
**Reason**: Using CPU
**Solution**: Install CUDA if you have NVIDIA GPU

### Very bad clips
**Solution**: Test other prompt types:
- Change in `prompt_type` in the UI
- Or create custom prompt

---

## 📊 EXAMPLE OUTPUT

```
exports/
└── My_Video_processed/                    # default base if you did not pick a folder
    └── Interview With Guest/              # folder name from metadata title (or filename)
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

## 🚀 PRACTICAL TIPS

1. **For better quality**:Use `-crf 15-18` (longer, better)
2. **For more clips**: Activate "Serious & High Value" (search for more content)
3. **For long videos**: Use 10 min chunks for better analysis
4. **History**: Check `processing_history.json` for statistics
5. **Customize**: Edit the prompts in `llm_analyzer.py` for your style

---
### 💙 [If you liked it, make a donation ☕](https://livepix.gg/ramompacheco)
