import os
import sys
from pathlib import Path


def _local_app_dir() -> Path:
    """Pasta persistente do utilizador (chaves API, exports instalados, histórico)."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path.home() / ".local" / "share"
    d = base / "AI_Viral_Clipper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()


def _whisper_language_from_env() -> str | None:
    raw = os.environ.get("WHISPER_LANGUAGE", "pt").strip().lower()
    if raw in ("", "auto", "none"):
        return None
    return raw


def _default_whisper_device() -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _default_whisper_compute_type(device: str) -> str:
    if device != "cuda":
        return "int8"
    # int8_float16 reduz VRAM no CUDA — ajuda quando Ollama já usa a mesma GPU
    if _env_bool("WHISPER_CUDA_LOW_VRAM") or _env_bool("WHISPER_SHARED_GPU_SAFE"):
        return "int8_float16"
    return "float16"


def whisper_device_effective() -> str:
    """
    Dispositivo efetivo do Whisper.
    WHISPER_SHARED_GPU_SAFE=1 força CPU para evitar disputa de VRAM com o Ollama na mesma GPU.
    """
    explicit = (os.environ.get("WHISPER_DEVICE") or "").strip().lower()
    if explicit in ("cpu", "cuda"):
        return explicit
    # Padrão: usar CPU para evitar disputa de VRAM/RAM com Ollama.
    # Para usar GPU, defina explicitamente `WHISPER_DEVICE=cuda`.
    if _env_bool("WHISPER_SHARED_GPU_SAFE") or _env_bool("WHISPER_PREFER_CPU"):
        return "cpu"
    return _default_whisper_device()

def _exports_root() -> Path:
    if getattr(sys, "frozen", False):
        p = _local_app_dir() / "exports"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return PROJECT_ROOT / "exports"


def _processing_history_file() -> Path:
    if getattr(sys, "frozen", False):
        return _local_app_dir() / "processing_history.json"
    return PROJECT_ROOT / "processing_history.json"


# Pasta padrão de exports (em desenvolvimento: raiz do repo; instalado: %LocalAppData%)
EXPORTS_ROOT = _exports_root()

# Histórico de processamento (mesma lógica)
PROCESSING_HISTORY_FILE = _processing_history_file()


def api_keys_storage_path() -> Path:
    """Ficheiro JSON com perfis de chaves API (nome + provedor + segredo)."""
    return _local_app_dir() / "api_keys.json"

# Parâmetros de clipes
class LLMParams:
    NUM_CTX = 4096
    NUM_PREDICT = 1024
    TEMPERATURE = 0.2
    TOP_P = 0.9

MIN_CLIP_SECONDS = 30.0
MAX_CLIP_SECONDS = 120.0

# Chunking da transcrição (~10 min por bloco; 600 s)
CHUNK_SECONDS = 600.0
# Sobreposição entre chunks (reduz cortes de ideia no meio entre blocos)
CHUNK_OVERLAP_SECONDS = 30.0

# ── Smart-snap: cortes mais limpos, independente do modelo ──
# Janela de procura (segundos em torno do ponto da IA) para encontrar pausa/frase.
SNAP_SEARCH_WINDOW_SEC = float(os.environ.get("CLIPMASTER_SNAP_WINDOW", "3.0"))
# Pausa mínima entre palavras (ms) para ser considerada "corte bom".
SNAP_MIN_PAUSE_MS = int(os.environ.get("CLIPMASTER_SNAP_PAUSE_MS", "300"))
# Peso: pausas longas valem mais no score de snapping.
SNAP_PAUSE_WEIGHT = float(os.environ.get("CLIPMASTER_SNAP_PAUSE_WEIGHT", "1.0"))
# Peso: começar/terminar em fim-de-frase (. ? !) vale mais.
SNAP_SENTENCE_WEIGHT = float(os.environ.get("CLIPMASTER_SNAP_SENTENCE_WEIGHT", "0.8"))
# Peso: proximidade ao ponto original da IA (menos desvio = mais seguro).
SNAP_PROXIMITY_WEIGHT = float(os.environ.get("CLIPMASTER_SNAP_PROXIMITY_WEIGHT", "0.5"))

# Faster-Whisper — qualidade de timestamp e texto
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = whisper_device_effective()
WHISPER_COMPUTE_TYPE = (
    os.environ.get("WHISPER_COMPUTE_TYPE")
    if os.environ.get("WHISPER_COMPUTE_TYPE")
    else _default_whisper_compute_type(WHISPER_DEVICE)
)
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
WHISPER_LANGUAGE = _whisper_language_from_env()

# Modelo padrão do Ollama (fallback)
DEFAULT_LLM_MODEL = "llama3.2:3b"

