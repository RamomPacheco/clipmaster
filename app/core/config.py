import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    return "cpu"

# Pasta padrão de exports (relativa à raiz do projeto antigo)
EXPORTS_ROOT = PROJECT_ROOT / "exports"

# Arquivo de histórico de processamento
PROCESSING_HISTORY_FILE = PROJECT_ROOT / "processing_history.json"

# Parâmetros de clipes
MIN_CLIP_SECONDS = 30.0
MAX_CLIP_SECONDS = 60.0

# Chunking da transcrição (10 minutos)
CHUNK_SECONDS = 600.0
# Sobreposição entre chunks (reduz cortes de ideia no meio entre blocos)
CHUNK_OVERLAP_SECONDS = 45.0

# Faster-Whisper — qualidade de timestamp e texto
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
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

