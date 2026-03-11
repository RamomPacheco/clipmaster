from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Pasta padrão de exports (relativa à raiz do projeto antigo)
EXPORTS_ROOT = PROJECT_ROOT / "exports"

# Arquivo de histórico de processamento
PROCESSING_HISTORY_FILE = PROJECT_ROOT / "processing_history.json"

# Parâmetros de clipes
MIN_CLIP_SECONDS = 30.0
MAX_CLIP_SECONDS = 60.0

# Chunking da transcrição (10 minutos)
CHUNK_SECONDS = 600.0

# Modelo padrão do Ollama (fallback)
DEFAULT_LLM_MODEL = "llama3.2:3b"

