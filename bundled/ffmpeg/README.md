# FFmpeg empacotado (ClipMaster)

Coloque aqui os binários **oficiais** ou de builds estáticos (ex.: [BtbN FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)) para a app funcionar **sem** instalar FFmpeg no sistema.

## Windows

Pasta destino:

`bundled/ffmpeg/windows/`

Ficheiros necessários:

- `ffmpeg.exe`
- `ffprobe.exe`

Passos típicos:

1. Descarregar o ficheiro `ffmpeg-master-latest-win64-gpl.zip` (ou variante LGPL se preferires).
2. Extrair e copiar `bin/ffmpeg.exe` e `bin/ffprobe.exe` para `bundled/ffmpeg/windows/`.

## macOS / Linux

Pasta: `bundled/ffmpeg/macos/` ou `bundled/ffmpeg/linux/` com executáveis `ffmpeg` e `ffprobe` (sem extensão), com permissão de execução.

## PyInstaller

No `.spec`, incluir a pasta da plataforma, por exemplo:

```python
from PyInstaller.utils.hooks import collect_data_files
import sys
from pathlib import Path

root = Path(__file__).parent
if sys.platform.startswith("win"):
    ffmpeg_data = [(str(root / "bundled" / "ffmpeg" / "windows"), "ffmpeg-bin")]
else:
    ffmpeg_data = []  # ajustar para macos/linux

a = Analysis(
    ...
    datas=ffmpeg_data,
)
```

## Variáveis de ambiente (opcional)

- `CLIPMASTER_FFMPEG_PATH` — caminho absoluto para `ffmpeg`
- `CLIPMASTER_FFPROBE_PATH` — caminho absoluto para `ffprobe`

Isto sobrepõe a pasta `bundled/`.
