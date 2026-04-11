# Descarrega FFmpeg GPL win64 (BtbN) para bundled/ffmpeg/windows/
# Executar na raiz do repositório:  powershell -ExecutionPolicy Bypass -File scripts\fetch_ffmpeg_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "main.py"))) {
    Write-Error "Execute a partir do repo ClipMaster (main.py não encontrado acima de scripts/)."
}

$Dest = Join-Path $Root "bundled\ffmpeg\windows"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$Url = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
$Zip = Join-Path $env:TEMP "clipmaster-ffmpeg-win64.zip"

Write-Host "A descarregar FFmpeg..."
Invoke-WebRequest -Uri $Url -OutFile $Zip

$Extract = Join-Path $env:TEMP "clipmaster-ffmpeg-extract"
if (Test-Path $Extract) { Remove-Item -Recurse -Force $Extract }
Expand-Archive -Path $Zip -DestinationPath $Extract -Force

$BinDir = Get-ChildItem -Path $Extract -Recurse -Directory -Filter "bin" | Where-Object {
    Test-Path (Join-Path $_.FullName "ffmpeg.exe")
} | Select-Object -First 1

if (-not $BinDir) {
    Write-Error "Não foi encontrada pasta bin com ffmpeg.exe no ZIP."
}

Copy-Item -Force (Join-Path $BinDir.FullName "ffmpeg.exe") (Join-Path $Dest "ffmpeg.exe")
Copy-Item -Force (Join-Path $BinDir.FullName "ffprobe.exe") (Join-Path $Dest "ffprobe.exe")

Remove-Item -Recurse -Force $Extract
Remove-Item -Force $Zip

Write-Host "OK: ffmpeg.exe e ffprobe.exe em $Dest"
