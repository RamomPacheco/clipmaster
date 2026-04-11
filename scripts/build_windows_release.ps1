# Build executável (PyInstaller onedir) e, se existir Inno Setup, o instalador .exe
# Uso (na raiz do repo, com venv ativo):
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows_release.ps1
# Opções:
#   -SkipInno     não corre ISCC
#   -SkipFFmpegCheck  não avisa se bundled/ffmpeg/windows estiver vazio

param(
    [switch] $SkipInno,
    [switch] $SkipFFmpegCheck,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path (Join-Path $Root "main.py"))) {
    Write-Error "main.py nao encontrado - execute o script a partir do repositorio ClipMaster."
}

$ffDir = Join-Path $Root "bundled\ffmpeg\windows"
$hasFF = (Test-Path (Join-Path $ffDir "ffmpeg.exe")) -and (Test-Path (Join-Path $ffDir "ffprobe.exe"))
if (-not $hasFF -and -not $SkipFFmpegCheck) {
    Write-Warning "FFmpeg nao encontrado em bundled\ffmpeg\windows\. O executavel nao incluira FFmpeg."
    Write-Warning "Corra: powershell -ExecutionPolicy Bypass -File scripts\fetch_ffmpeg_windows.ps1"
    if (-not $Force) {
        $cont = Read-Host "Continuar mesmo assim? (S/N)"
        if ($cont -notmatch '^[sSyY]') { exit 1 }
    }
}

Write-Host "PyInstaller..."
python -m PyInstaller (Join-Path $Root "ClipMaster.spec") --noconfirm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$distExe = Join-Path $Root "dist\ClipMaster\ClipMaster.exe"
if (-not (Test-Path $distExe)) {
    Write-Error "Build falhou: $distExe não existe."
}

Write-Host "Executável: $distExe"

if ($SkipInno) {
    Write-Host "Pasta portátil pronta: dist\ClipMaster\"
    exit 0
}

$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
    Write-Warning "Inno Setup 6 não encontrado. Instale a partir de https://jrsoftware.org/isdl.php"
    Write-Host "Pode distribuir a pasta dist\ClipMaster\ como ZIP (portátil)."
    exit 0
}

Write-Host "Inno Setup: $iscc"
& $iscc (Join-Path $Root "installer\ClipMaster.iss")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host 'Instalador: dist_installer\ClipMaster_Setup_*.exe'
