@echo off
chcp 65001 >nul
title ClipMaster

:: Sempre executa a partir da pasta onde o .bat está (raiz do projeto)
cd /d "%~dp0"

echo.
echo ========================================
echo   ClipMaster - iniciando...
echo ========================================
echo.

set "PY=%~dp0.venv\Scripts\python.exe"

if exist "%PY%" (
    echo [*] Python do venv: %PY%
) else (
    echo [!] Pasta .venv nao encontrada.
    echo     Primeira vez? Execute:  python setup_and_run.py
    echo     Ou crie o venv manualmente e instale: pip install -r requirements.txt
    set "PY=python"
    echo [*] Usando Python do PATH...
    echo.
)

"%PY%" main.py
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo [!] O app encerrou com erro ^(codigo %EXITCODE%^).
    pause
    exit /b %EXITCODE%
)

echo.
echo [*] App encerrado.
pause
exit /b 0
