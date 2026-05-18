@echo off
chcp 65001 >nul
title ClipMaster

cd /d "%~dp0"

echo.
echo ========================================
echo   ClipMaster - iniciando...
echo ========================================
echo.

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [!] Ambiente virtual nao encontrado: %~dp0.venv
    echo     Primeira vez? Execute:  python setup_and_run.py
    echo.
    pause
    exit /b 1
)

echo [*] Python: %PY%
echo [*] Iniciando...
echo.

"%PY%" main.py
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo [!] O app encerrou com erro ^(codigo %EXITCODE%^).
    echo     Se apareceu "No module named", reinstale: "%PY%" -m pip install -r requirements.txt
    pause
    exit /b %EXITCODE%
)

echo.
echo [*] App encerrado.
pause
exit /b 0
