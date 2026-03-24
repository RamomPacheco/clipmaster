@echo off
chcp 65001 >nul
title ClipMaster - AI Viral Clipper Pro

:: Sempre executa a partir da pasta onde o .bat está (raiz do projeto)
cd /d "%~dp0"

echo.
echo ========================================
echo   ClipMaster - iniciando...
echo ========================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo [*] Ambiente virtual ativado: .venv
) else (
    echo [!] Pasta .venv nao encontrada.
    echo     Primeira vez? Execute:  python setup_and_run.py
    echo     Ou crie o venv manualmente e instale: pip install -r requirements.txt
    echo [*] Usando Python do PATH...
    echo.
)

python main.py
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
