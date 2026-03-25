@echo off
title ClipMaster
echo [*] Ativando ambiente virtual Windows...
call .venv\Scripts\activate.bat
echo [*] Iniciando...
python main.py
pause