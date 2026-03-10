@echo off
title AI Viral Clipper Pro - Launcher
color 0A

echo [INFO] Ativando o ambiente virtual...
call .venv\Scripts\activate.bat

echo [INFO] Injetando bibliotecas CUDA (NVIDIA) no ambiente...
set PATH=E:\projetos_python\tiktoksele\.venv\Lib\site-packages\nvidia\cublas\bin;%PATH%
set PATH=E:\projetos_python\tiktoksele\.venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%

echo [INFO] Iniciando o aplicativo...
python main.py

echo.
