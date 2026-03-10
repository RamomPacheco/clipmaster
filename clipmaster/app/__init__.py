"""
Pacote principal da aplicação ClipMaster.

Organização em camadas:
- core: configuração, logging, setup de CUDA.
- models: schemas e validação de dados.
- services: regras de negócio e integrações externas.
- ui: interface PySide6.
- workers: tarefas em background (QThread).
"""

