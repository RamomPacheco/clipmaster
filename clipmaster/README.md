# ClipMaster

Aplicação desktop para encontrar e renderizar automaticamente clipes virais a partir de vídeos longos,
utilizando **Faster-Whisper** para transcrição, **Ollama** para análise de conteúdo e **FFmpeg** para os cortes.

## Estrutura de Pastas

```text
clipmaster/
├── pyproject.toml
├── README.md
├── tests/
│   ├── test_services/
│   └── test_ui/
└── app/
    ├── __init__.py
    ├── main.py
    ├── core/
    ├── models/
    ├── services/
    ├── ui/
    └── workers/
```

O código legado original permanece em `originproject/main.py` para referência histórica.

## Execução

Dentro da pasta `clipmaster/`:

```bash
python -m app.main
```

Ou, a partir da raiz do repositório, você pode executar o `main.py` raiz, que delega para `app.main`.

