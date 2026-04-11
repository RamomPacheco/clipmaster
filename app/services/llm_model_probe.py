"""
Testes mínimos contra a API para confirmar que o modelo escolhido (ou uma lista) responde.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from app.core.logger import logger


def probe_llm_model_minimal(
    provider: str,
    model_name: str,
    api_key: str | None,
) -> Tuple[bool, str]:
    """
    Envia um pedido mínimo (1–2 tokens) e devolve (sucesso, detalhe).
    Consome quota mínima; útil para validar chave + id do modelo.
    """
    p = provider.strip().lower()
    mid = (model_name or "").strip()
    if not mid:
        return False, "Indique um modelo (selecione ou escreva no campo)."

    try:
        if p == "ollama":
            import ollama

            ollama.chat(
                model=mid,
                messages=[{"role": "user", "content": "."}],
                options={"num_predict": 1, "temperature": 0},
            )
            return True, "Ollama respondeu ao pedido mínimo."

        if p == "gemini":
            import google.generativeai as genai

            k = (api_key or "").strip() or (os.environ.get("GOOGLE_API_KEY") or "").strip()
            if not k:
                return False, "Chave Gemini em falta (campo ou GOOGLE_API_KEY)."
            genai.configure(api_key=k)
            model = genai.GenerativeModel(mid)
            response = model.generate_content(
                "Responda apenas: A",
                generation_config={"max_output_tokens": 4, "temperature": 0},
            )
            try:
                _ = response.text
            except Exception as inner:  # noqa: BLE001
                fb = getattr(response, "prompt_feedback", None)
                return False, f"Sem texto na resposta ({inner})" + (f"; {fb}" if fb else "")
            return True, "Gemini respondeu ao pedido mínimo."

        if p == "groq":
            from app.services.llm_analyzer import _groq_chat_completion

            _groq_chat_completion(
                "Reply with a single letter only.",
                "Say A.",
                mid,
                (api_key or "").strip(),
                temperature=0,
                max_tokens=4,
                json_object=False,
            )
            return True, "Groq respondeu ao pedido mínimo."

        if p == "openai":
            from app.services.llm_analyzer import _openai_chat_completion

            _openai_chat_completion(
                "Reply with a single letter only.",
                "Say A.",
                mid,
                (api_key or "").strip(),
                temperature=0,
                max_tokens=4,
                json_object=False,
            )
            return True, "OpenAI respondeu ao pedido mínimo."

        if p == "openrouter":
            from app.services.llm_analyzer import _openrouter_chat_completion

            _openrouter_chat_completion(
                "Reply with a single letter only.",
                "Say A.",
                mid,
                (api_key or "").strip(),
                temperature=0,
                max_tokens=4,
                json_object=False,
            )
            return True, "OpenRouter respondeu ao pedido mínimo."

        if p == "transformers":
            return (
                False,
                "Não há teste rápido para Transformers local — use um processamento curto para validar.",
            )

    except Exception as e:  # noqa: BLE001
        logger.warning("probe_llm_model_minimal falhou (%s / %s): %s", p, mid, e)
        return False, str(e)[:400]

    return False, f"Provedor não suportado para teste: {p}"


def verify_models_from_list(
    provider: str,
    model_ids: List[str],
    api_key: str | None,
    *,
    max_models: int = 25,
) -> List[Tuple[str, bool, str]]:
    """
    Testa sequencialmente até ``max_models`` ids (primeiros da lista).
    Devolve lista de (model_id, ok, mensagem).
    """
    out: List[Tuple[str, bool, str]] = []
    for m in model_ids[:max_models]:
        ok, msg = probe_llm_model_minimal(provider, m, api_key)
        out.append((m, ok, msg))
    return out
