from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple
import ollama
from app.core.config import DEFAULT_LLM_MODEL
from app.core.logger import logger

# Reforça alinhamento aos timestamps reais do Whisper (reduz alucinação de segundos).
_TIMESTAMP_RULE = """
    REGRA DE TIMESTAMPS (OBRIGATÓRIA): Os valores de "start" e "end" DEVEM ser tempos que
    apareçam explicitamente nas linhas "[início - fim]" desta transcrição (use o número de
    início de uma linha para começar o clipe e o número de fim de uma linha para terminar,
    cobrindo uma ou várias linhas inteiras). Não invente segundos que não existam no texto.
"""


def _base_system_prompt() -> str:
    return (
        "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e "
        "YouTube Shorts. Sua ÚNICA função é extrair blocos de tempo. "
        "Retorne APENAS um array JSON puro."
    )


def build_prompts(prompt_type: str, text: str, custom_prompt: str | None) -> Tuple[str, str]:
    """
    Replica a lógica de _get_prompt_config do código original, extraída para serviço.
    """
    base_system = _base_system_prompt()

    if custom_prompt:
        hint = (
            "\n\n(Obrigatório: use apenas tempos de início/fim que existam nas linhas "
            "\"[x - y]\" da transcrição; não invente segundos fora desse texto.)"
        )
        return base_system, custom_prompt + hint

    base_user = f"""
    Analise esta fatiada da transcrição e encontre os momentos mais magnéticos.

    REGRAS DE OURO (CRÍTICAS):
    1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
    2. COERÊNCIA: O clipe deve começar no início exato do raciocínio e terminar na conclusão.
    3. FOCO: Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
    4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.
{_TIMESTAMP_RULE}
    --- TRANSCRIÇÃO ---
    {text}
    --- FIM DA TRANSCRIÇÃO ---

    Retorne APENAS o JSON rigoroso: 
    [
        {{"start": 10.5, "end": 55.0, "reason": "Motivo", "headline": "Título"}}
    ]
    """

    if prompt_type == "Padrão (Equilibrado)":
        return base_system, base_user

    if prompt_type == "Humor & Comédia":
        system = (
            "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e "
            "YouTube Shorts, focado em conteúdo humorístico e engraçado. Sua ÚNICA função é "
            "extrair blocos de tempo. Retorne APENAS um array JSON puro."
        )
        user = f"""
        Analise esta fatiada da transcrição e encontre os momentos mais engraçados e humorísticos.

        REGRAS DE OURO (CRÍTICAS):
        1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
        2. COERÊNCIA: O clipe deve começar no início exato da piada ou situação engraçada e terminar na conclusão.
        3. FOCO: Priorize momentos que gerem risadas, situações cômicas, ironia ou humor leve. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
        4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.
{_TIMESTAMP_RULE}
        --- TRANSCRIÇÃO ---
        {text}
        --- FIM DA TRANSCRIÇÃO ---

        Retorne APENAS o JSON rigoroso: 
        [
            {{"start": 10.5, "end": 55.0, "reason": "Motivo engraçado", "headline": "Título humorístico"}}
        ]
        """
        return system, user

    if prompt_type == "Sério & Alto Valor":
        system = (
            "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e "
            "YouTube Shorts, focado em conteúdo sério e de alto valor. Sua ÚNICA função é "
            "extrair blocos de tempo. Retorne APENAS um array JSON puro."
        )
        user = f"""
        Analise esta fatiada da transcrição e encontre os momentos mais sérios e valiosos.

        REGRAS DE OURO (CRÍTICAS):
        1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
        2. COERÊNCIA: O clipe deve começar no início exato do raciocínio sério e terminar na conclusão valiosa.
        3. FOCO: Priorize momentos que transmitam conhecimento profundo, insights valiosos, conselhos sérios ou conteúdo impactante. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
        4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.
{_TIMESTAMP_RULE}
        --- TRANSCRIÇÃO ---
        {text}
        --- FIM DA TRANSCRIÇÃO ---

        Retorne APENAS o JSON rigoroso: 
        [
            {{"start": 10.5, "end": 55.0, "reason": "Motivo sério e valioso", "headline": "Título impactante"}}
        ]
        """
        return system, user

    if prompt_type == "Storytelling & Emoção":
        system = (
            "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e "
            "YouTube Shorts, focado em storytelling emocional. Sua ÚNICA função é extrair "
            "blocos de tempo. Retorne APENAS um array JSON puro."
        )
        user = f"""
        Analise esta fatiada da transcrição e encontre os momentos mais emocionantes e narrativos.

        REGRAS DE OURO (CRÍTICAS):
        1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
        2. COERÊNCIA: O clipe deve começar no início exato da história ou emoção e terminar na conclusão emocional.
        3. FOCO: Priorize momentos que contem histórias, gerem emoção, inspiração ou conexão emocional. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
        4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.
{_TIMESTAMP_RULE}
        --- TRANSCRIÇÃO ---
        {text}
        --- FIM DA TRANSCRIÇÃO ---

        Retorne APENAS o JSON rigoroso: 
        [
            {{"start": 10.5, "end": 55.0, "reason": "Motivo emocional", "headline": "Título inspirador"}}
        ]
        """
        return system, user

    if prompt_type == "Educacional & Dicas":
        system = (
            "Você é um Diretor de Edição Sênior especialista em retenção para TikTok e "
            "YouTube Shorts, focado em conteúdo educacional. Sua ÚNICA função é extrair "
            "blocos de tempo. Retorne APENAS um array JSON puro."
        )
        user = f"""
        Analise esta fatiada da transcrição e encontre os momentos mais educacionais e com dicas práticas.

        REGRAS DE OURO (CRÍTICAS):
        1. DURAÇÃO (30s a 60s): O clipe DEVE ter no mínimo 30 segundos. Se a ideia precisar de mais tempo para ter coerência, você DEVE aumentar a duração, mas o LIMITE ABSOLUTO E MÁXIMO é 60 segundos. Não passe de 60s sob nenhuma hipótese.
        2. COERÊNCIA: O clipe deve começar no início exato da explicação ou dica e terminar na conclusão prática.
        3. FOCO: Priorize momentos que ensinem algo novo, deem dicas práticas, expliquem conceitos ou forneçam conhecimento útil. Retorne apenas clipes geniais. Se não houver nenhum, retorne [].
        4. NÃO DUPLICAR: Não gere clipes que se sobreponham significativamente (mais de 50%) ou sejam muito similares em conteúdo. Garanta que cada clipe seja único e distinto.
{_TIMESTAMP_RULE}
        --- TRANSCRIÇÃO ---
        {text}
        --- FIM DA TRANSCRIÇÃO ---

        Retorne APENAS o JSON rigoroso: 
        [
            {{"start": 10.5, "end": 55.0, "reason": "Motivo educacional", "headline": "Título instrutivo"}}
        ]
        """
        return system, user

    return base_system, base_user


def analyze_viral_potential(
    text: str,
    model_name: str | None,
    prompt_type: str,
    custom_prompt: str | None,
) -> List[Dict[str, Any]]:
    """
    Extrai clipes candidatos usando o Ollama, mantendo o comportamento do código original.
    """
    system_prompt, user_prompt = build_prompts(prompt_type, text, custom_prompt)
    model_to_use = model_name or DEFAULT_LLM_MODEL

    try:
        response = ollama.chat(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={
                "num_ctx": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )
        raw_content = response["message"]["content"]

        match = re.search(r"\[.*\]", raw_content, re.DOTALL)
        if not match:
            return []

        return json.loads(match.group(0).strip())

    except ollama.ResponseError as e:
        logger.error("Erro na resposta do Ollama: %s", e)
        return []
    except json.JSONDecodeError as e:
        logger.error("Erro ao decodificar JSON: %s", e)
        return []
    except Exception as e:  # noqa: BLE001
        logger.error("Falha ao extrair clipes deste capítulo: %s", e)
        return []

