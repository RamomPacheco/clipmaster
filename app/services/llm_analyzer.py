from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import ollama

from app.core.config import DEFAULT_LLM_MODEL
from app.core.logger import logger

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

# Reforça alinhamento aos timestamps reais do Whisper (reduz alucinação de segundos).
_TIMESTAMP_RULE = """
    REGRA DE TIMESTAMPS (OBRIGATÓRIA): Os valores de "start" e "end" DEVEM ser tempos que
    apareçam explicitamente nas linhas "[início - fim]" desta transcrição (use o número de
    início de uma linha para começar o clipe e o número de fim de uma linha para terminar,
    cobrindo uma ou várias linhas inteiras). Não invente segundos que não existam no texto.
"""

_HF_PIPELINE_CACHE: Dict[str, Any] = {}


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


def _extract_json_array(raw_content: str) -> List[Dict[str, Any]]:
    match = re.search(r"\[.*\]", raw_content, re.DOTALL)
    if not match:
        return []
    return json.loads(match.group(0).strip())


def _extract_json_object(raw_content: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return {}
    return json.loads(match.group(0).strip())


def _groq_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model_to_use: str,
    api_key: str,
    *,
    temperature: float,
    max_tokens: int,
    json_object: bool = False,
) -> str:
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(
            "Pacote 'httpx' não instalado. Execute: pip install httpx"
        ) from e

    key = (api_key or "").strip() or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("Chave API Groq em falta (campo na app ou GROQ_API_KEY).")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=body)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = resp.text[:500]
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"Groq API HTTP {resp.status_code}: {detail}") from e
        data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Resposta Groq sem choices.")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Resposta Groq sem texto.")
    return content


def _analyze_with_groq(
    system_prompt: str,
    user_prompt: str,
    model_to_use: str,
    api_key: str | None = None,
) -> List[Dict[str, Any]]:
    raw = _groq_chat_completion(
        system_prompt,
        user_prompt,
        model_to_use,
        (api_key or "").strip(),
        temperature=0.1,
        max_tokens=8192,
        json_object=False,
    )
    return _extract_json_array(raw)


def _analyze_with_ollama(
    system_prompt: str,
    user_prompt: str,
    model_to_use: str,
) -> List[Dict[str, Any]]:
    response = ollama.chat(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={
            "num_ctx": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
        },
    )
    raw_content = response["message"]["content"]
    return _extract_json_array(raw_content)


def _analyze_with_gemini(
    system_prompt: str,
    user_prompt: str,
    model_to_use: str,
    api_key: str | None = None,
) -> List[Dict[str, Any]]:
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError(
            "Pacote 'google-generativeai' não instalado. Instale para usar Gemini API."
        ) from e

    api_key_to_use = (api_key or "").strip() or os.environ.get("GOOGLE_API_KEY")
    if not api_key_to_use:
        raise RuntimeError("GOOGLE_API_KEY não definido no ambiente.")

    # Segue o padrão do exemplo fornecido pelo usuário.
    genai.configure(api_key=api_key_to_use)
    model = genai.GenerativeModel(model_to_use)
    response = model.generate_content(
        f"{system_prompt}\n\n{user_prompt}",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
        },
    )
    raw_content = getattr(response, "text", "") or ""
    return _extract_json_array(raw_content)


def _analyze_with_transformers(
    system_prompt: str,
    user_prompt: str,
    model_to_use: str,
    max_new_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "Pacotes de Transformers não instalados. Instale 'transformers', 'torch' e 'accelerate'."
        ) from e

    pipe = _HF_PIPELINE_CACHE.get(model_to_use)
    if pipe is None:
        pipe = pipeline(
            "text-generation",
            model=model_to_use,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _HF_PIPELINE_CACHE[model_to_use] = pipe

    prompt = f"{system_prompt}\n\n{user_prompt}\n\nRetorne apenas JSON válido."
    response = pipe(
        prompt,
        max_new_tokens=max(64, min(int(max_new_tokens or 700), 2048)),
        do_sample=False,
        temperature=0.1,
    )
    raw_content = ""
    if isinstance(response, list) and response:
        raw_content = str(response[0].get("generated_text", ""))
    return _extract_json_array(raw_content)


def analyze_viral_potential(
    text: str,
    model_name: str | None,
    prompt_type: str,
    custom_prompt: str | None,
    provider: str = "ollama",
    api_key: str | None = None,
    max_new_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Extrai clipes candidatos usando o Ollama, mantendo o comportamento do código original.
    """
    system_prompt, user_prompt = build_prompts(prompt_type, text, custom_prompt)
    model_to_use = model_name or DEFAULT_LLM_MODEL
    provider = provider.strip().lower()

    try:
        if provider == "gemini":
            return _analyze_with_gemini(system_prompt, user_prompt, model_to_use, api_key)
        if provider == "groq":
            return _analyze_with_groq(system_prompt, user_prompt, model_to_use, api_key)
        if provider == "transformers":
            return _analyze_with_transformers(
                system_prompt,
                user_prompt,
                model_to_use,
                max_new_tokens=max_new_tokens,
            )
        return _analyze_with_ollama(system_prompt, user_prompt, model_to_use)
    except ollama.ResponseError as e:
        logger.error("Erro na resposta do Ollama: %s", e)
        return []
    except json.JSONDecodeError as e:
        logger.error("Erro ao decodificar JSON: %s", e)
        return []
    except Exception as e:  # noqa: BLE001
        logger.error("Falha ao extrair clipes deste capítulo: %s", e)
        return []


def generate_social_package(
    narrative_context: str,
    clip_text: str,
    clip_start: float,
    clip_end: float,
    model_name: str | None,
    provider: str = "ollama",
    api_key: str | None = None,
    max_new_tokens: int | None = None,
) -> Dict[str, Any]:
    """
    Gera conteúdo social para um clipe já criado:
    - frase de impacto curta para capa
    - descrição para redes sociais
    - segundo relativo ideal para o frame da capa

    ``narrative_context``: transcrição do vídeo desde o início até o fim deste clipe
    (tempos absolutos), para o título fazer sentido no enredo.
    ``clip_text``: apenas o trecho do clipe, para ancorar o frame_second.
    """
    narrative = (narrative_context or "").strip()
    if not narrative:
        narrative = "Sem transcrição acumulada disponível."
    safe_clip = (clip_text or "").strip()
    if not safe_clip:
        safe_clip = "Sem transcrição detalhada apenas do clipe."

    duration = max(0.5, float(clip_end) - float(clip_start))
    model_to_use = model_name or DEFAULT_LLM_MODEL
    provider = provider.strip().lower()

    system_prompt = (
        "Você é um estrategista de conteúdo para TikTok e Shorts. "
        "Responda APENAS com JSON válido."
    )
    user_prompt = f"""
    Tarefa: gerar pacote social para um clipe já renderizado.

    CONTEXTO NARRATIVO (transcrição do vídeo ORIGINAL desde o início até o fim deste clipe,
    tempos absolutos em segundos — use isto para o título e a descrição fazerem sentido no conjunto):
    ---
    {narrative}
    ---

    Trecho APENAS deste clipe (referência para escolher o melhor instante visual):
    ---
    {safe_clip}
    ---

    DADOS:
    - início absoluto do clipe: {clip_start:.2f}s
    - fim absoluto do clipe: {clip_end:.2f}s
    - duração do clipe: {duration:.2f}s

    REGRAS:
    1) hook_phrase: frase curta e forte (máx. 90 caracteres), em português, sem emojis.
       Deve refletir o CONTEXTO NARRATIVO acima, não só as últimas frases do clipe.
    2) description: texto para redes sociais (2-4 linhas), com CTA de engajamento, alinhado ao contexto.
    3) frame_second: segundo RELATIVO dentro do clipe para tirar a capa.
       Deve estar entre 0 e {max(0.5, duration - 0.1):.2f}.
    4) Não invente fatos fora da transcrição.
    5) Retorne apenas este JSON:
    {{
      "hook_phrase": "texto",
      "description": "texto",
      "frame_second": 12.3
    }}
    """

    fallback = {
        "hook_phrase": "O momento que muda tudo",
        "description": (
            "Assista até o final e me diga se você concorda com esse ponto.\n"
            "Comenta sua opinião e compartilha com quem precisa ver isso."
        ),
        "frame_second": round(min(max(duration * 0.45, 0.0), max(0.0, duration - 0.1)), 2),
    }

    try:
        if provider == "gemini":
            import google.generativeai as genai

            api_key_to_use = (api_key or "").strip() or os.environ.get("GOOGLE_API_KEY")
            if not api_key_to_use:
                return fallback
            genai.configure(api_key=api_key_to_use)
            model = genai.GenerativeModel(model_to_use)
            response = model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={"temperature": 0.2, "top_p": 0.9},
            )
            raw_content = getattr(response, "text", "") or ""
            obj = _extract_json_object(raw_content)
        elif provider == "groq":
            try:
                raw_content = _groq_chat_completion(
                    system_prompt,
                    user_prompt,
                    model_to_use,
                    (api_key or "").strip(),
                    temperature=0.2,
                    max_tokens=4096,
                    json_object=True,
                )
            except Exception:  # noqa: BLE001
                raw_content = _groq_chat_completion(
                    system_prompt,
                    user_prompt,
                    model_to_use,
                    (api_key or "").strip(),
                    temperature=0.2,
                    max_tokens=4096,
                    json_object=False,
                )
            obj = _extract_json_object(raw_content)
        elif provider == "transformers":
            try:
                import torch
                from transformers import pipeline
            except ImportError:
                return fallback

            pipe = _HF_PIPELINE_CACHE.get(model_to_use)
            if pipe is None:
                pipe = pipeline(
                    "text-generation",
                    model=model_to_use,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                _HF_PIPELINE_CACHE[model_to_use] = pipe

            prompt = f"{system_prompt}\n\n{user_prompt}\n\nRetorne apenas JSON válido."
            response = pipe(
                prompt,
                max_new_tokens=max(64, min(int(max_new_tokens or 260), 1024)),
                do_sample=False,
                temperature=0.2,
            )
            raw_content = ""
            if isinstance(response, list) and response:
                raw_content = str(response[0].get("generated_text", ""))
            obj = _extract_json_object(raw_content)
        else:
            response = ollama.chat(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={
                    "num_ctx": 8192,
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            )
            raw_content = response["message"]["content"]
            obj = _extract_json_object(raw_content)
    except Exception as e:  # noqa: BLE001
        logger.warning("Falha ao gerar pacote social via IA: %s", e)
        return fallback

    hook = str(obj.get("hook_phrase", "")).strip() or fallback["hook_phrase"]
    description = str(obj.get("description", "")).strip() or fallback["description"]
    try:
        frame_second = float(obj.get("frame_second", fallback["frame_second"]))
    except Exception:  # noqa: BLE001
        frame_second = float(fallback["frame_second"])
    frame_second = max(0.0, min(frame_second, max(0.0, duration - 0.1)))

    return {
        "hook_phrase": hook[:140],
        "description": description,
        "frame_second": round(frame_second, 2),
    }

