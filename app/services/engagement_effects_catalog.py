"""
Catálogo de efeitos de retenção / engajamento para clipes verticais.

- IDs **implementados** no segundo passe MoviePy (quando a opção está ligada na UI).
- Lista **roadmap**: sugestões editoriais para a IA conhecer; ainda não aplicadas pelo encoder.
"""

from __future__ import annotations

from typing import Any, List, Sequence

# --- Pipeline MoviePy (video_engine + moviepy_engagement) ---
IMPLEMENTED_ENGAGEMENT_EFFECT_IDS = frozenset({"push_in_subtle", "fade_video_edges"})
EFFECT_NONE = "none"
DEFAULT_ENGAGEMENT_STACK: List[str] = ["push_in_subtle", "fade_video_edges"]

# Descrições para prompts e documentação (implementados)
ENGAGEMENT_EFFECTS_IMPLEMENTED: List[dict[str, str | bool]] = [
    {
        "id": EFFECT_NONE,
        "label": "Sem MoviePy neste clipe",
        "description": (
            "Não executar o segundo passe MoviePy; mantém só o vídeo já gerado pelo FFmpeg."
        ),
        "implemented": True,
    },
    {
        "id": "push_in_subtle",
        "label": "Push-in (Ken Burns discreto)",
        "description": (
            "Zoom lento e centrado ao longo do clipe (~3,8% no total), típico de retenção em Shorts."
        ),
        "implemented": True,
    },
    {
        "id": "fade_video_edges",
        "label": "Transição fade (vídeo)",
        "description": (
            "Dissolver suave no início e no fim apenas no vídeo; áudio permanece linear (sem duck/fade)."
        ),
        "implemented": True,
    },
]

# Sugestões para a IA (planejamento / copy / futuras features — não alteram o encode hoje)
ENGAGEMENT_EFFECTS_ROADMAP: List[dict[str, str]] = [
    {
        "id": "speed_ramp_intro",
        "label": "Acelerar leve o gancho inicial",
        "description": "Micro–speed ramp nos primeiros ~1–2s para densificar o hook (não implementado no encoder).",
    },
    {
        "id": "zoom_punch_emphasis",
        "label": "Punch de zoom em palavra-chave",
        "description": "Corte ou zoom rápido alinhado a um pico de fala (não implementado).",
    },
    {
        "id": "flash_subtle",
        "label": "Flash branco discreto",
        "description": "Um frame ou curva rápida para reengajar atenção em virada de ideia (não implementado).",
    },
    {
        "id": "stutter_replay",
        "label": "Replay / stutter cômico",
        "description": "Repetir 0,3–0,8s para ênfase ou humor (não implementado).",
    },
    {
        "id": "b_roll_placeholder",
        "label": "Sugestão de B-roll",
        "description": "Indicar no texto do clipe onde cobrir com imagem externa (não implementado).",
    },
    {
        "id": "caption_keyword_pop",
        "label": "Destaque tipográfico em palavras-chave",
        "description": "Complementar legendas TikTok com ênfase em termos de alta carga emocional (parcialmente via estilo ASS).",
    },
]


def normalize_engagement_effect_ids(raw: Any) -> List[str]:
    """Normaliza entrada da IA ou da UI para lista de IDs em minúsculas."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip().lower().replace(" ", "_")
        return [s] if s else []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: List[str] = []
        for x in raw:
            if x is None:
                continue
            s = str(x).strip().lower().replace(" ", "_")
            if s:
                out.append(s)
        return out
    return []


def resolve_moviepy_effect_ids(raw: Any) -> List[str] | None:
    """
    Converte o pedido da IA no conjunto de efeitos a aplicar no MoviePy.

    Retorna:
    - ``None``: não aplicar MoviePy neste clipe (``none`` explícito).
    - ``[]``: tratar como inválido → usar ``DEFAULT_ENGAGEMENT_STACK``.
    - lista não vazia: apenas IDs implementados; se ficar vazio após filtrar, usa o default.
    """
    ids = normalize_engagement_effect_ids(raw)
    if EFFECT_NONE in ids:
        return None
    if not ids:
        return list(DEFAULT_ENGAGEMENT_STACK)
    impl = [i for i in ids if i in IMPLEMENTED_ENGAGEMENT_EFFECT_IDS]
    if not impl:
        return list(DEFAULT_ENGAGEMENT_STACK)
    return impl


def engagement_effects_json_example_suffix() -> str:
    """Trecho a colar no exemplo de objeto JSON dos prompts da IA."""
    return ', "engagement_effects": ["push_in_subtle", "fade_video_edges"]'


def engagement_effects_prompt_instruction() -> str:
    """Bloco de instruções para o modelo escolher efeitos por clipe."""
    lines = [
        "",
        "EFEITOS DE RETENÇÃO (opcional, por clipe): inclua o campo "
        '"engagement_effects" como array de strings.',
        "Valores permitidos no pipeline de exportação:",
        f'  • "{EFFECT_NONE}" — não aplicar o segundo passe MoviePy neste clipe.',
        '  • "push_in_subtle" — zoom lento centrado (Ken Burns discreto).',
        '  • "fade_video_edges" — fade de entrada/saída só no vídeo (áudio intacto).',
        "Pode combinar os dois últimos (ex.: [\"push_in_subtle\", \"fade_video_edges\"]).",
        f'Se omitir o campo ou enviar [], usa-se o pacote padrão: {DEFAULT_ENGAGEMENT_STACK}.',
        "",
        "Outras ideias editoriais (speed ramp, punch zoom, flash, stutter, B-roll) podem ser "
        "mencionadas em \"reason\" ou \"headline\"; o app ainda não as aplica automaticamente no vídeo.",
    ]
    return "\n".join(lines)
