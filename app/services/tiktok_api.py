from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

from app.core.logger import logger

TIKTOK_API_BASE = "https://open.tiktokapis.com"

CHUNK_SIZE_DEFAULT = 10 * 1024 * 1024  # 10 MiB (documentação oficial)


class TikTokAPIError(Exception):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _parse_error_payload(data: Dict[str, Any]) -> Tuple[str, str | None]:
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message", "Erro TikTok")), str(err.get("code"))
    return "Resposta inválida da API TikTok.", None


def query_creator_info(client: httpx.Client, access_token: str) -> Dict[str, Any]:
    r = client.post(
        f"{TIKTOK_API_BASE}/v2/post/publish/creator_info/query/",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={},
        timeout=120.0,
    )
    try:
        data = r.json()
    except Exception as e:  # noqa: BLE001
        raise TikTokAPIError(f"Resposta não-JSON (HTTP {r.status_code}): {e}") from e
    if r.status_code >= 400:
        msg, code = _parse_error_payload(data)
        raise TikTokAPIError(msg or f"HTTP {r.status_code}", code)
    err = data.get("error")
    if isinstance(err, dict) and err.get("code") and err.get("code") != "ok":
        msg, code = _parse_error_payload(data)
        raise TikTokAPIError(msg, code)
    inner = data.get("data")
    if not isinstance(inner, dict):
        raise TikTokAPIError("Resposta sem data.creator_info.")
    return inner


def pick_privacy_level(preferred: str, options: List[str]) -> str:
    if preferred in options:
        return preferred
    for fb in ("SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS", "PUBLIC_TO_EVERYONE", "FOLLOWER_OF_CREATOR"):
        if fb in options:
            return fb
    return options[0] if options else "SELF_ONLY"


def init_video_upload(
    client: httpx.Client,
    access_token: str,
    video_path: Path,
    title: str,
    privacy_level: str,
) -> Tuple[str, str, int, int, int]:
    """
    Devolve ``publish_id``, ``upload_url``, ``video_size``, ``chunk_size``, ``total_chunk_count``.
    """
    path = video_path.resolve()
    if not path.is_file():
        raise TikTokAPIError(f"Ficheiro não encontrado: {path}")

    video_size = path.stat().st_size
    if video_size <= 0:
        raise TikTokAPIError("Ficheiro de vídeo vazio.")

    chunk_size = min(CHUNK_SIZE_DEFAULT, video_size)
    total_chunks = max(1, math.ceil(video_size / chunk_size))

    creator = query_creator_info(client, access_token)
    raw_opts = creator.get("privacy_level_options") or []
    options = [str(x) for x in raw_opts] if isinstance(raw_opts, list) else []
    privacy = pick_privacy_level(privacy_level, options)

    max_dur = creator.get("max_video_post_duration_sec")
    if isinstance(max_dur, int) and max_dur > 0:
        # Duração exata exigiria ffprobe — o utilizador deve respeitar limites do TikTok.
        logger.info("TikTok max_video_post_duration_sec=%s", max_dur)

    title_safe = (title or "ClipMaster")[:2200]

    body: Dict[str, Any] = {
        "post_info": {
            "title": title_safe,
            "privacy_level": privacy,
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": video_size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunks,
        },
    }

    r = client.post(
        f"{TIKTOK_API_BASE}/v2/post/publish/video/init/",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json=body,
        timeout=120.0,
    )
    try:
        data = r.json()
    except Exception as e:  # noqa: BLE001
        raise TikTokAPIError(f"Resposta não-JSON (HTTP {r.status_code}): {e}") from e
    if r.status_code >= 400:
        msg, code = _parse_error_payload(data)
        raise TikTokAPIError(msg or f"HTTP {r.status_code}", code)
    err = data.get("error")
    if isinstance(err, dict) and err.get("code") and err.get("code") != "ok":
        msg, code = _parse_error_payload(data)
        raise TikTokAPIError(msg, code)
    inner = data.get("data")
    if not isinstance(inner, dict):
        raise TikTokAPIError("Resposta sem data no init de vídeo.")
    publish_id = str(inner.get("publish_id", "")).strip()
    upload_url = str(inner.get("upload_url", "")).strip()
    if not publish_id or not upload_url:
        raise TikTokAPIError("Resposta sem publish_id ou upload_url.")
    return publish_id, upload_url, video_size, chunk_size, total_chunks


def put_video_chunks(
    client: httpx.Client,
    upload_url: str,
    video_path: Path,
    video_size: int,
    chunk_size: int,
    total_chunks: int,
) -> None:
    path = video_path.resolve()
    with path.open("rb") as f:
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, video_size)
            length = end - start
            f.seek(start)
            chunk = f.read(length)
            if len(chunk) != length:
                raise TikTokAPIError("Leitura incompleta do ficheiro de vídeo.")
            content_range = f"bytes {start}-{end - 1}/{video_size}"
            r = client.put(
                upload_url,
                content=chunk,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(length),
                    "Content-Range": content_range,
                },
                timeout=600.0,
            )
            if r.status_code >= 400:
                raise TikTokAPIError(
                    f"Upload falhou no segmento {i + 1}/{total_chunks} (HTTP {r.status_code}).",
                    None,
                )


def publish_video_file(access_token: str, video_path: Path, title: str, privacy_level: str) -> str:
    """
    Inicializa upload, envia chunks e devolve ``publish_id``.
    O processamento no TikTok é assíncrono; use o endpoint de estado se precisar de confirmar.
    """
    vp = video_path.resolve()
    with httpx.Client() as client:
        publish_id, upload_url, vsize, csize, nchunks = init_video_upload(
            client, access_token, vp, title, privacy_level
        )
        put_video_chunks(client, upload_url, vp, vsize, csize, nchunks)
    return publish_id
