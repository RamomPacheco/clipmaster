from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from app.core import config
from app.core.logger import logger


@dataclass
class TikTokCredentials:
    """Dados da app TikTok for Developers + token do utilizador (OAuth)."""

    client_key: str = ""
    client_secret: str = ""
    access_token: str = ""
    refresh_token: str = ""

    def has_upload_token(self) -> bool:
        return bool(self.access_token.strip())


class TikTokCredentialsStore:
    """Persistência simples em JSON (texto local — não partilhe o ficheiro)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or config.tiktok_credentials_path()
        self._data = TikTokCredentials()
        self.load()

    @property
    def path(self) -> Path:
        return self._path

    def get(self) -> TikTokCredentials:
        return self._data

    def load(self) -> None:
        if not self._path.is_file():
            self._data = TikTokCredentials()
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                self._data = TikTokCredentials()
                return
            self._data = TikTokCredentials(
                client_key=str(raw.get("client_key", "") or ""),
                client_secret=str(raw.get("client_secret", "") or ""),
                access_token=str(raw.get("access_token", "") or ""),
                refresh_token=str(raw.get("refresh_token", "") or ""),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Falha ao carregar credenciais TikTok: %s", e)
            self._data = TikTokCredentials()

    def save(self, creds: Optional[TikTokCredentials] = None) -> None:
        if creds is not None:
            self._data = creds
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "version": 1,
            **asdict(self._data),
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
