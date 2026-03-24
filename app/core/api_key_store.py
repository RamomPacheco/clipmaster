from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.logger import logger


@dataclass
class ApiKeyProfile:
    id: str
    label: str
    provider: str  # "gemini" | "groq"
    secret: str

    def to_json(self) -> Dict[str, str]:
        return {"id": self.id, "label": self.label, "provider": self.provider, "secret": self.secret}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> Optional["ApiKeyProfile"]:
        try:
            return ApiKeyProfile(
                id=str(obj["id"]),
                label=str(obj.get("label", "")).strip() or "Sem nome",
                provider=str(obj.get("provider", "")).strip().lower(),
                secret=str(obj.get("secret", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _default_store_path() -> Path:
    from app.core import config

    return config.api_keys_storage_path()


class ApiKeyStore:
    """Chaves API nomeadas persistidas em JSON (texto simples — evite partilhar o ficheiro)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _default_store_path()
        self._profiles: List[ApiKeyProfile] = []
        self._last_by_provider: Dict[str, str] = {}
        self.load()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        self._profiles = []
        self._last_by_provider = {}
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            for item in raw.get("profiles", []):
                if isinstance(item, dict):
                    p = ApiKeyProfile.from_json(item)
                    if p and p.provider in ("gemini", "groq") and p.secret.strip():
                        self._profiles.append(p)
            legacy = raw.get("last_profile_by_provider")
            if isinstance(legacy, dict):
                for k, v in legacy.items():
                    if isinstance(k, str) and isinstance(v, str) and self.get(v):
                        self._last_by_provider[k.strip().lower()] = v
            single = raw.get("last_profile_id")
            if isinstance(single, str) and single.strip():
                prof = self.get(single.strip())
                if prof and prof.provider not in self._last_by_provider:
                    self._last_by_provider[prof.provider] = prof.id
        except Exception as e:  # noqa: BLE001
            logger.warning("Falha ao carregar chaves API de %s: %s", self._path, e)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "profiles": [p.to_json() for p in self._profiles],
            "last_profile_by_provider": dict(self._last_by_provider),
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_all(self) -> List[ApiKeyProfile]:
        return sorted(self._profiles, key=lambda p: (p.provider, p.label.lower()))

    def list_for_provider(self, provider: str) -> List[ApiKeyProfile]:
        p = provider.strip().lower()
        if p not in ("gemini", "groq"):
            return []
        return sorted([x for x in self._profiles if x.provider == p], key=lambda x: x.label.lower())

    def get(self, profile_id: str) -> Optional[ApiKeyProfile]:
        for prof in self._profiles:
            if prof.id == profile_id:
                return prof
        return None

    def add(self, label: str, provider: str, secret: str) -> ApiKeyProfile:
        prof = ApiKeyProfile(
            id=str(uuid.uuid4()),
            label=(label or "").strip() or f"Chave API {provider}",
            provider=provider.strip().lower(),
            secret=secret.strip(),
        )
        if prof.provider not in ("gemini", "groq"):
            raise ValueError("provider deve ser gemini ou groq")
        if not prof.secret:
            raise ValueError("chave vazia")
        self._profiles.append(prof)
        self._last_by_provider[prof.provider] = prof.id
        self.save()
        return prof

    def remove(self, profile_id: str) -> bool:
        before = len(self._profiles)
        self._profiles = [p for p in self._profiles if p.id != profile_id]
        for prov, lid in list(self._last_by_provider.items()):
            if lid == profile_id:
                del self._last_by_provider[prov]
        if len(self._profiles) < before:
            self.save()
            return True
        return False

    def set_last_for_provider(self, provider: str, profile_id: Optional[str]) -> None:
        p = provider.strip().lower()
        if p not in ("gemini", "groq"):
            return
        if profile_id:
            self._last_by_provider[p] = profile_id
        else:
            self._last_by_provider.pop(p, None)
        self.save()

    def last_profile_id(self, provider: str) -> Optional[str]:
        p = provider.strip().lower()
        lid = self._last_by_provider.get(p)
        if lid and self.get(lid):
            return lid
        return None
