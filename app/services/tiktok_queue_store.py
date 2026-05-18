from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core import config
from app.core.logger import logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc_iso(s: str) -> datetime:
    t = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class TikTokScheduleJob:
    id: str
    video_path: str
    title: str
    privacy_level: str
    run_at_iso: str
    status: str  # pending | uploading | done | failed
    created_at_iso: str
    last_error: Optional[str] = None
    publish_id: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> Optional["TikTokScheduleJob"]:
        try:
            return TikTokScheduleJob(
                id=str(obj["id"]),
                video_path=str(obj["video_path"]),
                title=str(obj.get("title", "")),
                privacy_level=str(obj.get("privacy_level", "SELF_ONLY")),
                run_at_iso=str(obj["run_at_iso"]),
                status=str(obj.get("status", "pending")),
                created_at_iso=str(obj.get("created_at_iso", _utc_now_iso())),
                last_error=obj.get("last_error"),
                publish_id=obj.get("publish_id"),
            )
        except (KeyError, TypeError, ValueError):
            return None


class TikTokQueueStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or config.tiktok_schedule_queue_path()
        self._jobs: List[TikTokScheduleJob] = []
        self.load()

    def load(self) -> None:
        self._jobs = []
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            for item in raw.get("jobs", []):
                if isinstance(item, dict):
                    j = TikTokScheduleJob.from_json(item)
                    if j:
                        self._jobs.append(j)
        except Exception as e:  # noqa: BLE001
            logger.warning("Falha ao carregar fila TikTok: %s", e)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "jobs": [j.to_json() for j in self._jobs],
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_jobs(self) -> List[TikTokScheduleJob]:
        return list(self._jobs)

    def add_job(
        self,
        video_path: Path,
        title: str,
        privacy_level: str,
        run_at_iso: str,
    ) -> TikTokScheduleJob:
        job = TikTokScheduleJob(
            id=str(uuid.uuid4()),
            video_path=str(video_path.resolve()),
            title=title.strip() or "ClipMaster",
            privacy_level=privacy_level.strip() or "SELF_ONLY",
            run_at_iso=run_at_iso,
            status="pending",
            created_at_iso=_utc_now_iso(),
        )
        self._jobs.append(job)
        self.save()
        return job

    def claim_next_due(self) -> Optional[TikTokScheduleJob]:
        """Marca o próximo trabalho pendente cuja hora já passou como *uploading* e devolve-o."""
        now = datetime.now(timezone.utc)
        self.load()
        for j in self._jobs:
            if j.status != "pending":
                continue
            try:
                due = parse_utc_iso(j.run_at_iso)
            except ValueError:
                j.status = "failed"
                j.last_error = "Data/hora de agendamento inválida."
                self.save()
                continue
            if due <= now:
                j.status = "uploading"
                self.save()
                return j
        return None

    def mark_done(self, job_id: str, publish_id: str) -> None:
        for j in self._jobs:
            if j.id == job_id:
                j.status = "done"
                j.publish_id = publish_id
                j.last_error = None
                break
        self.save()

    def mark_failed(self, job_id: str, message: str) -> None:
        for j in self._jobs:
            if j.id == job_id:
                j.status = "failed"
                j.last_error = message[:2000]
                break
        self.save()

    def reset_stuck_uploading(self) -> None:
        """Devolve trabalhos *uploading* a *pending* (ex.: após encerrar a app a meio)."""
        changed = False
        for j in self._jobs:
            if j.status == "uploading":
                j.status = "pending"
                changed = True
        if changed:
            self.save()
