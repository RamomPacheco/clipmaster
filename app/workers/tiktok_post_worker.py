from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.core.logger import logger
from app.services.tiktok_api import TikTokAPIError, publish_video_file
from app.services.tiktok_queue_store import TikTokScheduleJob


class TikTokPublishWorker(QThread):
    """Executa upload para a TikTok numa thread separada (não bloqueia a UI)."""

    succeeded = Signal(str, str)  # job_id, publish_id
    failed = Signal(str, str)  # job_id, message

    def __init__(self, job: TikTokScheduleJob, access_token: str, parent=None) -> None:
        super().__init__(parent)
        self._job = job
        self._token = access_token.strip()

    def run(self) -> None:  # type: ignore[override]
        jid = self._job.id
        try:
            if not self._token:
                self.failed.emit(jid, "Access token TikTok em falta. Configure na aba Avançado.")
                return
            path = Path(self._job.video_path)
            pid = publish_video_file(
                self._token,
                path,
                title=self._job.title,
                privacy_level=self._job.privacy_level,
            )
            logger.info("TikTok upload OK job=%s publish_id=%s", jid, pid)
            self.succeeded.emit(jid, pid)
        except TikTokAPIError as e:
            logger.warning("TikTok API: %s", e)
            self.failed.emit(jid, str(e))
        except Exception as e:  # noqa: BLE001
            logger.exception("Falha inesperada no upload TikTok.")
            self.failed.emit(jid, str(e))
