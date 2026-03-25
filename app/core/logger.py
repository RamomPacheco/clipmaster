import logging
from collections.abc import Callable


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("clipmaster")


logger = configure_logging()


class ForwardingHandler(logging.Handler):
    """
    Envia cada registo já formatado para um callback (ex.: Signal.emit da UI).
    Usado no worker para espelhar logs do terminal no painel da aplicação.
    """

    def __init__(self, forward: Callable[[str], None]) -> None:
        super().__init__()
        self._forward = forward

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._forward(self.format(record))
        except Exception:
            self.handleError(record)

