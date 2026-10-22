"""Logging for C2.

Conventions:
- one ``key=value`` line per pipeline step, event name first (``materials``, ``graph`` ...);
- never log slide text, notes or file names (the file type is enough);
- concept ids and labels only at DEBUG;
- timings appear only in logs, never in API responses.

``configure_logging`` is never called at import time; the router calls it once on startup of
the cached pipeline. It never hijacks an existing logging setup.
"""

import contextvars
import logging
import uuid

ROOT_NAME = "edusync_c2"
FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"

request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "edusync_c2_request_id", default="-"
)

_HANDLER_MARK = "_edusync_c2_handler"


def set_request_id(value: str) -> contextvars.Token[str]:
    return request_id.set(value)


def reset_request_id(token: contextvars.Token[str]) -> None:
    request_id.reset(token)


def new_request_id() -> str:
    """Short id for correlating log lines; used in logs only."""
    return uuid.uuid4().hex[:8]


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def get_logger(name: str) -> logging.Logger:
    if name == ROOT_NAME or name.startswith(ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def configure_logging(level: str = "INFO") -> None:
    """Set the C2 log level and, if the root logger has no handlers, attach one handler."""
    log = logging.getLogger(ROOT_NAME)
    log.setLevel(level.upper())
    if logging.getLogger().handlers:
        return
    if any(getattr(h, _HANDLER_MARK, False) for h in log.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(RequestIdFilter())
    setattr(handler, _HANDLER_MARK, True)
    log.addHandler(handler)
