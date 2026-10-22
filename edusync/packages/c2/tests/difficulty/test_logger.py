import contextlib
import logging

import pytest

from edusync_c2 import logger as lg


@pytest.fixture
def c2_logger(monkeypatch):
    c2 = logging.getLogger("edusync_c2")
    monkeypatch.setattr(c2, "handlers", [])
    monkeypatch.setattr(c2, "level", c2.level)
    return c2


@contextlib.contextmanager
def bare_root(extra=()):
    """Root logger without handlers (pytest's capture handlers are put back afterwards)."""
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers[:] = list(extra)
    try:
        yield
    finally:
        root.handlers[:] = saved


def test_get_logger_names():
    assert lg.get_logger("edusync_c2.x.y").name == "edusync_c2.x.y"
    assert lg.get_logger("edusync_c2").name == "edusync_c2"
    assert lg.get_logger("other").name == "edusync_c2.other"
    assert lg.get_logger("edusync_c2x").name == "edusync_c2.edusync_c2x"


def test_request_id_helpers():
    assert lg.request_id.get() == "-"
    rid = lg.new_request_id()
    assert len(rid) == 8
    assert rid != lg.new_request_id()
    token = lg.set_request_id("abc")
    assert lg.request_id.get() == "abc"
    lg.reset_request_id(token)
    assert lg.request_id.get() == "-"


def test_configure_is_idempotent(c2_logger):
    with bare_root():
        lg.configure_logging("DEBUG")
        lg.configure_logging("warning")
    assert len(c2_logger.handlers) == 1
    assert c2_logger.level == logging.WARNING
    handler = c2_logger.handlers[0]
    token = lg.set_request_id("rid12345")
    try:
        rec = logging.LogRecord("edusync_c2.t", logging.INFO, "f", 1, "hello a=1", None, None)
        assert handler.filter(rec)
        assert "[rid12345] hello a=1" in handler.format(rec)
        assert "INFO edusync_c2.t" in handler.format(rec)
    finally:
        lg.reset_request_id(token)


def test_no_handler_when_root_has_one(c2_logger):
    with bare_root([logging.NullHandler()]):
        lg.configure_logging("ERROR")
    assert c2_logger.handlers == []
    assert c2_logger.level == logging.ERROR
