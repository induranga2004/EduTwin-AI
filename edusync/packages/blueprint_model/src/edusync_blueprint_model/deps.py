"""Lazy imports of optional dependencies with a helpful install hint."""

import importlib
import importlib.util
from types import ModuleType

INSTALL_CMD = "uv sync --inexact --package edusync-blueprint-model --extra {extra}"


class MissingExtraError(ImportError):
    """An optional dependency group is not installed."""


def install_command(extra: str) -> str:
    """The command that installs the given extra."""
    return INSTALL_CMD.format(extra=extra)


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise ``MissingExtraError`` naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"The '{module}' package is required for this step. Install it with: "
            f"{install_command(extra)}"
        ) from exc


def is_available(module: str) -> bool:
    """True if ``module`` can be found (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False
