"""Pseudonymous id rule shared with ``scripts/export_label_sheet.py --lecture-id``."""

import re

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def is_valid_id(value: str) -> bool:
    """1-32 letters, digits, '-' or '_'; no spaces or path separators."""
    return bool(ID_PATTERN.match(value))
