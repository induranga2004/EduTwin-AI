"""The two concept-id conventions, in one place.

* ``casefold_id``: the id C1 v0 graphs, the mock provider and the fallback path use. Casefolded
  label with whitespace runs collapsed to one space (``"Loss  Function"`` -> ``"loss function"``).
* ``slug_id``: the id C1 v1 graphs use (contract rule C1-018). Lowercase ASCII letters and digits
  joined by single hyphens (``"TCP/IP"`` -> ``"tcp-ip"``).
"""

import re

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def casefold_id(text: str) -> str:
    """Casefold, collapse whitespace and strip (the id rule of C1 v0 and the mock provider)."""
    return " ".join(text.casefold().split())


def slug_id(text: str) -> str:
    """Lowercase, runs of anything but ``[a-z0-9]`` become ``-``, no leading or trailing ``-``."""
    return _NON_SLUG.sub("-", text.casefold()).strip("-")
