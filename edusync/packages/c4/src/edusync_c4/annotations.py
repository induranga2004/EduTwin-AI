"""Extract text from BBB/tldraw whiteboard shape JSON (best effort)."""

import json
from typing import Any


def _collect(node: Any, parts: list[str]) -> None:
    """Walk nested rich-text/document content, collecting 'text' string leaves."""
    if isinstance(node, str):
        parts.append(node)
    elif isinstance(node, list):
        for child in node:
            _collect(child, parts)
    elif isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, str):
            parts.append(text)
        _collect(node.get("content"), parts)


def parse_annotation_text(annotation_info: dict | str | None) -> str | None:
    """Return the text of a shape, else None.

    Inspects ``props.text``, ``props.name``, ``props.questionText``, or nested ``props.richText``.
    Accepts a dict or a JSON string; malformed input yields None.
    """
    data = annotation_info
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    props = data.get("props")
    if not isinstance(props, dict):
        return None
    text = props.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    for q_key in ("questionText", "question"):
        q = props.get(q_key)
        if isinstance(q, str) and q.strip():
            return q.strip()
    for key in ("richText", "document"):
        parts: list[str] = []
        _collect(props.get(key), parts)
        joined = "".join(parts).strip()
        if joined:
            return joined
    return None
