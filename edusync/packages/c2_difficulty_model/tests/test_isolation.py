"""C2 must never import the trainer package (the dependency is one-way)."""

from pathlib import Path

C2 = Path(__file__).resolve().parents[2] / "c2"


def test_c2_does_not_reference_the_model_package():
    offenders = [
        str(path.relative_to(C2))
        for path in (C2 / "src").rglob("*.py")
        if "edusync_c2_model" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_c2_does_not_depend_on_the_model_package():
    text = (C2 / "pyproject.toml").read_text(encoding="utf-8")
    assert "edusync-c2-model" not in text
