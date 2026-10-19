"""Contract every EduSync component package (c1..c4) exposes to the API app."""

from dataclasses import dataclass

from fastapi import APIRouter


@dataclass(frozen=True)
class Component:
    """A mountable EduSync component.

    Each ``edusync_cN`` package exports ``COMPONENT = Component(...)``; the API app
    mounts ``router`` under ``prefix`` (``/api/<name>``).
    """

    name: str
    title: str
    description: str
    router: APIRouter

    @property
    def prefix(self) -> str:
        return f"/api/{self.name}"
