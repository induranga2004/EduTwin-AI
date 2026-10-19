from fastapi import APIRouter

from edusync_core import Component


def test_prefix():
    comp = Component(name="c1", title="T", description="d", router=APIRouter())
    assert comp.prefix == "/api/c1"
