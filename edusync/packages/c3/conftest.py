# blueprint_model/ is C3's own copy of the blueprint. It defines the same module name
# (edusync_blueprint_model) as the workspace package packages/blueprint_model, which is the one
# installed in the environment, so its tests cannot be collected in the workspace-wide run.
# Run them on their own, against the copy's source:
#   cd packages/c3/blueprint_model && PYTHONPATH=$PWD/src uv run --no-sync python -m pytest tests
collect_ignore = ["blueprint_model"]
