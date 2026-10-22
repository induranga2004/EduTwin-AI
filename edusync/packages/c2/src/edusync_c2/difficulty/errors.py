"""Errors of the model layer, shared by the model factory and the HTTP layer."""


class ModelArtifactError(Exception):
    """A trained-model artifact folder is missing files or does not match the contract."""


class ModelNotAvailableError(Exception):
    """The selected model cannot be used (unknown name, or not implemented yet)."""
