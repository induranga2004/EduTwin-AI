"""Errors of the model layer, shared by the model factory and the HTTP layer."""


class ModelArtifactError(Exception):
    """A trained-model artifact folder is missing files or does not match the contract."""


class ModelNotAvailableError(Exception):
    """The selected model cannot be used (unknown name, or not implemented yet)."""


class GraphProviderError(Exception):
    """The C1 graph could not be obtained, carrying the HTTP status the API should answer with.

    502 upstream failure or invalid response, 504 timeout, 422 C1 rejected the file, 500 a local
    graph file that cannot be loaded.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
