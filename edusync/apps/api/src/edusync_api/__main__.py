"""``python -m edusync_api``: run the app with uvicorn on port 8100."""

import os

import uvicorn


def main() -> None:
    """Serve on EDUSYNC_HOST (default 127.0.0.1), port 8100."""
    host = os.environ.get("EDUSYNC_HOST", "127.0.0.1")
    uvicorn.run("edusync_api.asgi:app", host=host, port=8100)


if __name__ == "__main__":
    main()
