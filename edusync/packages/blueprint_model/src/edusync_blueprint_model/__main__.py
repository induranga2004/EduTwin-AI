"""Allow ``python -m edusync_blueprint_model``."""

import sys

from edusync_blueprint_model.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
