# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
FastAPI application for the ICU-Guardian Environment.

Exposes the environment over HTTP and WebSocket endpoints,
compatible with OpenEnv client infrastructure.
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv_core.env_server.http_server import create_app

from models import ICUAction, ICUObservation
from server.icu_environment import ICUEnvironment

# Create the app using create_app (passes class for WebSocket session support)
app = create_app(
    ICUEnvironment, ICUAction, ICUObservation, env_name="icu_guardian"
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
