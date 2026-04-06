"""
ICU-Guardian: FastAPI Server

Creates the OpenEnv-compliant FastAPI application that exposes
/reset, /step, and /state endpoints.

When running in a Docker container with Python 3.11+, this uses the
official create_fastapi_app helper. For local development, it falls
back to a manual FastAPI setup.
"""

import os
import sys

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ICUAction, ICUObservation, ICUState
from server.icu_environment import ICUEnvironment

# Task can be configured via environment variable
task_name = os.getenv("ICU_TASK", "vital_stabilization")

# Create the environment
env = ICUEnvironment(task_name=task_name)

# Try to use official OpenEnv helper, fall back to manual FastAPI
try:
    from openenv_core.env_server import create_fastapi_app
    app = create_fastapi_app(env, ICUAction, ICUObservation)
except (ImportError, TypeError):
    # Manual FastAPI setup for local development / Python 3.9
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from dataclasses import asdict
    import json

    app = FastAPI(
        title="ICU-Guardian Environment",
        description="OpenEnv-compliant ICU patient monitoring environment",
        version="1.0.0",
    )

    @app.post("/reset")
    async def reset():
        obs = env.reset()
        return JSONResponse(content={
            "observation": asdict(obs),
            "reward": 0.0,
            "done": False,
        })

    @app.post("/step")
    async def step(action_data: dict = None):
        if action_data is None:
            action_data = {}
        action = ICUAction(
            action=action_data.get("action", "wait"),
            drug=action_data.get("drug"),
            dose=action_data.get("dose"),
            level=action_data.get("level"),
        )
        result = env.step(action)
        obs = result["observation"]
        return JSONResponse(content={
            "observation": asdict(obs),
            "reward": result["reward"],
            "done": result["done"],
            "score": result.get("score"),
        })

    @app.get("/state")
    async def state():
        s = env.state()
        return JSONResponse(content=asdict(s))

    @app.get("/")
    async def root():
        return {"status": "ok", "environment": "icu_guardian", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
