# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
FastAPI application for the ICU-Guardian Environment.

Uses openenv_core's create_fastapi_app helper to expose the environment
over HTTP endpoints compatible with OpenEnv clients.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ICUAction, ICUObservation
from server.icu_environment import ICUEnvironment

# Create environment instance
task_name = os.getenv("ICU_TASK", "vital_stabilization")
env = ICUEnvironment(task_name=task_name)

# Try using OpenEnv's create_app, fall back to create_fastapi_app, fall back to manual
try:
    from openenv_core.env_server.http_server import create_fastapi_app
    app = create_fastapi_app(env, ICUAction, ICUObservation)
except ImportError:
    # Manual FastAPI setup
    from fastapi import FastAPI, Body
    from fastapi.responses import JSONResponse
    from dataclasses import asdict
    from typing import Any, Dict

    app = FastAPI(
        title="ICU-Guardian Environment",
        description="OpenEnv-compliant ICU patient monitoring environment",
        version="1.0.0",
    )

    @app.post("/reset")
    async def reset(request: Dict[str, Any] = Body(default={})):
        obs = env.reset(
            seed=request.get("seed"),
            episode_id=request.get("episode_id"),
        )
        obs_dict = asdict(obs)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)
        obs_dict.pop("metadata", None)
        return {"observation": obs_dict, "reward": reward, "done": done}

    @app.post("/step")
    async def step(request: Dict[str, Any] = Body(...)):
        action_data = request.get("action", {})
        metadata = action_data.pop("metadata", {})
        action = ICUAction(**action_data)
        action.metadata = metadata
        obs = env.step(action)
        obs_dict = asdict(obs)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)
        obs_dict.pop("metadata", None)
        return {"observation": obs_dict, "reward": reward, "done": done}

    @app.get("/state")
    async def get_state():
        return asdict(env.state)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/")
    async def root():
        return {"status": "ok", "environment": "icu_guardian", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
