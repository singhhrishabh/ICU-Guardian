# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
FastAPI application for the ICU-Guardian Environment.

Provides HTTP endpoints compatible with OpenEnv HTTPEnvClient:
- POST /reset — Initialize new episode
- POST /step — Execute action
- GET /state — Episode metadata
- GET /health — Health check
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ICUAction, ICUObservation
from server.icu_environment import ICUEnvironment

# Create environment instance
task_name = os.getenv("ICU_TASK", "vital_stabilization")
env = ICUEnvironment(task_name=task_name)

app = FastAPI(
    title="OpenEnv Environment HTTP API",
    description="ICU-Guardian: OpenEnv-compliant ICU patient monitoring environment",
    version="1.0.0",
)

# Serve static files for the web dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serialize_observation(obs: ICUObservation) -> Dict[str, Any]:
    """Convert ICUObservation to HTTPEnvClient-compatible format."""
    obs_dict = asdict(obs)
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    obs_dict.pop("metadata", None)
    return {"observation": obs_dict, "reward": reward, "done": done}


@app.post("/reset")
async def reset(request: Dict[str, Any] = Body(default={})):
    """Reset the environment to initial state."""
    obs = env.reset(
        seed=request.get("seed"),
        episode_id=request.get("episode_id"),
    )
    return serialize_observation(obs)


@app.post("/step")
async def step(request: Dict[str, Any] = Body(...)):
    """Execute an action in the environment."""
    action_data = request.get("action", {})
    if isinstance(action_data, str):
        action_data = {"action": action_data}
    metadata = action_data.pop("metadata", {})
    action = ICUAction(**action_data)
    action.metadata = metadata
    obs = env.step(action)
    return serialize_observation(obs)


@app.get("/state")
async def get_state():
    """Return current environment state."""
    return asdict(env.state)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/schema")
async def schema():
    """Return environment action/observation schema."""
    return {
        "action_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["administer_meds", "adjust_oxygen", "trigger_code_sepsis", "wait"]},
                "drug": {"type": "string", "enum": ["vasopressor", "antihypertensive"], "nullable": True},
                "dose": {"type": "string", "enum": ["low", "high"], "nullable": True},
                "level": {"type": "string", "enum": ["increase", "decrease"], "nullable": True},
            },
            "required": ["action"],
        },
        "observation_schema": {
            "type": "object",
            "properties": {
                "HR": {"type": "integer"},
                "BP_sys": {"type": "integer"},
                "BP_dia": {"type": "integer"},
                "SpO2": {"type": "integer"},
                "Temp": {"type": "number"},
                "trend": {"type": "string"},
                "step_number": {"type": "integer"},
                "task_name": {"type": "string"},
            },
        },
    }


@app.get("/")
async def root():
    """Serve the web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return {
        "status": "ok",
        "environment": "icu_guardian",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/schema", "/docs"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
