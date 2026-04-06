"""
ICU-Guardian: FastAPI Server

Creates the OpenEnv-compliant FastAPI application that exposes
/reset, /step, and /state endpoints.

Compatible with both openenv_core (when available) and standalone mode.
"""

import os
import sys

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dataclasses import asdict

from models import ICUAction, ICUObservation, ICUState
from server.icu_environment import ICUEnvironment

# Task can be configured via environment variable
task_name = os.getenv("ICU_TASK", "vital_stabilization")

# Create the environment
env = ICUEnvironment(task_name=task_name)

app = FastAPI(
    title="ICU-Guardian Environment",
    description="OpenEnv-compliant ICU patient monitoring environment",
    version="1.0.0",
)


@app.post("/reset")
async def reset(request: Request = None):
    """Initialize a new patient episode."""
    # Accept optional task name in body
    body = {}
    try:
        if request:
            body = await request.json()
    except Exception:
        body = {}

    # Allow task switching via body
    if "task" in body:
        env.task_name = body["task"]

    obs = env.reset()
    return JSONResponse(content={
        "observation": asdict(obs),
        "reward": 0.0,
        "done": False,
    })


@app.post("/step")
async def step(request: Request):
    """Execute a clinical action on the patient."""
    try:
        action_data = await request.json()
    except Exception:
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
    """Return current episode metadata."""
    s = env.state()
    return JSONResponse(content=asdict(s))


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "environment": "icu_guardian", "version": "1.0.0"}


@app.get("/info")
async def info():
    """Return environment info including available tasks."""
    from server.tasks import TASKS
    return {
        "name": "icu_guardian",
        "version": "1.0.0",
        "tasks": {
            name: {
                "description": t.description,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
            }
            for name, t in TASKS.items()
        },
        "action_space": [
            "administer_meds(vasopressor, low|high)",
            "administer_meds(antihypertensive, low|high)",
            "adjust_oxygen(increase|decrease)",
            "trigger_code_sepsis()",
            "wait()",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
