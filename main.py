"""
FastAPI server for the Smart Home Energy Manager environment.
Exposes /reset, /step, /state, and /health endpoints for the OpenEnv framework.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.environment import SmartHomeEnv
from env.models import Action, Observation, StepResponse, State

app = FastAPI(
    title="Smart Home Energy Manager",
    description="RL environment for optimizing household energy consumption with solar panels and battery storage.",
    version="1.0.0",
)

# Single environment instance
env = SmartHomeEnv()


class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task: str


@app.get("/health")
def health():
    """Health check endpoint required by OpenEnv runtime."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest):
    """
    Reset the environment for a new episode.
    Accepts a task name: "easy", "medium", or "hard".
    Returns the initial Observation.
    """
    try:
        obs = env.reset(req.task)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    """
    Execute one time step with the given action.
    Returns StepResponse with observation, reward, done, and info.
    """
    if env.episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is finished. Call POST /reset to start a new one."
        )
    resp = env.step(action)
    return resp


@app.get("/state", response_model=State)
def get_state():
    """
    Get the current environment state with session metadata.
    Returns State model with episode_id, task_name, step_count, is_done, and observation.
    """
    return env.state()
