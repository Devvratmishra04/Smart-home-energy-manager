"""
Pydantic models for the Smart Home Energy Manager environment.
Strictly typed schemas for Observation, Action, Reward, and StepResponse.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Literal


class Observation(BaseModel):
    """What the agent sees at each time step."""
    hour_of_day: int = Field(..., ge=0, le=23, description="Current hour (0-23)")
    battery_soc: float = Field(..., ge=0.0, le=1.0, description="Battery state of charge (0.0-1.0)")
    current_demand_kw: float = Field(..., ge=0.0, description="Household power demand in kW")
    solar_generation_kw: float = Field(..., ge=0.0, description="Solar power generated in kW")
    grid_price_per_kwh: float = Field(..., ge=0.0, description="Current electricity price $/kWh")


class Action(BaseModel):
    """Semantic action the agent can take."""
    action_type: Literal[
        "charge_from_grid",
        "discharge_to_home",
        "sell_to_grid",
        "idle"
    ] = Field(..., description="The action to perform this step")


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(..., ge=0.0, le=1.0, description="Reward value bounded [0.0, 1.0]")
    is_invalid: bool = Field(default=False, description="Whether the action was invalid")
    description: str = Field(default="", description="Human-readable explanation of the reward")


class StepResponse(BaseModel):
    """Full response from a single environment step."""
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra info (final_score, total_cost on done)")


class State(BaseModel):
    """Environment session metadata — returned by GET /state."""
    episode_id: str = Field(..., description="Identifier for the current episode")
    task_name: str = Field(..., description="Active task name")
    step_count: int = Field(..., ge=0, description="Steps taken this episode")
    is_done: bool = Field(..., description="Whether current episode has ended")
    observation: Observation = Field(..., description="Current agent observation")
