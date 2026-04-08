"""
Dense, bounded reward calculator for the Smart Home Energy Manager.

Reward bounds:
  ~1.0  — demand met entirely from solar/battery (zero grid cost)
  ~0.0  — invalid action OR maximum penalty for high grid cost

Note: Values are clamped to the OPEN interval (0.0001, 0.9999) so that
the OpenEnv validator's strict 0 < score < 1 constraint is always satisfied.
"""

from .models import Reward

# Strict bounds required by the OpenEnv validator (exclusive of 0.0 and 1.0)
_MIN_REWARD = 0.01
_MAX_REWARD = 0.99


def _clamp(value: float) -> float:
    """Clamp reward to the open interval (_MIN_REWARD, _MAX_REWARD)."""
    return max(_MIN_REWARD, min(_MAX_REWARD, value))


def calculate_reward(cost: float, is_invalid: bool) -> Reward:
    """
    Calculate a bounded reward for a single step.

    Args:
        cost: The net monetary cost incurred this step.
              Negative cost means the agent earned money (sold to grid).
        is_invalid: Whether the action was physically impossible.

    Returns:
        A Reward object with value strictly in (0.0, 1.0).
    """
    if is_invalid:
        return Reward(
            value=_MIN_REWARD,
            is_invalid=True,
            description="Invalid action attempted (e.g., discharge empty battery)"
        )

    if cost <= 0.0:
        # Zero or negative cost = best outcome (met demand from solar/battery, or profited)
        return Reward(
            value=_MAX_REWARD,
            is_invalid=False,
            description="Demand met without grid cost, or profit from selling"
        )
    else:
        # Positive cost = penalty scaled by magnitude
        # Map: cost in (0, ∞) → reward in (_MIN_REWARD, _MAX_REWARD)
        reward_val = _clamp(round(1.0 - cost * 0.5, 4))
        return Reward(
            value=reward_val,
            is_invalid=False,
            description=f"Grid cost incurred: ${cost:.2f}"
        )
