"""
Dense, bounded reward calculator for the Smart Home Energy Manager.

Reward bounds:
  1.00  — demand met entirely from solar/battery (zero grid cost)
  0.00  — invalid action OR maximum penalty for high grid cost
"""

from .models import Reward


def calculate_reward(cost: float, is_invalid: bool) -> Reward:
    """
    Calculate a bounded reward for a single step.

    Args:
        cost: The net monetary cost incurred this step.
              Negative cost means the agent earned money (sold to grid).
        is_invalid: Whether the action was physically impossible.

    Returns:
        A Reward object with value in [0.0, 1.0].
    """
    if is_invalid:
        return Reward(
            value=0.0,
            is_invalid=True,
            description="Invalid action attempted (e.g., discharge empty battery)"
        )

    if cost <= 0.0:
        # Zero or negative cost = best outcome (met demand from solar/battery, or profited)
        return Reward(
            value=1.0,
            is_invalid=False,
            description="Demand met without grid cost, or profit from selling"
        )
    else:
        # Positive cost = penalty scaled by magnitude
        # Map: cost in (0, ∞) → reward in (0.0, 1.0)
        # Using: reward = max(0.0, 1.0 - cost * 0.5)
        reward_val = max(0.0, 1.0 - cost * 0.5)
        return Reward(
            value=round(reward_val, 4),
            is_invalid=False,
            description=f"Grid cost incurred: ${cost:.2f}"
        )
