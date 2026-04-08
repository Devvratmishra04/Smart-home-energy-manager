"""
Grader for the Smart Home Energy Manager.
Evaluates agent performance by comparing total cost against dynamically
computed baselines (naive and optimal) from the actual episode profiles.

Score formula:
    score = clamp((naive_cost - agent_cost) / (naive_cost - optimal_cost), 0.0001, 0.9999)

Note: Endpoints 0.0 and 1.0 are excluded to satisfy the OpenEnv validator's
strict requirement that each task score falls strictly within (0, 1).
"""

# Strict bounds required by the OpenEnv validator (exclusive of 0.0 and 1.0)
_MIN_SCORE = 0.01
_MAX_SCORE = 0.99


def calculate_score(naive_cost: float, optimal_cost: float, agent_cost: float) -> float:
    """
    Calculate a final score strictly between 0.0 and 1.0 (exclusive).

    Args:
        naive_cost: Cost if the agent bought all demand from grid (ignoring solar/battery).
        optimal_cost: Cost of an ideal agent with perfect foresight.
        agent_cost: The agent's total accumulated cost over the episode.

    Returns:
        A float score in (_MIN_SCORE, _MAX_SCORE) — strictly between 0 and 1.
        ~1.0 = matched or beat optimal, ~0.0 = worse than naive.
    """
    if naive_cost == optimal_cost:
        return 0.5  # Degenerate case — no spread; 0.5 satisfies strict bounds

    score = (naive_cost - agent_cost) / (naive_cost - optimal_cost)
    return max(_MIN_SCORE, min(_MAX_SCORE, round(score, 4)))
