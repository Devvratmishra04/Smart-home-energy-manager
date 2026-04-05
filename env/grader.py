"""
Grader for the Smart Home Energy Manager.
Evaluates agent performance by comparing total cost against dynamically
computed baselines (naive and optimal) from the actual episode profiles.

Score formula:
    score = clamp((naive_cost - agent_cost) / (naive_cost - optimal_cost), 0.0, 1.0)
"""


def calculate_score(naive_cost: float, optimal_cost: float, agent_cost: float) -> float:
    """
    Calculate a final score between 0.0 and 1.0.

    Args:
        naive_cost: Cost if the agent bought all demand from grid (ignoring solar/battery).
        optimal_cost: Cost of an ideal agent with perfect foresight.
        agent_cost: The agent's total accumulated cost over the episode.

    Returns:
        A float score in [0.0, 1.0].
        1.0 = matched or beat optimal, 0.0 = worse than naive.
    """
    if naive_cost == optimal_cost:
        return 0.5  # Degenerate case — no spread

    score = (naive_cost - agent_cost) / (naive_cost - optimal_cost)
    return max(0.0, min(1.0, round(score, 4)))
