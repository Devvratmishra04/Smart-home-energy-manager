"""
Task profile generator for Easy, Medium, and Hard tasks.
Each task produces 24-element lists: (demand_kw, solar_kw, price_per_kwh).

Profiles are DETERMINISTIC — each task uses a fixed random seed so that
profiles are identical across runs, making grading reproducible.
"""

import math
import random
from typing import Tuple, List

# Fixed seeds per task — ensures deterministic profiles
TASK_SEEDS = {
    "easy": 42,
    "medium": 123,
    "hard": 456,
}

VALID_TASKS = list(TASK_SEEDS.keys())

# Battery constants (mirrored from environment.py for baseline computation)
BATTERY_CAPACITY_KWH = 10.0
MAX_CHARGE_RATE_KW = 5.0
MAX_DISCHARGE_RATE_KW = 5.0
SELL_BACK_RATIO = 0.5


def generate_easy_profile() -> Tuple[List[float], List[float], List[float]]:
    """Sunny day, fixed pricing."""
    demand = []
    solar = []
    prices = []
    for h in range(24):
        # Gentle sine demand peaking in evening
        d = 1.0 + 0.5 * math.sin((h - 6) * math.pi / 12)
        demand.append(round(max(0.3, d), 3))

        # Strong solar bell curve 6am-6pm
        if 6 <= h <= 18:
            s = 4.0 * math.sin((h - 6) * math.pi / 12)
        else:
            s = 0.0
        solar.append(round(max(0.0, s), 3))

        # Fixed price
        prices.append(0.15)

    return demand, solar, prices


def generate_medium_profile() -> Tuple[List[float], List[float], List[float]]:
    """Cloudy day, Time-of-Use pricing."""
    demand = []
    solar = []
    prices = []
    for h in range(24):
        d = 1.5 + 1.0 * math.sin((h - 6) * math.pi / 12)
        demand.append(round(max(0.5, d), 3))

        if 6 <= h <= 18:
            base_solar = 2.5 * math.sin((h - 6) * math.pi / 12)
            s = base_solar * random.uniform(0.6, 1.0)  # Cloud cover
        else:
            s = 0.0
        solar.append(round(max(0.0, s), 3))

        # TOU: peak 4pm-9pm
        if 16 <= h <= 21:
            prices.append(0.45)
        else:
            prices.append(0.15)

    return demand, solar, prices


def generate_hard_profile() -> Tuple[List[float], List[float], List[float]]:
    """Random weather, dynamic pricing with extreme spikes."""
    demand = []
    solar = []
    prices = []
    for h in range(24):
        demand.append(round(random.uniform(1.0, 3.0), 3))

        if 6 <= h <= 18:
            base_solar = 3.0 * math.sin((h - 6) * math.pi / 12)
            s = base_solar * random.uniform(0.2, 1.0)
        else:
            s = 0.0
        solar.append(round(max(0.0, s), 3))

        # Dynamic pricing with random spikes
        if random.random() < 0.1:
            prices.append(round(random.uniform(0.80, 1.50), 3))
        elif 16 <= h <= 21:
            prices.append(0.50)
        else:
            prices.append(round(random.uniform(0.10, 0.20), 3))

    return demand, solar, prices


def compute_naive_cost(
    demand: List[float], solar: List[float], prices: List[float]
) -> float:
    """
    Compute cost for a naive agent that buys ALL demand from the grid,
    completely ignoring solar and battery.
    """
    return round(sum(d * p for d, p in zip(demand, prices)), 4)


def compute_optimal_cost(
    demand: List[float], solar: List[float], prices: List[float]
) -> float:
    """
    Compute cost for an ideal agent with perfect foresight.

    Uses a forward-looking greedy strategy:
      1. Use free solar first — store excess in battery, sell remainder.
      2. Pre-charge battery from the grid during cheap hours when future
         hours have strictly higher prices (arbitrage opportunity).
      3. Discharge battery during the most expensive hours to avoid
         buying from the grid at peak price.

    This produces a genuinely better baseline than idle for tasks
    with time-of-use or dynamic pricing (medium, hard).
    """
    battery = 0.0
    total_cost = 0.0

    # Pre-compute: max price in all future hours (for lookahead decisions)
    max_future_price = [0.0] * 24
    running_max = 0.0
    for h in range(23, -1, -1):
        max_future_price[h] = running_max
        running_max = max(running_max, prices[h])

    for h in range(24):
        net = demand[h] - solar[h]

        if net <= 0:
            # ---- Excess solar ----
            # Store as much as possible, sell the rest
            excess = -net
            space = BATTERY_CAPACITY_KWH - battery
            stored = min(excess, MAX_CHARGE_RATE_KW, space)
            battery += stored
            sold = excess - stored
            total_cost -= sold * prices[h] * SELL_BACK_RATIO
        else:
            # ---- Need power ----
            # Decide: discharge battery now, or save it for a more expensive hour?
            # Discharge if current price >= max price in remaining hours
            # (no better time to use the battery than now)
            if battery > 0 and prices[h] >= max_future_price[h]:
                discharge = min(MAX_DISCHARGE_RATE_KW, battery, net)
                battery -= discharge
                remaining = net - discharge
                total_cost += remaining * prices[h]
            else:
                # Cheaper now than future peak — buy from grid
                total_cost += net * prices[h]

            # ---- Pre-charge opportunity ----
            # If current price is strictly cheaper than the most expensive
            # future hour, charge the battery from the grid now.
            # The stored energy will be used later to avoid buying at peak.
            if prices[h] < max_future_price[h]:
                space = BATTERY_CAPACITY_KWH - battery
                charge = min(MAX_CHARGE_RATE_KW, space)
                if charge > 0:
                    battery += charge
                    total_cost += charge * prices[h]

    return round(total_cost, 4)


def generate_profile(task_name: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Main entry point. Returns (demand, solar, prices) for the given task.
    Each is a list of 24 floats (one per hour).

    Uses a fixed random seed per task for deterministic, reproducible profiles.
    """
    if task_name not in VALID_TASKS:
        raise ValueError(f"Unknown task: '{task_name}'. Valid tasks: {VALID_TASKS}")

    # Set deterministic seed for this task
    random.seed(TASK_SEEDS[task_name])

    if task_name == "easy":
        return generate_easy_profile()
    elif task_name == "medium":
        return generate_medium_profile()
    elif task_name == "hard":
        return generate_hard_profile()
    else:
        raise ValueError(f"Unknown task: '{task_name}'. Valid tasks: {VALID_TASKS}")
