"""
Core SmartHomeEnv environment class.
Implements reset(), step(), and state() for the OpenEnv framework.
"""

import uuid
from .models import Observation, Action, Reward, StepResponse, State
from .tasks import generate_profile, compute_naive_cost, compute_optimal_cost, VALID_TASKS
from .reward import calculate_reward
from .grader import calculate_score


class SmartHomeEnv:
    """
    Smart Home Energy Manager environment.

    The agent manages a household with solar panels and a battery over a 24-hour
    episode, deciding each hour whether to charge from grid, discharge to home,
    sell to grid, or idle.

    Battery specs:
        - Capacity: 10 kWh
        - Max charge rate: 5 kW per hour
        - Max discharge rate: 5 kW per hour
        - Sell-back price: 50% of current grid price
    """

    BATTERY_CAPACITY_KWH = 10.0
    MAX_CHARGE_RATE_KW = 5.0
    MAX_DISCHARGE_RATE_KW = 5.0
    SELL_BACK_RATIO = 0.5  # Grid buys back at 50% of current price

    def __init__(self):
        self.episode_id: str = str(uuid.uuid4())
        self.task_name: str = "easy"
        self.hour_of_day: int = 0
        self.battery_kwh: float = 0.0
        self.total_cost: float = 0.0
        self.episode_done: bool = True
        self.demand_profile: list = []
        self.solar_profile: list = []
        self.price_profile: list = []
        # Dynamic baselines — computed fresh from the actual generated profile
        self.naive_cost: float = 0.0
        self.optimal_cost: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_name: str) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            task_name: One of "easy", "medium", "hard".

        Returns:
            The initial Observation.
        """
        if task_name not in VALID_TASKS:
            raise ValueError(f"Unknown task: '{task_name}'. Valid tasks: {VALID_TASKS}")

        self.episode_id = str(uuid.uuid4())
        self.task_name = task_name
        self.hour_of_day = 0
        self.battery_kwh = 0.0
        self.total_cost = 0.0
        self.episode_done = False
        self.demand_profile, self.solar_profile, self.price_profile = generate_profile(task_name)

        # Compute baselines dynamically from the actual profile
        self.naive_cost = compute_naive_cost(
            self.demand_profile, self.solar_profile, self.price_profile
        )
        self.optimal_cost = compute_optimal_cost(
            self.demand_profile, self.solar_profile, self.price_profile
        )

        return self._get_observation()

    def step(self, action: Action) -> StepResponse:
        """
        Execute one time step (1 hour).

        Args:
            action: An Action with action_type in
                    {"charge_from_grid", "discharge_to_home", "sell_to_grid", "idle"}.

        Returns:
            StepResponse containing the new observation, reward, done flag, and info.
        """
        if self.episode_done:
            return StepResponse(
                observation=self._get_observation(),
                reward=Reward(value=0.0, is_invalid=True, description="Episode already done. Call /reset."),
                done=True,
                info={}
            )

        demand = self.demand_profile[self.hour_of_day]
        solar = self.solar_profile[self.hour_of_day]
        price = self.price_profile[self.hour_of_day]

        cost, is_invalid = self._apply_action(action.action_type, demand, solar, price)
        self.total_cost += cost

        reward = calculate_reward(cost, is_invalid)

        # Advance time
        self.hour_of_day += 1
        done = self.hour_of_day >= 24

        info = {}
        if done:
            self.episode_done = True
            info["final_score"] = calculate_score(
                self.naive_cost, self.optimal_cost, self.total_cost
            )
            info["total_cost"] = round(self.total_cost, 4)

        # Clamp hour for observation safety
        obs_hour = min(self.hour_of_day, 23)
        obs = Observation(
            hour_of_day=obs_hour,
            battery_soc=round(self.battery_kwh / self.BATTERY_CAPACITY_KWH, 4),
            current_demand_kw=self.demand_profile[obs_hour],
            solar_generation_kw=self.solar_profile[obs_hour],
            grid_price_per_kwh=self.price_profile[obs_hour],
        )

        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> State:
        """Return the current environment state with session metadata."""
        return State(
            episode_id=self.episode_id,
            task_name=self.task_name,
            step_count=self.hour_of_day,
            is_done=self.episode_done,
            observation=self._get_observation(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Observation:
        """Build an Observation from current internal state."""
        h = min(self.hour_of_day, 23)
        return Observation(
            hour_of_day=h,
            battery_soc=round(self.battery_kwh / self.BATTERY_CAPACITY_KWH, 4),
            current_demand_kw=self.demand_profile[h] if self.demand_profile else 0.0,
            solar_generation_kw=self.solar_profile[h] if self.solar_profile else 0.0,
            grid_price_per_kwh=self.price_profile[h] if self.price_profile else 0.0,
        )

    def _apply_action(
        self, action_type: str, demand: float, solar: float, price: float
    ) -> tuple:
        """
        Apply the chosen action and return (cost, is_invalid).

        Physics:
            net_demand = demand - solar
            > 0 means household needs more power than solar provides.
            < 0 means excess solar available.
        """
        net_demand = demand - solar
        is_invalid = False
        cost = 0.0

        if action_type == "charge_from_grid":
            # Charge battery from grid at max rate
            space = self.BATTERY_CAPACITY_KWH - self.battery_kwh
            charge_possible = min(self.MAX_CHARGE_RATE_KW, space)

            if charge_possible <= 0.01:
                # Battery full — invalid action
                is_invalid = True
                # Still must meet demand
                grid_draw = max(0.0, net_demand)
            else:
                self.battery_kwh += charge_possible
                # Grid must supply: positive net demand + charging
                grid_draw = max(0.0, net_demand) + charge_possible

            cost = grid_draw * price

        elif action_type == "discharge_to_home":
            if net_demand <= 0:
                # Solar already exceeds demand — discharging is useless
                is_invalid = True
                # Excess solar is wasted (or auto-stored via idle logic)
                cost = 0.0
            else:
                discharge_possible = min(
                    self.MAX_DISCHARGE_RATE_KW,
                    self.battery_kwh,
                    net_demand
                )
                if discharge_possible <= 0.01 and self.battery_kwh <= 0.01:
                    is_invalid = True

                self.battery_kwh = max(0.0, self.battery_kwh - discharge_possible)
                remaining = net_demand - discharge_possible
                cost = max(0.0, remaining) * price

        elif action_type == "sell_to_grid":
            # Sell battery power + excess solar to grid
            discharge_possible = min(self.MAX_DISCHARGE_RATE_KW, self.battery_kwh)
            self.battery_kwh = max(0.0, self.battery_kwh - discharge_possible)

            excess_solar = max(0.0, -net_demand)
            total_sold = discharge_possible + excess_solar

            # Must still buy from grid if net demand > 0
            grid_purchase = max(0.0, net_demand)
            sell_revenue = total_sold * price * self.SELL_BACK_RATIO

            cost = (grid_purchase * price) - sell_revenue

            if total_sold <= 0.01:
                # Nothing to sell — invalid
                is_invalid = True

        else:  # "idle" — let physics govern
            if net_demand > 0:
                # Need more than solar provides → buy from grid
                cost = net_demand * price
            else:
                # Excess solar → auto-store in battery, sell remainder
                excess = -net_demand
                space = self.BATTERY_CAPACITY_KWH - self.battery_kwh
                stored = min(excess, self.MAX_CHARGE_RATE_KW, space)
                self.battery_kwh += stored
                sold = excess - stored
                cost = -(sold * price * self.SELL_BACK_RATIO)

        return cost, is_invalid
