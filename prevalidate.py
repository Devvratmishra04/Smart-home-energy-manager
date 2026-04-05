"""
Prevalidation script for the Smart Home Energy Manager environment.
Uses FastAPI TestClient to validate all endpoints, schemas, and
OpenEnv submission gates locally.

Run:  python prevalidate.py
"""

import json
import random
import yaml
from starlette.testclient import TestClient
from main import app

client = TestClient(app)

TASKS = ["easy", "medium", "hard"]
VALID_ACTIONS = ["charge_from_grid", "discharge_to_home", "sell_to_grid", "idle"]
REQUIRED_YAML_FIELDS = [
    "spec_version", "name", "description", "version", "type",
    "runtime", "app", "port", "observation_space", "action_space",
    "reward_range", "tasks",
]
REQUIRED_STATE_FIELDS = ["episode_id", "task_name", "step_count", "is_done", "observation"]
REQUIRED_OBS_FIELDS = ["hour_of_day", "battery_soc", "current_demand_kw", "solar_generation_kw", "grid_price_per_kwh"]

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    """Print a pass/fail check result."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def validate_observation(obs: dict, label: str):
    """Validate that an observation dict has all required fields in range."""
    check(
        f"{label}: has all fields",
        all(k in obs for k in REQUIRED_OBS_FIELDS),
        f"keys={list(obs.keys())}"
    )
    check(f"{label}: hour_of_day in [0,23]", 0 <= obs.get("hour_of_day", -1) <= 23)
    check(f"{label}: battery_soc in [0,1]", 0.0 <= obs.get("battery_soc", -1) <= 1.0)
    check(f"{label}: demand >= 0", obs.get("current_demand_kw", -1) >= 0)
    check(f"{label}: solar >= 0", obs.get("solar_generation_kw", -1) >= 0)
    check(f"{label}: price >= 0", obs.get("grid_price_per_kwh", -1) >= 0)


def validate_state_response(data: dict, label: str):
    """Validate that a /state response has all required State model fields."""
    check(
        f"{label}: has all State fields",
        all(k in data for k in REQUIRED_STATE_FIELDS),
        f"keys={list(data.keys())}"
    )
    check(f"{label}: episode_id is string", isinstance(data.get("episode_id"), str))
    check(f"{label}: task_name is string", isinstance(data.get("task_name"), str))
    check(f"{label}: step_count is int >= 0", isinstance(data.get("step_count"), int) and data.get("step_count", -1) >= 0)
    check(f"{label}: is_done is bool", isinstance(data.get("is_done"), bool))
    if "observation" in data:
        validate_observation(data["observation"], f"{label}/observation")


def validate_step_response(data: dict, label: str):
    """Validate a step response has all required fields."""
    check(f"{label}: has observation", "observation" in data)
    check(f"{label}: has reward", "reward" in data)
    check(f"{label}: has done", "done" in data)
    check(f"{label}: has info", "info" in data)

    reward = data.get("reward", {})
    check(
        f"{label}: reward.value in [0.0, 1.0]",
        0.0 <= reward.get("value", -999) <= 1.0,
        f"got {reward.get('value')}"
    )
    check(f"{label}: reward.is_invalid is bool", isinstance(reward.get("is_invalid"), bool))


print("=" * 60)
print("Smart Home Energy Manager — Prevalidation")
print("=" * 60)

# ------------------------------------------------------------------
# Test 0: Validate openenv.yaml
# ------------------------------------------------------------------
print("\n[Test] openenv.yaml structure")
try:
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    check("openenv.yaml loads successfully", True)
    for field in REQUIRED_YAML_FIELDS:
        check(f"openenv.yaml has '{field}'", field in config, f"missing from config")

    # Validate reward_range
    rr = config.get("reward_range", [])
    check(
        "reward_range is [0.0, 1.0]",
        isinstance(rr, list) and len(rr) == 2 and rr[0] == 0.0 and rr[1] == 1.0,
        f"got {rr}"
    )

    # Validate tasks count
    tasks = config.get("tasks", [])
    check("openenv.yaml has 3 tasks", len(tasks) == 3, f"got {len(tasks)}")
except Exception as e:
    check("openenv.yaml loads successfully", False, str(e))

# ------------------------------------------------------------------
# Test 1: GET /health
# ------------------------------------------------------------------
print("\n[Test] GET /health")
resp = client.get("/health")
check("/health returns 200", resp.status_code == 200, f"got {resp.status_code}")
if resp.status_code == 200:
    health_data = resp.json()
    check("/health has status=ok", health_data.get("status") == "ok", f"got {health_data}")

# ------------------------------------------------------------------
# Test 2: GET /state before reset (should still return 200 with State model)
# ------------------------------------------------------------------
print("\n[Test] GET /state (initial)")
resp = client.get("/state")
check("/state returns 200", resp.status_code == 200, f"got {resp.status_code}")
if resp.status_code == 200:
    state_data = resp.json()
    validate_state_response(state_data, "/state (initial)")

# ------------------------------------------------------------------
# Test 3-5: Run each task for 24 steps
# ------------------------------------------------------------------
for task in TASKS:
    print(f"\n[Test] Task: {task}")

    # Reset
    resp = client.post("/reset", json={"task": task})
    check(f"POST /reset [{task}] returns 200", resp.status_code == 200, f"got {resp.status_code}")

    obs = resp.json()
    validate_observation(obs, f"reset [{task}]")

    # Check /state after reset
    resp = client.get("/state")
    check(f"GET /state after reset [{task}] returns 200", resp.status_code == 200)
    if resp.status_code == 200:
        state_data = resp.json()
        validate_state_response(state_data, f"/state after reset [{task}]")
        check(
            f"state.task_name matches [{task}]",
            state_data.get("task_name") == task,
            f"got {state_data.get('task_name')}"
        )
        check(
            f"state.step_count is 0 [{task}]",
            state_data.get("step_count") == 0,
            f"got {state_data.get('step_count')}"
        )
        check(
            f"state.is_done is False [{task}]",
            state_data.get("is_done") is False,
            f"got {state_data.get('is_done')}"
        )

    # Run 24 steps with random actions
    step_rewards = []
    final_done = False
    for step_num in range(1, 25):
        action_type = random.choice(VALID_ACTIONS)
        resp = client.post("/step", json={"action_type": action_type})
        check(
            f"step {step_num} [{task}] returns 200",
            resp.status_code == 200,
            f"got {resp.status_code}"
        )
        data = resp.json()
        validate_step_response(data, f"step {step_num} [{task}]")
        step_rewards.append(data["reward"]["value"])
        final_done = data["done"]

    check(f"done=true after 24 steps [{task}]", final_done is True)

    # Check final info
    info = data.get("info", {})
    if "final_score" in info:
        score = info["final_score"]
        check(
            f"final_score in [0,1] [{task}]",
            0.0 <= score <= 1.0,
            f"got {score}"
        )
    else:
        check(f"final_score present [{task}]", False, "missing from info")

    if "total_cost" in info:
        check(f"total_cost is number [{task}]", isinstance(info["total_cost"], (int, float)))

    # Confirm step after done returns error
    resp = client.post("/step", json={"action_type": "idle"})
    check(
        f"step after done returns 400 [{task}]",
        resp.status_code == 400,
        f"got {resp.status_code}"
    )

    # GET /state after episode — verify is_done
    resp = client.get("/state")
    check(f"GET /state after done [{task}] returns 200", resp.status_code == 200)
    if resp.status_code == 200:
        state_data = resp.json()
        check(
            f"state.is_done is True after episode [{task}]",
            state_data.get("is_done") is True,
            f"got {state_data.get('is_done')}"
        )

# ------------------------------------------------------------------
# Test 6: Determinism check — same task, same seed → same profiles
# ------------------------------------------------------------------
print("\n[Test] Determinism check (easy task)")
resp1 = client.post("/reset", json={"task": "easy"})
obs1 = resp1.json()
resp2 = client.post("/reset", json={"task": "easy"})
obs2 = resp2.json()
check(
    "deterministic: two resets of 'easy' produce identical observations",
    obs1 == obs2,
    f"obs1={obs1} vs obs2={obs2}"
)

# ------------------------------------------------------------------
# Test 7: Invalid task name
# ------------------------------------------------------------------
print("\n[Test] Invalid task name")
resp = client.post("/reset", json={"task": "impossible"})
check("POST /reset [invalid] returns 400", resp.status_code == 400, f"got {resp.status_code}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("*** ALL CHECKS PASSED -- Environment is valid! ***")
else:
    print(f"[!] {failed} check(s) failed. Review errors above.")
print("=" * 60)
