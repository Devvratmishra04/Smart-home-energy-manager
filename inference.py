"""
LLM-based inference agent for the Smart Home Energy Manager.

Uses AsyncOpenAI to query an LLM for decisions and httpx to interact
with the FastAPI environment server.

Required environment variables:
    API_BASE_URL   — LLM API endpoint (e.g., https://api.openai.com/v1)
    MODEL_NAME     — Model identifier (e.g., gpt-4o)
    HF_TOKEN       — Hugging Face Token (used as API key for OpenAI client)

Optional environment variables:
    ENV_BASE_URL   — Override environment server URL (for local dev).
                     Defaults to the deployed Hugging Face Space URL.

Log format (strict):
    [START] task=<task> env=SmartHomeEnv model=<model>
    [STEP] step=<n> action={"action_type": "..."} reward=<.2f> done=<bool> error=null
    [END] success=<bool> steps=<n> rewards=<comma-separated .2f>
"""

import os
import sys
import json
import asyncio
import httpx
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-model-api-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# --------------------------------------------------------------------------
# Environment URL — defaults to your deployed HF Space.
# For local development, override with:  ENV_BASE_URL=http://localhost:8000
# TODO: Replace the default below with your actual Hugging Face Space URL
#       before submission, e.g. "https://<your-username>-smart-home-manager.hf.space"
# --------------------------------------------------------------------------
ENV_BASE_URL = os.environ.get(
    "ENV_BASE_URL",
    "https://devvratmishra-smart-home-energy-manager.hf.space"
)

TASKS = ["easy", "medium", "hard"]

# Maximum retries when waiting for the environment server to become healthy
HEALTH_CHECK_RETRIES = 10
HEALTH_CHECK_INTERVAL_SECONDS = 3


SYSTEM_PROMPT = """You are an energy management agent controlling a smart home with solar panels and a battery.

Each step you receive an observation JSON with:
- hour_of_day: current hour (0-23)
- battery_soc: battery charge level (0.0 = empty, 1.0 = full)
- current_demand_kw: household power demand in kW
- solar_generation_kw: solar power being generated in kW
- grid_price_per_kwh: current electricity price in $/kWh

You must respond with ONLY a valid JSON object choosing one action:
{"action_type": "<ACTION>"}

Where <ACTION> is exactly one of:
- "charge_from_grid": Buy power from grid to charge the battery.
- "discharge_to_home": Use battery power to meet household demand.
- "sell_to_grid": Sell battery/excess solar power back to the grid.
- "idle": Do nothing special; solar goes to home, excess stored or sold.

Strategy guidance:
- When solar is high and battery is not full, consider "idle" to auto-store solar.
- When grid price is low, consider "charge_from_grid".
- When grid price is high and battery has charge, consider "discharge_to_home".
- When grid price is high and you have excess power, consider "sell_to_grid".
- Avoid discharging an empty battery or charging a full one (invalid actions).

Respond with ONLY the JSON object. No explanation, no markdown, no extra text."""


def format_bool(value: bool) -> str:
    """Format boolean as lowercase string."""
    return "true" if value else "false"


def format_reward(value: float) -> str:
    """Format reward to exactly 2 decimal places."""
    return f"{value:.2f}"


async def wait_for_env(http_client: httpx.AsyncClient) -> bool:
    """
    Poll the environment server's /health endpoint until it responds 200.

    Returns True if the server became healthy, False if all retries exhausted.
    """
    for attempt in range(1, HEALTH_CHECK_RETRIES + 1):
        try:
            resp = await http_client.get(f"{ENV_BASE_URL}/health")
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        if attempt < HEALTH_CHECK_RETRIES:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

    return False


async def query_llm(client: AsyncOpenAI, observation: dict) -> dict:
    """
    Send observation to LLM and parse the action response.

    Returns:
        A dict like {"action_type": "idle"}.
    """
    user_message = json.dumps(observation, indent=2)

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines).strip()

        action = json.loads(content)

        # Validate action_type
        valid_actions = {"charge_from_grid", "discharge_to_home", "sell_to_grid", "idle"}
        if action.get("action_type") not in valid_actions:
            action = {"action_type": "idle"}

        return action

    except Exception:
        # Fallback to idle on any error
        return {"action_type": "idle"}


async def run_task(
    client: AsyncOpenAI, http_client: httpx.AsyncClient, task_name: str
) -> None:
    """Run a single task episode and print structured logs."""
    # Print START line
    print(f"[START] task={task_name} env=SmartHomeEnv model={MODEL_NAME}")

    # Reset environment — wrapped in try/except for safety
    try:
        reset_resp = await http_client.post(
            f"{ENV_BASE_URL}/reset", json={"task": task_name}
        )
        if reset_resp.status_code != 200:
            print(f"[END] success=false steps=0 rewards=")
            print()
            return
        observation = reset_resp.json()
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=")
        print()
        return

    rewards_list = []
    step_num = 0
    success = True

    while True:
        step_num += 1

        # Query LLM for action
        action = await query_llm(client, observation)
        action_json = json.dumps(action, separators=(",", ":"))

        # Execute step
        error_str = "null"
        try:
            step_resp = await http_client.post(
                f"{ENV_BASE_URL}/step", json=action
            )
            if step_resp.status_code != 200:
                error_str = step_resp.text.replace('"', '\\"')
                reward_val = 0.0
                done = True
                success = False
            else:
                data = step_resp.json()
                observation = data["observation"]
                reward_val = data["reward"]["value"]
                done = data["done"]
        except Exception as e:
            error_str = str(e).replace('"', '\\"')
            reward_val = 0.0
            done = True
            success = False

        rewards_list.append(reward_val)

        # Print STEP line
        print(
            f"[STEP] step={step_num} "
            f"action={action_json} "
            f"reward={format_reward(reward_val)} "
            f"done={format_bool(done)} "
            f"error={error_str}"
        )

        if done:
            break

    # Print END line
    rewards_str = ",".join(format_reward(r) for r in rewards_list)
    print(f"[END] success={format_bool(success)} steps={step_num} rewards={rewards_str}")
    print()  # Blank line between tasks


async def main():
    """Main entry point: run inference on all three tasks."""
    # Initialize OpenAI client
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        # Wait for environment server to be healthy before running tasks
        # (no stdout here — dashboard expects only [START]/[STEP]/[END] lines)
        await wait_for_env(http_client)

        for task in TASKS:
            try:
                await run_task(client, http_client, task)
            except Exception as e:
                # Absolute last-resort catch — ensure the script never crashes
                print(f"[START] task={task} env=SmartHomeEnv model={MODEL_NAME}")
                print(f"[END] success=false steps=0 rewards=")
                print()


if __name__ == "__main__":
    asyncio.run(main())
