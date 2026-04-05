---
title: Smart Home Energy Manager
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Smart Home Energy Manager -- OpenEnv RL Environment

An RL environment where an agent optimizes household energy consumption with solar panels and battery storage under dynamic electricity pricing.

## Problem Motivation

Households with rooftop solar and home batteries face a complex real-time optimization problem. Electricity prices fluctuate throughout the day, solar generation depends on weather, and household demand varies by hour. A human cannot optimally decide -- every hour -- whether to charge the battery, discharge it, sell excess power, or simply idle.

**Reinforcement Learning** is the ideal approach because:
- The problem is **sequential**: each hour's decision affects future battery state.
- The environment is **stochastic**: weather and pricing are uncertain.
- The objective is **cost minimization**: a clear, quantifiable reward signal.

## Environment Design

### Observation Space
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hour_of_day` | int | 0-23 | Current hour of the day |
| `battery_soc` | float | 0.0-1.0 | Battery state of charge |
| `current_demand_kw` | float | >= 0 | Household power demand (kW) |
| `solar_generation_kw` | float | >= 0 | Solar power generated (kW) |
| `grid_price_per_kwh` | float | >= 0 | Current electricity price ($/kWh) |

### Action Space (Semantic)
| Action | Description |
|--------|-------------|
| `charge_from_grid` | Buy power from grid to charge battery |
| `discharge_to_home` | Use battery to meet household demand |
| `sell_to_grid` | Sell battery/excess solar to grid |
| `idle` | Let physics govern (solar -> home -> store/sell) |

### Battery Specifications
- **Capacity:** 10 kWh
- **Max charge rate:** 5 kW/hour
- **Max discharge rate:** 5 kW/hour
- **Sell-back price:** 50% of current grid price

## Tasks

| Task | Solar | Pricing | Challenge |
|------|-------|---------|---------| 
| **Easy** | Sunny (strong sine) | Fixed ($0.15/kWh) | Store solar by day, use at night |
| **Medium** | Cloudy (noisy) | Time-of-Use ($0.15/$0.45) | Pre-charge before peak hours |
| **Hard** | Random (20-100%) | Dynamic + spikes (up to $1.50) | Aggressive trading + safety buffer |

All task profiles are **deterministic** -- each task uses a fixed random seed, producing identical demand/solar/price profiles across runs. This ensures reproducible grading.

## Reward Design

Dense, bounded step rewards in the range **[0.0, 1.0]**:

| Outcome | Reward |
|---------|--------|
| Zero grid cost (solar/battery covers demand) | **1.0** |
| Grid cost incurred (scaled by magnitude) | **0.0 - 1.0** (higher = lower cost) |
| Invalid action (e.g., discharge empty battery) | **0.0** |

The reward uses a scaling formula: `reward = max(0.0, 1.0 - cost * 0.5)`. Negative costs (profit from selling) yield a reward of 1.0.

### Grader Formula
```
Score = clamp((Naive_Cost - Agent_Cost) / (Naive_Cost - Optimal_Cost), 0.0, 1.0)
```
- **Naive baseline:** Always buy ALL demand from grid, completely ignoring solar and battery.
- **Optimal baseline:** Computed dynamically from the actual generated profile using a forward-looking greedy strategy with perfect foresight (solar-first, grid pre-charging during cheap hours, battery discharge during expensive hours).

Both baselines are **computed fresh** from the seeded profile on every reset -- they are not hardcoded constants.

## How to Run

### Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Run Prevalidation
```bash
python prevalidate.py
```

### Run LLM Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-token-here"
# Optional: override env URL for local dev
# export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

### Docker Deployment
```bash
docker build -t smart-home-env .
docker run -p 8000:8000 smart-home-env
```

### Hugging Face Spaces
1. Push this repo to a Hugging Face Space (Docker type).
2. The `app_port: 8000` in this README's YAML frontmatter tells HF which port to route to.
3. Verify: `curl -X POST https://your-space.hf.space/reset -d '{"task":"easy"}'`
4. Update `ENV_BASE_URL` in `inference.py` with your deployed Space URL.

## API Endpoints

| Endpoint | Method | Request Body | Response |
|----------|--------|-------------|----------|
| `/health` | GET | -- | `{"status": "ok"}` |
| `/reset` | POST | `{"task": "easy"}` | Observation JSON |
| `/step` | POST | `{"action_type": "idle"}` | StepResponse JSON (observation, reward, done, info) |
| `/state` | GET | -- | State JSON (episode_id, task_name, step_count, is_done, observation) |

## Project Structure
```
├── env/
│   ├── __init__.py          # Package exports
│   ├── models.py            # Pydantic: Observation, Action, Reward, StepResponse, State
│   ├── tasks.py             # Deterministic task profiles + dynamic baseline computation
│   ├── reward.py            # Dense bounded reward calculator [0.0, 1.0]
│   ├── grader.py            # Final 0.0-1.0 scoring
│   └── environment.py       # SmartHomeEnv core class
├── main.py                  # FastAPI server
├── inference.py             # LLM agent (AsyncOpenAI + httpx)
├── prevalidate.py           # Local validation script
├── openenv.yaml             # OpenEnv config manifest
├── Dockerfile               # Container for deployment
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## License

MIT
