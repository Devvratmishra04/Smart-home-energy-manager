🚀 MVP Project Generation Prompt (RL + OpenEnv)

Generate a complete, implementation-ready MVP for a real-world Reinforcement Learning environment using the OpenEnv framework.

The output must be fully structured, modular, and directly usable by an AI coding agent to build the project with minimal ambiguity.

🎯 Goal

Create a hackathon-winning RL project that simulates a real-world task (NOT a game), with proper environment design, reward logic, and deployment.

📌 Requirements
1. Problem Definition

Clearly define a real-world problem

Explain:

Why it matters

Who benefits

Why RL is the right approach

2. Solution Overview

Describe how the agent interacts with the environment

Define:

Input → Decision → Output loop

Highlight real-world relevance

3. Environment Design (STRICT - OpenEnv Compatible)

Provide complete details for:

Observation Space

Structure, fields, data types

Action Space

All possible agent actions

State Representation

Internal environment state

Also include:

reset() logic

step(action) logic

state() logic

4. Task Design (MANDATORY – 3 Tasks)

Create 3 tasks with increasing difficulty:

Easy Task

Simple objective

Minimal steps

Medium Task

Multi-step reasoning

Hard Task

Complex decision-making with trade-offs

For each task include:

Input example

Expected behavior

Success criteria

5. Reward Function Design (CRITICAL)

Provide dense reward signals

Include:

Step-wise reward calculation

Partial progress rewards

Penalties (invalid actions, loops, inefficiency)

Ensure reward range is normalized (0–1)

6. Grader Design

Deterministic evaluation function

Outputs score between 0.0–1.0

Define exact scoring logic

7. MVP Architecture (VERY IMPORTANT)

Provide a clean folder structure, for example:

code
Code
download
content_copy
expand_less
project/
│── env/
│   ├── environment.py
│   ├── models.py
│   ├── tasks.py
│   ├── reward.py
│   ├── grader.py
│
│── inference.py
│── openenv.yaml
│── Dockerfile
│── requirements.txt
│── README.md

Explain purpose of each file.

8. Key Implementation Details

Provide:

Pseudocode OR actual code snippets for:

step() function

Reward calculation

Task grader

9. Baseline Agent Plan

How the agent will interact with environment

Strategy:

Rule-based / LLM-based / hybrid

How it improves over steps

10. Inference Script Requirements

Must follow OpenEnv loop:

reset → step → log → done

Should produce:

Reproducible results

Structured logs

11. Deployment Plan

Docker setup

Hugging Face Spaces deployment steps

Environment variables required

12. Validation Plan

Ensure compatibility with:

openenv validate

Prevalidation script

13. Stretch Features (Optional but Recommended)

Add uncertainty (stochastic environment)

Add memory/state tracking

Multi-agent extension

Visualization dashboard

⚡ Output Constraints

Must be:

Clear

Structured

Step-by-step

Implementation-ready

Avoid:

Vague ideas

Generic explanations

Missing components

🎯 Final Instruction

The output should be so detailed that a coding AI agent can directly generate the full project (code + deployment) without asking follow-up questions.
and make changes accordingly if it is necessary and also inform where the changes are being made and how are they beneficial for the project or hackathon