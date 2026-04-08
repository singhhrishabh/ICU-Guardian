"""
Inference Script — ICU-Guardian OpenEnv Environment
=====================================================
MANDATORY FORMAT COMPLIANCE:
- Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Uses OpenAI Client for all LLM calls
- Emits [START], [STEP], [END] stdout lines in exact required format
- Runs all 3 tasks and produces reproducible scores

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ICUAction
from server.icu_environment import ICUEnvironment

# ─── Environment Configuration ───────────────────────────────────────────────
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Exactly as required by checklist
API_KEY = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("ICU_BENCHMARK", "icu_guardian")

# ─── Task Configuration ──────────────────────────────────────────────────────
TASKS = [
    {"name": "vital_stabilization", "max_steps": 15},
    {"name": "bp_management", "max_steps": 20},
    {"name": "sepsis_detection", "max_steps": 25},
]

TEMPERATURE = 0.3
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.3

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an advanced ICU-Guardian AI, an expert clinical agent tasked with
    monitoring and stabilizing critically ill patients in an Intensive Care Unit.

    You operate in a continuous loop. At each step, you receive a JSON observation
    with the patient's vital signs and trends. You must choose the optimal action.

    YOUR OBJECTIVES:
    1. Prevent Hypoxia: Keep SpO2 >= 95%
    2. Manage Blood Pressure: Keep BP near 120/80 without overshooting
    3. Detect Sepsis: Rising Temp + dropping BP + spiking HR = trigger_code_sepsis

    SAFE RANGES:
    - HR: 60-100 bpm
    - BP_sys: 100-140 mmHg
    - BP_dia: 60-90 mmHg
    - SpO2: 95-100%
    - Temp: 36.0-37.8°C

    ACTION SPACE (respond with ONLY valid JSON, no markdown, no explanation):
    {"action": "administer_meds", "drug": "vasopressor", "dose": "low"}
    {"action": "administer_meds", "drug": "vasopressor", "dose": "high"}
    {"action": "administer_meds", "drug": "antihypertensive", "dose": "low"}
    {"action": "administer_meds", "drug": "antihypertensive", "dose": "high"}
    {"action": "adjust_oxygen", "level": "increase"}
    {"action": "adjust_oxygen", "level": "decrease"}
    {"action": "trigger_code_sepsis"}
    {"action": "wait"}

    STRATEGY:
    - If SpO2 < 95: increase oxygen
    - If BP_sys < 100: give vasopressor (low dose first)
    - If BP_sys > 140: give antihypertensive (low dose first)
    - If Temp rising AND BP dropping AND HR spiking over multiple steps: trigger sepsis
    - If vitals are all in safe range: wait
    - Prefer low doses to avoid overcorrection
    - Only trigger sepsis if you see a CLEAR multi-step pattern
""").strip()


# ─── Logging Functions ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # EXACT FORMAT: [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )



# ─── LLM Interaction ─────────────────────────────────────────────────────────

def build_user_prompt(obs_data: Dict, step: int, last_reward: float,
                      history: List[str]) -> str:
    """Format the observation as a user prompt for the LLM."""
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(f"""
        CURRENT OBSERVATION (Step {step}):
        {json.dumps(obs_data, indent=2)}

        Last reward: {last_reward:.2f}
        
        Recent history:
        {history_block}

        Choose your next action. Respond with ONLY a JSON object.
    """).strip()


def parse_action(response_text: str) -> Dict:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from surrounding text
    json_match = re.search(r'\{[^{}]+\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Default: wait
    return {"action": "wait"}


def get_model_action(client: OpenAI, obs_data: Dict, step: int,
                     last_reward: float, history: List[str]) -> Dict:
    """Call the LLM to get the next action."""
    user_prompt = build_user_prompt(obs_data, step, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action": "wait"}


def action_to_string(action_dict: Dict) -> str:
    """Convert action dict to a compact string for logging."""
    a = action_dict.get("action", "wait")
    if a == "administer_meds":
        return f"administer_meds({action_dict.get('drug', '?')},{action_dict.get('dose', '?')})"
    elif a == "adjust_oxygen":
        return f"adjust_oxygen({action_dict.get('level', '?')})"
    elif a == "trigger_code_sepsis":
        return "trigger_code_sepsis()"
    else:
        return "wait()"


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str, max_steps: int) -> None:
    """Run a single task episode using local environment."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = ICUEnvironment(task_name=task_name)
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            # Format observation for the LLM
            obs_data = {
                "HR": obs.HR,
                "BP_sys": obs.BP_sys,
                "BP_dia": obs.BP_dia,
                "SpO2": obs.SpO2,
                "Temp": obs.Temp,
                "Trend": obs.trend,
            }

            # Get action from LLM
            action_dict = get_model_action(client, obs_data, step, last_reward, history)
            action = ICUAction(
                action=action_dict.get("action", "wait"),
                drug=action_dict.get("drug"),
                dose=action_dict.get("dose"),
                level=action_dict.get("level"),
            )
            action_str = action_to_string(action_dict)

            # Execute step — returns ICUObservation directly
            obs = env.step(action)
            reward = obs.reward or 0.0
            done = obs.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error)

            history.append(
                f"Step {step}: {action_str} → reward={reward:.2f}, "
                f"HR={obs.HR}, BP={obs.BP_sys}/{obs.BP_dia}, "
                f"SpO2={obs.SpO2}, Temp={obs.Temp}"
            )

            if done:
                break

        score = env.get_score()
        # STICKTLY BETWEEN 0 and 1: Hackathon requirement (not 0.0 and not 1.0)
        score = max(0.0001, min(0.9999, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def run_task_docker(client: OpenAI, task_name: str, max_steps: int) -> None:
    """Run a single task episode using Docker-based environment."""
    from client import ICUGuardianEnv

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        os.environ["ICU_TASK"] = task_name
        env = await ICUGuardianEnv.from_docker_image(LOCAL_IMAGE_NAME)

        async with env:
            result = await env.reset()
            obs = result.observation
            last_reward = 0.0

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                obs_data = {
                    "HR": obs.HR,
                    "BP_sys": obs.BP_sys,
                    "BP_dia": obs.BP_dia,
                    "SpO2": obs.SpO2,
                    "Temp": obs.Temp,
                    "Trend": obs.trend,
                }

                action_dict = get_model_action(client, obs_data, step, last_reward, history)
                action = ICUAction(
                    action=action_dict.get("action", "wait"),
                    drug=action_dict.get("drug"),
                    dose=action_dict.get("dose"),
                    level=action_dict.get("level"),
                )
                action_str = action_to_string(action_dict)

                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = getattr(obs, 'last_action_error', None)

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action_str, reward=reward,
                         done=done, error=error)

                history.append(
                    f"Step {step}: {action_str} → reward={reward:.2f}, "
                    f"HR={obs.HR}, BP={obs.BP_sys}/{obs.BP_dia}, "
                    f"SpO2={obs.SpO2}, Temp={obs.Temp}"
                )

                if done:
                    break

            score = sum(rewards) / (max_steps * 0.8) if max_steps > 0 else 0.0001
            # STICKTLY BETWEEN 0 and 1: Hackathon requirement (not 0.0 and not 1.0)
            score = max(0.0001, min(0.9999, score))
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    """Run all tasks sequentially."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if LOCAL_IMAGE_NAME:
        # Docker-based execution
        async def _run_all():
            for task in TASKS:
                await run_task_docker(client, task["name"], task["max_steps"])
                print("", flush=True)
        asyncio.run(_run_all())
    else:
        # Local execution (direct environment)
        for task in TASKS:
            run_task(client, task["name"], task["max_steps"])
            print("", flush=True)


if __name__ == "__main__":
    main()
