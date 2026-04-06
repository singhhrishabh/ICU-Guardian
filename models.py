"""
ICU-Guardian: Typed models for Action, Observation, and State.

These dataclasses define the contract between agent and environment.
They are compatible with OpenEnv's base classes when running in Python 3.10+
Docker containers, and fall back gracefully for local development.
"""

from dataclasses import dataclass, field
from typing import Optional

# Compatibility shim: use OpenEnv base classes if available (Python 3.10+),
# otherwise use plain object. The Docker container will have Python 3.11.
try:
    from openenv_core.env_server.types import Action, Observation, State
except (ImportError, TypeError):
    # Fallback for Python < 3.10 or if openenv-core is not installed
    Action = object
    Observation = object
    State = object


@dataclass
class ICUAction(Action):
    """
    An action the AI agent can take in the ICU environment.

    Action Types:
        - "administer_meds": Give medication (requires drug + dose)
        - "adjust_oxygen": Change oxygen support (requires level)
        - "trigger_code_sepsis": Emergency sepsis alert (no params)
        - "wait": Monitor without intervention (no params)
    """

    action: str = "wait"
    drug: Optional[str] = None      # "vasopressor" | "antihypertensive"
    dose: Optional[str] = None      # "low" | "high"
    level: Optional[str] = None     # "increase" | "decrease"


@dataclass
class ICUObservation(Observation):
    """
    The patient's current vital signs and environment state.

    Vital Signs:
        HR: Heart rate (bpm), normal range 60-100
        BP_sys: Systolic blood pressure (mmHg), target ~120
        BP_dia: Diastolic blood pressure (mmHg), target ~80
        SpO2: Oxygen saturation (%), must stay >= 95
        Temp: Body temperature (°C), normal 36.5-37.5

    Context:
        trend: Natural language summary of the last 3 steps
        step_number: Current step in the episode
        task_name: Which task is active
        last_action_error: Error message if the last action was invalid
    """

    HR: int = 80
    BP_sys: int = 120
    BP_dia: int = 80
    SpO2: int = 98
    Temp: float = 37.0
    trend: str = "No prior data."
    step_number: int = 0
    task_name: str = "vital_stabilization"
    last_action_error: Optional[str] = None


@dataclass
class ICUState(State):
    """Episode metadata for the ICU environment."""

    episode_id: str = ""
    step_count: int = 0
    task_name: str = "vital_stabilization"
    max_steps: int = 15
    done: bool = False
