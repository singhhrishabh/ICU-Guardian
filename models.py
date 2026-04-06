# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
Data models for the ICU-Guardian Environment.

Uses Pydantic models inheriting from OpenEnv base types for full spec compliance.
"""

from typing import Optional
from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv_core.env_server.types import Action, Observation, State


class ICUAction(Action):
    """
    An action the AI agent can take in the ICU environment.

    Action Types:
        - "administer_meds": Give medication (requires drug + dose)
        - "adjust_oxygen": Change oxygen support (requires level)
        - "trigger_code_sepsis": Emergency sepsis alert
        - "wait": Monitor without intervention
    """

    action: str = Field(default="wait", description="Action type to perform")
    drug: Optional[str] = Field(default=None, description="Drug: vasopressor or antihypertensive")
    dose: Optional[str] = Field(default=None, description="Dose: low or high")
    level: Optional[str] = Field(default=None, description="Oxygen level: increase or decrease")


class ICUObservation(Observation):
    """
    The patient's current vital signs and environment state.

    Vital Signs:
        HR: Heart rate (bpm), normal range 60-100
        BP_sys: Systolic blood pressure (mmHg), target ~120
        BP_dia: Diastolic blood pressure (mmHg), target ~80
        SpO2: Oxygen saturation (%), must stay >= 95
        Temp: Body temperature (°C), normal 36.5-37.5
    """

    HR: int = Field(default=80, description="Heart Rate in bpm")
    BP_sys: int = Field(default=120, description="Systolic Blood Pressure mmHg")
    BP_dia: int = Field(default=80, description="Diastolic Blood Pressure mmHg")
    SpO2: int = Field(default=98, description="Oxygen Saturation %")
    Temp: float = Field(default=37.0, description="Body Temperature °C")
    trend: str = Field(default="No prior data.", description="Trend summary of last steps")
    step_number: int = Field(default=0, description="Current step in episode")
    task_name: str = Field(default="vital_stabilization", description="Active task name")
    last_action_error: Optional[str] = Field(default=None, description="Error from last action")
