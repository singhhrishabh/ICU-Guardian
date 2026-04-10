# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
Data models for the ICU-Guardian Environment.

Uses Python dataclasses inheriting from OpenEnv base types.
These are server-side models used with the OpenEnv framework.
Compatible with Python 3.9+ (no kw_only).
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

_KW = {"kw_only": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_KW)
class Action:
    """Base class for all environment actions."""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(**_KW)
class Observation:
    """Base class for all environment observations."""
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    """Base class for environment state."""
    episode_id: Optional[str] = None
    step_count: int = 0


@dataclass(**_KW)
class ICUAction(Action):
    """
    An action the AI agent can take in the ICU environment.

    Action Types:
        - "administer_meds": Give medication (requires drug + dose)
        - "adjust_oxygen": Change oxygen support (requires level)
        - "trigger_code_sepsis": Emergency sepsis alert
        - "wait": Monitor without intervention
    """

    action: str = "wait"
    drug: Optional[str] = None
    dose: Optional[str] = None
    level: Optional[str] = None
    rationale: Optional[str] = None  # Explainable AI functionality


@dataclass(**_KW)
class ICUObservation(Observation):
    """
    The patient's current vital signs and environment state.
    """

    HR: int = 80
    BP_sys: int = 120
    BP_dia: int = 80
    SpO2: int = 98
    Temp: float = 37.0
    lactate: float = 1.0  # Lab value for sepsis (mmol/L)
    organ_stress: float = 0.0  # Cumulative damage metric
    patient_profile: str = "Standard Adult"
    trend: str = "No prior data."
    step_number: int = 0
    task_name: str = "vital_stabilization"
    last_action_error: Optional[str] = None
