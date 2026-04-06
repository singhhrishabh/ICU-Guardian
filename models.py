# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
Data models for the ICU-Guardian Environment.

Uses Python dataclasses inheriting from OpenEnv base types.
These are server-side models used with the OpenEnv framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass(kw_only=True)
class Action:
    """Base class for all environment actions."""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)  
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class ICUObservation(Observation):
    """
    The patient's current vital signs and environment state.
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
