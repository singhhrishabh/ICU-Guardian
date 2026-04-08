# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
ICU-Guardian Environment Implementation.

Implements the OpenEnv Environment interface with:
- reset(): Initialize a new patient episode
- step(action): Apply clinical action, advance simulation, return observation
- state: Property returning episode metadata (State dataclass)
"""

import os
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dataclasses import dataclass, field

# Ensure parent directory is in path for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ICUAction, ICUObservation, Action, Observation, State
from server.simulator import ICUSimulator, SAFE_RANGES, TARGETS
from server.tasks import (
    TASKS,
    TaskConfig,
    grade_vital_stabilization,
    grade_bp_management,
    grade_sepsis_detection,
)


class ICUEnvironment:
    """
    ICU Patient Monitoring Environment.

    An AI agent monitors a critically ill patient's vital signs and must
    take clinical actions to stabilize the patient.
    """

    def __init__(self, task_name: Optional[str] = None):
        self._task_name = task_name or os.getenv("ICU_TASK", "vital_stabilization")
        self._task_config: Optional[TaskConfig] = None
        self._simulator: Optional[ICUSimulator] = None
        self._state = State(episode_id=str(uuid4())[:8], step_count=0)
        self._done: bool = False
        self._step_vitals: List[Dict] = []
        self._actions_taken: List[Dict] = []
        self._cumulative_reward: float = 0.0
        self._last_safe_fraction: float = 0.0
        self._sepsis_detection_step: Optional[int] = None
        self._last_action_error: Optional[str] = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> ICUObservation:
        """Initialize a new episode with a fresh patient."""
        if self._task_name not in TASKS:
            self._task_name = "vital_stabilization"

        self._task_config = TASKS[self._task_name]
        self._state = State(
            episode_id=episode_id or str(uuid4())[:8],
            step_count=0,
        )
        self._done = False
        self._step_vitals = []
        self._actions_taken = []
        self._cumulative_reward = 0.0
        self._last_safe_fraction = 0.0
        self._sepsis_detection_step = None
        self._last_action_error = None

        self._simulator = ICUSimulator(
            scenario=self._task_config.scenario,
            seed=seed or self._task_config.seed,
        )

        vitals = self._simulator.get_vitals()
        return ICUObservation(
            done=False,
            reward=0.01,
            HR=vitals.HR,
            BP_sys=vitals.BP_sys,
            BP_dia=vitals.BP_dia,
            SpO2=vitals.SpO2,
            Temp=vitals.Temp,
            trend="Initial assessment. No prior data.",
            step_number=0,
            task_name=self._task_name,
            last_action_error=None,
        )

    def step(self, action, timeout_s=None, **kwargs) -> ICUObservation:
        """Execute a clinical action and advance the simulation."""
        # Parse action - may come as ICUAction or generic Action
        if isinstance(action, ICUAction):
            icu_action = action
        elif hasattr(action, 'action'):
            icu_action = action
        else:
            icu_action = ICUAction(action="wait")

        if self._done:
            vitals = self._simulator.get_vitals()
            return self._make_observation(vitals, 0.0, True)

        self._state.step_count += 1

        action_dict = {
            "action": getattr(icu_action, 'action', 'wait'),
            "drug": getattr(icu_action, 'drug', None),
            "dose": getattr(icu_action, 'dose', None),
            "level": getattr(icu_action, 'level', None),
            "step": self._state.step_count,
        }
        self._actions_taken.append(action_dict)

        error = self._simulator.apply_action(
            action=action_dict["action"],
            drug=action_dict["drug"],
            dose=action_dict["dose"],
            level=action_dict["level"],
        )
        self._last_action_error = error

        if action_dict["action"] == "trigger_code_sepsis" and self._sepsis_detection_step is None:
            self._sepsis_detection_step = self._state.step_count

        self._simulator.advance()

        vitals = self._simulator.get_vitals()
        vitals_dict = {
            "HR": vitals.HR,
            "BP_sys": vitals.BP_sys,
            "BP_dia": vitals.BP_dia,
            "SpO2": vitals.SpO2,
            "Temp": vitals.Temp,
        }
        self._step_vitals.append(vitals_dict)

        reward = self._compute_reward(icu_action, vitals_dict, error)
        self._cumulative_reward += reward

        self._done = self._check_done()

        return self._make_observation(vitals, reward, self._done)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_score(self) -> float:
        """Compute the final graded score. Returns score in [0.0, 1.0]."""
        if not self._task_config or not self._simulator:
            return 0.01

        patient_crashed = self._simulator.is_patient_critical()

        raw_score = 0.0
        if self._task_name == "vital_stabilization":
            raw_score = grade_vital_stabilization(
                step_vitals=self._step_vitals,
                total_steps=self._state.step_count,
                max_steps=self._task_config.max_steps,
                patient_crashed=patient_crashed,
            )
        elif self._task_name == "bp_management":
            raw_score = grade_bp_management(
                step_vitals=self._step_vitals,
                actions_taken=self._actions_taken,
                total_steps=self._state.step_count,
                max_steps=self._task_config.max_steps,
                patient_crashed=patient_crashed,
            )
        elif self._task_name == "sepsis_detection":
            raw_score = grade_sepsis_detection(
                step_vitals=self._step_vitals,
                actions_taken=self._actions_taken,
                total_steps=self._state.step_count,
                max_steps=self._task_config.max_steps,
                sepsis_onset_step=self._simulator.patient.sepsis_onset_step,
                sepsis_detected=self._simulator.patient.sepsis_detected,
                sepsis_detection_step=self._sepsis_detection_step,
                patient_crashed=patient_crashed,
            )
        
        # STICKTLY BETWEEN 0 and 1: Hackathon requirement (not 0.0 and not 1.0)
        # Using [0.01, 0.99] to avoid rounding to 0.00 or 1.00 in logs
        return max(0.01, min(0.99, raw_score))

    def _compute_reward(self, action, vitals_dict, error):
        """Dense reward in [0.0, 1.0] range."""
        reward = 0.0
        current_safe = self._simulator.safe_zone_fraction()
        reward += current_safe * 0.6

        improvement = current_safe - self._last_safe_fraction
        if improvement > 0:
            reward += min(improvement, 1.0) * 0.2
        self._last_safe_fraction = current_safe

        action_quality = self._evaluate_action_quality(action, vitals_dict)
        reward += action_quality * 0.2

        if error:
            reward -= 0.05

        action_name = getattr(action, 'action', 'wait')
        if action_name == "trigger_code_sepsis":
            if self._task_name != "sepsis_detection":
                reward -= 0.3
            elif not self._simulator.patient.sepsis_active:
                reward -= 0.3

        # STICKTLY BETWEEN 0 and 1: Hackathon requirement (not 0.0 and not 1.0)
        # Using [0.01, 0.99] to avoid rounding to 0.00 or 1.00 in logs
        return max(0.01, min(0.99, reward))

    def _evaluate_action_quality(self, action, vitals):
        """Evaluate how appropriate the chosen action is."""
        quality = 0.0
        action_name = getattr(action, 'action', 'wait')
        drug = getattr(action, 'drug', None)
        level = getattr(action, 'level', None)

        if action_name == "wait":
            safe_frac = self._simulator.safe_zone_fraction()
            quality = 0.8 if safe_frac >= 0.8 else 0.3
        elif action_name == "administer_meds":
            if drug == "vasopressor":
                if vitals["BP_sys"] < SAFE_RANGES["BP_sys"][0]:
                    quality = 1.0
                elif vitals["BP_sys"] < TARGETS["BP_sys"]:
                    quality = 0.6
                else:
                    quality = 0.1
            elif drug == "antihypertensive":
                if vitals["BP_sys"] > SAFE_RANGES["BP_sys"][1]:
                    quality = 1.0
                elif vitals["BP_sys"] > TARGETS["BP_sys"]:
                    quality = 0.6
                else:
                    quality = 0.1
        elif action_name == "adjust_oxygen":
            if level == "increase":
                if vitals["SpO2"] < SAFE_RANGES["SpO2"][0]:
                    quality = 1.0
                elif vitals["SpO2"] < TARGETS["SpO2"]:
                    quality = 0.6
                else:
                    quality = 0.2
            else:
                if vitals["SpO2"] >= 99 and self._simulator.patient.oxygen_support_level > 0:
                    quality = 0.7
                else:
                    quality = 0.2
        elif action_name == "trigger_code_sepsis":
            if self._simulator.patient.sepsis_active and self._simulator.patient.sepsis_stage >= 1:
                quality = 1.0

        return quality

    def _check_done(self):
        """Check if the episode should end."""
        if self._state.step_count >= self._task_config.max_steps:
            return True
        if self._simulator.is_patient_critical():
            return True
        if self._task_name == "sepsis_detection":
            if self._simulator.patient.sepsis_detected:
                return True
            if self._simulator.patient.sepsis_stage >= 3:
                return True
        return False

    def _make_observation(self, vitals, reward, done):
        """Create an ICUObservation."""
        return ICUObservation(
            done=done,
            reward=round(reward, 4),
            HR=vitals.HR,
            BP_sys=vitals.BP_sys,
            BP_dia=vitals.BP_dia,
            SpO2=vitals.SpO2,
            Temp=vitals.Temp,
            trend=self._simulator.get_trend_summary() if self._simulator else "",
            step_number=self._state.step_count,
            task_name=self._task_name,
            last_action_error=self._last_action_error,
            metadata={"score": self.get_score()} if done else {},
        )
