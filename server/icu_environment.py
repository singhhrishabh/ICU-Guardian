# Copyright (c) ICU-Guardian Contributors
# Licensed under MIT License

"""
ICU-Guardian Environment Implementation.

Implements the OpenEnv Environment base class with:
- reset(): Initialize a new patient episode
- step(action): Apply clinical action, advance simulation, return results  
- state: Property returning episode metadata
"""

import os
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv_core.env_server.interfaces import Environment
    from openenv_core.env_server.types import Action, Observation, State

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ICUAction, ICUObservation
from server.simulator import ICUSimulator, SAFE_RANGES, TARGETS
from server.tasks import (
    TASKS,
    TaskConfig,
    grade_vital_stabilization,
    grade_bp_management,
    grade_sepsis_detection,
)


class ICUEnvironment(Environment):
    """
    ICU Patient Monitoring Environment.

    An AI agent monitors a critically ill patient's vital signs and must
    take clinical actions to stabilize the patient. The environment supports
    3 tasks of increasing difficulty.
    """

    def __init__(self, task_name: Optional[str] = None):
        self._task_name = task_name or os.getenv("ICU_TASK", "vital_stabilization")
        self._task_config: Optional[TaskConfig] = None
        self._simulator: Optional[ICUSimulator] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done: bool = False
        self._step_vitals: List[Dict] = []
        self._actions_taken: List[Dict] = []
        self._cumulative_reward: float = 0.0
        self._last_safe_fraction: float = 0.0
        self._sepsis_detection_step: Optional[int] = None
        self._last_action_error: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
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

        # Create simulator with task-specific scenario
        self._simulator = ICUSimulator(
            scenario=self._task_config.scenario,
            seed=seed or self._task_config.seed,
        )

        vitals = self._simulator.get_vitals()
        return ICUObservation(
            done=False,
            reward=0.0,
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

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a clinical action and advance the simulation."""
        # Parse the action - may come as ICUAction or generic Action
        if isinstance(action, ICUAction):
            icu_action = action
        else:
            # Try to convert from generic action dict/object
            action_data = action.model_dump() if hasattr(action, 'model_dump') else {}
            icu_action = ICUAction(
                action=action_data.get("action", "wait"),
                drug=action_data.get("drug"),
                dose=action_data.get("dose"),
                level=action_data.get("level"),
            )

        if self._done:
            vitals = self._simulator.get_vitals()
            return self._make_observation(vitals, 0.0, True)

        self._state.step_count += 1

        # Record the action
        action_dict = {
            "action": icu_action.action,
            "drug": icu_action.drug,
            "dose": icu_action.dose,
            "level": icu_action.level,
            "step": self._state.step_count,
        }
        self._actions_taken.append(action_dict)

        # Apply the action
        error = self._simulator.apply_action(
            action=icu_action.action,
            drug=icu_action.drug,
            dose=icu_action.dose,
            level=icu_action.level,
        )
        self._last_action_error = error

        # Track sepsis detection timing
        if icu_action.action == "trigger_code_sepsis" and self._sepsis_detection_step is None:
            self._sepsis_detection_step = self._state.step_count

        # Advance the simulation
        self._simulator.advance()

        # Record vitals after step
        vitals = self._simulator.get_vitals()
        vitals_dict = {
            "HR": vitals.HR,
            "BP_sys": vitals.BP_sys,
            "BP_dia": vitals.BP_dia,
            "SpO2": vitals.SpO2,
            "Temp": vitals.Temp,
        }
        self._step_vitals.append(vitals_dict)

        # Compute reward
        reward = self._compute_reward(icu_action, vitals_dict, error)
        self._cumulative_reward += reward

        # Check done conditions
        self._done = self._check_done()

        return self._make_observation(vitals, reward, self._done)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step used by the WebSocket handler."""
        return self.step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_score(self) -> float:
        """Compute the final graded score. Returns score in [0.0, 1.0]."""
        if not self._task_config or not self._simulator:
            return 0.0

        patient_crashed = self._simulator.is_patient_critical()

        if self._task_name == "vital_stabilization":
            return grade_vital_stabilization(
                step_vitals=self._step_vitals,
                total_steps=self._state.step_count,
                max_steps=self._task_config.max_steps,
                patient_crashed=patient_crashed,
            )
        elif self._task_name == "bp_management":
            return grade_bp_management(
                step_vitals=self._step_vitals,
                actions_taken=self._actions_taken,
                total_steps=self._state.step_count,
                max_steps=self._task_config.max_steps,
                patient_crashed=patient_crashed,
            )
        elif self._task_name == "sepsis_detection":
            return grade_sepsis_detection(
                step_vitals=self._step_vitals,
                actions_taken=self._actions_taken,
                total_steps=self._state.step_count,
                max_steps=self._task_config.max_steps,
                sepsis_onset_step=self._simulator.patient.sepsis_onset_step,
                sepsis_detected=self._simulator.patient.sepsis_detected,
                sepsis_detection_step=self._sepsis_detection_step,
                patient_crashed=patient_crashed,
            )
        return 0.0

    def _compute_reward(self, action: ICUAction, vitals_dict: Dict,
                        error: Optional[str]) -> float:
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

        if action.action == "trigger_code_sepsis":
            if self._task_name != "sepsis_detection":
                reward -= 0.3
            elif not self._simulator.patient.sepsis_active:
                reward -= 0.3

        return max(0.0, min(1.0, reward))

    def _evaluate_action_quality(self, action: ICUAction, vitals: Dict) -> float:
        """Evaluate how appropriate the chosen action is."""
        quality = 0.0

        if action.action == "wait":
            safe_frac = self._simulator.safe_zone_fraction()
            quality = 0.8 if safe_frac >= 0.8 else 0.3

        elif action.action == "administer_meds":
            if action.drug == "vasopressor":
                if vitals["BP_sys"] < SAFE_RANGES["BP_sys"][0]:
                    quality = 1.0
                elif vitals["BP_sys"] < TARGETS["BP_sys"]:
                    quality = 0.6
                else:
                    quality = 0.1
            elif action.drug == "antihypertensive":
                if vitals["BP_sys"] > SAFE_RANGES["BP_sys"][1]:
                    quality = 1.0
                elif vitals["BP_sys"] > TARGETS["BP_sys"]:
                    quality = 0.6
                else:
                    quality = 0.1

        elif action.action == "adjust_oxygen":
            if action.level == "increase":
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

        elif action.action == "trigger_code_sepsis":
            if self._simulator.patient.sepsis_active and self._simulator.patient.sepsis_stage >= 1:
                quality = 1.0
            else:
                quality = 0.0

        return quality

    def _check_done(self) -> bool:
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
