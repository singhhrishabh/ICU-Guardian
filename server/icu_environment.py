"""
ICU-Guardian: OpenEnv Environment Implementation

Implements the core Environment base class from openenv-core with:
- reset(): Initialize a new patient episode
- step(action): Apply clinical action, advance simulation, return results
- state(): Return episode metadata
"""

import os
import uuid
import sys
from typing import Dict, List, Optional

class Environment:
    """Base class for OpenEnv-style environments."""
    pass

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ICUAction, ICUObservation, ICUState
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
        self.task_name = task_name or os.getenv("ICU_TASK", "vital_stabilization")
        self.task_config: Optional[TaskConfig] = None
        self.simulator: Optional[ICUSimulator] = None
        self.episode_id: str = ""
        self.step_count: int = 0
        self.done: bool = False
        self.step_vitals: List[Dict] = []
        self.actions_taken: List[Dict] = []
        self.cumulative_reward: float = 0.0
        self.last_safe_fraction: float = 0.0
        self.sepsis_detection_step: Optional[int] = None
        self.last_action_error: Optional[str] = None

    def reset(self) -> ICUObservation:
        """Initialize a new episode with a fresh patient."""
        if self.task_name not in TASKS:
            self.task_name = "vital_stabilization"

        self.task_config = TASKS[self.task_name]
        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.done = False
        self.step_vitals = []
        self.actions_taken = []
        self.cumulative_reward = 0.0
        self.last_safe_fraction = 0.0
        self.sepsis_detection_step = None
        self.last_action_error = None

        # Create simulator with task-specific scenario
        self.simulator = ICUSimulator(
            scenario=self.task_config.scenario,
            seed=self.task_config.seed,
        )

        vitals = self.simulator.get_vitals()
        return ICUObservation(
            HR=vitals.HR,
            BP_sys=vitals.BP_sys,
            BP_dia=vitals.BP_dia,
            SpO2=vitals.SpO2,
            Temp=vitals.Temp,
            trend="Initial assessment. No prior data.",
            step_number=0,
            task_name=self.task_name,
            last_action_error=None,
        )

    def step(self, action: ICUAction) -> Dict:
        """
        Execute a clinical action and advance the simulation.

        Returns a dict with observation, reward, done, and optional score.
        """
        if self.done:
            vitals = self.simulator.get_vitals()
            return self._make_step_result(vitals, 0.0, True)

        self.step_count += 1

        # Record the action
        action_dict = {
            "action": action.action,
            "drug": action.drug,
            "dose": action.dose,
            "level": action.level,
            "step": self.step_count,
        }
        self.actions_taken.append(action_dict)

        # Apply the action
        error = self.simulator.apply_action(
            action=action.action,
            drug=action.drug,
            dose=action.dose,
            level=action.level,
        )
        self.last_action_error = error

        # Track sepsis detection timing
        if action.action == "trigger_code_sepsis" and self.sepsis_detection_step is None:
            self.sepsis_detection_step = self.step_count

        # Advance the simulation
        self.simulator.advance()

        # Record vitals after step
        vitals = self.simulator.get_vitals()
        vitals_dict = {
            "HR": vitals.HR,
            "BP_sys": vitals.BP_sys,
            "BP_dia": vitals.BP_dia,
            "SpO2": vitals.SpO2,
            "Temp": vitals.Temp,
        }
        self.step_vitals.append(vitals_dict)

        # Compute reward
        reward = self._compute_reward(action, vitals_dict, error)
        self.cumulative_reward += reward

        # Check done conditions
        self.done = self._check_done()

        return self._make_step_result(vitals, reward, self.done)

    def state(self) -> ICUState:
        """Return current episode metadata."""
        return ICUState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            task_name=self.task_name,
            max_steps=self.task_config.max_steps if self.task_config else 15,
            done=self.done,
        )

    def get_score(self) -> float:
        """
        Compute the final graded score for the episode.
        Called after episode ends. Returns score in [0.0, 1.0].
        """
        if not self.task_config or not self.simulator:
            return 0.0

        patient_crashed = self.simulator.is_patient_critical()

        if self.task_name == "vital_stabilization":
            return grade_vital_stabilization(
                step_vitals=self.step_vitals,
                total_steps=self.step_count,
                max_steps=self.task_config.max_steps,
                patient_crashed=patient_crashed,
            )
        elif self.task_name == "bp_management":
            return grade_bp_management(
                step_vitals=self.step_vitals,
                actions_taken=self.actions_taken,
                total_steps=self.step_count,
                max_steps=self.task_config.max_steps,
                patient_crashed=patient_crashed,
            )
        elif self.task_name == "sepsis_detection":
            return grade_sepsis_detection(
                step_vitals=self.step_vitals,
                actions_taken=self.actions_taken,
                total_steps=self.step_count,
                max_steps=self.task_config.max_steps,
                sepsis_onset_step=self.simulator.patient.sepsis_onset_step,
                sepsis_detected=self.simulator.patient.sepsis_detected,
                sepsis_detection_step=self.sepsis_detection_step,
                patient_crashed=patient_crashed,
            )
        return 0.0

    def _compute_reward(self, action: ICUAction, vitals_dict: Dict,
                        error: Optional[str]) -> float:
        """
        Compute per-step reward. Dense reward in [0.0, 1.0] range.

        Components:
        - Base: fraction of vitals in safe zone (0.0-0.6)
        - Improvement: positive change from last step (0.0-0.2)
        - Action quality: appropriate action for situation (0.0-0.2)
        """
        reward = 0.0

        # Base: safe zone fraction
        current_safe = self.simulator.safe_zone_fraction()
        reward += current_safe * 0.6

        # Improvement bonus
        improvement = current_safe - self.last_safe_fraction
        if improvement > 0:
            reward += min(improvement, 1.0) * 0.2
        self.last_safe_fraction = current_safe

        # Action quality
        action_quality = self._evaluate_action_quality(action, vitals_dict)
        reward += action_quality * 0.2

        # Penalties
        if error:
            reward -= 0.05

        if action.action == "trigger_code_sepsis":
            if self.task_name != "sepsis_detection":
                reward -= 0.3
            elif not self.simulator.patient.sepsis_active:
                reward -= 0.3

        return max(0.0, min(1.0, reward))

    def _evaluate_action_quality(self, action: ICUAction, vitals: Dict) -> float:
        """Evaluate how appropriate the chosen action is for the current state."""
        quality = 0.0

        if action.action == "wait":
            safe_frac = self.simulator.safe_zone_fraction()
            if safe_frac >= 0.8:
                quality = 0.8
            else:
                quality = 0.3

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
                if vitals["SpO2"] >= 99 and self.simulator.patient.oxygen_support_level > 0:
                    quality = 0.7
                else:
                    quality = 0.2

        elif action.action == "trigger_code_sepsis":
            if self.simulator.patient.sepsis_active and self.simulator.patient.sepsis_stage >= 1:
                quality = 1.0
            else:
                quality = 0.0

        return quality

    def _check_done(self) -> bool:
        """Check if the episode should end."""
        if self.step_count >= self.task_config.max_steps:
            return True

        if self.simulator.is_patient_critical():
            return True

        if self.task_name == "sepsis_detection":
            if self.simulator.patient.sepsis_detected:
                return True
            if self.simulator.patient.sepsis_stage >= 3:
                return True

        return False

    def _make_step_result(self, vitals, reward, done):
        """Create the step result dict."""
        obs = ICUObservation(
            HR=vitals.HR,
            BP_sys=vitals.BP_sys,
            BP_dia=vitals.BP_dia,
            SpO2=vitals.SpO2,
            Temp=vitals.Temp,
            trend=self.simulator.get_trend_summary() if self.simulator else "",
            step_number=self.step_count,
            task_name=self.task_name,
            last_action_error=self.last_action_error,
        )

        return {
            "observation": obs,
            "reward": round(reward, 4),
            "done": done,
            "score": self.get_score() if done else None,
        }
