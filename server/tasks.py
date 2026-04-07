"""
ICU-Guardian: Task Definitions and Graders

Defines 3 tasks of increasing difficulty:
1. Vital Sign Stabilization (Easy)
2. Post-Surgical BP Management (Medium)
3. Sepsis Detection & Response (Hard)

Each task has:
- Configuration (scenario, max steps, thresholds)
- A grader function that returns a score in [0.0, 1.0]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .simulator import ICUSimulator, SAFE_RANGES, TARGETS


@dataclass
class TaskConfig:
    """Configuration for an ICU task."""
    name: str
    description: str
    scenario: str           # Maps to simulator scenario
    max_steps: int
    difficulty: str         # "easy" | "medium" | "hard"
    seed: Optional[int] = None


# === Task Definitions ===

TASKS: Dict[str, TaskConfig] = {
    "vital_stabilization": TaskConfig(
        name="vital_stabilization",
        description=(
            "A patient presents with mild hypotension and low oxygen saturation. "
            "Stabilize all vital signs within safe ranges. "
            "This is an introductory task testing basic clinical reasoning."
        ),
        scenario="stable",
        max_steps=15,
        difficulty="easy",
        seed=42,
    ),
    "bp_management": TaskConfig(
        name="bp_management",
        description=(
            "A post-surgical patient exhibits volatile blood pressure swings. "
            "Carefully titrate medications to maintain stable BP without overcorrection. "
            "Excessive medication doses will crash the patient's blood pressure."
        ),
        scenario="post_surgical",
        max_steps=20,
        difficulty="medium",
        seed=123,
    ),
    "sepsis_detection": TaskConfig(
        name="sepsis_detection",
        description=(
            "A patient appears initially stable but will develop sepsis during the episode. "
            "Monitor for the multi-variable pattern (rising temperature + dropping BP + spiking HR) "
            "and trigger the sepsis protocol at the right moment. "
            "False alarms are penalized. Missing sepsis onset leads to episode failure."
        ),
        scenario="sepsis",
        max_steps=25,
        difficulty="hard",
        seed=456,
    ),
}


# === Grading Functions ===

def compute_vital_distance(vital_name: str, value: float) -> float:
    """
    Compute normalized distance of a vital sign from its target.
    Returns 0.0 (at target) to 1.0 (maximally deviated).
    """
    target = TARGETS[vital_name]
    safe_low, safe_high = SAFE_RANGES[vital_name]
    range_size = max(safe_high - safe_low, 1)
    distance = abs(value - target) / range_size
    return min(distance, 1.0)


def grade_vital_stabilization(
    step_vitals: List[Dict],
    total_steps: int,
    max_steps: int,
    patient_crashed: bool,
) -> float:
    """
    Grade the Vital Stabilization (Easy) task.
    ...
    Returns score in [0.0, 1.0]
    """
    if patient_crashed or total_steps == 0:
        return 0.0001

    # Component 1: Steps in safe zone (60%)
    safe_steps = 0
    for vitals in step_vitals:
        all_safe = all(
            SAFE_RANGES[k][0] <= vitals[k] <= SAFE_RANGES[k][1]
            for k in ["HR", "BP_sys", "BP_dia", "SpO2", "Temp"]
        )
        if all_safe:
            safe_steps += 1
    safe_fraction = safe_steps / len(step_vitals) if step_vitals else 0.0

    # Component 2: Final proximity to targets (25%)
    final_steps = step_vitals[-3:] if len(step_vitals) >= 3 else step_vitals
    avg_distance = 0.0
    for vitals in final_steps:
        step_dist = sum(
            compute_vital_distance(k, vitals[k])
            for k in ["HR", "BP_sys", "BP_dia", "SpO2", "Temp"]
        ) / 5
        avg_distance += step_dist
    avg_distance /= len(final_steps) if final_steps else 1
    proximity_score = max(0.0, 1.0 - avg_distance)

    # Component 3: Efficiency (15%)
    efficiency = max(0.0, 1.0 - (total_steps / max_steps))

    score = 0.60 * safe_fraction + 0.25 * proximity_score + 0.15 * efficiency
    # STRICTLY BETWEEN 0 and 1
    return round(min(max(score, 0.0001), 0.9999), 4)


def grade_bp_management(
    step_vitals: List[Dict],
    actions_taken: List[Dict],
    total_steps: int,
    max_steps: int,
    patient_crashed: bool,
) -> float:
    """
    Grade the BP Management (Medium) task.
    ...
    Returns score in [0.0, 1.0]
    """
    if patient_crashed or total_steps == 0:
        return 0.0001

    # Component 1: BP stability (45%)
    bp_values = [v["BP_sys"] for v in step_vitals]
    if len(bp_values) >= 2:
        bp_changes = [abs(bp_values[i] - bp_values[i - 1]) for i in range(1, len(bp_values))]
        avg_change = sum(bp_changes) / len(bp_changes)
        # Normalize: 0 change = 1.0, 20+ change = 0.0
        stability_score = max(0.0, 1.0 - avg_change / 20)
    else:
        stability_score = 0.5

    # Component 2: BP in safe range (30%)
    safe_bp_steps = sum(
        1 for v in step_vitals
        if SAFE_RANGES["BP_sys"][0] <= v["BP_sys"] <= SAFE_RANGES["BP_sys"][1]
        and SAFE_RANGES["BP_dia"][0] <= v["BP_dia"] <= SAFE_RANGES["BP_dia"][1]
    )
    bp_safe_fraction = safe_bp_steps / len(step_vitals) if step_vitals else 0.0

    # Component 3: Overcorrection penalty (15%)
    overcorrections = 0
    for i in range(1, len(bp_values)):
        # Detect BP crossing target in opposite direction
        if (bp_values[i - 1] < TARGETS["BP_sys"] and bp_values[i] > TARGETS["BP_sys"] + 15) or \
           (bp_values[i - 1] > TARGETS["BP_sys"] and bp_values[i] < TARGETS["BP_sys"] - 15):
            overcorrections += 1
    overcorrection_penalty = max(0.0, 1.0 - overcorrections * 0.25)

    # Component 4: Medication efficiency (10%)
    med_actions = sum(1 for a in actions_taken if a.get("action") == "administer_meds")
    med_ratio = med_actions / max(total_steps, 1)
    med_efficiency = max(0.0, 1.0 - med_ratio * 1.5)

    score = (0.45 * stability_score + 0.30 * bp_safe_fraction +
             0.15 * overcorrection_penalty + 0.10 * med_efficiency)
    # STRICTLY BETWEEN 0 and 1
    return round(min(max(score, 0.0001), 0.9999), 4)


def grade_sepsis_detection(
    step_vitals: List[Dict],
    actions_taken: List[Dict],
    total_steps: int,
    max_steps: int,
    sepsis_onset_step: int,
    sepsis_detected: bool,
    sepsis_detection_step: Optional[int],
    patient_crashed: bool,
) -> float:
    """
    Grade the Sepsis Detection (Hard) task.

    Score = weighted combination of:
    - 40%: Detection accuracy (did the agent correctly identify sepsis?)
    - 25%: Detection timing (how quickly after onset?)
    - 20%: Vital management during episode
    - 15%: False alarm avoidance

    Returns score in [0.0, 1.0]
    """
    if patient_crashed:
        return 0.05  # Minimal score for crashing

    # Component 1: Detection accuracy (40%)
    false_alarms = sum(
        1 for i, a in enumerate(actions_taken)
        if a.get("action") == "trigger_code_sepsis"
        and (i + 1) < sepsis_onset_step  # Triggered before sepsis started
    )

    if sepsis_detected and sepsis_detection_step is not None:
        # Must have detected after onset
        if sepsis_detection_step >= sepsis_onset_step:
            detection_score = 1.0
        else:
            detection_score = 0.1  # False alarm that happened to be before onset
    else:
        detection_score = 0.0  # Missed sepsis entirely

    # Component 2: Detection timing (25%)
    if sepsis_detected and sepsis_detection_step is not None and sepsis_detection_step >= sepsis_onset_step:
        delay = sepsis_detection_step - sepsis_onset_step
        # Ideal: detect within 2-4 steps of onset
        if delay <= 2:
            timing_score = 1.0
        elif delay <= 4:
            timing_score = 0.8
        elif delay <= 6:
            timing_score = 0.5
        else:
            timing_score = 0.2
    else:
        timing_score = 0.0

    # Component 3: Vital management (20%)
    safe_steps = 0
    for vitals in step_vitals:
        all_safe = all(
            SAFE_RANGES[k][0] <= vitals[k] <= SAFE_RANGES[k][1]
            for k in ["HR", "BP_sys", "BP_dia", "SpO2", "Temp"]
        )
        if all_safe:
            safe_steps += 1
    vital_score = safe_steps / len(step_vitals) if step_vitals else 0.0

    # Component 4: False alarm avoidance (15%)
    false_alarm_score = max(0.0, 1.0 - false_alarms * 0.5)

    score = (0.40 * detection_score + 0.25 * timing_score +
             0.20 * vital_score + 0.15 * false_alarm_score)
    # STRICTLY BETWEEN 0 and 1
    return round(min(max(score, 0.0001), 0.9999), 4)
