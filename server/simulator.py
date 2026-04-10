"""
ICU-Guardian: Physiological Patient Simulator

Simulates realistic ICU patient vital sign dynamics including:
- Inter-variable coupling (e.g., low BP → compensatory tachycardia)
- Medication pharmacokinetics (delayed onset, dose-dependent effects)
- Sepsis cascade progression
- Stochastic noise with bounded realism
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class VitalSigns:
    """Current patient vital signs."""
    HR: int = 80
    BP_sys: int = 120
    BP_dia: int = 80
    SpO2: int = 98
    Temp: float = 37.0


@dataclass
class MedicationEffect:
    """Tracks an active medication's effect over time."""
    drug: str
    dose: str
    steps_remaining: int
    peak_reached: bool = False


@dataclass
class PatientState:
    """Full internal state of the simulated patient."""
    vitals: VitalSigns = field(default_factory=VitalSigns)
    active_medications: List[MedicationEffect] = field(default_factory=list)
    oxygen_support_level: int = 0  # 0=room air, 1-5=supplemental levels
    sepsis_active: bool = False
    sepsis_stage: int = 0  # 0=none, 1=early, 2=developing, 3=severe
    sepsis_detected: bool = False
    profile_name: str = "Standard Adult"
    lactate: float = 1.0  # Normal < 2.0
    organ_stress: float = 0.0  # Accumulates based on critical state
    step_count: int = 0
    history: List[Dict] = field(default_factory=list)


# === SAFE RANGES ===
SAFE_RANGES = {
    "HR": (60, 100),
    "BP_sys": (100, 140),
    "BP_dia": (60, 90),
    "SpO2": (95, 100),
    "Temp": (36.0, 37.8),
}

# Target values for reward calculation
TARGETS = {
    "HR": 78,
    "BP_sys": 120,
    "BP_dia": 80,
    "SpO2": 98,
    "Temp": 37.0,
}


class ICUSimulator:
    """
    Simulates a patient in the ICU with realistic vital sign dynamics.

    The simulator models:
    - Natural vital sign drift with noise
    - Physiological coupling between variables
    - Medication pharmacokinetics
    - Sepsis cascade for hard-mode scenarios
    """

    def __init__(self, scenario: str = "stable", seed: Optional[int] = None):
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.patient = PatientState()
        self._configure_scenario()

    def _configure_scenario(self):
        """Set initial vitals based on the scenario."""
        if self.scenario == "stable":
            # Easy: Slightly abnormal vitals
            self.patient.profile_name = "Standard Adult"
            self.patient.vitals = VitalSigns(
                HR=self.rng.randint(88, 105),
                BP_sys=self.rng.randint(95, 108),
                BP_dia=self.rng.randint(55, 68),
                SpO2=self.rng.randint(90, 94),
                Temp=round(36.5 + self.rng.random() * 1.0, 1),
            )
            self.patient.oxygen_support_level = 0
            self.drift_intensity = 0.3
            self.noise_intensity = 0.5

        elif self.scenario == "post_surgical":
            # Medium: Volatile BP, otherwise okay
            self.patient.profile_name = "Elderly with COPD" # Example profile altering baseline
            self.patient.vitals = VitalSigns(
                HR=self.rng.randint(90, 110),
                BP_sys=self.rng.randint(85, 100),
                BP_dia=self.rng.randint(50, 65),
                SpO2=self.rng.randint(93, 96),
                Temp=round(37.0 + self.rng.random() * 0.8, 1),
            )
            self.patient.oxygen_support_level = 1
            self.drift_intensity = 0.6
            self.noise_intensity = 1.0
            # BP will have periodic swings
            self._bp_swing_phase = self.rng.random() * math.pi * 2
            self._bp_swing_amplitude = self.rng.randint(8, 15)

        elif self.scenario == "sepsis":
            # Hard: Starts relatively stable, then sepsis develops
            self.patient.profile_name = "Immunocompromised"
            self.patient.vitals = VitalSigns(
                HR=self.rng.randint(75, 90),
                BP_sys=self.rng.randint(110, 125),
                BP_dia=self.rng.randint(70, 82),
                SpO2=self.rng.randint(95, 98),
                Temp=round(37.0 + self.rng.random() * 0.3, 1),
            )
            self.patient.oxygen_support_level = 0
            self.drift_intensity = 0.4
            self.noise_intensity = 0.6
            # Sepsis will begin between steps 5-8
            self.patient.sepsis_onset_step = self.rng.randint(5, 8)

        elif self.scenario == "weaning":
            # Advanced: Patient starts on max oxygen support but has improving intrinsic lung function.
            # Agent must titrate down O2 level safely.
            self.patient.profile_name = "Recovering Pneumonia"
            self.patient.vitals = VitalSigns(
                HR=self.rng.randint(75, 85),
                BP_sys=self.rng.randint(110, 125),
                BP_dia=self.rng.randint(70, 80),
                SpO2=self.rng.randint(97, 100),
                Temp=round(37.0 + self.rng.random() * 0.2, 1),
            )
            # High initial oxygen
            self.patient.oxygen_support_level = 5
            self.drift_intensity = 0.2
            self.noise_intensity = 0.4

    def get_vitals(self) -> VitalSigns:
        """Return current vital signs (copy)."""
        v = self.patient.vitals
        return VitalSigns(
            HR=v.HR, BP_sys=v.BP_sys, BP_dia=v.BP_dia,
            SpO2=v.SpO2, Temp=v.Temp,
        )

    def get_trend_summary(self) -> str:
        """Generate a natural language summary of the last 3 steps."""
        history = self.patient.history
        if len(history) == 0:
            return "No prior data. Initial assessment."

        recent = history[-3:]
        parts = []
        for i, entry in enumerate(recent):
            step_label = f"Step {entry['step']}"
            changes = []
            if entry.get("hr_delta", 0) != 0:
                direction = "↑" if entry["hr_delta"] > 0 else "↓"
                changes.append(f"HR {direction}{abs(entry['hr_delta'])}")
            if entry.get("bp_sys_delta", 0) != 0:
                direction = "↑" if entry["bp_sys_delta"] > 0 else "↓"
                changes.append(f"BP_sys {direction}{abs(entry['bp_sys_delta'])}")
            if entry.get("spo2_delta", 0) != 0:
                direction = "↑" if entry["spo2_delta"] > 0 else "↓"
                changes.append(f"SpO2 {direction}{abs(entry['spo2_delta'])}")
            if entry.get("temp_delta", 0) != 0:
                direction = "↑" if entry["temp_delta"] > 0 else "↓"
                changes.append(f"Temp {direction}{abs(entry['temp_delta']):.1f}")

            if changes:
                parts.append(f"{step_label}: {', '.join(changes)}")
            else:
                parts.append(f"{step_label}: stable")

        return " | ".join(parts)

    def apply_action(self, action: str, drug: Optional[str] = None,
                     dose: Optional[str] = None, level: Optional[str] = None) -> Optional[str]:
        """
        Apply a clinical action to the patient.

        Returns an error message if the action is invalid, None otherwise.
        """
        if action == "wait":
            pass  # No intervention

        elif action == "administer_meds":
            valid_drugs = ("vasopressor", "antihypertensive", "antibiotics", "fluids", "sedative")
            if drug not in valid_drugs:
                return f"Invalid drug: {drug}. Must be one of {valid_drugs}."
            if dose not in ("low", "high") and drug not in ("antibiotics", "fluids"):
                return f"Invalid dose for {drug}. Use 'low' or 'high'."

            # Check for dangerous drug interactions
            active_drugs = [m.drug for m in self.patient.active_medications]
            if drug == "vasopressor" and "antihypertensive" in active_drugs:
                return "WARNING: Conflicting medications — vasopressor given with active antihypertensive."
            if drug == "antihypertensive" and "vasopressor" in active_drugs:
                return "WARNING: Conflicting medications — antihypertensive given with active vasopressor."

            duration = 3 if dose == "low" else 4
            if drug == "antibiotics": duration = 10
            if drug == "fluids": duration = 2
            
            self.patient.active_medications.append(
                MedicationEffect(drug=drug, dose=dose or "standard", steps_remaining=duration)
            )

        elif action == "adjust_oxygen":
            if level not in ("increase", "decrease"):
                return f"Invalid level: {level}. Must be 'increase' or 'decrease'."
            if level == "increase":
                self.patient.oxygen_support_level = min(5, self.patient.oxygen_support_level + 1)
            else:
                self.patient.oxygen_support_level = max(0, self.patient.oxygen_support_level - 1)

        elif action == "trigger_code_sepsis":
            self.patient.sepsis_detected = True

        else:
            return f"Unknown action: {action}. Valid: wait, administer_meds, adjust_oxygen, trigger_code_sepsis."

        return None

    def advance(self):
        """
        Advance the simulation by one time step.

        This applies:
        1. Natural drift + noise
        2. Physiological coupling
        3. Medication effects
        4. Oxygen support effects
        5. Sepsis progression (if applicable)
        """
        v = self.patient.vitals
        prev_hr = v.HR
        prev_bp_sys = v.BP_sys
        prev_bp_dia = v.BP_dia
        prev_spo2 = v.SpO2
        prev_temp = v.Temp
        self.patient.step_count += 1

        # 1. Natural drift toward homeostasis + noise
        hr_drift = (TARGETS["HR"] - v.HR) * 0.05 * self.drift_intensity
        bp_sys_drift = (TARGETS["BP_sys"] - v.BP_sys) * 0.04 * self.drift_intensity
        bp_dia_drift = (TARGETS["BP_dia"] - v.BP_dia) * 0.04 * self.drift_intensity
        spo2_drift = (TARGETS["SpO2"] - v.SpO2) * 0.03 * self.drift_intensity
        temp_drift = (TARGETS["Temp"] - v.Temp) * 0.02 * self.drift_intensity

        hr_noise = self.rng.gauss(0, 2 * self.noise_intensity)
        bp_sys_noise = self.rng.gauss(0, 3 * self.noise_intensity)
        bp_dia_noise = self.rng.gauss(0, 2 * self.noise_intensity)
        spo2_noise = self.rng.gauss(0, 0.5 * self.noise_intensity)
        temp_noise = self.rng.gauss(0, 0.1 * self.noise_intensity)

        # 2. Physiological coupling
        coupling_hr = 0
        coupling_spo2 = 0
        # Low BP → compensatory tachycardia
        if v.BP_sys < 95:
            coupling_hr += (95 - v.BP_sys) * 0.3
        # Low SpO2 → tachycardia
        if v.SpO2 < 93:
            coupling_hr += (93 - v.SpO2) * 1.5
        # High HR → increased oxygen demand → SpO2 drops
        if v.HR > 110:
            coupling_spo2 -= (v.HR - 110) * 0.15

        # 3. Medication effects
        med_hr_delta = 0
        med_bp_sys_delta = 0
        med_bp_dia_delta = 0
        meds_to_remove = []

        for i, med in enumerate(self.patient.active_medications):
            # Effect ramps up in first step, then steady
            if not med.peak_reached:
                effect_multiplier = 0.6
                med.peak_reached = True
            else:
                effect_multiplier = 1.0

            if med.drug == "vasopressor":
                magnitude = 8 if med.dose == "low" else 16
                med_bp_sys_delta += magnitude * effect_multiplier
                med_bp_dia_delta += magnitude * 0.5 * effect_multiplier
                med_hr_delta += 3 * effect_multiplier  # Slight HR increase

            elif med.drug == "antihypertensive":
                magnitude = 7 if med.dose == "low" else 14
                med_bp_sys_delta -= magnitude * effect_multiplier
                med_bp_dia_delta -= magnitude * 0.5 * effect_multiplier
                med_hr_delta -= 2 * effect_multiplier

            elif med.drug == "fluids":
                med_bp_sys_delta += 4 * effect_multiplier
                med_bp_dia_delta += 2 * effect_multiplier

            elif med.drug == "sedative":
                med_hr_delta -= 5 * effect_multiplier
                med_bp_sys_delta -= 5 * effect_multiplier
                
            elif med.drug == "antibiotics":
                # Antibiotics slowly resolve sepsis active state
                if self.patient.sepsis_active and med.steps_remaining <= 5:
                    self.patient.sepsis_active = False

            med.steps_remaining -= 1
            if med.steps_remaining <= 0:
                meds_to_remove.append(i)

        for i in reversed(meds_to_remove):
            self.patient.active_medications.pop(i)

        # 4. Oxygen support → SpO2
        oxygen_boost = self.patient.oxygen_support_level * 1.2

        # 5. Scenario-specific effects
        scenario_bp_delta = 0
        if self.scenario == "post_surgical":
            # Periodic BP swings
            self._bp_swing_phase += 0.6
            scenario_bp_delta = self._bp_swing_amplitude * math.sin(self._bp_swing_phase)

        # 6. Sepsis progression
        sepsis_hr_delta = 0
        sepsis_bp_delta = 0
        sepsis_temp_delta = 0
        sepsis_spo2_delta = 0

        if (self.scenario == "sepsis" and
                self.patient.step_count >= self.patient.sepsis_onset_step and
                not self.patient.sepsis_detected):

            steps_since_onset = self.patient.step_count - self.patient.sepsis_onset_step

            if steps_since_onset <= 2:
                self.patient.sepsis_stage = 1  # Early
                sepsis_temp_delta = 0.3 + steps_since_onset * 0.2
                sepsis_hr_delta = 5 + steps_since_onset * 3
                sepsis_bp_delta = -3 - steps_since_onset * 2
            elif steps_since_onset <= 5:
                self.patient.sepsis_stage = 2  # Developing
                sepsis_temp_delta = 0.8 + (steps_since_onset - 2) * 0.3
                sepsis_hr_delta = 12 + (steps_since_onset - 2) * 5
                sepsis_bp_delta = -8 - (steps_since_onset - 2) * 4
                sepsis_spo2_delta = -1 - (steps_since_onset - 2)
            else:
                self.patient.sepsis_stage = 3  # Severe
                sepsis_temp_delta = 1.5 + (steps_since_onset - 5) * 0.2
                sepsis_hr_delta = 25 + (steps_since_onset - 5) * 3
                sepsis_bp_delta = -20 - (steps_since_onset - 5) * 5
                sepsis_spo2_delta = -3 - (steps_since_onset - 5) * 2

            self.patient.sepsis_active = True

        # === Apply all deltas ===
        v.HR = int(round(v.HR + hr_drift + hr_noise + coupling_hr + med_hr_delta + sepsis_hr_delta))
        v.BP_sys = int(round(v.BP_sys + bp_sys_drift + bp_sys_noise + med_bp_sys_delta + sepsis_bp_delta + scenario_bp_delta))
        v.BP_dia = int(round(v.BP_dia + bp_dia_drift + bp_dia_noise + med_bp_dia_delta + sepsis_bp_delta * 0.5))
        v.SpO2 = int(round(v.SpO2 + spo2_drift + spo2_noise + coupling_spo2 + oxygen_boost + sepsis_spo2_delta))
        v.Temp = round(v.Temp + temp_drift + temp_noise + sepsis_temp_delta, 1)

        # === Clamp to physiological limits ===
        v.HR = max(30, min(200, v.HR))
        v.BP_sys = max(50, min(220, v.BP_sys))
        v.BP_dia = max(30, min(130, v.BP_dia))
        v.SpO2 = max(60, min(100, v.SpO2))
        v.Temp = max(34.0, min(42.0, round(v.Temp, 1)))

        # Ensure BP_sys > BP_dia
        if v.BP_sys <= v.BP_dia:
            v.BP_sys = v.BP_dia + 20

        # === Calculate Labs and Organ Stress ===
        # Lactate rises if tissues are under-perfused or severe sepsis
        lactate_drift = 0.0
        if v.BP_sys < 90:
            lactate_drift += 0.1
        if v.SpO2 < 92:
            lactate_drift += 0.1
        if getattr(self.patient, 'sepsis_stage', 0) >= 2: # use getattr for safety
            lactate_drift += 0.2
        if lactate_drift == 0.0:
            lactate_drift -= 0.1 # Clear lactate if perfusion is okay
            
        self.patient.lactate = max(0.5, min(15.0, self.patient.lactate + lactate_drift))
        
        # Organ stress accumulates while vitals are heavily out of bounds
        critical_vitals = sum([
            1 if v.HR < 50 or v.HR > 140 else 0,
            1 if v.BP_sys < 80 or v.BP_sys > 180 else 0,
            1 if v.SpO2 < 88 else 0
        ])
        
        if critical_vitals > 0:
            self.patient.organ_stress += critical_vitals * 0.5
        elif self.patient.organ_stress > 0:
            # Slow recovery
            self.patient.organ_stress -= 0.1
            
        self.patient.organ_stress = max(0.0, float(self.patient.organ_stress))

        # Ensure BP_sys > BP_dia
        if v.BP_sys <= v.BP_dia:
            v.BP_sys = v.BP_dia + 20

        # === Record history ===
        self.patient.history.append({
            "step": self.patient.step_count,
            "hr_delta": v.HR - prev_hr,
            "bp_sys_delta": v.BP_sys - prev_bp_sys,
            "bp_dia_delta": v.BP_dia - prev_bp_dia,
            "spo2_delta": v.SpO2 - prev_spo2,
            "temp_delta": round(v.Temp - prev_temp, 1),
        })

    def is_patient_critical(self) -> bool:
        """Check if the patient is in a life-threatening state."""
        v = self.patient.vitals
        return (v.HR < 40 or v.HR > 180 or
                v.BP_sys < 60 or v.BP_sys > 200 or
                v.SpO2 < 70 or
                v.Temp > 41.0)

    def vitals_in_safe_zone(self) -> Dict[str, bool]:
        """Check each vital against its safe range."""
        v = self.patient.vitals
        return {
            "HR": SAFE_RANGES["HR"][0] <= v.HR <= SAFE_RANGES["HR"][1],
            "BP_sys": SAFE_RANGES["BP_sys"][0] <= v.BP_sys <= SAFE_RANGES["BP_sys"][1],
            "BP_dia": SAFE_RANGES["BP_dia"][0] <= v.BP_dia <= SAFE_RANGES["BP_dia"][1],
            "SpO2": SAFE_RANGES["SpO2"][0] <= v.SpO2 <= SAFE_RANGES["SpO2"][1],
            "Temp": SAFE_RANGES["Temp"][0] <= v.Temp <= SAFE_RANGES["Temp"][1],
        }

    def safe_zone_fraction(self) -> float:
        """Fraction of vitals currently in safe zone (0.0 to 1.0)."""
        zone = self.vitals_in_safe_zone()
        return sum(1 for v in zone.values() if v) / len(zone)
