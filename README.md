---
title: Icu Guardian
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# 🏥 ICU-Guardian: OpenEnv Environment for Critical Care AI

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0+-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live-green)](https://huggingface.co/spaces/singhhrishabh/icu-guardian)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Meta PyTorch OpenEnv Hackathon Submission** — Train AI agents to monitor and stabilize critically ill ICU patients through the standard OpenEnv `step()` / `reset()` / `state()` API.

### 🎬 Live Demo

**🔗 Try it live:** [singhhrishabh-icu-guardian.hf.space](https://singhhrishabh-icu-guardian.hf.space)

https://github.com/user-attachments/assets/live_api_demo.webp

*Interactive ICU monitoring dashboard with real-time vital signs, clinical actions, and reward tracking.*

## 🎯 Overview

ICU-Guardian simulates a realistic Intensive Care Unit where an AI agent must continuously monitor a patient's vital signs and take clinical actions to maintain stability. The environment models physiological dynamics including inter-variable coupling, medication pharmacokinetics, and sepsis cascade progression.

This is a **real-world task** that clinicians perform daily — making life-or-death decisions based on evolving multi-variable data under time pressure.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Agent (LLM / RL Policy)                                │
│  ┌─────────────────────────────────────────┐            │
│  │ Observes vitals → Decides action        │            │
│  └──────────────┬──────────────────────────┘            │
└─────────────────┼───────────────────────────────────────┘
                  │ step(ICUAction) / reset() / state()
┌─────────────────▼───────────────────────────────────────┐
│  ICU-Guardian Environment (Docker Container)            │
│  ┌──────────────────┐  ┌────────────────────────┐       │
│  │  FastAPI Server   │  │  Patient Simulator     │       │
│  │  /reset           │──│  • Vital dynamics      │       │
│  │  /step            │  │  • Drug pharmacology   │       │
│  │  /state           │  │  • Sepsis cascade      │       │
│  └──────────────────┘  └────────────────────────┘       │
│  ┌──────────────────┐  ┌────────────────────────┐       │
│  │  Task Configs     │  │  Graders               │       │
│  │  3 difficulty     │──│  Score: [0.0, 1.0]     │       │
│  │  levels           │  │  Per-task evaluation   │       │
│  └──────────────────┘  └────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Observation Space

The agent receives a JSON observation at each step:

| Field | Type | Description | Normal Range |
|-------|------|-------------|--------------|
| `HR` | int | Heart Rate (bpm) | 60–100 |
| `BP_sys` | int | Systolic Blood Pressure (mmHg) | 100–140 |
| `BP_dia` | int | Diastolic Blood Pressure (mmHg) | 60–90 |
| `SpO2` | int | Oxygen Saturation (%) | 95–100 |
| `Temp` | float | Body Temperature (°C) | 36.0–37.8 |
| `trend` | str | Natural language trend summary | — |
| `step_number` | int | Current step in episode | — |
| `task_name` | str | Active task identifier | — |
| `last_action_error` | str? | Error from invalid action | — |

**Example observation:**
```json
{
  "HR": 105,
  "BP_sys": 95,
  "BP_dia": 58,
  "SpO2": 91,
  "Temp": 37.2,
  "trend": "Step 1: HR ↑5, BP_sys ↓8, SpO2 ↓2 | Step 2: HR ↑3, SpO2 ↓1 | Step 3: stable",
  "step_number": 4,
  "task_name": "vital_stabilization",
  "last_action_error": null
}
```

## 🎮 Action Space

The agent must respond with exactly **one** action per step:

### 1. Administer Medication
```json
{"action": "administer_meds", "drug": "vasopressor", "dose": "low"}
{"action": "administer_meds", "drug": "vasopressor", "dose": "high"}
{"action": "administer_meds", "drug": "antihypertensive", "dose": "low"}
{"action": "administer_meds", "drug": "antihypertensive", "dose": "high"}
```
- **Vasopressor**: Raises blood pressure (onset: 1 step, duration: 3-4 steps)
- **Antihypertensive**: Lowers blood pressure
- Low dose = gentler effect; High dose = stronger but risks overcorrection

### 2. Adjust Oxygen Support
```json
{"action": "adjust_oxygen", "level": "increase"}
{"action": "adjust_oxygen", "level": "decrease"}
```
- Increase: Boosts SpO2 by ~1.2% per level (max 5 levels)
- Decrease: Appropriate when SpO2 is already high

### 3. Emergency Sepsis Alert
```json
{"action": "trigger_code_sepsis"}
```
- Triggers the sepsis protocol
- **Only use when sepsis pattern detected** (rising Temp + dropping BP + spiking HR)
- False alarms are heavily penalized

### 4. Wait / Monitor
```json
{"action": "wait"}
```
- Appropriate when vitals are stable

## 📋 Tasks

### Task 1: Vital Sign Stabilization (Easy)
- **Scenario**: Patient with mild hypotension and low SpO2
- **Goal**: Stabilize all vitals within safe ranges
- **Max Steps**: 15
- **Key Skills**: Basic clinical reasoning, oxygen management

### Task 2: Post-Surgical BP Management (Medium)
- **Scenario**: Post-surgical patient with volatile BP swings
- **Goal**: Maintain stable BP without overcorrection
- **Max Steps**: 20
- **Key Skills**: Medication titration, avoiding overshoot

### Task 3: Sepsis Detection & Response (Hard)
- **Scenario**: Initially stable patient developing sepsis
- **Goal**: Detect sepsis onset pattern and trigger protocol at the right time
- **Max Steps**: 25
- **Key Skills**: Multi-variable pattern recognition, timing precision

## 🏆 Reward Function

Dense per-step rewards in [0.0, 1.0]:

| Component | Weight | Description |
|-----------|--------|-------------|
| Safe Zone | 60% | Fraction of vitals in safe range |
| Improvement | 20% | Positive change from previous step |
| Action Quality | 20% | Appropriateness of chosen action |

**Penalties:**
- Invalid action: -0.05
- False sepsis alarm: -0.30
- Unnecessary medication: reduced action quality

**Final Episode Score** (from grader, 0.0–1.0):
- Task 1: Safe zone coverage + final proximity + efficiency
- Task 2: BP stability + range compliance + overcorrection avoidance
- Task 3: Detection accuracy + timing + vital management

## 🚀 Quick Start

### Installation
```bash
pip install openenv-core
cd ICU-Guardian
pip install -e .
```

### Run Locally (without Docker)
```bash
cd server
python app.py
```

### Run with Docker
```bash
docker build -t icu-guardian .
docker run -p 8000:8000 icu-guardian
```

### Use the Environment
```python
import asyncio
from client import ICUGuardianEnv
from models import ICUAction

async def main():
    async with ICUGuardianEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(f"Initial HR: {result.observation.HR}")
        
        result = await env.step(ICUAction(
            action="adjust_oxygen", level="increase"
        ))
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

### Run Baseline Inference
```bash
export HF_TOKEN="your-hf-token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

## 📊 Baseline Results (Qwen2.5-72B-Instruct)

Actual inference results using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Difficulty | Score | Steps | Key Actions |
|------|-----------|-------|-------|-------------|
| vital_stabilization | Easy | **0.284** | 15 | `adjust_oxygen(increase)` ×2, then monitor |
| bp_management | Medium | **0.586** | 20 | `vasopressor(low)`, `antihypertensive(low)` — correct titration |
| sepsis_detection | Hard | **0.050** | 9 | (API credit limit hit — no LLM actions) |

<details>
<summary>📋 Full inference output (click to expand)</summary>

```
[START] task=vital_stabilization env=icu_guardian model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=adjust_oxygen(increase) reward=0.68 done=false error=null
[STEP] step=2 action=adjust_oxygen(increase) reward=0.64 done=false error=null
[STEP] step=3 action=wait() reward=0.64 done=false error=null
...
[STEP] step=15 action=wait() reward=0.76 done=true error=null
[END] success=false steps=15 score=0.284 rewards=0.68,0.64,0.64,...,0.80,0.76,0.76

[START] task=bp_management env=icu_guardian model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=administer_meds(vasopressor,low) reward=0.68 done=false error=null
[STEP] step=3 action=wait() reward=0.80 done=false error=null
[STEP] step=6 action=administer_meds(antihypertensive,low) reward=0.72 done=false error=null
[STEP] step=11 action=administer_meds(vasopressor,low) reward=0.68 done=false error=null
...
[STEP] step=20 action=wait() reward=0.80 done=true error=null
[END] success=true steps=20 score=0.586 rewards=0.68,0.68,0.80,...,0.64,0.80

[START] task=sepsis_detection env=icu_guardian model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=wait() reward=0.96 done=false error=null
...
[STEP] step=9 action=wait() reward=0.06 done=true error=null
[END] success=false steps=9 score=0.050 rewards=0.96,0.76,...,0.30,0.06
```
</details>

**Analysis:** The LLM correctly identified low SpO₂ (increased oxygen) and low BP (gave vasopressor), demonstrating genuine clinical reasoning. The BP management task shows nuanced titration — switching between vasopressor and antihypertensive as BP fluctuated. Sepsis detection scored low due to API rate limits and the inherent difficulty of multi-step pattern recognition.

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ICU_TASK` | No | `vital_stabilization` | Active task |
| `API_BASE_URL` | No | HF Router | LLM API endpoint |
| `MODEL_NAME` | No | Qwen2.5-72B | Model for inference |
| `HF_TOKEN` | Yes* | — | HuggingFace API key |
| `IMAGE_NAME` | No | — | Docker image name |

## 📁 Project Structure
```
ICU-Guardian/
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml         # Package configuration
├── models.py              # Typed Action/Observation/State
├── client.py              # EnvClient subclass
├── __init__.py            # Package exports
├── inference.py           # Baseline inference script
├── Dockerfile             # Container definition
├── README.md              # This file
└── server/
    ├── app.py             # FastAPI server
    ├── icu_environment.py # Environment implementation
    ├── simulator.py       # Patient physiology simulator
    ├── tasks.py           # Task definitions & graders
    └── requirements.txt   # Server dependencies
```

## 🧪 Physiological Model

The simulator models realistic ICU dynamics:

- **Homeostatic drift**: Vitals naturally trend toward normal ranges
- **Inter-variable coupling**: Low BP → compensatory tachycardia; Low SpO2 → HR increase
- **Medication pharmacokinetics**: 1-step onset delay, 3-4 step duration, dose-dependent magnitude
- **Sepsis cascade**: Progressive multi-stage deterioration (early → developing → severe)
- **Stochastic noise**: Gaussian noise scaled by scenario difficulty

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
