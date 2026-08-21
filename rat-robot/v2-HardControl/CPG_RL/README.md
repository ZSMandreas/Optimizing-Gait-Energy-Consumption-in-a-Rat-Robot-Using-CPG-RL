# CPG_RL

Thesis: **Optimizing Gait Energy Consumption in a Rat Robot Using CPG-RL**

Master’s thesis code for **energy-aware CPG locomotion** on a MuJoCo rat robot (Nermo-style).

**Method (Ours):** **CDER** energy-shaped reward, learned with PPO on a CPG action space.

Related display materials: `../../../project_page/`

---

## Naming (methods)

| Name | Meaning | Typical artifact |
|------|---------|------------------|
| **ours** | CDER RL controller (final method) | `logs/ours/`, `env/ours_cder_v31_env.py` |
| **baseline** | Open-loop CPG planner (three-way eval) **or** RL energy-shaped checkpoint used in closure (`logs/baseline/`) | see scripts / logs |
| **baseline_simplified** | Simplified open-loop CPG planner (three-way eval) | `scripts/eval_*` |

Three-way comparison tags in scripts: `ours` / `baseline` / `baseline_simplified`  
(formerly M2 / M3 / M4).

---

## Can you run it?

**Structurally yes** — entrypoints and checkpoints for the main path are present. You need a Python env with:

- `mujoco`, `gymnasium`, `numpy`, `matplotlib`
- `stable-baselines3` (repo also vendors `../../../stable-baselines3/`)
- MuJoCo XML from `../TrotGait/models/` (e.g. `dynamic_4l_kp2.xml`)

**Caveats:**

| Item | Status |
|------|--------|
| Closure (`run_cder_closure_validation.py`) | Needs `logs/baseline` + `logs/ours` zips (present) |
| Single-speed three-way | Runnable if deps + TrotGait XML OK |
| Multi-speed / fair CoT | Scripts expect optional `logs/ours_vcmd0.06_*` / `0.09_*` (not kept in this tree; only main `logs/ours`) |
| Absolute paths in some train/eval defaults | Prefer running from `CPG_RL/` so relative/config paths win |
| Current bare system Python | Missing `gymnasium` / `stable_baselines3` until you install them |

```bash
cd rat-robot/v2-HardControl/CPG_RL
python experiments/run_cder_closure_validation.py
python scripts/eval_single_speed_three_way.py
```

---

## Layout

```
CPG_RL/
├── env/            # Gymnasium envs, train/eval, energy utils
├── experiments/    # Closure validation & ablation
├── scripts/        # Multi-speed / CoT / figures
├── analysis/       # Post-process tables & chapter figures
├── configs/        # Final env kwargs (5 JSON)
├── LegModel/
└── logs/
    ├── ours/
    ├── baseline/                 # RL energy-shaped (closure)
    ├── ours_ablation_no_norm/
    └── ours_ablation_equal_weights/
```

### Key files

| Role | Path |
|------|------|
| Ours env | `env/ours_cder_v31_env.py` (`OursCderEnvV31`) |
| Baseline env (energy-shaped parent) | `env/baseline_energy_shaped_env.py` |
| Closure M1–M9 | `experiments/run_cder_closure_validation.py` |
| Three-way eval | `scripts/eval_single_speed_three_way.py`, `eval_multi_velocity_three_way.py` |
| Ours checkpoint | `logs/ours/seed0/rat_cpg_ppo_route_a.zip` |
| Baseline RL checkpoint | `logs/baseline/seed1/checkpoints/rat_cpg_ppo_route_a_1500000_steps.zip` |

### `configs/`

| File | Use |
|------|-----|
| `ours_cder_v31_env_kwargs.json` | Main ours |
| `v31_vcmd0.06_thesis_scaled_v2.json` | Multi-speed ours @ 0.06 |
| `v31_vcmd0.09_thesis_scaled.json` | Multi-speed ours @ 0.09 |
| `ablation_no_norm_env_kwargs.json` | Ablation A |
| `ablation_equal_weights_env_kwargs.json` | Ablation B |
