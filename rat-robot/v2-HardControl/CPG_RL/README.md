# CPG_RL

Thesis: **Optimizing Gait Energy Consumption in a Rat Robot Using CPG-RL**

Master’s thesis code for **energy-aware CPG locomotion** on a MuJoCo rat robot (Nermo-style).

**Method (Ours):** **CDER** — Contact / Dynamics Energy Redistribution shaped reward (v3.1), learned with PPO on a CPG action space.

Related display materials: `../../../project_page/`

---

## What this folder is

| Role | Path |
|------|------|
| Final Ours environment | `env/ours_cder_v31_env.py` (`OursCderEnvV31`) |
| Closure validation (M1–M9) | `experiments/run_cder_closure_validation.py` |
| Multi-velocity three-way eval | `scripts/eval_multi_velocity_three_way.py` |
| Final checkpoints | `logs/w3_ours_v3.1/`, `logs/w2_baseline/`, ablation logs |
| MuJoCo XML (shared) | `../TrotGait/models/` (esp. `dynamic_4l_kp2.xml`) |

---

## Layout

```
CPG_RL/
├── env/            # Gymnasium envs, train/eval entrypoints, energy utils
├── experiments/    # Closure validation & ablation protocol
├── scripts/        # Multi-speed / CoT / thesis figures
├── analysis/       # Post-process tables & chapter figures
├── configs/        # Final env kwargs only (5 JSON files)
├── LegModel/       # Leg kinematics helpers
└── logs/           # Trained PPO checkpoints
```

### `configs/` (kept)

| File | Use |
|------|-----|
| `ours_cder_v31_env_kwargs.json` | Main Ours (~0.12 / chapter-5 ref) |
| `v31_vcmd0.06_thesis_scaled_v2.json` | Multi-speed Ours @ 0.06 |
| `v31_vcmd0.09_thesis_scaled.json` | Multi-speed Ours @ 0.09 |
| `ablation_no_norm_env_kwargs.json` | Ablation A |
| `ablation_equal_weights_env_kwargs.json` | Ablation B |

### `logs/` (kept)

| Directory | Role |
|-----------|------|
| `w3_ours_v3.1/` | **Final CDER Ours** checkpoint |
| `w2_baseline/` | Energy-shaped baseline |
| `w3_ours_v3.1_ablation_no_norm/` | Ablation A |
| `w3_ours_v3.1_ablation_equal_weights/` | Ablation B |

Main Ours zip: `logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip`

---

## Dependencies (typical)

- Python 3.10+
- `mujoco`, `gymnasium`, `numpy`, `matplotlib`
- `stable-baselines3` (PPO)
- Shared robot assets under `../TrotGait/models/`

```bash
cd rat-robot/v2-HardControl/CPG_RL
```

---

## Common commands

```bash
python experiments/run_cder_closure_validation.py
python scripts/eval_single_speed_three_way.py
python scripts/eval_multi_velocity_three_way.py
python scripts/eval_fair_cot_table.py
```

---

## Notes

- Folder history: `CPG_TrotGait` → `CDER_RatGait` → **`CPG_RL`**.
- Repo title: **Optimizing Gait Energy Consumption in a Rat Robot Using CPG-RL**.
- Multi-speed scripts may also expect per-speed checkpoints under `logs/w3_ours_vcmd0.06_*` / `0.09_*` if those runs are re-evaluated; this cleaned tree keeps the main `w3_ours_v3.1` set by default.
