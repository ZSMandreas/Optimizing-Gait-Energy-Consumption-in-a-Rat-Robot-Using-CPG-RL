# Optimizing Gait Energy Consumption in a Rat Robot Using CPG-RL

Master’s thesis code and project materials for **CPG-RL (CDER)** energy-aware trot locomotion on a MuJoCo rat robot.

**Author:** Simiao Zhuang · Technical University of Munich

---

## Repository layout

| Path | Description |
|------|-------------|
| [`rat-robot/v2-HardControl/CPG_RL/`](rat-robot/v2-HardControl/CPG_RL/) | **Main thesis code** (envs, train/eval, closure & multi-velocity eval, checkpoints) |
| [`rat-robot/v2-HardControl/TrotGait/`](rat-robot/v2-HardControl/TrotGait/) | Shared MuJoCo models / original hard-control stack |
| [`project_page/`](project_page/) | Thesis project webpage (figures, videos, abstract) |
| `stable-baselines3/` | Vendored PPO dependency |
| `pyGLFW/`, `Data/` | Supporting assets from the original rat-robot stack |

See **[`CPG_RL/README.md`](rat-robot/v2-HardControl/CPG_RL/README.md)** for method details, configs, logs, and how to run experiments.

---

## Method (short)

- **CPG** action space on a quadruped rat model  
- **CDER (Ours) v3.1** energy-shaped reward → lower cost of transport vs CPG planner baselines  
- Closure validation, ablations, and multi-velocity evaluation scripts live under `CPG_RL/`

---

## Citation

```bibtex
@mastersthesis{zhuang_cpg_rl,
  title   = {Optimizing Gait Energy Consumption in a Rat Robot Using CPG-RL},
  author  = {Zhuang, Simiao},
  school  = {Technical University of Munich},
  year    = {2026}
}
```
