# Ablation B — equal CDER weights (0.50) — training summary (seed 0)

- Log directory: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/ours_ablation_equal_weights/seed0`
- Train log: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/ours_ablation_equal_weights/seed0/train_20260517_010333.log`
- TensorBoard: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/ours_ablation_equal_weights/seed0/tb_logs/PPO_1`
- Checkpoint: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/ours_ablation_equal_weights/seed0/rat_cpg_ppo_route_a.zip`
- Env config: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/configs/ablation_equal_weights_env_kwargs.json`
- M1–M9 eval: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/outputs/thesis_experiments_v3.1_ablation_equal_weights/`

## Experiment intent

Sets α_Wp=α_Wn=α_damp=α_fric=α_norm=0.50 (v3.1 uses 0.30/0.60/0.50/0.50/1.00).

Shared with v3.1 seed0: ShapeV3 env, 1.5M steps, 4 parallel envs, PPO defaults (n_steps=1024, batch=128, lr=3e-4, ent_coef=0.01), kinematic weights vel=1.0, pose=0.2, x=0.1, xvel_boost=5.0.

## CDER weights (`ours_weights`)

| component | this run | v3.1 ref |
|-----------|---------:|---------:|
| `alpha_W_pos` | 0.5 | 0.3 |
| `alpha_W_neg` | 0.5 | 0.6 |
| `alpha_E_damp` | 0.5 | 0.5 |
| `alpha_E_fric` | 0.5 | 0.5 |
| `alpha_E_norm` | 0.5 | 1.0 |

## Wall-clock

- Start (log filename): 2026-05-17T01:03:33
- End (log mtime): 2026-05-17T10:16:44.144657
- Duration: **9.22 hours** (33191 s)

## Final SB3 log block (last table in train log)

| key | value |
|-----|------:|
| `E_damp_J` | 0.0422 |
| `E_fric_J` | 0.0104 |
| `E_norm_J` | 0.018 |
| `W_neg_J` | 0.0463 |
| `W_pos_J` | 0.086 |
| `approx_kl` | 0.0904366 |
| `clip_fraction` | 0.554 |
| `ep_len_mean` | 82 |
| `ep_rew_mean` | 35.6 |
| `explained_variance` | 0.936 |
| `fps` | 45 |
| `r_energy` | -0.101 |
| `r_pose` | -0.00192 |
| `r_vel` | 0.651 |
| `r_xvel` | -0.0338 |
| `std` | 0.205 |
| `time_elapsed` | 3.318e+04 |
| `total_timesteps` | 1.503e+06 |

## TensorBoard — last 100 logged points (training end)

| tag | mean | std | last | step |
|-----|-----:|-----:|-----:|-----:|
| `rollout/ep_rew_mean` | 34.53 | 0.8354 | 35.57 | 1503232 |
| `rollout/ep_len_mean` | 82 | 0 | 82 | 1503232 |
| `train/clip_fraction` | 0.5406 | 0.02177 | 0.5543 | 1503232 |
| `reward/r_energy` | -0.07451 | 0.02501 | -0.1014 | 1503232 |
| `reward/r_vel` | 0.6523 | 0.05015 | 0.6509 | 1503232 |
| `reward/r_pose` | -0.002175 | 0.0001118 | -0.001922 | 1503232 |
| `reward/r_xvel` | -0.03235 | 0.01626 | -0.03383 | 1503232 |
| `energy/W_pos_J` | 0.06385 | 0.01833 | 0.08596 | 1503232 |
| `energy/W_neg_J` | 0.03465 | 0.01392 | 0.04631 | 1503232 |
| `energy/E_damp_J` | 0.03215 | 0.009698 | 0.04215 | 1503232 |
| `energy/E_fric_J` | 0.01078 | 0.0003582 | 0.0104 | 1503232 |
| `energy/E_norm_J` | 0.007612 | 0.008334 | 0.01798 | 1503232 |
| `train/approx_kl` | 0.0795 | 0.01312 | 0.09044 | 1503232 |
| `train/explained_variance` | 0.93 | 0.01699 | 0.9359 | 1503232 |
| `train/std` | 0.2127 | 0.006556 | 0.2053 | 1503232 |

## Quick sanity (5 ep, deterministic)

| metric | this run | v3.1 seed0 ref |
|--------|---------:|---------------:|
| episode return | 37.06 | 47.24 |
| forward speed (mm/s) | 75.0 | 97.9 |
| mean b (mm) | 1.7 | 1.02 |
| ep length | 82.0 | 82 |

Source: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/ours_ablation_equal_weights/seed0/sanity_check.md`

## M1–M9 standard eval (20 ep)

| metric | this run | v3.1 ref |
|--------|---------:|---------:|
| M4 forward speed (mm/s) | 76.2 | 97.9 |
| M9 COT mean | 4.411 | 3.544 |

## CPG actions (last env-step, 20 eval episodes)

- b: mean **2.43** mm (std 1.232)
- a: mean **10.00** mm (std 0.000)

## Convergence note

- `ep_len_mean` reaches **82** early (~4k steps) and stays flat.
- Final `ep_rew_mean` ≈ **35.6** (v3.1 seed0 ≈ 45.0); policy settles in a **slower** basin than v3.1.
- `clip_fraction` ≈ **0.55** (higher than v3.1 ~0.43 → more aggressive PPO updates).

## Provenance

- Env: `RatCpgEnvEnergySubstep50ShapeV3`
- Launch: `launch_ablation_equal_weights_seed0.sh`
- Eval driver: `thesis_experiments_cder_ablation_eval.py`
- `train_config.json` saved at training time
