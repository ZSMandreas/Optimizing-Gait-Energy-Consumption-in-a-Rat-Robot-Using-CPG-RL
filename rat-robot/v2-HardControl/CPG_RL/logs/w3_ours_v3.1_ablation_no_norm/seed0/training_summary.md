# Ablation A — α_E_norm = 0 — training summary (seed 0)

- Log directory: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/w3_ours_v3.1_ablation_no_norm/seed0`
- Train log: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/w3_ours_v3.1_ablation_no_norm/seed0/train_20260517_010330.log`
- TensorBoard: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/w3_ours_v3.1_ablation_no_norm/seed0/tb_logs/PPO_1`
- Checkpoint: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/w3_ours_v3.1_ablation_no_norm/seed0/rat_cpg_ppo_route_a.zip`
- Env config: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/configs/ablation_no_norm_env_kwargs.json`
- M1–M9 eval: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/outputs/thesis_experiments_v3.1_ablation_no_norm/`

## Experiment intent

Removes normal-contact energy penalty; all other settings match v3.1 seed0.

Shared with v3.1 seed0: ShapeV3 env, 1.5M steps, 4 parallel envs, PPO defaults (n_steps=1024, batch=128, lr=3e-4, ent_coef=0.01), kinematic weights vel=1.0, pose=0.2, x=0.1, xvel_boost=5.0.

## CDER weights (`ours_weights`)

| component | this run | v3.1 ref |
|-----------|---------:|---------:|
| `alpha_W_pos` | 0.3 | 0.3 |
| `alpha_W_neg` | 0.6 | 0.6 |
| `alpha_E_damp` | 0.5 | 0.5 |
| `alpha_E_fric` | 0.5 | 0.5 |
| `alpha_E_norm` | 0.0 | 1.0 |

## Wall-clock

- Start (log filename): 2026-05-17T01:03:30
- End (log mtime): 2026-05-17T10:13:06.331126
- Duration: **9.16 hours** (32976 s)

## Final SB3 log block (last table in train log)

| key | value |
|-----|------:|
| `E_damp_J` | 0.039 |
| `E_fric_J` | 0.0107 |
| `E_norm_J` | 0.0186 |
| `W_neg_J` | 0.0431 |
| `W_pos_J` | 0.0776 |
| `approx_kl` | 0.093538 |
| `clip_fraction` | 0.55 |
| `ep_len_mean` | 82 |
| `ep_rew_mean` | 34.8 |
| `explained_variance` | 0.857 |
| `fps` | 45 |
| `r_energy` | -0.074 |
| `r_pose` | -0.0029 |
| `r_vel` | 0.602 |
| `r_xvel` | -0.0706 |
| `std` | 0.186 |
| `time_elapsed` | 3.297e+04 |
| `total_timesteps` | 1.503e+06 |

## TensorBoard — last 100 logged points (training end)

| tag | mean | std | last | step |
|-----|-----:|-----:|-----:|-----:|
| `rollout/ep_rew_mean` | 33.31 | 1.155 | 34.77 | 1503232 |
| `rollout/ep_len_mean` | 82 | 0 | 82 | 1503232 |
| `train/clip_fraction` | 0.5235 | 0.02157 | 0.5504 | 1503232 |
| `reward/r_energy` | -0.0585 | 0.01859 | -0.07401 | 1503232 |
| `reward/r_vel` | 0.5993 | 0.04348 | 0.6021 | 1503232 |
| `reward/r_pose` | -0.002831 | 0.0001395 | -0.002901 | 1503232 |
| `reward/r_xvel` | -0.05254 | 0.01764 | -0.07059 | 1503232 |
| `energy/W_pos_J` | 0.06383 | 0.0171 | 0.07761 | 1503232 |
| `energy/W_neg_J` | 0.03197 | 0.01437 | 0.04312 | 1503232 |
| `energy/E_damp_J` | 0.02963 | 0.009813 | 0.03903 | 1503232 |
| `energy/E_fric_J` | 0.0107 | 0.0002624 | 0.01068 | 1503232 |
| `energy/E_norm_J` | 0.007882 | 0.008274 | 0.01859 | 1503232 |
| `train/approx_kl` | 0.07281 | 0.01113 | 0.09354 | 1503232 |
| `train/explained_variance` | 0.8944 | 0.02541 | 0.8566 | 1503232 |
| `train/std` | 0.2076 | 0.01177 | 0.1864 | 1503232 |

## Quick sanity (5 ep, deterministic)

| metric | this run | v3.1 seed0 ref |
|--------|---------:|---------------:|
| episode return | 37.64 | 47.24 |
| forward speed (mm/s) | 71.5 | 97.9 |
| mean b (mm) | 1.76 | 1.02 |
| ep length | 82.0 | 82 |

Source: `/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL/logs/w3_ours_v3.1_ablation_no_norm/seed0/sanity_check.md`

## M1–M9 standard eval (20 ep)

| metric | this run | v3.1 ref |
|--------|---------:|---------:|
| M4 forward speed (mm/s) | 74.3 | 97.9 |
| M9 COT mean | 3.860 | 3.544 |

## CPG actions (last env-step, 20 eval episodes)

- b: mean **2.01** mm (std 0.712)
- a: mean **10.00** mm (std 0.000)

## Convergence note

- `ep_len_mean` reaches **82** early (~4k steps) and stays flat.
- Final `ep_rew_mean` ≈ **34.8** (v3.1 seed0 ≈ 45.0); policy settles in a **slower** basin than v3.1.
- `clip_fraction` ≈ **0.55** (higher than v3.1 ~0.43 → more aggressive PPO updates).

## Provenance

- Env: `RatCpgEnvEnergySubstep50ShapeV3`
- Launch: `launch_ablation_no_norm_seed0.sh`
- Eval driver: `thesis_experiments_cder_ablation_eval.py`
- `train_config.json` saved at training time
