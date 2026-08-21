#!/usr/bin/env python3
"""
Extract converged CPG parameters from Ours v3.1 seed0 for Planner Baseline reference.

Read-only rollout: 20 deterministic episodes × 82 env-steps.
Outputs under outputs/thesis_experiments_planner_baseline/ours_cpg_reference/
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "env"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3  # noqa: E402

XML_REL = "../TrotGait/models/dynamic_4l_kp2.xml"
MODEL_PATH = str((ROOT / XML_REL).resolve())
CKPT = ROOT / "logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip"
OUT_DIR = ROOT / "outputs/thesis_experiments_planner_baseline/ours_cpg_reference"

LEG_NAMES = ["FL", "FR", "RL", "RR"]
FOOT_BODIES = ["foot_fl", "foot_fr", "foot_rl", "foot_rr"]
FN_CONTACT_THRESHOLD = 0.1  # N, same as run_cder_closure_validation

N_EPISODES = 20
ENV_STEPS_PER_EP = 82
TWO_PI = 2.0 * math.pi

# Trot reference relative phases (rad, mod 2π) vs FL
TROT_REF = {"FL": 0.0, "FR": math.pi, "RL": math.pi, "RR": 0.0}

# M2/M4 sanity reference (Ours v3.1 seed0)
SANITY_REF = {
    "forward_speed_mm_s": 97.3,
    "b_mean_mm": 1.0,
    "r_mean_pooled": 0.88,
    "theta_td_dev_rad": 0.015,
}


def _body_id_to_leg(model: mujoco.MjModel) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for leg, name in enumerate(FOOT_BODIES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            out[bid] = leg
    return out


def mod_2pi(x: np.ndarray | float) -> np.ndarray:
    return np.mod(np.asarray(x, dtype=np.float64), TWO_PI)


def phase_rel(theta_other: np.ndarray, theta_fl: np.ndarray) -> np.ndarray:
    return mod_2pi(theta_other - theta_fl)


def circular_mean(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=np.float64).ravel()
    if a.size == 0:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def circular_std(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=np.float64).ravel()
    if a.size == 0:
        return float("nan")
    R = np.sqrt(np.mean(np.cos(a)) ** 2 + np.mean(np.sin(a)) ** 2)
    return float(np.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))))


def stats_linear(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"n": 0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "p5": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def stats_circular(angles: np.ndarray) -> Dict[str, float]:
    a = np.asarray(angles, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    return {
        "circular_mean": circular_mean(a),
        "circular_std": circular_std(a),
        "n": int(a.size),
    }


def detect_touchdowns_contact(contact: np.ndarray) -> np.ndarray:
    """Rising edges on binarized contact series (0/1)."""
    c = contact.astype(np.int8)
    if c.size < 2:
        return np.zeros(0, dtype=np.int64)
    d = np.diff(c)
    return np.where(d == 1)[0] + 1


def detect_touchdowns_phase(theta: np.ndarray) -> np.ndarray:
    """Touchdown = swing→stance (θ crossing π upward), same as thesis M2."""
    if theta.size == 0:
        return np.zeros(0, dtype=np.int64)
    in_sw = theta < math.pi
    d = np.diff(in_sw.astype(np.int8))
    return np.where(d == -1)[0] + 1


def collect_rollout() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not CKPT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    env = RatCpgEnvEnergySubstep50ShapeV3(
        model_path=MODEL_PATH,
        max_episode_steps=ENV_STEPS_PER_EP * 50,
        render_mode=None,
        enable_csv_log=False,
        xvel_boost_multiplier=5.0,
    )
    ppo = PPO.load(str(CKPT), env=None)
    e = env.unwrapped
    model = e.model
    mouse_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    foot_to_leg = _body_id_to_leg(model)

    per_envstep_rows: List[Dict[str, Any]] = []
    all_theta_td: List[List[float]] = [[] for _ in range(4)]
    all_theta_td_phase: List[List[float]] = [[] for _ in range(4)]
    ep_meta: List[Dict[str, Any]] = []
    substep_fwd_vy: List[float] = []

    # Substep buffers for phase-pattern plot (pooled over episodes)
    sub_theta_pool: List[np.ndarray] = []
    sub_contact_pool: List[np.ndarray] = []

    orig_mj_step = mujoco.mj_step
    sub_theta: List[np.ndarray] = []
    sub_fn: List[np.ndarray] = []
    ep_y_start: float | None = None

    def mj_wrap(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        orig_mj_step(m, d)
        sub_theta.append(
            np.array([float(e.foot_path.get_leg_phase(i)) for i in range(4)], dtype=np.float64)
        )
        v6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, mouse_bid, v6, 0)
        substep_fwd_vy.append(-float(v6[4]))
        fn_leg = np.zeros(4, dtype=np.float64)
        for ic in range(int(d.ncon)):
            contact = d.contact[ic]
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            b1 = int(m.geom_bodyid[g1])
            b2 = int(m.geom_bodyid[g2])
            if (b1 == 0 and b2 == 0) or (b1 != 0 and b2 != 0):
                continue
            foot_bid = b1 if b1 != 0 else b2
            leg = foot_to_leg.get(foot_bid, -1)
            if leg < 0:
                continue
            cforce = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(m, d, ic, cforce)
            fn_leg[leg] += abs(float(cforce[0]))
        sub_fn.append(fn_leg)

    mujoco.mj_step = mj_wrap  # type: ignore[assignment]

    try:
        for ep in range(N_EPISODES):
            sub_theta.clear()
            sub_fn.clear()
            ep_y_start = None
            ep_rows_start = len(per_envstep_rows)
            terminated_early = False
            trunc_early = False
            steps_done = 0

            obs, _ = env.reset(seed=ep)
            ep_y_start = float(env.unwrapped.data.xpos[mouse_bid, 1])
            done = False
            while not done and steps_done < ENV_STEPS_PER_EP:
                act, _ = ppo.predict(obs, deterministic=True)
                obs, _r, term, trunc, _info = env.step(act)
                done = bool(term or trunc)
                steps_done += 1
                if term:
                    terminated_early = True
                if trunc and steps_done < ENV_STEPS_PER_EP:
                    trunc_early = True

                f4 = np.array(e._cur_f4, dtype=np.float64)
                mu4 = np.array(e._cur_mu4, dtype=np.float64)
                a_m = float(e._cur_a)
                b_m = float(e._cur_b)
                r4 = np.sqrt(np.clip(mu4, 0.0, None))

                K = int(e.n_substeps)
                th_end = sub_theta[-1] if sub_theta else np.zeros(4)
                fn_sub = np.stack(sub_fn[-K:], axis=0) if len(sub_fn) >= K else np.zeros((0, 4))
                contact_sub = (fn_sub > FN_CONTACT_THRESHOLD).astype(np.float64)

                # Touchdown phases this env-step (physical contact rising edges)
                if contact_sub.size > 0:
                    base_idx = len(sub_theta) - contact_sub.shape[0]
                    for leg in range(4):
                        td_local = detect_touchdowns_contact(contact_sub[:, leg])
                        for tloc in td_local:
                            gidx = base_idx + int(tloc)
                            if 0 <= gidx < len(sub_theta):
                                all_theta_td[leg].append(float(sub_theta[gidx][leg]))
                        th_slice = np.stack(
                            sub_theta[base_idx : base_idx + contact_sub.shape[0]], axis=0
                        )
                        td_phase = detect_touchdowns_phase(th_slice[:, leg])
                        for tloc in td_phase:
                            gidx = base_idx + int(tloc)
                            if 0 <= gidx < len(sub_theta):
                                all_theta_td_phase[leg].append(float(sub_theta[gidx][leg]))

                # Body state at end of env-step
                d = e.data
                body_z = float(d.xpos[mouse_bid, 2])
                v6 = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, mouse_bid, v6, 0)
                body_vx = float(v6[3])
                body_vy = float(v6[4])
                forward_v = -body_vy  # training convention: forward along -Y

                row = {
                    "episode": ep,
                    "env_step": steps_done - 1,
                    "f_FL": f4[0],
                    "f_FR": f4[1],
                    "f_RL": f4[2],
                    "f_RR": f4[3],
                    "mu_FL": mu4[0],
                    "mu_FR": mu4[1],
                    "mu_RL": mu4[2],
                    "mu_RR": mu4[3],
                    "a_m": a_m,
                    "b_m": b_m,
                    "r_FL": r4[0],
                    "r_FR": r4[1],
                    "r_RL": r4[2],
                    "r_RR": r4[3],
                    "a_mm": a_m * 1000.0,
                    "b_mm": b_m * 1000.0,
                    "theta_FL": th_end[0],
                    "theta_FR": th_end[1],
                    "theta_RL": th_end[2],
                    "theta_RR": th_end[3],
                    "phase_FR_rel": float(phase_rel(th_end[1], th_end[0])),
                    "phase_RL_rel": float(phase_rel(th_end[2], th_end[0])),
                    "phase_RR_rel": float(phase_rel(th_end[3], th_end[0])),
                    "body_z_m": body_z,
                    "body_z_mm": body_z * 1000.0,
                    "body_vx_mps": body_vx,
                    "body_vy_mps": body_vy,
                    "forward_v_mps": forward_v,
                    "forward_v_mmps": forward_v * 1000.0,
                }
                per_envstep_rows.append(row)

            # Episode-level substep data for phase pattern
            if sub_theta and sub_fn:
                th_arr = np.stack(sub_theta, axis=0)
                fn_arr = np.stack(sub_fn, axis=0)
                sub_theta_pool.append(th_arr)
                sub_contact_pool.append((fn_arr > FN_CONTACT_THRESHOLD).astype(np.float64))

            ep_body_z = [r["body_z_mm"] for r in per_envstep_rows[ep_rows_start:]]
            ep_y_end = float(e.data.xpos[mouse_bid, 1])
            T_ep = ENV_STEPS_PER_EP * 50 * float(model.opt.timestep)
            dist_fwd_m = float(ep_y_start - ep_y_end) if ep_y_start is not None else float("nan")
            ep_meta.append(
                {
                    "episode": ep,
                    "env_steps_completed": steps_done,
                    "completed_82": steps_done >= ENV_STEPS_PER_EP and not terminated_early,
                    "terminated": terminated_early,
                    "truncated_before_82": trunc_early,
                    "body_z_range_mm": float(np.max(ep_body_z) - np.min(ep_body_z)) if ep_body_z else float("nan"),
                    "mean_forward_v_mmps_envstep": float(
                        np.mean([r["forward_v_mmps"] for r in per_envstep_rows[ep_rows_start:]])
                    )
                    if ep_rows_start < len(per_envstep_rows)
                    else float("nan"),
                    "forward_speed_mmps_displacement": dist_fwd_m / T_ep * 1000.0,
                }
            )
            print(f"ep {ep}: steps={steps_done}, completed_82={ep_meta[-1]['completed_82']}")
    finally:
        mujoco.mj_step = orig_mj_step  # type: ignore[assignment]
        env.close()

    rollout_meta = {
        "checkpoint": str(CKPT),
        "n_episodes": N_EPISODES,
        "env_steps_per_episode": ENV_STEPS_PER_EP,
        "episodes": ep_meta,
        "substep_theta": sub_theta_pool,
        "substep_contact": sub_contact_pool,
        "theta_td_per_leg": all_theta_td,
        "theta_td_per_leg_phase_based": all_theta_td_phase,
        "substep_forward_vy_mmps": substep_fwd_vy,
    }
    return per_envstep_rows, rollout_meta


def compute_statistics(rows: List[Dict[str, Any]], rollout_meta: Dict[str, Any]) -> Dict[str, Any]:
    n = len(rows)
    assert n == N_EPISODES * ENV_STEPS_PER_EP, f"Expected {N_EPISODES * ENV_STEPS_PER_EP} rows, got {n}"

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=np.float64)

    scalar_keys = []
    for leg in LEG_NAMES:
        scalar_keys += [f"f_{leg}", f"mu_{leg}", f"r_{leg}"]
    scalar_keys += ["a_m", "b_m", "a_mm", "b_mm"]

    linear_stats: Dict[str, Any] = {k: stats_linear(col(k)) for k in scalar_keys}
    f_means_per_leg = [linear_stats[f"f_{leg}"]["mean"] for leg in LEG_NAMES]

    phase_stats: Dict[str, Any] = {}
    for leg in LEG_NAMES:
        phase_stats[f"theta_{leg}"] = stats_circular(col(f"theta_{leg}"))
    for rel in ("FR", "RL", "RR"):
        phase_stats[f"phase_{rel}_rel"] = stats_circular(col(f"phase_{rel}_rel"))

    # Trot closeness: circular distance from reference relative phases
    trot_devs = []
    for rel, ref in [("FR", TROT_REF["FR"]), ("RL", TROT_REF["RL"]), ("RR", TROT_REF["RR"])]:
        d = np.abs(mod_2pi(col(f"phase_{rel}_rel") - ref))
        d = np.minimum(d, TWO_PI - d)
        trot_devs.append(float(np.mean(d)))
    trot_closeness_mean_dev = float(np.mean(trot_devs))

    td_stats: Dict[str, Any] = {}
    td_phase_stats: Dict[str, Any] = {}
    for i, leg in enumerate(LEG_NAMES):
        td = np.array(rollout_meta["theta_td_per_leg"][i], dtype=np.float64)
        s = stats_linear(td)
        s["deviation_from_pi_rad_mean"] = (
            float(np.mean(np.abs(td - math.pi))) if td.size else float("nan")
        )
        td_stats[leg] = s
        td_p = np.array(rollout_meta["theta_td_per_leg_phase_based"][i], dtype=np.float64)
        sp = stats_linear(td_p)
        sp["deviation_from_pi_rad_mean"] = (
            float(np.mean(np.abs(td_p - math.pi))) if td_p.size else float("nan")
        )
        td_phase_stats[leg] = sp

    body_z = col("body_z_mm")
    lat = col("body_vx_mps") * 1000.0
    ep_z_ranges = [m["body_z_range_mm"] for m in rollout_meta["episodes"]]
    fwd_substep = np.array(rollout_meta["substep_forward_vy_mmps"], dtype=np.float64) * 1000.0
    fwd_disp = np.array(
        [m["forward_speed_mmps_displacement"] for m in rollout_meta["episodes"]], dtype=np.float64
    )

    r_pooled = np.concatenate([col(f"r_{leg}") for leg in LEG_NAMES])
    f_pooled = np.concatenate([col(f"f_{leg}") for leg in LEG_NAMES])

    body_stats = {
        "mean_body_z_mm": float(np.mean(body_z)),
        "body_z_range_per_ep_mean_mm": float(np.mean(ep_z_ranges)),
        "mean_forward_speed_mmps_substep": float(np.mean(fwd_substep)),
        "mean_forward_speed_mmps_displacement": float(np.mean(fwd_disp)),
        "std_forward_speed_mmps_displacement": float(np.std(fwd_disp, ddof=1)),
        "mean_lateral_speed_mmps": float(np.mean(np.abs(lat))),
    }

    # Sanity checks
    checks: Dict[str, Any] = {}
    n_complete = sum(1 for m in rollout_meta["episodes"] if m["completed_82"])
    checks["all_20_episodes_complete_82_steps"] = n_complete == N_EPISODES
    checks["episodes_completed_count"] = n_complete

    f_ok = bool(np.all((np.array(f_means_per_leg) >= 1.0) & (np.array(f_means_per_leg) <= 5.0)))
    r_ok = bool(np.all((r_pooled >= 0.0) & (r_pooled <= 1.0)))
    a_mm = col("a_mm")
    b_mm = col("b_mm")
    a_ok = bool(np.all((a_mm >= 9.0) & (a_mm <= 11.0)))
    b_ok = bool(np.all((b_mm >= 0.5) & (b_mm <= 8.0)))  # action upper bound 8 mm
    nan_ok = not any(
        np.any(~np.isfinite(col(k)))
        for k in ["f_FL", "mu_FL", "a_mm", "b_mm", "theta_FL", "body_z_mm", "forward_v_mmps"]
    )

    # Phase trot: diagonal pairs in-phase (FR~RL, FL~RR offsets ~π between pairs)
    fr_dev = trot_devs[0]
    rl_dev = trot_devs[1]
    rr_dev = trot_devs[2]
    phase_trot_ok = trot_closeness_mean_dev < 0.35  # generous; trot ref is exact

    checks["f_in_1_5_Hz"] = f_ok
    checks["r_in_0_1"] = r_ok
    checks["a_in_9_11_mm"] = a_ok
    checks["b_in_0_5_5_mm"] = b_ok
    checks["no_NaN"] = nan_ok
    checks["phase_pattern_trot_like"] = phase_trot_ok
    checks["any_failed"] = not all(
        [checks["all_20_episodes_complete_82_steps"], f_ok, r_ok, a_ok, b_ok, nan_ok, phase_trot_ok]
    )

    # Compare to M2/M4 reference
    td_dev_mean_phase = float(
        np.mean([td_phase_stats[leg]["deviation_from_pi_rad_mean"] for leg in LEG_NAMES])
    )
    fwd_m4 = body_stats["mean_forward_speed_mmps_substep"]  # matches M4: mean(-base_vy)
    sanity_flags = {
        "forward_speed_mm_s": {
            "measured": fwd_m4,
            "measured_displacement": body_stats["mean_forward_speed_mmps_displacement"],
            "reference": SANITY_REF["forward_speed_mm_s"],
            "ok": abs(fwd_m4 - SANITY_REF["forward_speed_mm_s"]) < 5.0,
        },
        "b_mean_mm": {
            "measured": linear_stats["b_mm"]["mean"],
            "reference": SANITY_REF["b_mean_mm"],
            "ok": abs(linear_stats["b_mm"]["mean"] - SANITY_REF["b_mean_mm"]) < 0.5,
        },
        "r_mean_pooled": {
            "measured": float(np.mean(r_pooled)),
            "reference": SANITY_REF["r_mean_pooled"],
            "ok": abs(float(np.mean(r_pooled)) - SANITY_REF["r_mean_pooled"]) < 0.08,
        },
        "theta_td_dev_rad": {
            "measured_contact_based": float(
                np.mean([td_stats[leg]["deviation_from_pi_rad_mean"] for leg in LEG_NAMES])
            ),
            "measured_phase_based_M2": td_dev_mean_phase,
            "reference": SANITY_REF["theta_td_dev_rad"],
            "ok": abs(td_dev_mean_phase - SANITY_REF["theta_td_dev_rad"]) < 0.05,
        },
    }
    checks["m2_m4_sanity"] = sanity_flags
    checks["m2_m4_all_ok"] = all(v["ok"] for v in sanity_flags.values())

    planner = {
        "a_planner_mm": linear_stats["a_mm"]["mean"],
        "b_planner_mm": linear_stats["b_mm"]["mean"],
        "f_planner_Hz": float(np.mean([linear_stats[f"f_{leg}"]["mean"] for leg in LEG_NAMES])),
        "r_planner": {leg: linear_stats[f"r_{leg}"]["mean"] for leg in LEG_NAMES},
        "phase_offsets_rad_vs_FL": {
            "FL": 0.0,
            "FR": circular_mean(col("phase_FR_rel")),
            "RL": circular_mean(col("phase_RL_rel")),
            "RR": circular_mean(col("phase_RR_rel")),
        },
        "theta_td_rad_verification_phase_M2": {
            leg: td_phase_stats[leg]["mean"] for leg in LEG_NAMES
        },
    }

    return {
        "n_samples": n,
        "linear": linear_stats,
        "phase": phase_stats,
        "trot_closeness_mean_deviation_rad": trot_closeness_mean_dev,
        "touchdown_theta_td_contact": td_stats,
        "touchdown_theta_td_phase_M2": td_phase_stats,
        "body": body_stats,
        "sanity_checks": checks,
        "planner_recommended": planner,
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_phase_pattern(rollout_meta: Dict[str, Any], path: Path, n_bins: int = 72) -> None:
    theta_all = np.concatenate(rollout_meta["substep_theta"], axis=0)
    contact_all = np.concatenate(rollout_meta["substep_contact"], axis=0)
    edges = np.linspace(0.0, TWO_PI, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.ravel()
    for leg, ax in enumerate(axes):
        th = theta_all[:, leg]
        c = contact_all[:, leg]
        duty = float(np.mean(c)) if c.size else 0.0
        mean_c = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            mask = (th >= edges[b]) & (th < edges[b + 1])
            if b == n_bins - 1:
                mask = mask | (th >= edges[b])
            mean_c[b] = float(np.mean(c[mask])) if np.any(mask) else 0.0
        ax.plot(centers, mean_c, "b-", lw=1.5)
        ax.fill_between(centers, 0, mean_c, alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, TWO_PI)
        ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, TWO_PI])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.set_title(f"{LEG_NAMES[leg]}  (duty={duty:.2f})")
        ax.set_ylabel("contact (avg)")
        ax.grid(True, alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("CPG phase θ (rad)")
    fig.suptitle("Ours v3.1 seed0 — contact vs CPG phase (pooled substeps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(rows: List[Dict[str, Any]], stats: Dict[str, Any], path: Path) -> None:
    L = stats["linear"]
    P = stats["phase"]
    T = stats["touchdown_theta_td_contact"]
    Tm2 = stats["touchdown_theta_td_phase_M2"]
    B = stats["body"]
    Pl = stats["planner_recommended"]
    Ch = stats["sanity_checks"]

    def rng(s: Dict[str, float]) -> str:
        return f"{s['p5']:.2f}--{s['p95']:.2f}"

    md: List[str] = []
    md.append("# Ours (v3.1 seed 0) — CPG Reference for Planner Baseline\n\n")
    md.append("## Source\n")
    md.append(f"- Checkpoint: `{CKPT.relative_to(ROOT)}`\n")
    md.append(
        f"- Evaluation: {N_EPISODES} deterministic episodes × {ENV_STEPS_PER_EP} env-steps "
        f"= {stats['n_samples']} samples\n"
    )
    md.append("- v_cmd: 0.12 m/s\n\n")

    md.append("## Per-leg frequency (Hz)\n\n")
    md.append("| Leg | mean | std | median | 5%-95% range |\n")
    md.append("|-----|-----:|----:|-------:|-------------:|\n")
    for leg in LEG_NAMES:
        s = L[f"f_{leg}"]
        md.append(
            f"| {leg} | {s['mean']:.2f} | {s['std']:.2f} | {s['median']:.2f} | {rng(s)} |\n"
        )

    md.append("\n## Per-leg amplitude scaling r = sqrt(mu)\n\n")
    md.append("| Leg | mean r | std r | median r | 5%-95% range |\n")
    md.append("|-----|-------:|------:|---------:|-------------:|\n")
    for leg in LEG_NAMES:
        s = L[f"r_{leg}"]
        md.append(
            f"| {leg} | {s['mean']:.2f} | {s['std']:.2f} | {s['median']:.2f} | {rng(s)} |\n"
        )

    sa, sb = L["a_mm"], L["b_mm"]
    md.append("\n## Forward foot amplitude a (mm)\n\n")
    md.append(f"  mean: {sa['mean']:.2f} mm\n")
    md.append(f"  std:  {sa['std']:.2f} mm\n")
    md.append(f"  range: {sa['min']:.2f}--{sa['max']:.2f}\n\n")

    md.append("## Vertical foot amplitude b (mm)\n\n")
    md.append(f"  mean: {sb['mean']:.2f} mm\n")
    md.append(f"  std:  {sb['std']:.2f} mm\n")
    md.append(f"  range: {sb['min']:.2f}--{sb['max']:.2f}\n")
    md.append("  (Note: should be at action lower bound 1mm)\n\n")

    md.append("## Relative phase offsets (radians, mod 2π)\n\n")
    for rel in ("FR", "RL", "RR"):
        ps = P[f"phase_{rel}_rel"]
        md.append(
            f"  phase_{rel} vs FL: mean {ps['circular_mean']:.2f}  "
            f"std {ps['circular_std']:.2f}\n"
        )
    md.append("\n  Trot reference:\n")
    md.append("    FL:0, FR:π, RL:π, RR:0\n")
    md.append("    (i.e. FL/RR same phase, FR/RL same phase, offset π between pairs)\n\n")
    md.append(
        f"  Closeness to trot: mean deviation {stats['trot_closeness_mean_deviation_rad']:.3f} rad\n\n"
    )

    md.append("## Touchdown phase per leg (radians)\n\n")
    md.append("Contact-based (rising edge on fn > threshold):\n\n")
    md.append("| Leg | mean theta_td | std | deviation from π |\n")
    md.append("|-----|--------------:|----:|-----------------:|\n")
    for leg in LEG_NAMES:
        s = T[leg]
        md.append(
            f"| {leg} | {s['mean']:.2f} | {s['std']:.2f} | "
            f"{s['deviation_from_pi_rad_mean']:.2f} |\n"
        )
    md.append("\nPhase-based (M2 protocol; swing→stance at θ≈π):\n\n")
    md.append("| Leg | mean theta_td | std | deviation from π |\n")
    md.append("|-----|--------------:|----:|-----------------:|\n")
    for leg in LEG_NAMES:
        s = Tm2[leg]
        md.append(
            f"| {leg} | {s['mean']:.2f} | {s['std']:.4f} | "
            f"{s['deviation_from_pi_rad_mean']:.4f} |\n"
        )

    md.append("\n## Body state\n\n")
    md.append(f"  Mean body height: {B['mean_body_z_mm']:.2f} mm\n")
    md.append(f"  Body z range (per ep mean): {B['body_z_range_per_ep_mean_mm']:.2f} mm\n")
    md.append(
        f"  Forward speed: {B['mean_forward_speed_mmps_substep']:.1f} mm/s "
        f"(substep mean −v_y; M4 protocol)\n"
    )
    md.append(
        f"  Forward speed (displacement): {B['mean_forward_speed_mmps_displacement']:.1f} mm/s\n"
    )
    md.append(f"  Lateral speed: {B['mean_lateral_speed_mmps']:.1f} mm/s\n\n")

    md.append("## Recommended Planner Configuration\n\n")
    md.append("For the Planner Baseline (see plan in §6.X), use these constants:\n\n")
    md.append(f"  a_planner       = {Pl['a_planner_mm']:.1f} mm        (mean of Ours, fixed)\n")
    md.append(f"  b_planner       = {Pl['b_planner_mm']:.1f} mm        (mean of Ours, fixed)\n")
    md.append(f"  f_planner       = {Pl['f_planner_Hz']:.2f} Hz       (mean of Ours, fixed; same f for all legs)\n\n")
    md.append("  Per-leg r_planner:\n")
    for leg in LEG_NAMES:
        md.append(f"    r_{leg} = {Pl['r_planner'][leg]:.2f}\n")
    po = Pl["phase_offsets_rad_vs_FL"]
    md.append("\n  Phase offsets:\n")
    md.append("    FL: 0\n")
    md.append(f"    FR: {po['FR']:.2f} rad\n")
    md.append(f"    RL: {po['RL']:.2f} rad\n")
    md.append(f"    RR: {po['RR']:.2f} rad\n\n")
    md.append(
        "  Touchdown phases (phase-based M2, verification): "
        f"{Tm2['FL']['mean']:.2f}, {Tm2['FR']['mean']:.2f}, "
        f"{Tm2['RL']['mean']:.2f}, {Tm2['RR']['mean']:.2f}\n\n"
    )

    md.append("## Sanity check\n\n")
    md.append("The collected values should match the existing M2/M4 outputs for Ours v3.1 seed 0:\n")
    for key, ref in SANITY_REF.items():
        sf = Ch["m2_m4_sanity"][key]
        flag = "OK" if sf["ok"] else "**MISMATCH**"
        if key == "theta_td_dev_rad":
            md.append(
                f"  - {key}: phase-based {sf['measured_phase_based_M2']:.4f}, "
                f"contact-based {sf['measured_contact_based']:.3f}, reference {ref} — {flag}\n"
            )
        elif key == "forward_speed_mm_s":
            md.append(
                f"  - {key}: measured {sf['measured']:.1f} "
                f"(displacement {sf['measured_displacement']:.1f}), reference {ref} — {flag}\n"
            )
        else:
            md.append(
                f"  - {key}: measured {sf['measured']:.3f}, reference {ref} — {flag}\n"
            )
    md.append("\n")
    if Ch.get("any_failed"):
        md.append("### ⚠️ SANITY CHECK FAILURES\n\n")
        if not Ch["all_20_episodes_complete_82_steps"]:
            md.append(
                f"- Only {Ch['episodes_completed_count']}/{N_EPISODES} episodes completed 82 steps\n"
            )
        for k, v in [
            ("f in [1,5] Hz", Ch["f_in_1_5_Hz"]),
            ("r in [0,1]", Ch["r_in_0_1"]),
            ("a in [9,11] mm", Ch["a_in_9_11_mm"]),
            ("b in [0.5,5] mm", Ch["b_in_0_5_5_mm"]),
            ("no NaN", Ch["no_NaN"]),
            ("phase trot-like", Ch["phase_pattern_trot_like"]),
        ]:
            if not v:
                md.append(f"- Failed: {k}\n")
    else:
        md.append("All automated sanity checks passed.\n")

    path.write_text("".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting rollout...")
    rows, rollout_meta = collect_rollout()
    print(f"Collected {len(rows)} env-step rows")

    stats = compute_statistics(rows, rollout_meta)

    # JSON (exclude large arrays)
    json_out = {k: v for k, v in stats.items()}
    json_out["source"] = {
        "checkpoint": str(CKPT),
        "n_episodes": N_EPISODES,
        "env_steps_per_episode": ENV_STEPS_PER_EP,
        "v_cmd_mps": 0.12,
        "contact_threshold_N": FN_CONTACT_THRESHOLD,
    }
    json_out["episodes"] = rollout_meta["episodes"]
    (OUT_DIR / "ours_cpg_summary_statistics.json").write_text(
        json.dumps(json_out, indent=2), encoding="utf-8"
    )

    write_csv(rows, OUT_DIR / "ours_cpg_per_envstep.csv")
    write_markdown(rows, stats, OUT_DIR / "ours_cpg_summary.md")
    plot_phase_pattern(rollout_meta, OUT_DIR / "ours_cpg_phase_pattern.png")

    print(f"\nWrote outputs to {OUT_DIR}")
    if stats["sanity_checks"].get("any_failed"):
        print("WARNING: some sanity checks failed — see ours_cpg_summary.md")


if __name__ == "__main__":
    main()
