#!/usr/bin/env python3
"""Thesis §5.2 single-speed symmetry / gait-structure metrics for M2/M3/M4.

Uses Phase-1 stride_2cycles_*.npz + *_extras.npz (episode 0, speed-matched).
Lightweight deterministic re-run of episode 0 only for CPG theta / Ours mu in window.
No retraining. Does not modify existing npz files.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CPG_ROOT = Path(__file__).resolve().parents[1]
TROT_SRC = (CPG_ROOT / "../TrotGait/src").resolve()
os.chdir(CPG_ROOT)
sys.path.insert(0, str(CPG_ROOT))
sys.path.insert(0, str(CPG_ROOT / "env"))
sys.path.insert(0, str(CPG_ROOT / "experiments"))
sys.path.insert(0, str(CPG_ROOT / "scripts"))
sys.path.insert(0, str(TROT_SRC))

from run_cder_closure_validation import LEG_NAMES  # noqa: E402

DT = 0.002
TAU_MAX = 0.157
SAT_THRESH = 0.95 * TAU_MAX
NOMINAL_PHASE = {"FL": 0.0, "FR": math.pi, "RL": math.pi, "RR": 0.0}
JOINT_NAMES = [
    "FL_hip", "FL_knee", "FR_hip", "FR_knee",
    "RL_hip", "RL_knee", "RR_hip", "RR_knee",
]
BASE = CPG_ROOT / "outputs/final_result/single_speed"

CONTROLLERS = [
    {
        "json_key": "Planner",
        "subdir": "m3_planner",
        "tag": "planner",
        "speed_key": "m3",
    },
    {
        "json_key": "Planner-simplified",
        "subdir": "m4_planner_simplified",
        "tag": "planner_simplified",
        "speed_key": "m4",
    },
    {
        "json_key": "Ours",
        "subdir": "m2_ours",
        "tag": "ours",
        "speed_key": "m2",
    },
]

ACTION_RANGES = {"f": (0.3, 2.5), "mu": (0.2, 1.0)}


def _phase_dev(theta: float, nominal: float) -> float:
    d = (theta - nominal + math.pi) % (2.0 * math.pi) - math.pi
    return abs(float(d))


def _envelope_filter(x: np.ndarray, win: int = 15) -> np.ndarray:
    """Simple moving-average envelope on 0/1 contact signal."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    w = max(3, int(win) | 1)
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, k, mode="same")


def _touchdown_indices_fz(fz: np.ndarray, lo: float = 0.1, hi: float = 0.5) -> np.ndarray:
    """Touchdown when |Fz| crosses from < lo to > hi (stance onset)."""
    fn = np.abs(fz)
    rising = (fn[1:] >= hi) & (fn[:-1] < lo)
    return np.where(rising)[0] + 1


def _load_stride_bundle(subdir: str, tag: str) -> Dict[str, Any]:
    d = np.load(BASE / subdir / f"stride_2cycles_{tag}.npz", allow_pickle=True)
    ex_path = BASE / subdir / f"stride_2cycles_{tag}_extras.npz"
    ex = np.load(ex_path) if ex_path.exists() else None
    ev = json.loads((BASE / subdir / "eval_summary.json").read_text(encoding="utf-8"))
    return {"base": d, "extras": ex, "eval": ev}


def _r_per_leg(cfg: Dict[str, Any], tag: str, mu_window: Optional[np.ndarray]) -> Tuple[List[float], str]:
    """Return r per leg [FL,FR,RL,RR] and note."""
    if tag == "planner":
        # TrotGait MouseController: uniform ellipse scaling (a_scale=b_scale=1 default).
        return [1.0, 1.0, 1.0, 1.0], "open-loop uniform ellipse (nominal r=1.0)"
    if tag == "planner_simplified":
        r_cfg = cfg.get("config", {}).get("r", {})
        if r_cfg:
            return [float(r_cfg[l]) for l in LEG_NAMES], "frozen from Ours seed0 (eval_summary config.r)"
        return [0.96, 0.97, 0.81, 0.79], "default R_M4 fallback"
    # Ours
    if mu_window is not None and mu_window.size:
        r = [float(np.sqrt(max(float(np.mean(mu_window[:, i])), 0.0))) for i in range(4)]
        return r, "mean applied mu4 over 2-stride window (episode 0 re-run)"
    return [float("nan")] * 4, "mu unavailable — see v31_action_stats seed0 episode means"


def _collect_ours_mu_f_window(i0: int, i1: int) -> Tuple[np.ndarray, float]:
    """Re-run M2 ep0; return mu4 (T_sub, 4) and mean f4 (Hz) for [i0:i1)."""
    from stable_baselines3 import PPO
    from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3
    from eval_single_speed_three_way import CKPT_M2, ENV_KWARGS_M2, ENV_STEPS, load_env_kwargs

    env = RatCpgEnvEnergySubstep50ShapeV3(**load_env_kwargs(ENV_KWARGS_M2))
    policy = PPO.load(str(CKPT_M2), env=env)
    e = env.unwrapped
    mu_rows: List[np.ndarray] = []
    f_rows: List[np.ndarray] = []
    obs, _ = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < ENV_STEPS:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        steps += 1
        mu_rows.append(np.array(e._cur_mu4, dtype=np.float64))
        f_rows.append(np.array(e._cur_f4, dtype=np.float64))
    env.close()
    mu_env = np.stack(mu_rows, axis=0)
    f_env = np.stack(f_rows, axis=0)
    mu_sub = np.repeat(mu_env, 50, axis=0)[:4100]
    f_sub = np.repeat(f_env, 50, axis=0)[:4100]
    sl = mu_sub[i0:i1]
    sf = f_sub[i0:i1]
    return sl, float(np.mean(sf))


def _collect_theta_episode(tag: str, fre_m3: float, f_m4: float) -> np.ndarray:
    """Return theta (4100, 4) for episode 0 at substep resolution."""
    from eval_single_speed_three_way import (
        A_M4,
        CKPT_M2,
        DT_SUBSTEP,
        ENV_KWARGS_M2,
        ENV_STEPS,
        MODEL_XML,
        N_SUBSTEPS,
        PHASE_OFFSETS_M4,
        _configure_m4_cpg,
        _make_m4_env,
        _m4_substep,
        load_env_kwargs,
        override_kp_kv,
    )

    th: List[np.ndarray] = []
    if tag == "ours":
        from stable_baselines3 import PPO
        from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3
        import mujoco

        env = RatCpgEnvEnergySubstep50ShapeV3(**load_env_kwargs(ENV_KWARGS_M2))
        policy = PPO.load(str(CKPT_M2), env=env)
        e = env.unwrapped
        orig = mujoco.mj_step

        def wrap(m, d):
            orig(m, d)
            th.append(np.array(e.foot_path.cpg.theta, dtype=np.float64).copy())

        mujoco.mj_step = wrap  # type: ignore[assignment]
        try:
            obs, _ = env.reset(seed=0)
            done = False
            steps = 0
            while not done and steps < ENV_STEPS:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
                steps += 1
        finally:
            mujoco.mj_step = orig  # type: ignore[assignment]
            env.close()
        return np.stack(th, axis=0)

    if tag == "planner":
        from Controller import MouseController  # noqa: E402
        from ToSim import SimModel  # noqa: E402

        sim = SimModel(str(MODEL_XML))
        override_kp_kv(sim.model, kp=2.0, kv=0.0)
        controller = MouseController(fre_m3, DT_SUBSTEP, 20)
        settle = int(2.0 / DT_SUBSTEP)
        warmup = int(2.0 / DT_SUBSTEP)
        for _ in range(settle):
            sim.runStep([0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, 0, 0, 0, 0], DT_SUBSTEP, realtime=False)
        sim.initializing()
        for _ in range(warmup):
            c, _ = controller.runStep()
            sim.runStep(c, DT_SUBSTEP, realtime=False)
        sim.initializing()
        stenum = float(controller.SteNum)
        for _ in range(N_SUBSTEPS):
            c, _ = controller.runStep()
            sim.runStep(c, DT_SUBSTEP, realtime=False)
            cur = float(controller.curStep)
            theta4 = np.array(
                [
                    2.0 * math.pi * ((cur + controller.stepDiff[i]) % stenum) / stenum
                    for i in range(4)
                ],
                dtype=np.float64,
            )
            th.append(theta4)
        return np.stack(th, axis=0)

    if tag == "planner_simplified":
        env = _make_m4_env()
        _configure_m4_cpg(env, seed=0, f_planner=f_m4, a_planner=A_M4)
        e = env.unwrapped
        for k in range(N_SUBSTEPS):
            _m4_substep(env, k, f_m4)
            th.append(np.array(e.foot_path.cpg.theta, dtype=np.float64).copy())
        env.close()
        return np.stack(th, axis=0)

    raise ValueError(tag)


def compute_one(ctrl: Dict[str, Any], theta_ep: np.ndarray, mu_window: Optional[np.ndarray], cpg_f_override: Optional[float] = None) -> Dict[str, Any]:
    sub, tag = ctrl["subdir"], ctrl["tag"]
    bundle = _load_stride_bundle(sub, tag)
    d, ex, ev = bundle["base"], bundle["extras"], bundle["eval"]
    i0 = int(d["start_idx_global"])
    stride_len = int(d["stride_length_used_substeps"][0]) if "stride_length_used_substeps" in d.files else int(
        d.get("stride_length_mean_substeps", 0)
    )

    tau = np.asarray(d["tau"], dtype=np.float64)
    body_pos = np.asarray(d["body_pos"], dtype=np.float64)
    body_vel = np.asarray(d["body_vel"], dtype=np.float64)
    contact = np.asarray(d["contact"], dtype=bool)
    grf = np.asarray(d["grf_world"], dtype=np.float64)
    foot_pb = np.asarray(d["foot_pos_body"], dtype=np.float64)

    T = tau.shape[0]
    theta_w = theta_ep[i0 : i0 + T]

    # [1] r per leg
    r_list, r_note = _r_per_leg(ev, tag, mu_window)

    # [2] phase deviation at touchdown
    phase_dev = []
    td_stride_lens = []
    for li, leg in enumerate(LEG_NAMES):
        td = _touchdown_indices_fz(grf[:, li, 2])
        if td.size == 0:
            phase_dev.append(float("nan"))
            td_stride_lens.append([])
            continue
        devs = [_phase_dev(float(theta_w[t, li]), NOMINAL_PHASE[leg]) for t in td]
        phase_dev.append(float(np.mean(devs)))
        if td.size >= 2:
            td_stride_lens.append(list(np.diff(td).astype(int)))
        else:
            td_stride_lens.append([])

    # [3] foot lift pp (mm) — prefer extras foot_pos_world
    lift_pp = []
    lift_note = "world frame from stride_extras foot_pos_world"
    if ex is not None and "foot_pos_world" in ex.files:
        fpw = np.asarray(ex["foot_pos_world"], dtype=np.float64)
        for li in range(4):
            z = fpw[:, li, 2] * 1000.0
            lift_pp.append(float(z.max() - z.min()))
    else:
        lift_note = "approximate (body rotation ignored; foot_pos_body z only)"
        for li in range(4):
            z = foot_pb[:, li, 2] * 1000.0
            lift_pp.append(float(z.max() - z.min()))

    # [4] torque saturation (%)
    sat = {}
    for ji, jname in enumerate(JOINT_NAMES):
        frac = float(np.mean(np.abs(tau[:, ji]) > SAT_THRESH)) * 100.0
        sat[jname] = round(frac, 4)

    # [5] body bounce
    bz = body_pos[:, 2] * 1000.0
    bvz = body_vel[:, 2] * 1000.0
    body = {
        "body_z_pp_mm": round(float(bz.max() - bz.min()), 4),
        "body_vz_pp_mm_per_s": round(float(bvz.max() - bvz.min()), 4),
        "body_vz_rms_mm_per_s": round(float(np.sqrt(np.mean(bvz ** 2))), 4),
    }

    # [6] gait structure
    duty = []
    for li in range(4):
        env_f = _envelope_filter(contact[:, li].astype(np.float64))
        duty.append(round(float(np.mean(env_f)), 4))
    stride_period_s = stride_len * DT
    gait_freq = 1.0 / stride_period_s if stride_period_s > 0 else float("nan")
    cfg = ev.get("config", {})
    cpg_f = cpg_f_override if cpg_f_override is not None else (cfg.get("fre") or cfg.get("f") or float("nan"))

    out = {
        "speed_mm_per_s": round(float(ev["fwd_speed_mm_s"]["mean"]), 4),
        "r_per_leg": [round(r_list[i], 4) for i in range(4)],
        "r_per_leg_note": r_note,
        "phase_dev_per_leg": [round(phase_dev[i], 4) for i in range(4)],
        "foot_lift_pp_per_leg": [round(lift_pp[i], 4) for i in range(4)],
        "foot_lift_note": lift_note,
        "torque_saturation_per_joint": sat,
        **body,
        "gait_freq_Hz": round(gait_freq, 4),
        "stride_len_substep": int(stride_len),
        "stride_period_s": round(stride_period_s, 4),
        "cpg_f_Hz": round(float(cpg_f), 4) if np.isfinite(float(cpg_f)) else None,
        "duty_factor_per_leg": [duty[i] for i in range(4)],
        "_leg_order": LEG_NAMES,
        "_internal": {
            "window": [int(i0), int(i0 + T)],
            "td_stride_lens_per_leg": {
                LEG_NAMES[i]: [int(x) for x in td_stride_lens[i]] for i in range(4)
            },
            "r_per_leg_dict": {LEG_NAMES[i]: round(r_list[i], 4) for i in range(4)},
        },
    }
    return out


def sanity_checks(results: Dict[str, Any]) -> Dict[str, Any]:
    chk: Dict[str, Any] = {}
    for name, d in results.items():
        if name.startswith("_"):
            continue
        r_vals = list(d["r_per_leg"])
        pd_vals = [v for v in d["phase_dev_per_leg"] if np.isfinite(v)]
        lift_vals = list(d["foot_lift_pp_per_leg"])
        sat_vals = list(d["torque_saturation_per_joint"].values())
        td_lens = d["_internal"]["td_stride_lens_per_leg"]
        stride_ref = d["stride_len_substep"]

        leg_periods = []
        for leg in LEG_NAMES:
            lens = td_lens[leg]
            if lens:
                leg_periods.extend(lens)
        chk[name] = {
            "stride_len_substep": stride_ref,
            "td_period_mean_per_leg": {
                leg: (round(float(np.mean(td_lens[leg])), 1) if td_lens[leg] else None)
                for leg in LEG_NAMES
            },
            "td_period_spread_vs_ref": {
                leg: (round(float(np.std(td_lens[leg])), 2) if len(td_lens[leg]) > 1 else 0.0)
                for leg in LEG_NAMES
            },
            "r_sigma_across_legs": round(float(np.std(r_vals)), 4),
            "phase_dev_mean_rad": round(float(np.mean(pd_vals)), 4) if pd_vals else None,
            "foot_lift_asymmetry_(max-min)/mean": round(
                (max(lift_vals) - min(lift_vals)) / max(np.mean(lift_vals), 1e-9), 4
            ),
            "torque_sat_mean_pct": round(float(np.mean(sat_vals)), 4),
            "torque_sat_max_joint": max(d["torque_saturation_per_joint"], key=d["torque_saturation_per_joint"].get),
            "body_z_pp_mm": d["body_z_pp_mm"],
        }
    # cross-controller ordering hints
    names = [k for k in results if not k.startswith("_")]
    chk["_ordering"] = {
        "r_sigma_smaller_is_more_symmetric": sorted(names, key=lambda n: chk[n]["r_sigma_across_legs"]),
        "phase_dev_smaller_is_better": sorted(
            names, key=lambda n: chk[n]["phase_dev_mean_rad"] or 999.0
        ),
        "torque_sat_higher_is_worse": sorted(names, key=lambda n: -chk[n]["torque_sat_mean_pct"]),
    }
    return chk


def main() -> None:
    # CPG fre/f from Phase-1 eval configs
    m3_cfg = json.loads((BASE / "m3_planner/eval_summary.json").read_text())["config"]
    m4_cfg = json.loads((BASE / "m4_planner_simplified/eval_summary.json").read_text())["config"]
    fre_m3 = float(m3_cfg["fre"])
    f_m4 = float(m4_cfg["f"])

    print("Collecting CPG theta (episode 0 re-run, read-only)...")
    theta_cache = {
        "planner": _collect_theta_episode("planner", fre_m3, f_m4),
        "planner_simplified": _collect_theta_episode("planner_simplified", fre_m3, f_m4),
        "ours": _collect_theta_episode("ours", fre_m3, f_m4),
    }

    print("Collecting Ours mu over 2-stride window...")
    ours_bundle = _load_stride_bundle("m2_ours", "ours")
    i0 = int(ours_bundle["base"]["start_idx_global"])
    i1 = int(ours_bundle["base"]["end_idx_global"])
    mu_window, ours_f_hz = _collect_ours_mu_f_window(i0, i1)

    results: Dict[str, Any] = {}
    for ctrl in CONTROLLERS:
        tag = ctrl["tag"]
        print(f"Computing {ctrl['json_key']}...")
        mu_w = mu_window if tag == "ours" else None
        f_ov = ours_f_hz if tag == "ours" else None
        block = compute_one(ctrl, theta_cache[tag], mu_w, cpg_f_override=f_ov)
        # strip internal for JSON top-level but keep in sanity
        internal = block.pop("_internal")
        block["_internal"] = internal  # keep for sanity, user can ignore
        results[ctrl["json_key"]] = block

    sanity = sanity_checks(results)
    payload = {**results, "_sanity_checks": sanity, "_meta": {
        "episode": 0,
        "dt_substep_s": DT,
        "tau_max_Nm": TAU_MAX,
        "saturation_threshold_Nm": SAT_THRESH,
        "data_source": "Phase-1 stride_2cycles_* + extras; theta/mu from ep0 eval re-run",
    }}

    # Public-facing copy without _internal
    public = {}
    for k, v in results.items():
        public[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}

    out_paths = [
        BASE / "thesis_metrics_3way.json",
        CPG_ROOT / "outputs/thesis_metrics_3way.json",
    ]
    for op in out_paths:
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(public, indent=2), encoding="utf-8")
        print(f"Wrote {op}")

    full_path = BASE / "thesis_metrics_3way_full.json"
    full_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {full_path} (includes sanity + _internal)")

    # Try user-requested upload path
    upload = Path("/mnt/user-data/uploads/thesis_metrics_3way.json")
    try:
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_text(json.dumps(public, indent=2), encoding="utf-8")
        print(f"Wrote {upload}")
    except OSError as e:
        print(f"Note: could not write {upload}: {e}")

    print("\n=== Sanity checks ===")
    print(json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
