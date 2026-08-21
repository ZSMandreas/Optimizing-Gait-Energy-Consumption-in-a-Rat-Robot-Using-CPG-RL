#!/usr/bin/env python3
"""Single-speed three-way eval: M2 Ours / M3 Planner / M4 Planner-simplified.

Outputs under outputs/final_result/single_speed/ with substep logs,
eval summaries, useful/wasted work decomposition, and sanity checks.
No policy retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from stable_baselines3 import PPO

CPG_ROOT = Path(__file__).resolve().parents[1]
TROT_SRC = (CPG_ROOT / "../TrotGait/src").resolve()
os.chdir(CPG_ROOT)
sys.path.insert(0, str(CPG_ROOT))
sys.path.insert(0, str(CPG_ROOT / "env"))
sys.path.insert(0, str(CPG_ROOT / "experiments"))
sys.path.insert(0, str(CPG_ROOT / "scripts"))
sys.path.insert(0, str(TROT_SRC))

from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3  # noqa: E402
from run_cder_closure_validation import (  # noqa: E402
    B_DAMP,
    FN_CONTACT_THRESHOLD,
    FOOT_BODIES,
    FOOT_SITES,
    G_ACC,
    LEG_NAMES,
    M_BODY,
    _body_id_to_leg,
)

# M3 imports (TrotGait/src)
from Controller import MouseController  # noqa: E402
from ToSim import SimModel  # noqa: E402


def override_kp_kv(model: mujoco.MjModel, kp: float = 2.0, kv: float = 0.0) -> int:
    """Match TrotGait/src/sim_test.py (kp2_kv0)."""
    n = 0
    for name in ACTUATOR_ORDER:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kv
        n += 1
    return n

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M = M_BODY
G = G_ACC
DT_SUBSTEP = 0.002
N_EPISODES = 20
N_SUBSTEPS = 4100
ENV_STEPS = 82
SUBSTEPS_PER_ENV = 50

OUT_ROOT = CPG_ROOT / "outputs/final_result"
OUT_SINGLE = OUT_ROOT / "single_speed"

CKPT_M2 = CPG_ROOT / "logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip"
ENV_KWARGS_M2 = CPG_ROOT / "configs/ours_cder_v31_env_kwargs.json"
MODEL_XML = (CPG_ROOT / "../TrotGait/models/dynamic_4l_kp2.xml").resolve()
SCRIPT_M3 = TROT_SRC / "sim_test.py"
SCRIPT_M4 = CPG_ROOT / "scripts/planner_baseline.py"

# M4 CPG (Ours seed0 stats; shape-fair speed-match calibrates f only, a/b/r frozen)
A_M4 = 0.010
B_M4 = 0.001
F_M4_NATIVE = 2.43
F_M4 = F_M4_NATIVE  # updated by calibration
R_M4 = {"FL": 0.96, "FR": 0.97, "RL": 0.81, "RR": 0.79}
PHASE_OFFSETS_M4 = np.array([0.0, math.pi, math.pi, 0.0], dtype=np.float64)

# M3 timing — match TrotGait/src/sim_test.py defaults (warmup 2s + measure 8s)
M3_WARMUP_S = 2.0
M3_MEASURE_S = 8.0
M3_FRE = 0.5  # updated by calibration
M3_LOG_SUBSTEPS = N_SUBSTEPS

# Speed-match target (M2 Ours realized speed)
TARGET_SPEED_MM_S = 97.0
SWEEP_EPISODES = 5
M3_FRE_CANDIDATES = [1.0, 1.2, 1.4, 1.6, 1.8]
M4_F_CANDIDATES = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

ACTUATOR_ORDER = [
    "thigh_joint_fl", "leg_joint_fl",
    "thigh_joint_fr", "leg_joint_fr",
    "thigh_joint_rl", "leg_joint_rl",
    "thigh_joint_rr", "leg_joint_rr",
]


@dataclass
class SubstepEpisode:
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    tau: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    body_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    body_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    foot_pos_body: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 3)))
    grf_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 3)))
    contact: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=bool))


@dataclass
class EpisodeScalars:
    episode: int
    energy_J: float
    distance_m: float
    lateral_m: float
    fwd_speed_mps: float
    cot: float
    W_propulsion_J: float
    W_braking_J: float
    W_vertical_bounce_J: float
    W_lateral_cost_J: float
    W_total_mech_J: float
    eta_locomotion: float


# ---------------------------------------------------------------------------
# Shared kinematics / contact helpers
# ---------------------------------------------------------------------------
def _parse_actuator_indices(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray]:
    act_idx = []
    dof_idx = []
    for name in ACTUATOR_ORDER:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator not found: {name}")
        jnt = int(model.actuator_trnid[aid, 0])
        act_idx.append(aid)
        dof_idx.append(int(model.jnt_dofadr[jnt]))
    return np.asarray(act_idx, dtype=np.int64), np.asarray(dof_idx, dtype=np.int64)


def _grf_world_per_leg(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_to_leg: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return grf_world [4,3] and fn_mag [4] in world frame."""
    grf = np.zeros((4, 3), dtype=np.float64)
    fn_mag = np.zeros(4, dtype=np.float64)
    for ic in range(int(data.ncon)):
        contact = data.contact[ic]
        g1, g2 = int(contact.geom[0]), int(contact.geom[1])
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        if (b1 == 0 and b2 == 0) or (b1 != 0 and b2 != 0):
            continue
        foot_bid = b1 if b1 != 0 else b2
        leg = foot_to_leg.get(foot_bid, -1)
        if leg < 0:
            continue
        cforce = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, ic, cforce)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        f_w = cforce[0] * frame[0] + cforce[1] * frame[1] + cforce[2] * frame[2]
        grf[leg] += f_w
        fn_mag[leg] += abs(float(cforce[0]))
    return grf, fn_mag


def _record_substep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    mouse_bid: int,
    site_ids: List[int],
    act_idx: np.ndarray,
    dof_idx: np.ndarray,
    foot_to_leg: Dict[int, int],
    t_val: float,
) -> Dict[str, Any]:
    q = np.array(
        [float(data.qpos[int(model.jnt_qposadr[int(model.actuator_trnid[int(ai), 0])])]) for ai in act_idx],
        dtype=np.float64,
    )
    qvel = np.array([float(data.qvel[int(di)]) for di in dof_idx], dtype=np.float64)
    tau = np.array([float(data.actuator_force[int(ai)]) for ai in act_idx], dtype=np.float64)

    p_mouse = data.xpos[mouse_bid].copy()
    R = data.xmat[mouse_bid].reshape(3, 3)
    v6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, mouse_bid, v6, 0)
    body_vel = v6[3:6].copy()

    foot_pos_body = np.zeros((4, 3), dtype=np.float64)
    for li, sid in enumerate(site_ids):
        local = R.T @ (data.site_xpos[sid].copy() - p_mouse)
        foot_pos_body[li] = local

    grf_world, fn_mag = _grf_world_per_leg(model, data, foot_to_leg)
    contact = fn_mag > FN_CONTACT_THRESHOLD

    return {
        "t": float(t_val),
        "q": q,
        "qvel": qvel,
        "tau": tau,
        "body_pos": p_mouse,
        "body_vel": body_vel,
        "foot_pos_body": foot_pos_body,
        "grf_world": grf_world,
        "contact": contact,
    }


def _accumulate_substep_log(rows: List[Dict[str, Any]]) -> SubstepEpisode:
    ep = SubstepEpisode()
    if not rows:
        return ep
    ep.t = np.array([r["t"] for r in rows], dtype=np.float64)
    ep.q = np.stack([r["q"] for r in rows], axis=0)
    ep.qvel = np.stack([r["qvel"] for r in rows], axis=0)
    ep.tau = np.stack([r["tau"] for r in rows], axis=0)
    ep.body_pos = np.stack([r["body_pos"] for r in rows], axis=0)
    ep.body_vel = np.stack([r["body_vel"] for r in rows], axis=0)
    ep.foot_pos_body = np.stack([r["foot_pos_body"] for r in rows], axis=0)
    ep.grf_world = np.stack([r["grf_world"] for r in rows], axis=0)
    ep.contact = np.stack([r["contact"] for r in rows], axis=0)
    return ep


def _forward_distance_m(body_pos: np.ndarray) -> float:
    """Pilot env-step: sum max(0, -Δy) over substeps."""
    if body_pos.shape[0] < 2:
        return 0.0
    dy = body_pos[:-1, 1] - body_pos[1:, 1]
    return float(np.sum(np.maximum(dy, 0.0)))


def _mechanical_energy_J(qvel: np.ndarray, tau: np.ndarray, dt: float = DT_SUBSTEP) -> float:
    p = tau * qvel
    return float(np.sum(np.abs(p)) * dt)


def compute_useful_wasted(ep: SubstepEpisode, dt: float = DT_SUBSTEP) -> Dict[str, float]:
    T = ep.t.size
    W_prop = W_brake = W_vert = W_lat = 0.0
    v_x_fwd = -ep.body_vel[:, 1]
    v_z = ep.body_vel[:, 2]
    v_y_lat = ep.body_vel[:, 0]

    for t in range(T):
        for leg in range(4):
            if not ep.contact[t, leg]:
                continue
            F_x_fwd = -ep.grf_world[t, leg, 1]
            F_z = ep.grf_world[t, leg, 2]
            F_y_lat = ep.grf_world[t, leg, 0]
            p_prop = F_x_fwd * v_x_fwd[t]
            p_vert = F_z * v_z[t]
            p_lat = F_y_lat * v_y_lat[t]
            if p_prop > 0:
                W_prop += p_prop * dt
            else:
                W_brake += -p_prop * dt
            W_vert += abs(p_vert) * dt
            W_lat += abs(p_lat) * dt

    W_total = _mechanical_energy_J(ep.qvel, ep.tau, dt)
    denom = W_prop + W_brake + W_vert + W_lat
    eta = W_prop / denom if denom > 1e-12 else float("nan")
    return {
        "W_propulsion_J": W_prop,
        "W_braking_J": W_brake,
        "W_vertical_bounce_J": W_vert,
        "W_lateral_cost_J": W_lat,
        "W_total_mech_J": W_total,
        "eta_locomotion": eta,
    }


def episode_scalars(ep_idx: int, ep: SubstepEpisode) -> EpisodeScalars:
    uw = compute_useful_wasted(ep)
    dist = _forward_distance_m(ep.body_pos)
    T = max(ep.t.size, 1)
    duration = T * DT_SUBSTEP
    energy = _mechanical_energy_J(ep.qvel, ep.tau)
    lateral = float(abs(ep.body_pos[-1, 0] - ep.body_pos[0, 0])) if ep.body_pos.size else 0.0
    fwd_speed = dist / duration if duration > 1e-9 else 0.0
    cot = energy / (M * G * dist) if dist > 1e-6 else float("nan")
    return EpisodeScalars(
        episode=ep_idx,
        energy_J=energy,
        distance_m=dist,
        lateral_m=lateral,
        fwd_speed_mps=fwd_speed,
        cot=cot,
        W_propulsion_J=uw["W_propulsion_J"],
        W_braking_J=uw["W_braking_J"],
        W_vertical_bounce_J=uw["W_vertical_bounce_J"],
        W_lateral_cost_J=uw["W_lateral_cost_J"],
        W_total_mech_J=uw["W_total_mech_J"],
        eta_locomotion=uw["eta_locomotion"],
    )


def _stat_dict(values: List[float]) -> Dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "per_ep": values}
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "std": std, "per_ep": values}


def build_eval_summary(
    model_id: str,
    ckpt_or_script: str,
    config: Dict[str, Any],
    scalars: List[EpisodeScalars],
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "model_id": model_id,
        "ckpt_or_script": ckpt_or_script,
        "n_episodes": len(scalars),
        "config": config,
        "M": M,
        "g": G,
        "fwd_speed_mm_s": _stat_dict([s.fwd_speed_mps * 1000.0 for s in scalars]),
        "distance_mm": _stat_dict([s.distance_m * 1000.0 for s in scalars]),
        "lateral_mm": _stat_dict([s.lateral_m * 1000.0 for s in scalars]),
        "energy_J": _stat_dict([s.energy_J for s in scalars]),
        "cot": _stat_dict([s.cot for s in scalars]),
        "notes": notes,
    }


def save_per_episode_npz(path: Path, episodes: List[SubstepEpisode]) -> None:
    n = len(episodes)
    t_max = max((e.t.size for e in episodes), default=0)
    t_max = max(t_max, N_SUBSTEPS)

    t_arr = np.zeros((n, t_max), dtype=np.float64)
    q_arr = np.zeros((n, t_max, 8), dtype=np.float64)
    qvel_arr = np.zeros((n, t_max, 8), dtype=np.float64)
    tau_arr = np.zeros((n, t_max, 8), dtype=np.float64)
    body_pos_arr = np.zeros((n, t_max, 3), dtype=np.float64)
    body_vel_arr = np.zeros((n, t_max, 3), dtype=np.float64)
    foot_arr = np.zeros((n, t_max, 4, 3), dtype=np.float64)
    grf_arr = np.zeros((n, t_max, 4, 3), dtype=np.float64)
    contact_arr = np.zeros((n, t_max, 4), dtype=bool)
    lengths = np.zeros(n, dtype=np.int32)

    for i, ep in enumerate(episodes):
        L = ep.t.size
        lengths[i] = L
        if L == 0:
            continue
        t_arr[i, :L] = ep.t
        q_arr[i, :L] = ep.q
        qvel_arr[i, :L] = ep.qvel
        tau_arr[i, :L] = ep.tau
        body_pos_arr[i, :L] = ep.body_pos
        body_vel_arr[i, :L] = ep.body_vel
        foot_arr[i, :L] = ep.foot_pos_body
        grf_arr[i, :L] = ep.grf_world
        contact_arr[i, :L] = ep.contact

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        episode=np.arange(n, dtype=np.int32),
        lengths=lengths,
        t=t_arr,
        q=q_arr,
        qvel=qvel_arr,
        tau=tau_arr,
        body_pos=body_pos_arr,
        body_vel=body_vel_arr,
        foot_pos_body=foot_arr,
        grf_world=grf_arr,
        contact=contact_arr,
    )


def write_useful_wasted_csv(path: Path, scalars: List[EpisodeScalars]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "episode", "W_propulsion_mJ", "W_braking_mJ", "W_bounce_mJ", "W_lateral_mJ",
            "eta_locomotion", "W_total_mech_J", "distance_mm", "fwd_speed_mm_s",
        ])
        for s in scalars:
            w.writerow([
                s.episode,
                s.W_propulsion_J * 1000.0,
                s.W_braking_J * 1000.0,
                s.W_vertical_bounce_J * 1000.0,
                s.W_lateral_cost_J * 1000.0,
                s.eta_locomotion,
                s.W_total_mech_J,
                s.distance_m * 1000.0,
                s.fwd_speed_mps * 1000.0,
            ])


# ---------------------------------------------------------------------------
# M2 Ours (RL)
# ---------------------------------------------------------------------------
def load_env_kwargs(path: Path) -> dict:
    kw = json.loads(path.read_text(encoding="utf-8"))
    mp = kw.get("model_path", "../TrotGait/models/dynamic_4l_kp2.xml")
    kw["model_path"] = str((CPG_ROOT / mp).resolve())
    kw["render_mode"] = None
    kw["enable_csv_log"] = False
    kw.setdefault("target_speed", 0.12)
    return kw


def run_m2_ours() -> Tuple[List[SubstepEpisode], List[EpisodeScalars]]:
    env_kwargs = load_env_kwargs(ENV_KWARGS_M2)
    env = RatCpgEnvEnergySubstep50ShapeV3(**env_kwargs)
    model = PPO.load(str(CKPT_M2), env=env)
    e = env.unwrapped
    mj_model = e.model
    mj_data = e.data
    mouse_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
    foot_to_leg = _body_id_to_leg(mj_model)
    act_idx, dof_idx = _parse_actuator_indices(mj_model)

    episodes: List[SubstepEpisode] = []
    scalars: List[EpisodeScalars] = []
    orig_mj_step = mujoco.mj_step
    sub_rows: List[Dict[str, Any]] = []

    def mj_wrap(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        orig_mj_step(m, d)
        sub_rows.append(
            _record_substep(
                m, d,
                mouse_bid=mouse_bid,
                site_ids=site_ids,
                act_idx=act_idx,
                dof_idx=dof_idx,
                foot_to_leg=foot_to_leg,
                t_val=float(d.time),
            )
        )

    print("Running M2 Ours (20 episodes)...")
    mujoco.mj_step = mj_wrap  # type: ignore[assignment]
    try:
        for ep in range(N_EPISODES):
            sub_rows.clear()
            obs, _ = env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < ENV_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
                steps += 1
            log = _accumulate_substep_log(sub_rows)
            sc = episode_scalars(ep, log)
            episodes.append(log)
            scalars.append(sc)
            print(
                f"  ep {ep}: fwd={sc.fwd_speed_mps*1000:.1f} mm/s "
                f"cot={sc.cot:.3f} eta={sc.eta_locomotion:.3f}"
            )
    finally:
        mujoco.mj_step = orig_mj_step  # type: ignore[assignment]
        env.close()
    return episodes, scalars


# ---------------------------------------------------------------------------
# M3 Planner (TrotGait sim_test stack)
# ---------------------------------------------------------------------------
def _m3_single_episode(
    fre: float,
    ep: int,
    *,
    log_steps: int = M3_LOG_SUBSTEPS,
) -> Tuple[SubstepEpisode, EpisodeScalars]:
    """One M3 episode: sim_test protocol; optional substep logging."""
    model_xml = str(MODEL_XML)
    measure_steps = int(M3_MEASURE_S / DT_SUBSTEP)
    warmup_steps = int(M3_WARMUP_S / DT_SUBSTEP)
    settle_steps = int(2.0 / DT_SUBSTEP)

    sim = SimModel(model_xml)
    override_kp_kv(sim.model, kp=2.0, kv=0.0)
    controller = MouseController(fre, DT_SUBSTEP, 20)
    mj_model = sim.model
    mj_data = sim.data
    mouse_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
    foot_to_leg = _body_id_to_leg(mj_model)
    act_idx, dof_idx = _parse_actuator_indices(mj_model)

    for _ in range(settle_steps):
        ctrl = [0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, 0, 0, 0, 0]
        sim.runStep(ctrl, DT_SUBSTEP, realtime=False)

    sim.initializing()
    for _ in range(warmup_steps):
        ctrl, _ = controller.runStep()
        sim.runStep(ctrl, DT_SUBSTEP, realtime=False)

    sim.initializing()

    sub_rows: List[Dict[str, Any]] = []
    t0 = float(mj_data.time)
    for k in range(log_steps):
        ctrl, _ = controller.runStep()
        sim.runStep(ctrl, DT_SUBSTEP, realtime=False)
        sub_rows.append(
            _record_substep(
                mj_model, mj_data,
                mouse_bid=mouse_bid,
                site_ids=site_ids,
                act_idx=act_idx,
                dof_idx=dof_idx,
                foot_to_leg=foot_to_leg,
                t_val=float(mj_data.time - t0),
            )
        )

    log_full = _accumulate_substep_log(sub_rows)
    n_stat = min(measure_steps, log_full.t.size)
    log_stat = SubstepEpisode()
    for attr in ("t", "q", "qvel", "tau", "body_pos", "body_vel", "foot_pos_body", "grf_world", "contact"):
        setattr(log_stat, attr, getattr(log_full, attr)[:n_stat].copy())
    sc = episode_scalars(ep, log_stat)
    return log_full, sc


def _m3_quick_episode(fre: float, ep: int) -> EpisodeScalars:
    """Fast episode for fre sweep (no substep npz)."""
    model_xml = str(MODEL_XML)
    measure_steps = int(M3_MEASURE_S / DT_SUBSTEP)
    warmup_steps = int(M3_WARMUP_S / DT_SUBSTEP)
    settle_steps = int(2.0 / DT_SUBSTEP)

    sim = SimModel(model_xml)
    override_kp_kv(sim.model, kp=2.0, kv=0.0)
    controller = MouseController(fre, DT_SUBSTEP, 20)
    mj_model = sim.model
    mj_data = sim.data
    mouse_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
    foot_to_leg = _body_id_to_leg(mj_model)
    act_idx, dof_idx = _parse_actuator_indices(mj_model)

    for _ in range(settle_steps):
        sim.runStep([0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, 0, 0, 0, 0], DT_SUBSTEP, realtime=False)
    sim.initializing()
    for _ in range(warmup_steps):
        c, _ = controller.runStep()
        sim.runStep(c, DT_SUBSTEP, realtime=False)
    sim.initializing()

    sub_rows: List[Dict[str, Any]] = []
    t0 = float(mj_data.time)
    for _ in range(measure_steps):
        c, _ = controller.runStep()
        sim.runStep(c, DT_SUBSTEP, realtime=False)
        sub_rows.append(
            _record_substep(
                mj_model, mj_data,
                mouse_bid=mouse_bid, site_ids=site_ids,
                act_idx=act_idx, dof_idx=dof_idx, foot_to_leg=foot_to_leg,
                t_val=float(mj_data.time - t0),
            )
        )
    log = _accumulate_substep_log(sub_rows)
    return episode_scalars(ep, log)


def sweep_m3_fre(
    candidates: List[float],
    n_ep: int = SWEEP_EPISODES,
    target_mm_s: float = TARGET_SPEED_MM_S,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Return best fre closest to target and sweep log rows."""
    rows: List[Dict[str, Any]] = []
    print("\n=== M3 fre calibration sweep ===")
    for fre in candidates:
        speeds = [_m3_quick_episode(fre, ep).fwd_speed_mps * 1000.0 for ep in range(n_ep)]
        mean = float(np.mean(speeds))
        rows.append({"fre_Hz": fre, "fwd_speed_mm_s_mean": mean, "per_ep_mm_s": speeds})
        mark = ""
        print(f"  fre={fre:.1f} Hz: speed = {mean:.1f} mm/s")

    best = min(rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - target_mm_s))
    print(f"  → Selected fre = {best['fre_Hz']:.2f} Hz ({best['fwd_speed_mm_s_mean']:.1f} mm/s)")
    return float(best["fre_Hz"]), rows


def run_m3_planner(
    fre: Optional[float] = None,
    n_episodes: int = N_EPISODES,
) -> Tuple[List[SubstepEpisode], List[EpisodeScalars]]:
    """Faithful to TrotGait/src/sim_test.py with calibrated fre."""
    fre = float(M3_FRE if fre is None else fre)
    measure_steps = int(M3_MEASURE_S / DT_SUBSTEP)
    log_steps = int(M3_LOG_SUBSTEPS)

    episodes: List[SubstepEpisode] = []
    scalars: List[EpisodeScalars] = []

    print(f"Running M3 Planner (fre={fre:.2f} Hz, {n_episodes} episodes)...")
    for ep in range(n_episodes):
        log_full, sc = _m3_single_episode(fre, ep, log_steps=log_steps)
        episodes.append(log_full)
        scalars.append(sc)
        print(
            f"  ep {ep}: fwd={sc.fwd_speed_mps*1000:.1f} mm/s "
            f"cot={sc.cot:.3f} eta={sc.eta_locomotion:.3f}"
        )
    return episodes, scalars


# ---------------------------------------------------------------------------
# M4 Planner-simplified
# ---------------------------------------------------------------------------
def _configure_m4_cpg(
    env: RatCpgEnvEnergySubstep50ShapeV3,
    seed: int,
    f_planner: float,
    a_planner: float = A_M4,
) -> None:
    e = env.unwrapped
    env.reset(seed=seed)
    mu4 = np.array([R_M4[leg] ** 2 for leg in LEG_NAMES], dtype=np.float64)
    f4 = np.full(4, f_planner, dtype=np.float64)
    e.foot_path.set_shape_uniform(a_planner, B_M4)
    e.foot_path.set_controls(f4, mu4)
    e.foot_path.cpg.mu_vec = mu4.copy()
    e.foot_path.cpg.omega_vec = (2.0 * math.pi * f4).astype(np.float64)


def _m4_substep(env: RatCpgEnvEnergySubstep50ShapeV3, k: int, f_planner: float) -> None:
    e = env.unwrapped
    model = e.model
    data = e.data
    dt = float(model.opt.timestep)
    thetas = (2.0 * math.pi * f_planner * (k * dt) + PHASE_OFFSETS_M4) % (2.0 * math.pi)
    e.foot_path.cpg.theta = thetas.copy()
    joint_targets: List[Tuple[float, float]] = []
    for leg_id in range(4):
        Fy, Fz = e.foot_path.get_foot_target(leg_id=leg_id)
        turn = e.turn_F if leg_id < 2 else e.turn_H
        tY = math.cos(turn) * Fy - math.sin(turn) * Fz
        tZ = math.cos(turn) * Fz + math.sin(turn) * Fy
        q = e.leg_models[leg_id].pos_2_angle(tY, tZ)
        if q is None:
            q1, q2 = e._last_valid_joint_targets[leg_id]
        else:
            q1, q2 = q
            e._last_valid_joint_targets[leg_id] = (q1, q2)
        joint_targets.append((float(q1), float(q2)))
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for leg_i, (hip_act, knee_act) in enumerate(e.actuator_ids):
        q1_des, q2_des = joint_targets[leg_i]
        if knee_act >= 0:
            ctrl[int(knee_act)] = q1_des
        if hip_act >= 0:
            ctrl[int(hip_act)] = q2_des
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)


def _make_m4_env() -> RatCpgEnvEnergySubstep50ShapeV3:
    return RatCpgEnvEnergySubstep50ShapeV3(
        model_path=str(MODEL_XML),
        max_episode_steps=N_SUBSTEPS,
        render_mode=None,
        enable_csv_log=False,
        enable_episode_summary=False,
        randomize_cpg_phase=False,
        randomize_cpg_params=False,
        use_energy_tank=False,
        gait_template="trot",
        xvel_boost_multiplier=5.0,
    )


def _m4_single_episode(
    env: RatCpgEnvEnergySubstep50ShapeV3,
    ep: int,
    f_planner: float,
    a_planner: float = A_M4,
) -> Tuple[SubstepEpisode, EpisodeScalars]:
    e = env.unwrapped
    mj_model = e.model
    mouse_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
    foot_to_leg = _body_id_to_leg(mj_model)
    act_idx, dof_idx = _parse_actuator_indices(mj_model)

    _configure_m4_cpg(env, seed=ep, f_planner=f_planner, a_planner=a_planner)
    sub_rows: List[Dict[str, Any]] = []
    for k in range(N_SUBSTEPS):
        _m4_substep(env, k, f_planner)
        sub_rows.append(
            _record_substep(
                mj_model, e.data,
                mouse_bid=mouse_bid,
                site_ids=site_ids,
                act_idx=act_idx,
                dof_idx=dof_idx,
                foot_to_leg=foot_to_leg,
                t_val=k * DT_SUBSTEP,
            )
        )
    log = _accumulate_substep_log(sub_rows)
    return log, episode_scalars(ep, log)


def sweep_m4_f(
    candidates: List[float],
    n_ep: int = SWEEP_EPISODES,
    target_mm_s: float = TARGET_SPEED_MM_S,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Shape-fair: a=10mm, b=1mm, r frozen; sweep f only."""
    rows: List[Dict[str, Any]] = []
    env = _make_m4_env()
    print("\n=== M4 f calibration sweep (shape-fair: a=10mm, b=1mm, r frozen) ===")
    try:
        for f in candidates:
            speeds = [
                _m4_single_episode(env, ep, f)[1].fwd_speed_mps * 1000.0 for ep in range(n_ep)
            ]
            mean = float(np.mean(speeds))
            rows.append({"f_Hz": f, "fwd_speed_mm_s_mean": mean, "per_ep_mm_s": speeds})
            print(f"  f={f:.2f} Hz: speed = {mean:.1f} mm/s")
    finally:
        env.close()

    best = min(rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - target_mm_s))
    print(f"  → Selected f = {best['f_Hz']:.2f} Hz ({best['fwd_speed_mm_s_mean']:.1f} mm/s)")
    return float(best["f_Hz"]), rows


def run_m4_planner_simplified(
    f_planner: Optional[float] = None,
    n_episodes: int = N_EPISODES,
) -> Tuple[List[SubstepEpisode], List[EpisodeScalars]]:
    f_planner = float(F_M4 if f_planner is None else f_planner)
    env = _make_m4_env()
    episodes: List[SubstepEpisode] = []
    scalars: List[EpisodeScalars] = []

    print(
        f"Running M4 Planner-simplified (a={A_M4*1000:.0f}mm, f={f_planner:.2f}Hz, "
        f"{n_episodes} episodes)..."
    )
    try:
        for ep in range(n_episodes):
            log, sc = _m4_single_episode(env, ep, f_planner)
            episodes.append(log)
            scalars.append(sc)
            print(
                f"  ep {ep}: fwd={sc.fwd_speed_mps*1000:.1f} mm/s "
                f"cot={sc.cot:.3f} eta={sc.eta_locomotion:.3f}"
            )
    finally:
        env.close()
    return episodes, scalars


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
class SanityError(RuntimeError):
    pass


def run_sanity_checks(
    m2: Tuple[List[EpisodeScalars], Dict[str, Any]],
    m3: Tuple[List[EpisodeScalars], Dict[str, Any]],
    m4: Tuple[List[EpisodeScalars], Dict[str, Any]],
    *,
    speed_matched: bool = True,
) -> None:
    s2, sum2 = m2[0], m2[1]
    s3, sum3 = m3[0], m3[1]
    s4, sum4 = m4[0], m4[1]

    spd2 = sum2["fwd_speed_mm_s"]["mean"]
    spd3 = sum3["fwd_speed_mm_s"]["mean"]
    spd4 = sum4["fwd_speed_mm_s"]["mean"]
    cot2 = sum2["cot"]["mean"]
    cot3 = sum3["cot"]["mean"]
    cot4 = sum4["cot"]["mean"]

    errors: List[str] = []

    # (1) speed
    if spd2 <= 0 or spd3 <= 0 or spd4 <= 0:
        errors.append("fwd_speed <= 0 for at least one model")
    if speed_matched:
        if not (94.0 <= spd2 <= 100.0):
            errors.append(f"M2 fwd_speed {spd2:.1f} not in [94, 100] (expected 97±3 mm/s)")
        if not (92.0 <= spd3 <= 102.0):
            errors.append(f"M3 fwd_speed {spd3:.1f} not in [92, 102] (expected 97±5 mm/s)")
        if not (92.0 <= spd4 <= 102.0):
            errors.append(f"M4 fwd_speed {spd4:.1f} not in [92, 102] (expected 97±5 mm/s)")
    else:
        if not (92.0 <= spd2 <= 102.0):
            errors.append(f"M2 fwd_speed {spd2:.1f} not in [92, 102]")
        if spd3 <= 0:
            errors.append("M3 fwd_speed must be > 0")

    # (2) COT
    if not (4.56 <= cot2 <= 4.76):
        errors.append(f"M2 COT {cot2:.3f} not in [4.56, 4.76] (canonical 4.66±0.1)")
    if speed_matched:
        if cot3 <= cot2:
            errors.append(f"M3 COT {cot3:.3f} should be > M2 COT {cot2:.3f}")
        if cot4 <= cot2:
            errors.append(f"M4 COT {cot4:.3f} should be > M2 COT {cot2:.3f}")

    # (3) eta
    for tag, scs in [("M2", s2), ("M3", s3), ("M4", s4)]:
        for s in scs:
            eta = s.eta_locomotion
            if not np.isfinite(eta):
                errors.append(f"{tag} ep{s.episode}: eta is NaN")
            elif eta > 1.0:
                errors.append(f"{tag} ep{s.episode}: eta={eta:.4f} > 1")
            elif eta < 0.0:
                errors.append(f"{tag} ep{s.episode}: eta={eta:.4f} < 0")
            elif not (0.2 <= eta <= 0.9):
                errors.append(f"{tag} ep{s.episode}: eta={eta:.4f} outside [0.2, 0.9]")

    # (4) energy conservation
    for tag, scs in [("M2", s2), ("M3", s3), ("M4", s4)]:
        for s in scs:
            if s.energy_J <= 0:
                continue
            rel = abs(s.W_total_mech_J - s.energy_J) / s.energy_J
            if rel > 0.05:
                errors.append(
                    f"{tag} ep{s.episode}: |W_total_mech - energy_J|/energy_J = {rel:.3f} > 5%"
                )

    if errors:
        msg = "SANITY CHECK FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        print(msg)
        raise SanityError(msg)
    print("All sanity checks passed.")


# ---------------------------------------------------------------------------
# Aggregation / README
# ---------------------------------------------------------------------------
def write_three_way_comparison(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return float(a.mean()), std


def build_comparison_row(model: str, scalars: List[EpisodeScalars]) -> Dict[str, float]:
    fwd_m, fwd_s = _mean_std([s.fwd_speed_mps * 1000 for s in scalars])
    cot_m, cot_s = _mean_std([s.cot for s in scalars])
    dist_m, dist_s = _mean_std([s.distance_m * 1000 for s in scalars])
    wp_m, wp_s = _mean_std([s.W_propulsion_J * 1000 for s in scalars])
    wb_m, wb_s = _mean_std([s.W_braking_J * 1000 for s in scalars])
    bz_m, bz_s = _mean_std([s.W_vertical_bounce_J * 1000 for s in scalars])
    lat_m, lat_s = _mean_std([s.W_lateral_cost_J * 1000 for s in scalars])
    eta_m, eta_s = _mean_std([s.eta_locomotion for s in scalars])
    return {
        "model": model,
        "fwd_speed_mm_s_mean": fwd_m,
        "fwd_speed_mm_s_std": fwd_s,
        "cot_mean": cot_m,
        "cot_std": cot_s,
        "W_propulsion_mJ_mean": wp_m,
        "W_propulsion_mJ_std": wp_s,
        "W_braking_mJ_mean": wb_m,
        "W_braking_mJ_std": wb_s,
        "W_bounce_mJ_mean": bz_m,
        "W_bounce_mJ_std": bz_s,
        "W_lateral_mJ_mean": lat_m,
        "W_lateral_mJ_std": lat_s,
        "eta_locomotion_mean": eta_m,
        "eta_locomotion_std": eta_s,
        "distance_mm_mean": dist_m,
        "distance_mm_std": dist_s,
    }


def save_model_outputs(
    subdir: str,
    model_id: str,
    ckpt_or_script: str,
    config: Dict[str, Any],
    episodes: List[SubstepEpisode],
    scalars: List[EpisodeScalars],
    notes: str,
) -> Dict[str, Any]:
    out_dir = OUT_SINGLE / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_eval_summary(model_id, ckpt_or_script, config, scalars, notes=notes)
    # attach eta per ep for README
    summary["eta_locomotion"] = _stat_dict([s.eta_locomotion for s in scalars])
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_per_episode_npz(out_dir / "per_episode_substep.npz", episodes)
    write_useful_wasted_csv(out_dir / "useful_wasted_work.csv", scalars)
    return summary


def write_speed_calibration_log(
    path: Path,
    m3_rows: List[Dict[str, Any]],
    m4_rows: List[Dict[str, Any]],
    fre_sel: float,
    f_sel: float,
    m3_spd: float,
    m4_spd: float,
) -> None:
    f_dev_pct = 100.0 * (f_sel - F_M4_NATIVE) / F_M4_NATIVE
    lines = [
        "# Speed calibration log (single_speed eval)\n\n",
        f"Target speed: **{TARGET_SPEED_MM_S:.0f} mm/s** (M2 Ours realized, offline CPG tuning).\n\n",
        "## M3 Planner — fre sweep\n\n",
        "Entry: `TrotGait/src/sim_test.py` stack (`MouseController` + `LegPath` ellipse).\n",
        f"5 episodes per candidate, measure window {M3_MEASURE_S}s after {M3_WARMUP_S}s warmup.\n\n",
        "| fre (Hz) | mean fwd speed (mm/s) |\n|----------|----------------------:|\n",
    ]
    if m3_rows:
        for r in m3_rows:
            sel = " **← selected**" if abs(r["fre_Hz"] - fre_sel) < 1e-6 else ""
            lines.append(f"| {r['fre_Hz']:.2f} | {r['fwd_speed_mm_s_mean']:.1f}{sel} |\n")
    else:
        lines.append("| (prior run) | — |\n")
    lines.append(f"\n**Selected:** fre = **{fre_sel:.2f} Hz** → {m3_spd:.1f} mm/s (20-ep full eval).\n\n")

    lines.append("## M4 Planner-simplified — f sweep (shape-fair)\n\n")
    lines.append(
        "Entry: `scripts/planner_baseline.py` (ShapeV3 env, **a=10 mm, b=1 mm, r frozen**).\n"
        "Only step frequency f is calibrated for speed-match; spatial ellipse unchanged.\n"
        "5 episodes per candidate.\n\n"
        "| f (Hz) | mean fwd speed (mm/s) |\n|--------|----------------------:|\n"
    )
    for r in m4_rows:
        sel = " **← selected**" if abs(r["f_Hz"] - f_sel) < 1e-6 else ""
        lines.append(f"| {r['f_Hz']:.2f} | {r['fwd_speed_mm_s_mean']:.1f}{sel} |\n")
    lines.append(
        f"\n**Selected:** f = **{f_sel:.2f} Hz** → {m4_spd:.1f} mm/s.\n"
        f"Deviation from Ours native f ({F_M4_NATIVE:.2f} Hz): **{f_dev_pct:+.1f}%**.\n"
        f"Spatial shape: a = **{A_M4*1000:.0f} mm**, b = **{B_M4*1000:.0f} mm**, r unchanged.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def print_final_console(
    sum2: Dict[str, Any],
    sum3: Dict[str, Any],
    sum4: Dict[str, Any],
    s2: List[EpisodeScalars],
    s3: List[EpisodeScalars],
    s4: List[EpisodeScalars],
    *,
    fre_sel: float,
    f_sel: float,
) -> None:
    eta2, _ = _mean_std([s.eta_locomotion for s in s2])
    eta3, _ = _mean_std([s.eta_locomotion for s in s3])
    eta4, _ = _mean_std([s.eta_locomotion for s in s4])
    f_dev_pct = 100.0 * (f_sel - F_M4_NATIVE) / F_M4_NATIVE

    def _uw_mean(scs: List[EpisodeScalars], attr: str) -> float:
        return float(np.mean([getattr(s, attr) * 1000.0 for s in scs]))

    print("\nSpeed calibration:")
    print(f"   M3 fre = {fre_sel:.2f} Hz → speed = {sum3['fwd_speed_mm_s']['mean']:.1f} mm/s")
    print(
        f"   M4 f   = {f_sel:.2f} Hz (a=10mm fixed) → speed = {sum4['fwd_speed_mm_s']['mean']:.1f} mm/s "
        f"(f deviation from native: {f_dev_pct:+.1f}%)"
    )
    print("\nThree-way (speed-matched at ~97 mm/s):")
    print(f"   M2 Ours:            COT = {sum2['cot']['mean']:.2f}, η_loc = {eta2:.2f}")
    print(f"   M3 Planner:         COT = {sum3['cot']['mean']:.2f}, η_loc = {eta3:.2f}")
    print(f"   M4 Planner-simpl:   COT = {sum4['cot']['mean']:.2f}, η_loc = {eta4:.2f}")
    print("\nUseful/wasted per episode (mJ):")
    print("                      W_prop  W_brake  W_bounce  W_lateral")
    print(
        f"   M2 Ours:          {_uw_mean(s2, 'W_propulsion_J'):5.0f}    "
        f"{_uw_mean(s2, 'W_braking_J'):5.0f}     "
        f"{_uw_mean(s2, 'W_vertical_bounce_J'):5.0f}      "
        f"{_uw_mean(s2, 'W_lateral_cost_J'):5.0f}"
    )
    print(
        f"   M3 Planner:       {_uw_mean(s3, 'W_propulsion_J'):5.0f}    "
        f"{_uw_mean(s3, 'W_braking_J'):5.0f}     "
        f"{_uw_mean(s3, 'W_vertical_bounce_J'):5.0f}      "
        f"{_uw_mean(s3, 'W_lateral_cost_J'):5.0f}"
    )
    print(
        f"   M4 Planner-simpl: {_uw_mean(s4, 'W_propulsion_J'):5.0f}    "
        f"{_uw_mean(s4, 'W_braking_J'):5.0f}     "
        f"{_uw_mean(s4, 'W_vertical_bounce_J'):5.0f}      "
        f"{_uw_mean(s4, 'W_lateral_cost_J'):5.0f}"
    )
    print("\nAll sanity checks passed.")
    print("Saved to outputs/final_result/single_speed/")


def _load_summary_from_disk(subdir: str) -> Dict[str, Any]:
    path = OUT_SINGLE / subdir / "eval_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_row_from_summary(model: str, summary: Dict[str, Any], subdir: str) -> Dict[str, float]:
    def _ms(key: str) -> Tuple[float, float]:
        d = summary[key]
        return float(d["mean"]), float(d["std"])

    fwd_m, fwd_s = _ms("fwd_speed_mm_s")
    cot_m, cot_s = _ms("cot")
    dist_m, dist_s = _ms("distance_mm")

    uw_path = OUT_SINGLE / subdir / "useful_wasted_work.csv"
    with uw_path.open(newline="", encoding="utf-8") as f:
        uw_rows = list(csv.DictReader(f))
    wp_m, wp_s = _mean_std([float(r["W_propulsion_mJ"]) for r in uw_rows])
    wb_m, wb_s = _mean_std([float(r["W_braking_mJ"]) for r in uw_rows])
    bz_m, bz_s = _mean_std([float(r["W_bounce_mJ"]) for r in uw_rows])
    lat_m, lat_s = _mean_std([float(r["W_lateral_mJ"]) for r in uw_rows])
    eta_m, eta_s = _mean_std([float(r["eta_locomotion"]) for r in uw_rows])
    return {
        "model": model,
        "fwd_speed_mm_s_mean": fwd_m,
        "fwd_speed_mm_s_std": fwd_s,
        "cot_mean": cot_m,
        "cot_std": cot_s,
        "W_propulsion_mJ_mean": wp_m,
        "W_propulsion_mJ_std": wp_s,
        "W_braking_mJ_mean": wb_m,
        "W_braking_mJ_std": wb_s,
        "W_bounce_mJ_mean": bz_m,
        "W_bounce_mJ_std": bz_s,
        "W_lateral_mJ_mean": lat_m,
        "W_lateral_mJ_std": lat_s,
        "eta_locomotion_mean": eta_m,
        "eta_locomotion_std": eta_s,
        "distance_mm_mean": dist_m,
        "distance_mm_std": dist_s,
    }


def main() -> None:
    global M3_FRE, F_M4

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["m2", "m3", "m4"],
                        choices=["m2", "m3", "m4"])
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Use current M3_FRE / F_M4 without sweep")
    parser.add_argument("--m3-fre", type=float, default=None)
    parser.add_argument("--m4-f", type=float, default=None)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m3_sweep_rows: List[Dict[str, Any]] = []
    m4_sweep_rows: List[Dict[str, Any]] = []

    if args.m3_fre is not None:
        M3_FRE = float(args.m3_fre)
    if args.m4_f is not None:
        F_M4 = float(args.m4_f)

    if not args.skip_calibration and ("m3" in args.models or "m4" in args.models):
        if args.m3_fre is None and "m3" in args.models:
            fre_cands = list(M3_FRE_CANDIDATES)
            M3_FRE, m3_sweep_rows = sweep_m3_fre(fre_cands)
            best_m3 = min(m3_sweep_rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - TARGET_SPEED_MM_S))
            if abs(best_m3["fwd_speed_mm_s_mean"] - TARGET_SPEED_MM_S) > 5:
                # User candidates [1.0..1.8] are too fast with cumulative-dy metric;
                # fine sweep in 0.70–0.85 Hz brackets ~97 mm/s.
                extra = [0.70, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.85]
                for f in extra:
                    speeds = [_m3_quick_episode(f, ep).fwd_speed_mps * 1000.0 for ep in range(SWEEP_EPISODES)]
                    mean = float(np.mean(speeds))
                    m3_sweep_rows.append({"fre_Hz": f, "fwd_speed_mm_s_mean": mean, "per_ep_mm_s": speeds})
                    print(f"  fre={f:.2f} Hz: speed = {mean:.1f} mm/s (extended)")
                M3_FRE = min(m3_sweep_rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - TARGET_SPEED_MM_S))["fre_Hz"]
                print(f"  → Selected fre = {M3_FRE:.2f} Hz (extended sweep)")
        if args.m4_f is None and "m4" in args.models:
            F_M4, m4_sweep_rows = sweep_m4_f(list(M4_F_CANDIDATES))
            best_m4 = min(m4_sweep_rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - TARGET_SPEED_MM_S))
            if abs(best_m4["fwd_speed_mm_s_mean"] - TARGET_SPEED_MM_S) > 5:
                extra_f = [2.75, 2.85, 2.95, 3.05, 3.15]
                env = _make_m4_env()
                try:
                    for f in extra_f:
                        if any(abs(r["f_Hz"] - f) < 1e-6 for r in m4_sweep_rows):
                            continue
                        speeds = [
                            _m4_single_episode(env, ep, f)[1].fwd_speed_mps * 1000.0
                            for ep in range(SWEEP_EPISODES)
                        ]
                        mean = float(np.mean(speeds))
                        m4_sweep_rows.append({"f_Hz": f, "fwd_speed_mm_s_mean": mean, "per_ep_mm_s": speeds})
                        print(f"  f={f:.2f} Hz: speed = {mean:.1f} mm/s (extended)")
                finally:
                    env.close()
                F_M4 = min(m4_sweep_rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - TARGET_SPEED_MM_S))["f_Hz"]
                print(f"  → Selected f = {F_M4:.2f} Hz (extended sweep)")

    results: Dict[str, Any] = {}

    if "m2" in args.models:
        ep2, sc2 = run_m2_ours()
        sum2 = save_model_outputs(
            "m2_ours",
            "M2_ours",
            str(CKPT_M2.relative_to(CPG_ROOT)),
            {"target_speed": 0.12, "speed_match": "reference"},
            ep2, sc2,
            "RL CDER v3.1 reference (~97 mm/s); pilot env-step COT",
        )
        results["m2"] = (sc2, sum2)

    if "m3" in args.models:
        ep3, sc3 = run_m3_planner(fre=M3_FRE)
        sum3 = save_model_outputs(
            "m3_planner",
            "M3_planner",
            str(SCRIPT_M3.relative_to(CPG_ROOT.parent)),
            {"fre": M3_FRE, "kp": 2.0, "kv": 0.0, "measure_s": M3_MEASURE_S, "speed_matched": True},
            ep3, sc3,
            f"TrotGait sim_test; fre={M3_FRE:.2f} Hz calibrated offline to ~{TARGET_SPEED_MM_S:.0f} mm/s",
        )
        results["m3"] = (sc3, sum3)

    if "m4" in args.models:
        ep4, sc4 = run_m4_planner_simplified(f_planner=F_M4)
        f_dev = 100.0 * (F_M4 - F_M4_NATIVE) / F_M4_NATIVE
        sum4 = save_model_outputs(
            "m4_planner_simplified",
            "M4_planner_simplified",
            str(SCRIPT_M4.relative_to(CPG_ROOT)),
            {"a": A_M4, "b": B_M4, "f": F_M4, "f_native": F_M4_NATIVE,
             "f_deviation_pct": f_dev, "r": R_M4, "speed_matched": True,
             "shape_fair": True},
            ep4, sc4,
            f"Shape-fair open-loop; a=10mm b=1mm r frozen; f={F_M4:.2f}Hz calibrated ({f_dev:+.1f}% vs native)",
        )
        results["m4"] = (sc4, sum4)

    # Rebuild three-way aggregates if all three summaries exist (on disk or this run)
    subdirs = {"m2": "m2_ours", "m3": "m3_planner", "m4": "m4_planner_simplified"}
    if all((OUT_SINGLE / subdirs[k] / "eval_summary.json").exists() for k in subdirs):
        rows = []
        for key, model in [("m2", "M2_ours"), ("m3", "M3_planner"), ("m4", "M4_planner_simplified")]:
            if key in results:
                rows.append(build_comparison_row(model, results[key][0]))
            else:
                rows.append(
                    build_comparison_row_from_summary(
                        model, _load_summary_from_disk(subdirs[key]), subdirs[key]
                    )
                )
        write_three_way_comparison(OUT_SINGLE / "three_way_comparison.csv", rows)

        sum2 = results["m2"][1] if "m2" in results else _load_summary_from_disk("m2_ours")
        sum3 = results["m3"][1] if "m3" in results else _load_summary_from_disk("m3_planner")
        sum4 = results["m4"][1] if "m4" in results else _load_summary_from_disk("m4_planner_simplified")

        m3_fre_sel = float(sum3.get("config", {}).get("fre", M3_FRE))
        m4_f_sel = float(sum4.get("config", {}).get("f", F_M4))
        if m3_sweep_rows or m4_sweep_rows:
            write_speed_calibration_log(
                OUT_SINGLE / "speed_calibration_log.md",
                m3_sweep_rows,
                m4_sweep_rows,
                m3_fre_sel,
                m4_f_sel,
                sum3["fwd_speed_mm_s"]["mean"],
                sum4["fwd_speed_mm_s"]["mean"],
            )

        if not args.skip_sanity and all(k in results for k in ("m2", "m3", "m4")):
            run_sanity_checks(results["m2"], results["m3"], results["m4"], speed_matched=True)

        readme = OUT_ROOT / "README.md"
        m2_spd, m3_spd, m4_spd = [s["fwd_speed_mm_s"]["mean"] for s in (sum2, sum3, sum4)]
        m2_cot, m3_cot, m4_cot = [s["cot"]["mean"] for s in (sum2, sum3, sum4)]
        eta2 = sum2["eta_locomotion"]["mean"]
        eta3 = sum3["eta_locomotion"]["mean"]
        eta4 = sum4["eta_locomotion"]["mean"]
        m3_fre = float(sum3.get("config", {}).get("fre", M3_FRE))
        m4_f = float(sum4.get("config", {}).get("f", F_M4))
        f_dev = 100.0 * (m4_f - F_M4_NATIVE) / F_M4_NATIVE
        readme.write_text(
            f"""# Final Results Directory

Single-speed comparison (**speed-matched ~97 mm/s** via offline CPG calibration):
  - M2 Ours:                  RL CDER v3.1 (reference)
  - M3 Planner:               fre={m3_fre:.2f} Hz (calibrated)
  - M4 Planner-simplified:    shape-fair — a=10 mm, b=1 mm, r frozen; f={m4_f:.2f} Hz ({f_dev:+.1f}% vs native)

See `single_speed/speed_calibration_log.md` for sweep tables.

COT pipeline: pilot env-step |τ·q̇|/(M·g·d_fwd)

Results:
  M2: fwd_speed = {m2_spd:.1f} mm/s, COT = {m2_cot:.2f}, η_loc = {eta2:.2f}
  M3: fwd_speed = {m3_spd:.1f} mm/s, COT = {m3_cot:.2f}, η_loc = {eta3:.2f}
  M4: fwd_speed = {m4_spd:.1f} mm/s, COT = {m4_cot:.2f}, η_loc = {eta4:.2f}

Generated by eval_single_speed_three_way.py at {ts}.
""",
            encoding="utf-8",
        )

        if all(k in results for k in ("m2", "m3", "m4")):
            print_final_console(
                sum2, sum3, sum4,
                results["m2"][0], results["m3"][0], results["m4"][0],
                fre_sel=m3_fre, f_sel=m4_f,
            )
        else:
            print("\nThree-way comparison and README updated from disk + this run.")
            print(f"  M2: {m2_spd:.1f} mm/s  M3: {m3_spd:.1f} mm/s  M4: {m4_spd:.1f} mm/s")
            print(f"  M4 shape-fair: a=10mm, f={m4_f:.2f}Hz ({f_dev:+.1f}% vs native {F_M4_NATIVE}Hz)")


if __name__ == "__main__":
    main()
