#!/usr/bin/env python3
"""
Thesis-experiments master measurement campaign.

Two policies are rolled out for 20 deterministic episodes (82 env-steps × 50 substeps = 4100 substeps each):
  baseline (W2): logs/w2_baseline/seed1/checkpoints/rat_cpg_ppo_route_a_1500000_steps.zip
  cder   (V3.1): logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip

For each policy, per-substep kinematics + per-substep energy decomposition
(W+, W-, E_damp, E_fric, E_norm, with per-leg attribution) are recorded, then
the 9 tasks M1..M9 are computed and saved under outputs/thesis_experiments/.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for _p in (ROOT, ROOT / "env", ROOT / "experiments", ROOT / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from rat_cpg_env_energy_substep50 import FootPathFixed  # noqa: E402
from w2_energy_shaped_env import RatCpgEnvEnergySubstep50ShapeV2  # noqa: E402
from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3  # noqa: E402

XML_REL = "../TrotGait/models/dynamic_4l_kp2.xml"
MODEL_PATH = str((ROOT / XML_REL).resolve())
OUT_DIR = ROOT / "outputs" / "thesis_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_BASELINE = ROOT / "logs/w2_baseline/seed1/checkpoints/rat_cpg_ppo_route_a_1500000_steps.zip"
CKPT_CDER = ROOT / "logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip"

TB_LOGDIR_CDER = ROOT / "logs/w3_ours_v3.1/seed0/tb_logs/PPO_1"

LEG_NAMES = ["FL", "FR", "RL", "RR"]
FOOT_BODIES = ["foot_fl", "foot_fr", "foot_rl", "foot_rr"]
FOOT_SITES = ["ankle_fl", "ankle_fr", "ankle_rl", "ankle_rr"]
HIP_SITES = ["thigh_link_fl", "thigh_link_fr", "thigh_link_rl", "thigh_link_rr"]
G_ACC = 9.81  # m/s^2
M_BODY = 0.2895  # kg (from previous mass partition)
B_DAMP = 0.005  # N·m·s/rad

# Contact threshold for binarising fnormal -> contact/no-contact.
# Robot weight ~ M_BODY * G_ACC ≈ 2.84 N; single-leg support ≈ 0.7 N.
# 0.1 N ≈ 3.5% of body weight; strict enough to remove numerical noise and bouncing.
FN_CONTACT_THRESHOLD = 0.1

# Morphological gap-filling: ignore "air" intervals shorter than this many
# substeps (each substep = 1 ms). Bridges sub-20 ms foot-bounce chatter into
# a continuous stance so gait diagrams stay readable without falsifying physics.
MIN_AIR_MS = 20
# Similarly drop ultra-short "stance" runs that are clearly noise (< 4 ms).
MIN_STANCE_MS = 4


def compute_hip_offsets_body_frame() -> Dict[int, np.ndarray]:
    """Per-leg (x, y, z) of `thigh_link_*` site in mouse-body local frame at qpos0.

    Used to translate the leg-IK reference (which has the hip at its origin) into the
    mouse-body frame so that `Fy_ref / Fz_ref` overlay correctly onto the ankle position
    expressed in the mouse body frame (which is what `foot_y_body / foot_z_body` use).
    """
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    mbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    p_mouse = d.xpos[mbid].copy()
    R_T = d.xmat[mbid].reshape(3, 3).T
    out: Dict[int, np.ndarray] = {}
    for leg, name in enumerate(HIP_SITES):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            out[leg] = np.zeros(3, dtype=np.float64)
            continue
        p_world = d.site_xpos[sid].copy()
        out[leg] = R_T @ (p_world - p_mouse)
    return out


def _binarize_contact(fnormal: np.ndarray, threshold: float = FN_CONTACT_THRESHOLD,
                      min_air_ms: int = MIN_AIR_MS,
                      min_stance_ms: int = MIN_STANCE_MS) -> np.ndarray:
    """Threshold + morphological cleanup. fnormal shape (T, 4), returns int8 (T, 4)."""
    out = (fnormal > threshold).astype(np.int8)
    for leg in range(out.shape[1]):
        c = out[:, leg].copy()
        if c.size == 0:
            continue
        # Fill short "air" gaps (0-runs shorter than min_air_ms surrounded by 1).
        i = 0
        while i < c.size:
            if c[i] == 0:
                j = i
                while j < c.size and c[j] == 0:
                    j += 1
                run_len = j - i
                if 0 < i and j < c.size and run_len < min_air_ms:
                    c[i:j] = 1
                i = j
            else:
                i += 1
        # Drop short "stance" runs (1-runs shorter than min_stance_ms).
        i = 0
        while i < c.size:
            if c[i] == 1:
                j = i
                while j < c.size and c[j] == 1:
                    j += 1
                if j - i < min_stance_ms:
                    c[i:j] = 0
                i = j
            else:
                i += 1
        out[:, leg] = c
    return out

# How many env-steps to roll out per episode (1 env-step = 50 substeps = 0.05 s).
EVN_STEPS_PER_EP = 82
N_EPISODES = 20

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    # per substep
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (S,)
    foot_z_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    foot_y_body: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    foot_z_body: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    theta: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    fnormal: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    q8: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    qvel8: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    tau8: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    qref8: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    base_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    base_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # per substep energy: shape (S, 4) per leg
    Wp_leg_sub: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    Wn_leg_sub: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    Edamp_leg_sub: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    Efric_leg_sub: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    Enorm_leg_sub: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    # per env-step aggregates
    t_env: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (E,)
    mu_env: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    f_env: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    a_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # per env-step energies (totals = sum over legs)
    Wp_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Wn_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Edamp_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Efric_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Enorm_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # per env-step KE and PE of base (for closure)
    KE_env: np.ndarray = field(default_factory=lambda: np.zeros(0))
    PE_env: np.ndarray = field(default_factory=lambda: np.zeros(0))

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def make_env(policy_kind: str):
    if policy_kind == "cder":
        env = RatCpgEnvEnergySubstep50ShapeV3(
            model_path=MODEL_PATH,
            max_episode_steps=4100,
            render_mode=None,
            enable_csv_log=False,
            xvel_boost_multiplier=5.0,
        )
    else:
        env = RatCpgEnvEnergySubstep50ShapeV2(
            model_path=MODEL_PATH,
            max_episode_steps=4100,
            render_mode=None,
            enable_csv_log=False,
        )
    return env


def _body_id_to_leg(model: mujoco.MjModel) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for leg, name in enumerate(FOOT_BODIES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            out[bid] = leg
    return out


def collect_rollout(policy_kind: str, ckpt: Path, n_eps: int = N_EPISODES) -> List[EpisodeLog]:
    env = make_env(policy_kind)
    ppo = PPO.load(str(ckpt), env=None)
    e = env.unwrapped

    model = e.model
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
    mouse_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    foot_to_leg = _body_id_to_leg(model)
    dt_sim = float(model.opt.timestep)

    # Cache leg actuator/dof index pairs (hip, knee) per leg = 2 entries per leg.
    act_idx = np.asarray(e._actuator_idx_leg, dtype=np.int64)  # (8,)
    dof_idx = np.asarray(e._dof_idx_leg, dtype=np.int64)  # (8,)

    logs: List[EpisodeLog] = []
    for ep in range(n_eps):
        log = EpisodeLog()
        sub_t: List[float] = []
        sub_fzw: List[np.ndarray] = []
        sub_fyb: List[np.ndarray] = []
        sub_fzb: List[np.ndarray] = []
        sub_theta: List[np.ndarray] = []
        sub_fn: List[np.ndarray] = []
        sub_q: List[np.ndarray] = []
        sub_qv: List[np.ndarray] = []
        sub_tau: List[np.ndarray] = []
        sub_qref: List[np.ndarray] = []
        sub_base_xyz: List[np.ndarray] = []
        sub_base_vel: List[np.ndarray] = []
        sub_Wp_leg: List[np.ndarray] = []
        sub_Wn_leg: List[np.ndarray] = []
        sub_Ed_leg: List[np.ndarray] = []
        sub_Ef_leg: List[np.ndarray] = []
        sub_En_leg: List[np.ndarray] = []

        env_t: List[float] = []
        env_mu: List[np.ndarray] = []
        env_f: List[np.ndarray] = []
        env_a: List[float] = []
        env_b: List[float] = []
        env_Wp: List[float] = []
        env_Wn: List[float] = []
        env_Ed: List[float] = []
        env_Ef: List[float] = []
        env_En: List[float] = []
        env_KE: List[float] = []
        env_PE: List[float] = []

        # Reset per-env-step contact accumulators tracker: we read after each env-step
        prev_Efric = 0.0
        prev_Enorm = 0.0

        orig_mj_step = mujoco.mj_step

        def mj_wrap(m: mujoco.MjModel, d: mujoco.MjData) -> None:
            # PRE: ctrl was set by env. Capture qref (= ctrl) BEFORE mj_step so it's the commanded value.
            qref_now = np.array([float(d.ctrl[i]) if i < d.ctrl.size else 0.0 for i in act_idx], dtype=np.float64)
            orig_mj_step(m, d)
            sub_t.append(float(d.time))
            # foot world z
            fzw = np.array([float(d.site_xpos[sid][2]) for sid in site_ids], dtype=np.float64)
            sub_fzw.append(fzw)
            # foot in mouse body frame (y,z)
            p_mouse = d.xpos[mouse_bid].copy()
            R = d.xmat[mouse_bid].reshape(3, 3)
            fyb = np.zeros(4, dtype=np.float64)
            fzb = np.zeros(4, dtype=np.float64)
            for k, sid in enumerate(site_ids):
                p_foot = d.site_xpos[sid].copy()
                local = R.T @ (p_foot - p_mouse)
                fyb[k] = float(local[1])
                fzb[k] = float(local[2])
            sub_fyb.append(fyb)
            sub_fzb.append(fzb)
            # CPG phase per leg
            sub_theta.append(np.array([float(e.foot_path.get_leg_phase(i)) for i in range(4)], dtype=np.float64))
            # joint state
            q = np.array([float(d.qpos[int(model.jnt_qposadr[int(model.actuator_trnid[ai, 0])])]) for ai in act_idx], dtype=np.float64)
            qv = np.array([float(d.qvel[int(di)]) for di in dof_idx], dtype=np.float64)
            tau = np.array([float(d.actuator_force[int(ai)]) for ai in act_idx], dtype=np.float64)
            sub_q.append(q)
            sub_qv.append(qv)
            sub_tau.append(tau)
            sub_qref.append(qref_now)
            # base (mouse) world position and velocity (use mouse body com)
            sub_base_xyz.append(d.xpos[mouse_bid].copy())
            v6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, mouse_bid, v6, 0)
            sub_base_vel.append(v6[3:6].copy())
            # contact forces aggregated by foot
            fn_leg = np.zeros(4, dtype=np.float64)
            efric_leg = np.zeros(4, dtype=np.float64)
            enorm_leg = np.zeros(4, dtype=np.float64)
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
                f_n = float(cforce[0])
                f_t = float(np.linalg.norm(cforce[1:3]))
                fn_leg[leg] += abs(f_n)
                # friction energy via tangential velocity
                v6f = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, foot_bid, v6f, 0)
                v_foot_world = v6f[3:6]
                frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                n_hat = frame[0, :]
                v_tan = v_foot_world - np.dot(v_foot_world, n_hat) * n_hat
                v_t = float(np.linalg.norm(v_tan))
                efric_leg[leg] += f_t * v_t * dt_sim
                v_n = float(np.dot(v_foot_world, n_hat))
                p_norm = -f_n * v_n
                if p_norm > 0.0:
                    enorm_leg[leg] += p_norm * dt_sim
            sub_fn.append(fn_leg)
            sub_Ef_leg.append(efric_leg)
            sub_En_leg.append(enorm_leg)
            # joint per-leg energy (paired hip+knee)
            p_inst = tau * qv  # (8,)
            Wp_inst = np.maximum(p_inst, 0.0) * dt_sim
            Wn_inst = np.maximum(-p_inst, 0.0) * dt_sim
            Ed_inst = B_DAMP * (qv ** 2) * dt_sim
            Wp_leg = np.array([Wp_inst[2 * i] + Wp_inst[2 * i + 1] for i in range(4)], dtype=np.float64)
            Wn_leg = np.array([Wn_inst[2 * i] + Wn_inst[2 * i + 1] for i in range(4)], dtype=np.float64)
            Ed_leg = np.array([Ed_inst[2 * i] + Ed_inst[2 * i + 1] for i in range(4)], dtype=np.float64)
            sub_Wp_leg.append(Wp_leg)
            sub_Wn_leg.append(Wn_leg)
            sub_Ed_leg.append(Ed_leg)

        mujoco.mj_step = mj_wrap  # type: ignore[assignment]
        try:
            obs, _ = env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < EVN_STEPS_PER_EP:
                # Snapshot positions BEFORE step for ΔKE/ΔPE if needed (we use first/last)
                act, _ = ppo.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(act)
                done = bool(term or trunc)
                steps += 1
                env_t.append(float(e.data.time))
                env_mu.append(np.array(e._cur_mu4, dtype=np.float64).copy())
                env_f.append(np.array(e._cur_f4, dtype=np.float64).copy())
                env_a.append(float(getattr(e, "_cur_a", float("nan"))))
                env_b.append(float(getattr(e, "_cur_b", float("nan"))))
                # Aggregate per-env-step from accumulated per-substep arrays for this env-step:
                # We accumulate the last K substeps (= n_substeps) into the env-step.
                K = int(e.n_substeps)
                Wp_arr = np.stack(sub_Wp_leg[-K:], axis=0).sum(axis=0)  # (4,)
                Wn_arr = np.stack(sub_Wn_leg[-K:], axis=0).sum(axis=0)
                Ed_arr = np.stack(sub_Ed_leg[-K:], axis=0).sum(axis=0)
                Ef_arr = np.stack(sub_Ef_leg[-K:], axis=0).sum(axis=0)
                En_arr = np.stack(sub_En_leg[-K:], axis=0).sum(axis=0)
                env_Wp.append(float(np.sum(Wp_arr)))
                env_Wn.append(float(np.sum(Wn_arr)))
                env_Ed.append(float(np.sum(Ed_arr)))
                env_Ef.append(float(np.sum(Ef_arr)))
                env_En.append(float(np.sum(En_arr)))
                # KE, PE of base (mouse) at end of env-step
                vlast = sub_base_vel[-1]
                xyz_last = sub_base_xyz[-1]
                env_KE.append(0.5 * M_BODY * float(np.dot(vlast, vlast)))
                env_PE.append(M_BODY * G_ACC * float(xyz_last[2]))
        finally:
            mujoco.mj_step = orig_mj_step  # type: ignore[assignment]

        # Stack
        log.t = np.array(sub_t, dtype=np.float64)
        log.foot_z_world = np.stack(sub_fzw, axis=0) if sub_fzw else np.zeros((0, 4))
        log.foot_y_body = np.stack(sub_fyb, axis=0) if sub_fyb else np.zeros((0, 4))
        log.foot_z_body = np.stack(sub_fzb, axis=0) if sub_fzb else np.zeros((0, 4))
        log.theta = np.stack(sub_theta, axis=0) if sub_theta else np.zeros((0, 4))
        log.fnormal = np.stack(sub_fn, axis=0) if sub_fn else np.zeros((0, 4))
        log.q8 = np.stack(sub_q, axis=0) if sub_q else np.zeros((0, 8))
        log.qvel8 = np.stack(sub_qv, axis=0) if sub_qv else np.zeros((0, 8))
        log.tau8 = np.stack(sub_tau, axis=0) if sub_tau else np.zeros((0, 8))
        log.qref8 = np.stack(sub_qref, axis=0) if sub_qref else np.zeros((0, 8))
        log.base_xyz = np.stack(sub_base_xyz, axis=0) if sub_base_xyz else np.zeros((0, 3))
        log.base_vel = np.stack(sub_base_vel, axis=0) if sub_base_vel else np.zeros((0, 3))
        log.Wp_leg_sub = np.stack(sub_Wp_leg, axis=0) if sub_Wp_leg else np.zeros((0, 4))
        log.Wn_leg_sub = np.stack(sub_Wn_leg, axis=0) if sub_Wn_leg else np.zeros((0, 4))
        log.Edamp_leg_sub = np.stack(sub_Ed_leg, axis=0) if sub_Ed_leg else np.zeros((0, 4))
        log.Efric_leg_sub = np.stack(sub_Ef_leg, axis=0) if sub_Ef_leg else np.zeros((0, 4))
        log.Enorm_leg_sub = np.stack(sub_En_leg, axis=0) if sub_En_leg else np.zeros((0, 4))
        log.t_env = np.array(env_t, dtype=np.float64)
        log.mu_env = np.stack(env_mu, axis=0) if env_mu else np.zeros((0, 4))
        log.f_env = np.stack(env_f, axis=0) if env_f else np.zeros((0, 4))
        log.a_env = np.array(env_a, dtype=np.float64)
        log.b_env = np.array(env_b, dtype=np.float64)
        log.Wp_env = np.array(env_Wp, dtype=np.float64)
        log.Wn_env = np.array(env_Wn, dtype=np.float64)
        log.Edamp_env = np.array(env_Ed, dtype=np.float64)
        log.Efric_env = np.array(env_Ef, dtype=np.float64)
        log.Enorm_env = np.array(env_En, dtype=np.float64)
        log.KE_env = np.array(env_KE, dtype=np.float64)
        log.PE_env = np.array(env_PE, dtype=np.float64)
        logs.append(log)
        print(f"[{policy_kind}] ep {ep}: {log.t.size} substeps, {log.t_env.size} env-steps")
    env.close()
    return logs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"n": 0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def detect_swing_blocks(theta: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return list of (stance_start, stance_end, swing_start, swing_end) for each swing cycle
    that has a non-empty preceding stance."""
    in_sw = theta < math.pi
    blocks: List[Tuple[int, int, int, int]] = []
    if in_sw.size == 0:
        return blocks
    d = np.diff(in_sw.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if in_sw[0]:
        starts.insert(0, 0)
    if in_sw[-1]:
        ends.append(in_sw.size)
    prev_swing_end = 0
    for s, e2 in zip(starts, ends):
        # preceding stance: from prev_swing_end (or 0) to s
        stance_start = prev_swing_end
        stance_end = s
        if stance_end <= stance_start:
            prev_swing_end = e2
            continue
        blocks.append((stance_start, stance_end, s, e2))
        prev_swing_end = e2
    return blocks


def detect_touchdowns(theta: np.ndarray) -> np.ndarray:
    """Touchdown = transition swing -> stance (θ crossing π upward).
    Returns indices of touchdown substeps."""
    if theta.size == 0:
        return np.zeros(0, dtype=np.int64)
    in_sw = theta < math.pi
    d = np.diff(in_sw.astype(np.int8))
    return np.where(d == -1)[0] + 1  # transitions to stance


# ---------------------------------------------------------------------------
# Task M1: CDER closure equation residual
# ---------------------------------------------------------------------------

def task_m1(logs_cder: List[EpisodeLog]) -> Dict[str, Any]:
    residuals: List[float] = []
    Wp_totals: List[float] = []
    Wn_totals: List[float] = []
    Ed_totals: List[float] = []
    Ef_totals: List[float] = []
    En_totals: List[float] = []
    dKE: List[float] = []
    dPE: List[float] = []
    for L in logs_cder:
        if L.Wp_env.size == 0:
            continue
        Wp = float(np.sum(L.Wp_env))
        Wn = float(np.sum(L.Wn_env))
        Ed = float(np.sum(L.Edamp_env))
        Ef = float(np.sum(L.Efric_env))
        En = float(np.sum(L.Enorm_env))
        # ΔKE, ΔPE between first and last substep snapshots; first KE/PE not stored explicitly,
        # but we can use first base velocity / z from substep arrays.
        v0 = L.base_vel[0]
        v1 = L.base_vel[-1]
        z0 = float(L.base_xyz[0, 2])
        z1 = float(L.base_xyz[-1, 2])
        KE0 = 0.5 * M_BODY * float(np.dot(v0, v0))
        KE1 = 0.5 * M_BODY * float(np.dot(v1, v1))
        PE0 = M_BODY * G_ACC * z0
        PE1 = M_BODY * G_ACC * z1
        dK = KE1 - KE0
        dP = PE1 - PE0
        res = Wp - (Wn + Ed + Ef + En + dK + dP)
        residuals.append(res)
        Wp_totals.append(Wp)
        Wn_totals.append(Wn)
        Ed_totals.append(Ed)
        Ef_totals.append(Ef)
        En_totals.append(En)
        dKE.append(dK)
        dPE.append(dP)
    residuals_arr = np.array(residuals, dtype=np.float64)
    Wp_arr = np.array(Wp_totals, dtype=np.float64)
    pct = 100.0 * residuals_arr / np.maximum(Wp_arr, 1e-12)
    return {
        "per_episode": [
            {
                "ep": i,
                "Wp_total_J": Wp_totals[i],
                "Wn_total_J": Wn_totals[i],
                "Edamp_total_J": Ed_totals[i],
                "Efric_total_J": Ef_totals[i],
                "Enorm_total_J": En_totals[i],
                "dKE_J": dKE[i],
                "dPE_J": dPE[i],
                "residual_J": residuals[i],
                "residual_pct_of_Wp": float(pct[i]),
            }
            for i in range(len(residuals))
        ],
        "summary": {
            "n_episodes": int(residuals_arr.size),
            "mean_residual_J": float(np.mean(residuals_arr)),
            "std_residual_J": float(np.std(residuals_arr, ddof=1)) if residuals_arr.size > 1 else 0.0,
            "mean_residual_pct_of_Wp": float(np.mean(pct)),
            "std_residual_pct_of_Wp": float(np.std(pct, ddof=1)) if pct.size > 1 else 0.0,
            "median_residual_pct_of_Wp": float(np.median(pct)),
            "min_residual_pct_of_Wp": float(np.min(pct)),
            "max_residual_pct_of_Wp": float(np.max(pct)),
        },
    }


# ---------------------------------------------------------------------------
# Task M2: CDER per-leg metrics
# ---------------------------------------------------------------------------

def task_m2(logs_cder: List[EpisodeLog]) -> Dict[str, Any]:
    # r = sqrt(mu) per env-step (pool across episodes)
    R_per_leg: List[List[float]] = [[] for _ in range(4)]
    for L in logs_cder:
        for leg in range(4):
            R_per_leg[leg].extend(np.sqrt(np.clip(L.mu_env[:, leg], 0.0, None)).tolist())

    # Foot lift: per swing cycle, peak_swing_world_z - mean_stance_world_z
    lifts_per_leg: List[List[float]] = [[] for _ in range(4)]
    # theta_td: CPG phase at touchdown = transition swing→stance (phase wraps so it's near π).
    td_per_leg: List[List[float]] = [[] for _ in range(4)]
    # Foot trajectory LS fit on swing samples in mouse-body y/z; (y0,z0) fixed from FootPathFixed defaults
    Y0Z0 = {0: (-0.0, -0.045), 1: (-0.0, -0.045), 2: (-0.005, -0.05), 3: (-0.005, -0.05)}
    fit_per_leg: Dict[int, Dict[str, Any]] = {}

    # For LS fit, use one representative episode (episode 0).
    L0 = logs_cder[0]
    for leg in range(4):
        th = L0.theta[:, leg]
        swing = th < math.pi
        c = np.cos(th[swing])
        s = np.sin(th[swing])
        y0, z0 = Y0Z0[leg]
        FY = L0.foot_y_body[:, leg]
        FZ = L0.foot_z_body[:, leg]
        a_eff = float(np.dot(c, FY[swing] - y0) / max(np.dot(c, c), 1e-12))
        b_eff = float(np.dot(s, FZ[swing] - z0) / max(np.dot(s, s), 1e-12))
        py = y0 + a_eff * c
        pz = z0 + b_eff * s
        fit_per_leg[leg] = {
            "y0": y0,
            "z0": z0,
            "a_eff": a_eff,
            "b_eff": b_eff,
            "rmse_y_m": float(np.sqrt(np.mean((FY[swing] - py) ** 2))),
            "rmse_z_m": float(np.sqrt(np.mean((FZ[swing] - pz) ** 2))),
            "n_swing_samples": int(swing.sum()),
        }

    # Aggregate lifts and theta_td across all 20 episodes
    for L in logs_cder:
        for leg in range(4):
            blocks = detect_swing_blocks(L.theta[:, leg])
            for stance_s, stance_e, swing_s, swing_e in blocks:
                peak = float(np.max(L.foot_z_world[swing_s:swing_e, leg]))
                stance_mean = float(np.mean(L.foot_z_world[stance_s:stance_e, leg]))
                lifts_per_leg[leg].append(peak - stance_mean)
            # Touchdowns
            td_idx = detect_touchdowns(L.theta[:, leg])
            # Get the CPG phase at the substep AFTER swing→stance transition. By definition it's >= π.
            for k in td_idx:
                td_per_leg[leg].append(float(L.theta[k, leg]))

    out: Dict[str, Any] = {"per_leg": {}, "fit": {}}
    for leg in range(4):
        name = LEG_NAMES[leg]
        out["per_leg"][name] = {
            "r": _stats1d(np.array(R_per_leg[leg])),
            "lift_m": _stats1d(np.array(lifts_per_leg[leg])),
            "theta_td_rad": {
                **_stats1d(np.array(td_per_leg[leg])),
                "deviation_from_pi_rad_mean": float(np.mean(np.abs(np.array(td_per_leg[leg]) - math.pi))) if td_per_leg[leg] else float("nan"),
            },
        }
        out["fit"][name] = fit_per_leg[leg]
    return out


# ---------------------------------------------------------------------------
# Task M3: Baseline gait diagram
# ---------------------------------------------------------------------------

def task_m3_baseline_gait(logs_base: List[EpisodeLog], out_dir: Path) -> Dict[str, Any]:
    # Pick a 2-second mid-episode window from episode 0
    L = logs_base[0]
    t0 = 2.0
    t1 = 4.0
    if L.t[-1] < t1:
        t0, t1 = max(0.0, L.t[-1] - 2.0), L.t[-1]
    mask = (L.t >= t0) & (L.t <= t1)
    Tw = L.t[mask] - t0
    contact = _binarize_contact(L.fnormal[mask])
    fn = L.fnormal[mask]

    csv_path = out_dir / "M3_baseline_contact_pattern.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "leg", "fnormal", "contact"])
        for k in range(Tw.size):
            for leg in range(4):
                w.writerow([float(Tw[k] + t0), LEG_NAMES[leg], float(fn[k, leg]), int(contact[k, leg])])

    # Plot gait diagram: 4 rows, shaded contact bars
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for leg in range(4):
        row_y = 3 - leg  # FL on top
        in_contact = contact[:, leg]
        # find runs
        d = np.diff(np.concatenate([[0], in_contact, [0]]).astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            ax.barh(row_y, Tw[e - 1] - Tw[s] if e > s else 0.0, left=Tw[s], height=0.7, color="#3060B0", edgecolor="none")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(LEG_NAMES)
    ax.set_xlabel("Time (s) within window")
    ax.set_xlim(0.0, Tw[-1] if Tw.size > 0 else 1.0)
    ax.set_title(
        f"Baseline gait diagram (ep 0, window [{t0:.2f}, {t1:.2f}] s; "
        f"thr={FN_CONTACT_THRESHOLD:g} N, fill air<{MIN_AIR_MS}ms, drop stance<{MIN_STANCE_MS}ms)"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / "M3_baseline_gait_diagram.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "window_s": [t0, t1],
        "duty_per_leg": {LEG_NAMES[leg]: float(np.mean(contact[:, leg])) for leg in range(4)},
        "n_contact_events_per_leg": {
            LEG_NAMES[leg]: int(np.sum(np.diff(contact[:, leg].astype(np.int8)) == 1)) for leg in range(4)
        },
    }


# ---------------------------------------------------------------------------
# Task M4: Baseline detailed diagnostic snapshot (6-panel)
# ---------------------------------------------------------------------------

def task_m4_baseline_diag(
    logs: List[EpisodeLog],
    out_dir: Path,
    policy_label: str = "baseline",
    file_tag: str = "baseline",
    hip_offsets: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Any]:
    L = logs[0]
    if hip_offsets is None:
        hip_offsets = compute_hip_offsets_body_frame()
    # Window: 0.7 s mid-episode
    t0 = 3.75
    t1 = 4.45
    if L.t[-1] < t1:
        t0, t1 = max(0.0, L.t[-1] - 0.7), L.t[-1]
    mask = (L.t >= t0) & (L.t <= t1)
    Tw = L.t[mask] - t0
    # Reference foot Fy, Fz are nominal CPG targets via foot_path; we approximate using mu_env and a_env/b_env.
    # Since baseline used ShapeV2 (smoothing_alpha default = 1; so applied = decoded).
    # For baseline, a/b are also (a, b) shared. We have L.a_env, L.b_env per env-step. Re-index to substeps.
    K = 50
    # env step index per substep:
    env_idx = np.minimum(np.arange(Tw.size, dtype=np.int64) // K + int(t0 / (K * 0.001)), L.a_env.size - 1)
    # safer: use times to find nearest env-step
    env_idx = np.searchsorted(L.t_env, L.t[mask], side="right") - 1
    env_idx = np.clip(env_idx, 0, L.t_env.size - 1)
    a_sub = L.a_env[env_idx]
    b_sub = L.b_env[env_idx]
    mu_sub = L.mu_env[env_idx]
    theta_w = L.theta[mask]
    # Reference Fy/Fz per substep per leg.
    # The CPG formula `y0 + a*r*cos(phi)` / `z0 + b*r*sin(phi)` is expressed in the
    # leg-IK frame (origin at the leg's hip). The actual foot position arrays
    # `foot_y_body / foot_z_body` are expressed in the mouse-body frame.
    # We therefore add the constant hip-to-body offset (computed at qpos0) so the
    # ref and actual curves overlay correctly for ALL legs (especially RL/RR
    # whose hips sit ~120 mm behind the mouse-body origin).
    Y0Z0 = {0: (-0.0, -0.045), 1: (-0.0, -0.045), 2: (-0.005, -0.05), 3: (-0.005, -0.05)}
    Fy_ref = np.zeros((Tw.size, 4), dtype=np.float64)
    Fz_ref = np.zeros((Tw.size, 4), dtype=np.float64)
    for leg in range(4):
        y0, z0 = Y0Z0[leg]
        r = np.sqrt(np.clip(mu_sub[:, leg], 0.0, None))
        hip_dy = float(hip_offsets[leg][1])
        hip_dz = float(hip_offsets[leg][2])
        Fy_ref[:, leg] = hip_dy + y0 + a_sub * r * np.cos(theta_w[:, leg])
        # Use simple sin (no asym shaping) to match what FootPathFixed default does (z_asym_alpha=0)
        Fz_ref[:, leg] = hip_dz + z0 + b_sub * r * np.sin(theta_w[:, leg])

    # Joint torque limit
    tau_max = 0.157

    # Energy: per env-step within window
    em = (L.t_env >= t0) & (L.t_env <= t1)
    Tew = L.t_env[em] - t0
    Wp_w = L.Wp_env[em]
    Wn_w = L.Wn_env[em]
    Ed_w = L.Edamp_env[em]
    Ef_w = L.Efric_env[em]
    En_w = L.Enorm_env[em]

    # Mean forward velocity ≈ -base_vy in this codebase (forward = -Y)
    base_v = L.base_vel[mask]
    base_xyz = L.base_xyz[mask]
    fwd_speed = float(-np.mean(base_v[:, 1]))

    # CSV with underlying data
    csv_path = out_dir / f"M4_{file_tag}_diagnostic.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["t", "leg", "Fy_body", "Fz_body", "Fy_ref", "Fz_ref",
             "q_hip", "q_knee", "qref_hip", "qref_knee",
             "tau_hip", "tau_knee", "fnormal",
             "base_x", "base_y", "base_z", "base_vx", "base_vy", "base_vz",
             "Wp_total", "Wn_total", "Edamp_total", "Efric_total", "Enorm_total"]
        )
        q = L.q8[mask]
        qref = L.qref8[mask]
        tau = L.tau8[mask]
        fy = L.foot_y_body[mask]
        fz = L.foot_z_body[mask]
        fn = L.fnormal[mask]
        for k in range(Tw.size):
            tt = float(Tw[k] + t0)
            # Look up matching env-step energies
            ek = int(np.clip(np.searchsorted(L.t_env, tt + 1e-9) - 1, 0, L.t_env.size - 1))
            for leg in range(4):
                w.writerow([
                    tt, LEG_NAMES[leg],
                    float(fy[k, leg]), float(fz[k, leg]),
                    float(Fy_ref[k, leg]), float(Fz_ref[k, leg]),
                    float(q[k, 2 * leg + 0]), float(q[k, 2 * leg + 1]),
                    float(qref[k, 2 * leg + 0]), float(qref[k, 2 * leg + 1]),
                    float(tau[k, 2 * leg + 0]), float(tau[k, 2 * leg + 1]),
                    float(fn[k, leg]),
                    float(base_xyz[k, 0]), float(base_xyz[k, 1]), float(base_xyz[k, 2]),
                    float(base_v[k, 0]), float(base_v[k, 1]), float(base_v[k, 2]),
                    float(L.Wp_env[ek]), float(L.Wn_env[ek]), float(L.Edamp_env[ek]),
                    float(L.Efric_env[ek]), float(L.Enorm_env[ek]),
                ])

    # Multi-panel plot
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(6, 4, hspace=0.55, wspace=0.35)
    title_str = (
        f"{policy_label} policy diagnostic — ep0 — T={L.t[-1]:.2f}s, "
        f"mean fwd speed≈{fwd_speed * 1000:.1f} mm/s"
    )
    fig.suptitle(title_str, fontsize=14)

    # (a) Foot Fy, Fz body frame, 4 legs
    for leg in range(4):
        ax = fig.add_subplot(gs[0, leg])
        ax.plot(Tw, L.foot_y_body[mask, leg] * 1000.0, "b-", lw=0.8, label="Fy actual")
        ax.plot(Tw, Fy_ref[:, leg] * 1000.0, "b:", lw=0.8, label="Fy ref")
        ax.plot(Tw, L.foot_z_body[mask, leg] * 1000.0, "r-", lw=0.8, label="Fz actual")
        ax.plot(Tw, Fz_ref[:, leg] * 1000.0, "r:", lw=0.8, label="Fz ref")
        ax.set_title(f"(a) {LEG_NAMES[leg]} foot body Fy/Fz (mm)")
        ax.grid(alpha=0.3)
        if leg == 0:
            ax.legend(fontsize=7, loc="best")
    # (b) Joint positions (hip, knee, ref + actual)
    q = L.q8[mask]
    qref = L.qref8[mask]
    for leg in range(4):
        ax = fig.add_subplot(gs[1, leg])
        ax.plot(Tw, q[:, 2 * leg + 0], "b-", lw=0.8, label="hip act")
        ax.plot(Tw, qref[:, 2 * leg + 0], "b:", lw=0.8, label="hip ref")
        ax.plot(Tw, q[:, 2 * leg + 1], "r-", lw=0.8, label="knee act")
        ax.plot(Tw, qref[:, 2 * leg + 1], "r:", lw=0.8, label="knee ref")
        ax.set_title(f"(b) {LEG_NAMES[leg]} joint q (rad)")
        ax.grid(alpha=0.3)
        if leg == 0:
            ax.legend(fontsize=7, loc="best")
    # (c) Torques with ±τ_max
    tau = L.tau8[mask]
    for leg in range(4):
        ax = fig.add_subplot(gs[2, leg])
        ax.plot(Tw, tau[:, 2 * leg + 0], "b-", lw=0.8, label="τ hip")
        ax.plot(Tw, tau[:, 2 * leg + 1], "r-", lw=0.8, label="τ knee")
        ax.axhline(tau_max, color="k", ls="--", lw=0.5)
        ax.axhline(-tau_max, color="k", ls="--", lw=0.5)
        ax.set_title(f"(c) {LEG_NAMES[leg]} τ (N·m)")
        ax.set_ylim(-tau_max * 1.4, tau_max * 1.4)
        ax.grid(alpha=0.3)
        if leg == 0:
            ax.legend(fontsize=7, loc="best")
    # (d) base velocities + height
    ax = fig.add_subplot(gs[3, :2])
    ax.plot(Tw, base_v[:, 0] * 1000.0, label="vx (mm/s)")
    ax.plot(Tw, base_v[:, 1] * 1000.0, label="vy (mm/s)")
    ax.plot(Tw, base_v[:, 2] * 1000.0, label="vz (mm/s)")
    ax.set_title("(d) base velocity (mm/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(gs[3, 2:])
    ax2.plot(Tw, base_xyz[:, 2] * 1000.0, "g-")
    ax2.set_title("(d) base height z (mm)")
    ax2.grid(alpha=0.3)
    # (e) Gait pattern
    ax = fig.add_subplot(gs[4, :])
    contact = _binarize_contact(L.fnormal[mask])
    for leg in range(4):
        row_y = 3 - leg
        in_c = contact[:, leg]
        d = np.diff(np.concatenate([[0], in_c, [0]]).astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            ax.barh(row_y, Tw[e - 1] - Tw[s] if e > s else 0.0, left=Tw[s], height=0.7, color="#3060B0")
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(LEG_NAMES)
    ax.set_title(
        f"(e) Gait contact pattern (fn>{FN_CONTACT_THRESHOLD:g}N; "
        f"fill air<{MIN_AIR_MS}ms, drop stance<{MIN_STANCE_MS}ms)"
    )
    ax.set_xlim(0.0, Tw[-1] if Tw.size > 0 else 1.0)
    ax.grid(axis="x", alpha=0.3)
    # (f) Energy components per env-step
    ax = fig.add_subplot(gs[5, :])
    ax.plot(Tew, Wp_w * 1000.0, label="W+ (mJ)")
    ax.plot(Tew, Wn_w * 1000.0, label="|W−| (mJ)")
    ax.plot(Tew, Ed_w * 1000.0, label="E_damp (mJ)")
    ax.plot(Tew, Ef_w * 1000.0, label="E_fric (mJ)")
    ax.plot(Tew, En_w * 1000.0, label="E_norm (mJ)")
    ax.set_title("(f) Energy per env-step (mJ)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5)
    ax.set_xlabel("t (s) within window")

    png_path = out_dir / f"M4_{file_tag}_diagnostic.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "window_s": [t0, t1],
        "mean_fwd_speed_mm_per_s": fwd_speed * 1000.0,
        "hip_offsets_body_frame_xyz_m": {
            LEG_NAMES[i]: hip_offsets[i].tolist() for i in range(4)
        },
    }


# ---------------------------------------------------------------------------
# Task M5: Reward component time series (both policies)
# ---------------------------------------------------------------------------

def task_m5_components(logs: List[EpisodeLog], out_csv: Path) -> Dict[str, Any]:
    L = logs[0]
    # Use 1-second mid-episode window.
    t0 = 2.5
    t1 = 3.5
    if L.t_env[-1] < t1:
        t0, t1 = max(0.0, L.t_env[-1] - 1.0), L.t_env[-1]
    em = (L.t_env >= t0) & (L.t_env <= t1)
    Tew = L.t_env[em]
    # Per-env-step per-leg components: sum subs in each env-step
    K = int((L.t.size) / max(1, L.t_env.size))  # ≈ 50
    # We have per-substep arrays; aggregate per env-step.
    Wp_leg = np.zeros((L.t_env.size, 4), dtype=np.float64)
    Wn_leg = np.zeros_like(Wp_leg)
    Ed_leg = np.zeros_like(Wp_leg)
    Ef_leg = np.zeros_like(Wp_leg)
    En_leg = np.zeros_like(Wp_leg)
    for ek in range(L.t_env.size):
        s = ek * K
        e2 = min((ek + 1) * K, L.t.size)
        if s >= e2:
            continue
        Wp_leg[ek] = L.Wp_leg_sub[s:e2].sum(axis=0)
        Wn_leg[ek] = L.Wn_leg_sub[s:e2].sum(axis=0)
        Ed_leg[ek] = L.Edamp_leg_sub[s:e2].sum(axis=0)
        Ef_leg[ek] = L.Efric_leg_sub[s:e2].sum(axis=0)
        En_leg[ek] = L.Enorm_leg_sub[s:e2].sum(axis=0)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "leg", "Wp", "Wn", "Edamp", "Efric", "Enorm"])
        for ek in range(L.t_env.size):
            if not em[ek]:
                continue
            t = float(L.t_env[ek])
            for leg in range(4):
                w.writerow([t, LEG_NAMES[leg], float(Wp_leg[ek, leg]), float(Wn_leg[ek, leg]),
                            float(Ed_leg[ek, leg]), float(Ef_leg[ek, leg]), float(En_leg[ek, leg])])
            # totals row
            w.writerow([t, "total",
                        float(np.sum(Wp_leg[ek])), float(np.sum(Wn_leg[ek])),
                        float(np.sum(Ed_leg[ek])), float(np.sum(Ef_leg[ek])),
                        float(np.sum(En_leg[ek]))])
    return {
        "csv_path": str(out_csv),
        "window_s": [t0, t1],
        "n_env_steps_in_window": int(np.sum(em)),
    }


# ---------------------------------------------------------------------------
# Task M6: Per-leg per-component aggregation
# ---------------------------------------------------------------------------

def task_m6_per_leg_per_component(logs: List[EpisodeLog], out_csv: Path) -> Dict[str, Any]:
    # Sum over all episodes, all substeps, per leg, per component.
    Wp = np.zeros(4, dtype=np.float64)
    Wn = np.zeros(4, dtype=np.float64)
    Ed = np.zeros(4, dtype=np.float64)
    Ef = np.zeros(4, dtype=np.float64)
    En = np.zeros(4, dtype=np.float64)
    n_strides_total = 0
    # Count strides using touchdowns on leg 0 (FL) as the stride boundary.
    for L in logs:
        Wp += L.Wp_leg_sub.sum(axis=0)
        Wn += L.Wn_leg_sub.sum(axis=0)
        Ed += L.Edamp_leg_sub.sum(axis=0)
        Ef += L.Efric_leg_sub.sum(axis=0)
        En += L.Enorm_leg_sub.sum(axis=0)
        tds = detect_touchdowns(L.theta[:, 0])
        n_strides_total += max(0, len(tds) - 1)
    norm = max(1, n_strides_total)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["leg", "Wp_per_stride_J", "Wn_per_stride_J", "Edamp_per_stride_J",
                    "Efric_per_stride_J", "Enorm_per_stride_J"])
        for leg in range(4):
            w.writerow([LEG_NAMES[leg],
                        float(Wp[leg] / norm), float(Wn[leg] / norm),
                        float(Ed[leg] / norm), float(Ef[leg] / norm), float(En[leg] / norm)])
    return {
        "csv_path": str(out_csv),
        "n_strides_used_for_normalization": n_strides_total,
        "totals_J": {
            "Wp": Wp.tolist(),
            "Wn": Wn.tolist(),
            "Edamp": Ed.tolist(),
            "Efric": Ef.tolist(),
            "Enorm": En.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Task M7: Training-time component evolution (TB logs)
# ---------------------------------------------------------------------------

def task_m7_tb(tb_dir: Path, out_csv: Path) -> Dict[str, Any]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        return {"error": f"tensorboard import failed: {exc}"}
    if not tb_dir.is_dir():
        return {"error": f"tb dir not found: {tb_dir}"}
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    needed = ["energy/W_pos_J", "energy/W_neg_J", "energy/E_damp_J", "energy/E_fric_J", "energy/E_norm_J"]
    missing = [t for t in needed if t not in tags]
    if missing:
        return {"error": f"missing tags: {missing}", "available": sorted(tags)}
    series = {t: acc.Scalars(t) for t in needed}
    # Build aligned per-step table (by step)
    step_set = set()
    for ev in series.values():
        for e in ev:
            step_set.add(int(e.step))
    steps_sorted = sorted(step_set)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["step", "W_pos_J", "W_neg_J", "E_damp_J", "E_fric_J", "E_norm_J"]
        w.writerow(cols)
        lookup = {t: {int(e.step): float(e.value) for e in series[t]} for t in needed}
        for s in steps_sorted:
            w.writerow([s,
                        lookup["energy/W_pos_J"].get(s, float("nan")),
                        lookup["energy/W_neg_J"].get(s, float("nan")),
                        lookup["energy/E_damp_J"].get(s, float("nan")),
                        lookup["energy/E_fric_J"].get(s, float("nan")),
                        lookup["energy/E_norm_J"].get(s, float("nan"))])
    return {
        "csv_path": str(out_csv),
        "n_steps_logged": len(steps_sorted),
        "step_range": [steps_sorted[0], steps_sorted[-1]] if steps_sorted else [None, None],
    }


# ---------------------------------------------------------------------------
# Task M8: Per-stride energy decomposition
# ---------------------------------------------------------------------------

def task_m8_per_stride(logs: List[EpisodeLog], out_csv: Path, n_strides: int = 5) -> Dict[str, Any]:
    L = logs[0]
    tds = detect_touchdowns(L.theta[:, 0])  # use FL as reference
    if len(tds) < 2:
        return {"error": "fewer than 2 touchdowns; no complete strides found"}
    n = min(n_strides, len(tds) - 1)
    rows: List[List[Any]] = []
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stride_id", "t_start", "t_end", "Wp_J", "Wn_J", "Edamp_J", "Efric_J", "Enorm_J"])
        for k in range(n):
            s = int(tds[k])
            e2 = int(tds[k + 1])
            Wp = float(L.Wp_leg_sub[s:e2].sum())
            Wn = float(L.Wn_leg_sub[s:e2].sum())
            Ed = float(L.Edamp_leg_sub[s:e2].sum())
            Ef = float(L.Efric_leg_sub[s:e2].sum())
            En = float(L.Enorm_leg_sub[s:e2].sum())
            t_s = float(L.t[s])
            t_e = float(L.t[e2 - 1]) if e2 > s else float(L.t[s])
            row = [k, t_s, t_e, Wp, Wn, Ed, Ef, En]
            rows.append(row)
            w.writerow(row)
    return {"csv_path": str(out_csv), "n_strides": n}


# ---------------------------------------------------------------------------
# Task M9: Per-episode COT distribution
# ---------------------------------------------------------------------------

def task_m9_cot(logs: List[EpisodeLog]) -> Dict[str, Any]:
    """COT = W+_total / (M * g * total_distance)."""
    cots: List[float] = []
    Wp_per_ep: List[float] = []
    dist_per_ep: List[float] = []
    for L in logs:
        if L.base_xyz.shape[0] == 0:
            continue
        Wp = float(np.sum(L.Wp_env))
        x0 = L.base_xyz[0]
        x1 = L.base_xyz[-1]
        # Distance: forward-axis (-Y) primary, but use 2D distance to be robust
        d = float(np.linalg.norm(x1[:2] - x0[:2]))
        if d < 1e-6:
            continue
        cot = Wp / (M_BODY * G_ACC * d)
        cots.append(cot)
        Wp_per_ep.append(Wp)
        dist_per_ep.append(d)
    cot_arr = np.array(cots, dtype=np.float64)
    return {
        "per_episode_cot": cots,
        "per_episode_Wp_J": Wp_per_ep,
        "per_episode_distance_m": dist_per_ep,
        "summary": {
            "n_episodes": int(cot_arr.size),
            "mean": float(np.mean(cot_arr)) if cot_arr.size else 0.0,
            "median": float(np.median(cot_arr)) if cot_arr.size else 0.0,
            "std": float(np.std(cot_arr, ddof=1)) if cot_arr.size > 1 else 0.0,
            "p10": float(np.percentile(cot_arr, 10)) if cot_arr.size else 0.0,
            "p25": float(np.percentile(cot_arr, 25)) if cot_arr.size else 0.0,
            "p75": float(np.percentile(cot_arr, 75)) if cot_arr.size else 0.0,
            "p90": float(np.percentile(cot_arr, 90)) if cot_arr.size else 0.0,
            "min": float(np.min(cot_arr)) if cot_arr.size else 0.0,
            "max": float(np.max(cot_arr)) if cot_arr.size else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check_ckpts() -> Tuple[Path, Path]:
    if not CKPT_BASELINE.is_file():
        raise FileNotFoundError(f"baseline ckpt not found: {CKPT_BASELINE}")
    if not CKPT_CDER.is_file():
        raise FileNotFoundError(f"cder ckpt not found: {CKPT_CDER}")
    return CKPT_BASELINE, CKPT_CDER


def main() -> None:
    ckpt_base, ckpt_cder = _check_ckpts()
    print("Baseline ckpt:", ckpt_base)
    print("CDER ckpt:    ", ckpt_cder)
    print("Output dir:   ", OUT_DIR)

    print("\n=== Rollout: baseline (W2, seed1) ===")
    logs_base = collect_rollout("baseline", ckpt_base)
    print("\n=== Rollout: CDER (v3.1, seed0) ===")
    logs_cder = collect_rollout("cder", ckpt_cder)

    # Save raw NPZs (compact: per-episode env-step level + small per-substep summaries)
    def _pack(logs: List[EpisodeLog], path: Path) -> None:
        np.savez_compressed(
            path,
            t_env=np.stack([L.t_env for L in logs], axis=0),
            mu_env=np.stack([L.mu_env for L in logs], axis=0),
            f_env=np.stack([L.f_env for L in logs], axis=0),
            a_env=np.stack([L.a_env for L in logs], axis=0),
            b_env=np.stack([L.b_env for L in logs], axis=0),
            Wp_env=np.stack([L.Wp_env for L in logs], axis=0),
            Wn_env=np.stack([L.Wn_env for L in logs], axis=0),
            Edamp_env=np.stack([L.Edamp_env for L in logs], axis=0),
            Efric_env=np.stack([L.Efric_env for L in logs], axis=0),
            Enorm_env=np.stack([L.Enorm_env for L in logs], axis=0),
            KE_env=np.stack([L.KE_env for L in logs], axis=0),
            PE_env=np.stack([L.PE_env for L in logs], axis=0),
        )

    _pack(logs_base, OUT_DIR / "rollout_baseline_env_level.npz")
    _pack(logs_cder, OUT_DIR / "rollout_cder_env_level.npz")

    # ---- M1 ----
    print("\n=== M1: CDER closure residual ===")
    m1 = task_m1(logs_cder)
    (OUT_DIR / "M1_closure_residual_cder.json").write_text(json.dumps(m1, indent=2), encoding="utf-8")
    m1_md = ["# M1 — CDER closure equation residual\n\n"]
    m1_md.append("`residual = W+_total − (W−_total + E_damp_total + E_fric_total + E_norm_total + ΔKE + ΔPE)`\n\n")
    s = m1["summary"]
    m1_md.append(
        f"- n_episodes: **{s['n_episodes']}**\n"
        f"- mean residual: **{s['mean_residual_J']:.6f} J** (std {s['std_residual_J']:.6f})\n"
        f"- mean residual as % of W+_total: **{s['mean_residual_pct_of_Wp']:.2f}%** "
        f"(std {s['std_residual_pct_of_Wp']:.2f}%)\n"
        f"- median %: {s['median_residual_pct_of_Wp']:.2f}%, "
        f"range [{s['min_residual_pct_of_Wp']:.2f}%, {s['max_residual_pct_of_Wp']:.2f}%]\n\n"
        f"**Comparison to baseline 4.9%:** CDER residual is **{s['mean_residual_pct_of_Wp']:.2f}%** vs baseline 4.9%.\n\n"
    )
    m1_md.append("| ep | W+ (J) | W− (J) | E_damp (J) | E_fric (J) | E_norm (J) | ΔKE (J) | ΔPE (J) | residual (J) | residual/W+ (%) |\n")
    m1_md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in m1["per_episode"]:
        m1_md.append(
            f"| {r['ep']} | {r['Wp_total_J']:.4f} | {r['Wn_total_J']:.4f} | "
            f"{r['Edamp_total_J']:.4f} | {r['Efric_total_J']:.4f} | {r['Enorm_total_J']:.4f} "
            f"| {r['dKE_J']:+.5f} | {r['dPE_J']:+.5f} | {r['residual_J']:+.5f} | {r['residual_pct_of_Wp']:+.2f}% |\n"
        )
    (OUT_DIR / "M1_closure_summary.md").write_text("".join(m1_md), encoding="utf-8")

    # ---- M2 ----
    print("\n=== M2: CDER per-leg metrics ===")
    m2 = task_m2(logs_cder)
    (OUT_DIR / "M2_cder_per_leg_metrics.json").write_text(json.dumps(m2, indent=2), encoding="utf-8")
    md = ["# M2 — CDER per-leg metrics\n\n"]
    md.append("## r = sqrt(mu) per env-step\n\n")
    md.append("| Leg | mean | median | std | p10 | p90 | min | max | n |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name in LEG_NAMES:
        s = m2["per_leg"][name]["r"]
        md.append(
            f"| {name} | {s['mean']:.4f} | {s['median']:.4f} | {s['std']:.4f} "
            f"| {s['p10']:.4f} | {s['p90']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['n']} |\n"
        )
    md.append("\n## Foot lift per swing cycle (peak swing − mean preceding stance, world z, m)\n\n")
    md.append("| Leg | mean | median | std | p10 | p90 | min | max | n |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name in LEG_NAMES:
        s = m2["per_leg"][name]["lift_m"]
        md.append(
            f"| {name} | {s['mean']:+.5f} | {s['median']:+.5f} | {s['std']:.5f} "
            f"| {s['p10']:+.5f} | {s['p90']:+.5f} | {s['min']:+.5f} | {s['max']:+.5f} | {s['n']} |\n"
        )
    md.append("\n## Touchdown CPG phase θ_td (rad; nominal touchdown = π)\n\n")
    md.append("| Leg | mean | median | std | dev from π (mean abs) | n |\n|---|---:|---:|---:|---:|---:|\n")
    for name in LEG_NAMES:
        s = m2["per_leg"][name]["theta_td_rad"]
        md.append(
            f"| {name} | {s.get('mean', float('nan')):.4f} | {s.get('median', float('nan')):.4f} "
            f"| {s.get('std', float('nan')):.4f} | {s.get('deviation_from_pi_rad_mean', float('nan')):.4f} "
            f"| {s.get('n', 0)} |\n"
        )
    md.append("\n## Foot trajectory LS fit (ep 0; mouse-body y/z; swing φ∈[0,π))\n\n")
    md.append("| Leg | a_eff (m) | b_eff (m) | RMSE_y (m) | RMSE_z (m) | n_swing |\n|---|---:|---:|---:|---:|---:|\n")
    for name in LEG_NAMES:
        f = m2["fit"][name]
        md.append(
            f"| {name} | {f['a_eff']:+.5f} | {f['b_eff']:+.5f} | {f['rmse_y_m']:.5f} | {f['rmse_z_m']:.5f} | {f['n_swing_samples']} |\n"
        )
    (OUT_DIR / "M2_cder_summary.md").write_text("".join(md), encoding="utf-8")

    # ---- M3 ----
    print("\n=== M3: Baseline gait diagram ===")
    m3 = task_m3_baseline_gait(logs_base, OUT_DIR)

    # ---- M4 ----
    print("\n=== M4: Diagnostic snapshots (baseline + CDER) ===")
    hip_offsets = compute_hip_offsets_body_frame()
    for leg in range(4):
        print(f"  hip_offset[{LEG_NAMES[leg]}] = (x,y,z) m: "
              f"({hip_offsets[leg][0]:+.5f}, {hip_offsets[leg][1]:+.5f}, {hip_offsets[leg][2]:+.5f})")
    m4 = task_m4_baseline_diag(
        logs_base, OUT_DIR, policy_label="Baseline (W2 seed1)",
        file_tag="baseline", hip_offsets=hip_offsets,
    )
    m4_cder = task_m4_baseline_diag(
        logs_cder, OUT_DIR, policy_label="CDER (v3.1 seed0)",
        file_tag="cder", hip_offsets=hip_offsets,
    )

    # M4 fix note
    fix_md: List[str] = []
    fix_md.append("# M4 Fy/Fz reference fix — hip-offset correction\n\n")
    fix_md.append(
        "**Bug**: Earlier `M4_*_diagnostic.png` panel (a) showed `Fy_ref` / `Fz_ref` "
        "(CPG target) expressed in the **leg-IK frame** (origin at each leg's hip), "
        "while `Fy_body` / `Fz_body` (actual ankle position) are expressed in the "
        "**mouse-body frame**. The two differ by the hip's position in the mouse-body "
        "frame. For front legs (FL/FR) this offset is small, so the curves looked "
        "consistent; for hind legs (RL/RR) the y-offset is ≈ +0.12 m, which shifted "
        "`Fy_body` to ~+120 mm and made the ref vs actual curves visually misaligned.\n\n"
    )
    fix_md.append(
        "**Fix**: Computed each leg's `thigh_link_*` site position in mouse-body frame "
        "at qpos0, then added it to `Fy_ref` and `Fz_ref` in the M4 diagnostic only. "
        "The control loop itself is unaffected (it operates in joint-space via per-leg "
        "IK, which is correct as-is and verified by panel (b) joint tracking and the "
        "M1 closure residual).\n\n"
    )
    fix_md.append("## Per-leg hip offsets in the mouse-body frame (m), at qpos0\n\n")
    fix_md.append("| Leg | hip_x (m) | hip_y (m) | hip_z (m) | site name |\n")
    fix_md.append("|---|---:|---:|---:|---|\n")
    for leg in range(4):
        ox, oy, oz = (float(v) for v in hip_offsets[leg])
        fix_md.append(
            f"| {LEG_NAMES[leg]} | {ox:+.5f} | {oy:+.5f} | {oz:+.5f} | `{HIP_SITES[leg]}` |\n"
        )
    fix_md.append(
        "\nThe corrected `Fy_ref / Fz_ref` shown in `M4_*_diagnostic.png` panel (a) is\n\n"
        "```\n"
        "Fy_ref(t) = hip_y + y0_leg + a · r · cos(theta(t))\n"
        "Fz_ref(t) = hip_z + z0_leg + b · r · sin(theta(t))\n"
        "```\n\n"
        "where `hip_y`, `hip_z` come from the table above; `y0_leg`, `z0_leg` are the "
        "nominal swing-branch values from `FootPathFixed.para_FU` (front legs) or "
        "`para_HU` (hind legs).\n\n"
        "## Scope of the fix\n\n"
        "- **Modified**: `task_m4_baseline_diag(...)` in `run_cder_closure_validation.py` "
        "(the only place `Fy_ref / Fz_ref` were rendered in body-frame plots).\n"
        "- **Regenerated**: `M4_baseline_diagnostic.png`, `M4_baseline_diagnostic.csv`, "
        "`M4_cder_diagnostic.png`, `M4_cder_diagnostic.csv` (same 0.7 s windows as before).\n"
        "- **NOT touched**: control / training code; M1, M3, M5–M9; the M2 LS fit "
        "(its hind-leg `RMSE_y ≈ 0.13 m` artefact has the same root cause and could be "
        "fixed analogously if you want a tighter fit in the leg-IK frame; current "
        "M2 z-fits remain valid since z-offset is the same for both `Fz_body` and the "
        "fit's `z0_leg` constant — only y was used for the cosine fit so it sees the "
        "offset).\n"
    )
    (OUT_DIR / "M4_fix_note.md").write_text("".join(fix_md), encoding="utf-8")

    # ---- M5 ----
    print("\n=== M5: Reward component time series ===")
    m5_base = task_m5_components(logs_base, OUT_DIR / "M5_reward_components_baseline.csv")
    m5_cder = task_m5_components(logs_cder, OUT_DIR / "M5_reward_components_cder.csv")

    # ---- M6 ----
    print("\n=== M6: Per-leg per-component aggregation ===")
    m6_base = task_m6_per_leg_per_component(logs_base, OUT_DIR / "M6_per_leg_per_component_baseline.csv")
    m6_cder = task_m6_per_leg_per_component(logs_cder, OUT_DIR / "M6_per_leg_per_component_cder.csv")

    # ---- M7 ----
    print("\n=== M7: Training-time component evolution (TB logs) ===")
    m7 = task_m7_tb(TB_LOGDIR_CDER, OUT_DIR / "M7_training_component_evolution.csv")

    # ---- M8 ----
    print("\n=== M8: Per-stride energy decomposition ===")
    m8_base = task_m8_per_stride(logs_base, OUT_DIR / "M8_per_stride_baseline.csv")
    m8_cder = task_m8_per_stride(logs_cder, OUT_DIR / "M8_per_stride_cder.csv")

    # ---- M9 ----
    print("\n=== M9: Per-episode COT distribution ===")
    m9_base = task_m9_cot(logs_base)
    m9_cder = task_m9_cot(logs_cder)
    (OUT_DIR / "M9_cot_distribution_baseline.json").write_text(json.dumps(m9_base, indent=2), encoding="utf-8")
    (OUT_DIR / "M9_cot_distribution_cder.json").write_text(json.dumps(m9_cder, indent=2), encoding="utf-8")
    md9 = ["# M9 — Per-episode COT distribution\n\n"]
    md9.append("`COT = W+_total / (M · g · total_distance)`, M = 0.2895 kg, g = 9.81 m/s².\n\n")
    md9.append("| policy | n | mean | median | std | p10 | p25 | p75 | p90 | min | max |\n")
    md9.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, m in [("baseline", m9_base), ("cder", m9_cder)]:
        s = m["summary"]
        md9.append(
            f"| {name} | {s['n_episodes']} | {s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} "
            f"| {s['p10']:.3f} | {s['p25']:.3f} | {s['p75']:.3f} | {s['p90']:.3f} "
            f"| {s['min']:.3f} | {s['max']:.3f} |\n"
        )
    (OUT_DIR / "M9_cot_summary.md").write_text("".join(md9), encoding="utf-8")

    # ---- Master report ----
    print("\n=== Master report ===")
    master: List[str] = []
    master.append("# Thesis experiments — master report\n\n")
    master.append(f"- Baseline checkpoint: `{ckpt_base}`\n")
    master.append(f"- CDER checkpoint:    `{ckpt_cder}`\n")
    master.append(f"- Output dir:         `{OUT_DIR}`\n")
    master.append(f"- Episodes per policy: {N_EPISODES} (deterministic, seed = ep index)\n")
    master.append(f"- Env-steps per episode: {EVN_STEPS_PER_EP} (50 substeps each)\n")
    master.append(f"- Master script: `{Path(__file__).resolve()}`\n\n")
    master.append("## Task statuses & outputs\n\n")

    s1 = m1["summary"]
    master.append(
        f"### M1 — CDER closure residual  (status: success)\n"
        f"- Files: `M1_closure_residual_cder.json`, `M1_closure_summary.md`\n"
        f"- CDER mean residual = **{s1['mean_residual_J']:.4f} J** "
        f"(**{s1['mean_residual_pct_of_Wp']:.2f}% of W+_total**, std {s1['std_residual_pct_of_Wp']:.2f}%); baseline reference is **4.9%**.\n\n"
    )

    master.append(
        f"### M2 — CDER per-leg metrics  (status: success)\n"
        f"- Files: `M2_cder_per_leg_metrics.json`, `M2_cder_summary.md`\n"
        f"- Per-leg r/lift/θ_td/foot-fit tables (FL/FR/RL/RR).\n\n"
    )

    master.append(
        f"### M3 — Baseline gait diagram  (status: success)\n"
        f"- Files: `M3_baseline_contact_pattern.csv`, `M3_baseline_gait_diagram.png`\n"
        f"- Window: {m3['window_s']} s. Duty per leg: {m3['duty_per_leg']}.\n\n"
    )

    master.append(
        f"### M4 — Detailed diagnostic snapshots (baseline + CDER)  (status: success)\n"
        f"- Files: `M4_baseline_diagnostic.png`/`.csv`, `M4_cder_diagnostic.png`/`.csv`\n"
        f"- Window: {m4['window_s']} s\n"
        f"- Baseline mean forward speed ≈ {m4['mean_fwd_speed_mm_per_s']:.1f} mm/s\n"
        f"- CDER mean forward speed ≈ {m4_cder['mean_fwd_speed_mm_per_s']:.1f} mm/s\n\n"
    )

    master.append(
        f"### M5 — Reward component time series  (status: success)\n"
        f"- Files: `M5_reward_components_baseline.csv` (window {m5_base['window_s']} s, "
        f"{m5_base['n_env_steps_in_window']} env-steps), "
        f"`M5_reward_components_cder.csv` (window {m5_cder['window_s']} s, "
        f"{m5_cder['n_env_steps_in_window']} env-steps).\n\n"
    )

    master.append(
        f"### M6 — Per-leg per-component aggregation  (status: success)\n"
        f"- Files: `M6_per_leg_per_component_baseline.csv` (n_strides_norm={m6_base['n_strides_used_for_normalization']}), "
        f"`M6_per_leg_per_component_cder.csv` (n_strides_norm={m6_cder['n_strides_used_for_normalization']}).\n\n"
    )

    if "error" in m7:
        master.append(f"### M7 — Training-time component evolution  (status: failed)\n- Error: {m7['error']}\n\n")
    else:
        master.append(
            f"### M7 — Training-time component evolution  (status: success)\n"
            f"- File: `M7_training_component_evolution.csv` ({m7['n_steps_logged']} points, "
            f"step range {m7['step_range']}).\n\n"
        )

    if "error" in m8_base or "error" in m8_cder:
        master.append(
            f"### M8 — Per-stride energy decomposition  (status: partial)\n"
            f"- baseline: {'ok ('+str(m8_base.get('n_strides'))+' strides)' if 'csv_path' in m8_base else m8_base.get('error')}\n"
            f"- cder: {'ok ('+str(m8_cder.get('n_strides'))+' strides)' if 'csv_path' in m8_cder else m8_cder.get('error')}\n\n"
        )
    else:
        master.append(
            f"### M8 — Per-stride energy decomposition  (status: success)\n"
            f"- Files: `M8_per_stride_baseline.csv` ({m8_base.get('n_strides')} strides), "
            f"`M8_per_stride_cder.csv` ({m8_cder.get('n_strides')} strides).\n\n"
        )

    sb = m9_base["summary"]; sc = m9_cder["summary"]
    master.append(
        f"### M9 — Per-episode COT distribution  (status: success)\n"
        f"- Files: `M9_cot_distribution_baseline.json`, `M9_cot_distribution_cder.json`, `M9_cot_summary.md`\n"
        f"- Baseline COT mean = **{sb['mean']:.3f}** (median {sb['median']:.3f}, std {sb['std']:.3f}), "
        f"CDER mean = **{sc['mean']:.3f}** (median {sc['median']:.3f}, std {sc['std']:.3f}).\n\n"
    )

    master.append("## Notes / Caveats\n\n")
    master.append(
        "- Energy components: per-substep `W+, W-, E_damp` derived from `tau · qvel` and `b_damp · qvel²` "
        "(b_damp = 0.005 N·m·s/rad); `E_fric, E_norm` from contact tangential/normal power times substep dt.\n"
        "- Per-leg `E_fric` and `E_norm` attributed by routing each foot-vs-floor contact pair to the corresponding leg.\n"
        "- Closure residual includes only the **mouse body** ΔKE/ΔPE (M = 0.2895 kg, the full simulator mass). "
        "Limb ΔKE/ΔPE is neglected, so residuals contain limb kinetic-energy churn.\n"
        "- M2 LS fits use the **mouse-body y/z** frame (same as before); hind-leg RMSE_y will be large for the same reason "
        "as in the previous follow-up — that frame is not the leg-IK Fy/Fz frame.\n"
        "- M4 reference `Fy_ref / Fz_ref` is computed with the **simple** `y₀ + a·r·cos(θ) / z₀ + b·r·sin(θ)` formula "
        "(`z_asym_alpha=0`, the project default); the actual CPG can add z-asym shaping that this reference ignores.\n"
        "- M7 uses tensorboard scalars logged at ~4 k-step intervals (367 points over 0→1.5 M env-steps).\n"
        "- M9 COT uses `W+_total / (M g d)` with d = 2-D base displacement from first to last substep of each episode.\n\n"
    )
    (OUT_DIR / "run_cder_closure_validation_report.md").write_text("".join(master), encoding="utf-8")
    print("Wrote master report:", OUT_DIR / "run_cder_closure_validation_report.md")


if __name__ == "__main__":
    main()
