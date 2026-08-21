#!/usr/bin/env python3
"""Fair three-way COT table: identical steady-state measurement for M2/M3/M4.

Problem: M3 warms up (settle 2s + warmup 2s) then measures only steady state,
while M2 (Ours) and M4 log COT from t=0 (startup transient included). The
transient inflates COT ~20-25%, unfairly penalizing M2/M4 vs M3.

Fix: apply the SAME protocol to all three -> discard the first WARMUP_S seconds
of every rollout, then compute COT on the steady-state window.

  - M2: per-speed RL ckpt, deterministic rollout, slice t>=WARMUP_S.
  - M4: open-loop CPG (calibrated f), slice t>=WARMUP_S.
  - M3: already steady (faithful sim_test warmup); re-run with calibrated fre.

Outputs: outputs/final_result/multi_velocity/fair_cot_table.{csv,md}
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import mujoco
from stable_baselines3 import PPO

CPG_ROOT = Path(__file__).resolve().parents[1]
for _p in (CPG_ROOT, CPG_ROOT / "env", CPG_ROOT / "experiments", CPG_ROOT / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import eval_single_speed_three_way as rss  # noqa: E402
from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3  # noqa: E402
OUT = CPG_ROOT / "outputs/final_result/multi_velocity"
CUR_CSV = OUT / "three_way_multi_velocity.csv"

WARMUP_S = 2.0          # discard startup transient (applied to M2 & M4)
N_EP_M2 = 5
N_EP_M4 = 3
V_CMDS = (0.06, 0.09, 0.12)

M2_PER_SPEED = {
    0.06: (CPG_ROOT / "logs/w3_ours_vcmd0.06_thesis_scaled_v2/seed0/rat_cpg_ppo_route_a.zip",
           CPG_ROOT / "configs/v31_vcmd0.06_thesis_scaled_v2.json"),
    0.09: (CPG_ROOT / "logs/w3_ours_vcmd0.09_thesis_scaled/seed0/rat_cpg_ppo_route_a.zip",
           CPG_ROOT / "configs/v31_vcmd0.09_thesis_scaled.json"),
    0.12: (CPG_ROOT / "logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip",
           CPG_ROOT / "configs/ours_cder_v31_env_kwargs.json"),
}


def _slice(log, t0: float):
    t = log.t - log.t[0]
    mask = t >= t0
    sub = rss.SubstepEpisode()
    for attr in ("t", "q", "qvel", "tau", "body_pos", "body_vel",
                 "foot_pos_body", "grf_world", "contact"):
        setattr(sub, attr, getattr(log, attr)[mask].copy())
    sub.t = sub.t - sub.t[0]
    return sub


def _stats(scs: List[rss.EpisodeScalars]) -> Dict[str, float]:
    cot = [s.cot for s in scs]
    spd = [s.fwd_speed_mps * 1000.0 for s in scs]
    wprop = [s.W_propulsion_J for s in scs]
    wwaste = [s.W_braking_J + s.W_vertical_bounce_J + s.W_lateral_cost_J for s in scs]
    eta = [s.eta_locomotion for s in scs]
    return {"cot": float(np.mean(cot)), "cot_std": float(np.std(cot)),
            "spd": float(np.mean(spd)), "spd_std": float(np.std(spd)),
            "w_useful": float(np.mean(wprop)), "w_wasted": float(np.mean(wwaste)),
            "eta_useful": float(np.mean(eta)),
            "w_brake": float(np.mean([s.W_braking_J for s in scs])),
            "w_vert": float(np.mean([s.W_vertical_bounce_J for s in scs])),
            "w_lat": float(np.mean([s.W_lateral_cost_J for s in scs]))}


def measure_m2(v_cmd: float) -> Dict[str, float]:
    ckpt, env_kw_path = M2_PER_SPEED[v_cmd]
    env = RatCpgEnvEnergySubstep50ShapeV3(**rss.load_env_kwargs(env_kw_path))
    model = PPO.load(str(ckpt), env=env)
    e = env.unwrapped; mj = e.model
    mouse_bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, n) for n in rss.FOOT_SITES]
    foot_to_leg = rss._body_id_to_leg(mj)
    act_idx, dof_idx = rss._parse_actuator_indices(mj)
    orig = mujoco.mj_step
    rows: List[Dict[str, Any]] = []

    def wrap(m, d):
        orig(m, d)
        rows.append(rss._record_substep(m, d, mouse_bid=mouse_bid, site_ids=site_ids,
                                        act_idx=act_idx, dof_idx=dof_idx,
                                        foot_to_leg=foot_to_leg, t_val=float(d.time)))
    scs = []
    mujoco.mj_step = wrap
    try:
        for ep in range(N_EP_M2):
            rows.clear()
            obs, _ = env.reset(seed=ep)
            done, steps = False, 0
            while not done and steps < rss.ENV_STEPS:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(a)
                done = bool(term or trunc); steps += 1
            log = rss._accumulate_substep_log(rows)
            scs.append(rss.episode_scalars(ep, _slice(log, WARMUP_S)))
    finally:
        mujoco.mj_step = orig
        env.close()
    return _stats(scs)


def measure_m4(v_cmd: float, f_hz: float) -> Dict[str, float]:
    env = rss._make_m4_env()
    e = env.unwrapped; mj = e.model
    mouse_bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, n) for n in rss.FOOT_SITES]
    foot_to_leg = rss._body_id_to_leg(mj)
    act_idx, dof_idx = rss._parse_actuator_indices(mj)
    scs = []
    try:
        for ep in range(N_EP_M4):
            rss._configure_m4_cpg(env, seed=ep, f_planner=f_hz)
            rows: List[Dict[str, Any]] = []
            for k in range(rss.N_SUBSTEPS):
                rss._m4_substep(env, k, f_hz)
                rows.append(rss._record_substep(mj, e.data, mouse_bid=mouse_bid,
                            site_ids=site_ids, act_idx=act_idx, dof_idx=dof_idx,
                            foot_to_leg=foot_to_leg, t_val=k * rss.DT_SUBSTEP))
            log = rss._accumulate_substep_log(rows)
            scs.append(rss.episode_scalars(ep, _slice(log, WARMUP_S)))
    finally:
        env.close()
    return _stats(scs)


def measure_m3(fre: float) -> Dict[str, float]:
    # M3 already warms up + measures steady state internally (faithful sim_test).
    scs = [rss._m3_quick_episode(fre, ep) for ep in range(3)]
    return _stats(scs)


def _read_cur() -> Dict[Tuple[float, str], Dict[str, Any]]:
    out: Dict[Tuple[float, str], Dict[str, Any]] = {}
    with CUR_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(float(r["v_cmd_mps"]), r["model"])] = r
    return out


def main() -> None:
    cur = _read_cur()
    table: List[Dict[str, Any]] = []
    for v in V_CMDS:
        m3_fre = float(cur[(v, "M3_planner")]["m3_fre"])
        m4_f = float(cur[(v, "M4_planner_simplified")]["m4_f_hz"])
        print(f"\n===== v_cmd={v:.2f} (M3 fre={m3_fre}, M4 f={m4_f}) =====", flush=True)
        def _p(tag, st):
            print(f"  {tag} steady: COT={st['cot']:.3f} @ {st['spd']:.1f} mm/s | "
                  f"useful={st['w_useful']:.3f}J wasted={st['w_wasted']:.3f}J "
                  f"eta(useful%)={st['eta_useful']*100:.1f}%", flush=True)
        m2 = measure_m2(v); _p("M2", m2)
        m3 = measure_m3(m3_fre); _p("M3", m3)
        m4 = measure_m4(v, m4_f); _p("M4", m4)
        for model, st, full in (("M2_ours", m2, cur[(v, "M2_ours")]),
                                ("M3_planner", m3, cur[(v, "M3_planner")]),
                                ("M4_planner_simplified", m4, cur[(v, "M4_planner_simplified")])):
            table.append({"v_cmd": v, "model": model,
                          "cot_fair": st["cot"], "cot_std": st["cot_std"],
                          "spd_fair": st["spd"],
                          "w_useful_J": st["w_useful"], "w_wasted_J": st["w_wasted"],
                          "eta_useful": st["eta_useful"],
                          "w_brake_J": st["w_brake"], "w_vert_J": st["w_vert"], "w_lat_J": st["w_lat"],
                          "cot_old": float(full["cot_mean"]), "spd_old": float(full["fwd_speed_mm_s_mean"])})

    # CSV
    with (OUT / "fair_cot_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)

    # Markdown
    lines = ["# Multi-velocity steady-state metrics (COT / speed / useful-work fraction)", "",
             f"All models: discard first {WARMUP_S:.0f}s transient, measure steady state.",
             "M2/M4 were previously measured from t=0 (transient included); M3 already steady.",
             "",
             "Useful work = forward propulsion; Wasted = braking + vertical bounce + lateral.",
             "eta = useful / (useful + wasted).", "",
             "| v_cmd | Model | COT | Speed mm/s | Useful J | Wasted J | useful% (eta) | COT(old) |",
             "|------:|-------|----:|-----------:|---------:|---------:|--------------:|---------:|"]
    for v in V_CMDS:
        grp = [t for t in table if t["v_cmd"] == v]
        best = min(grp, key=lambda t: t["cot_fair"])["model"]
        for t in grp:
            win = " ✅" if t["model"] == best else ""
            lines.append(f"| {v:.2f} | {t['model']}{win} | **{t['cot_fair']:.2f}** | {t['spd_fair']:.1f} "
                         f"| {t['w_useful_J']:.3f} | {t['w_wasted_J']:.3f} | {t['eta_useful']*100:.1f}% "
                         f"| {t['cot_old']:.2f} |")
    (OUT / "fair_cot_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n================ FAIR COT TABLE ================")
    for v in V_CMDS:
        grp = [t for t in table if t["v_cmd"] == v]
        best = min(grp, key=lambda t: t["cot_fair"])
        print(f"v={v:.2f}: " + "  ".join(f"{t['model'].split('_')[0]}={t['cot_fair']:.2f}(was {t['cot_old']:.2f})"
                                          for t in grp) + f"   winner={best['model'].split('_')[0]}")
    print(f"\nWrote {OUT/'fair_cot_table.csv'} and fair_cot_table.md")


if __name__ == "__main__":
    main()
