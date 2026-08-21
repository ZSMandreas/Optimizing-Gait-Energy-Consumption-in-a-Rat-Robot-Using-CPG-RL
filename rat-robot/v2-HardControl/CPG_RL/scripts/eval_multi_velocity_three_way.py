#!/usr/bin/env python3
"""Multi-velocity three-way eval: M2 Ours / M3 Planner / M4 Planner-simplified.

M2: per-speed RL ckpt — pilot JSON or unified env-step COT (|tau·qdot|/(M·g·d)).
M3/M4: offline CPG calibration to M2 realized speed at each v_cmd, then 20-ep eval.

Outputs: outputs/final_result/multi_velocity/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CPG_ROOT = Path(__file__).resolve().parents[1]
os.chdir(CPG_ROOT)
sys.path.insert(0, str(CPG_ROOT))
sys.path.insert(0, str(CPG_ROOT / "env"))
sys.path.insert(0, str(CPG_ROOT / "experiments"))
sys.path.insert(0, str(CPG_ROOT / "scripts"))

import mujoco
import numpy as np
from stable_baselines3 import PPO

from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3
import eval_single_speed_three_way as rss  # noqa: E402

MULTI_VCMD = CPG_ROOT / "outputs/thesis_experiments_multi_vcmd"
OUT_ROOT = CPG_ROOT / "outputs/final_result/multi_velocity"
CSV_PATH = OUT_ROOT / "three_way_multi_velocity.csv"

V_CMDS = (0.06, 0.09, 0.12)

M2_EVAL_JSON: Dict[float, Path] = {
    0.06: MULTI_VCMD / "pilot_vcmd0.06_thesis_scaled_v2/final_eval_ours_1500k.json",
    0.09: MULTI_VCMD / "pilot_vcmd0.09_thesis_scaled/final_eval_ours_1500k.json",
    0.12: MULTI_VCMD / "pilot_v_cmd_0.12/final_eval_ours_1500k.json",
}

M2_PER_SPEED: Dict[float, Dict[str, Path]] = {
    0.06: {
        "ckpt": CPG_ROOT / "logs/w3_ours_vcmd0.06_thesis_scaled_v2/seed0/rat_cpg_ppo_route_a.zip",
        "env_kwargs": CPG_ROOT / "configs/v31_vcmd0.06_thesis_scaled_v2.json",
    },
    0.09: {
        "ckpt": CPG_ROOT / "logs/w3_ours_vcmd0.09_thesis_scaled/seed0/rat_cpg_ppo_route_a.zip",
        "env_kwargs": CPG_ROOT / "configs/v31_vcmd0.09_thesis_scaled.json",
    },
    0.12: {
        "ckpt": CPG_ROOT / "logs/w3_ours_v3.1/seed0/rat_cpg_ppo_route_a.zip",
        "env_kwargs": CPG_ROOT / "configs/ours_cder_v31_env_kwargs.json",
    },
}


def _load_m2_from_json(v_cmd: float) -> Dict[str, Any]:
    path = M2_EVAL_JSON[v_cmd]
    if not path.is_file():
        raise FileNotFoundError(f"M2 eval JSON missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    per_ep = data.get("per_episode", [])
    speeds = [float(e["fwd_vel_mean_mps"]) * 1000.0 for e in per_ep]
    cots = [float(e["cot"]) for e in per_ep]
    dists = [float(e["distance_m"]) * 1000.0 for e in per_ep]
    return {
        "v_cmd_mps": v_cmd,
        "model": "M2_ours",
        "source": str(path.relative_to(CPG_ROOT)),
        "ckpt": data.get("ckpt", ""),
        "n_episodes": int(data.get("n_episodes", len(per_ep))),
        "fwd_speed_mm_s_mean": float(data["fwd_speed_mm_s"]["mean"]),
        "fwd_speed_mm_s_std": float(data["fwd_speed_mm_s"]["std"]),
        "cot_mean": float(data["cot"]["mean"]),
        "cot_std": float(data["cot"]["std"]),
        "distance_mm_mean": float(np.mean(dists)),
        "distance_mm_std": float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
        "per_ep_speed_mm_s": speeds,
        "per_ep_cot": cots,
        "target_speed_mm_s": float(data["fwd_speed_mm_s"]["mean"]),
        "m3_fre": None,
        "m4_f_hz": None,
    }


def run_m2_envstep(v_cmd: float, n_episodes: int = rss.N_EPISODES) -> Tuple[List[rss.SubstepEpisode], List[rss.EpisodeScalars]]:
    """M2 per-speed ckpt with env-step COT (same pipeline as M3/M4)."""
    spec = M2_PER_SPEED[v_cmd]
    ckpt = spec["ckpt"]
    env_kwargs = rss.load_env_kwargs(spec["env_kwargs"])
    env = RatCpgEnvEnergySubstep50ShapeV3(**env_kwargs)
    model = PPO.load(str(ckpt), env=env)
    e = env.unwrapped
    mj_model = e.model
    mouse_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mouse")
    site_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in rss.FOOT_SITES]
    foot_to_leg = rss._body_id_to_leg(mj_model)
    act_idx, dof_idx = rss._parse_actuator_indices(mj_model)

    episodes: List[rss.SubstepEpisode] = []
    scalars: List[rss.EpisodeScalars] = []
    orig_mj_step = mujoco.mj_step
    sub_rows: List[Dict[str, Any]] = []

    def mj_wrap(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        orig_mj_step(m, d)
        sub_rows.append(
            rss._record_substep(
                m, d,
                mouse_bid=mouse_bid,
                site_ids=site_ids,
                act_idx=act_idx,
                dof_idx=dof_idx,
                foot_to_leg=foot_to_leg,
                t_val=float(d.time),
            )
        )

    print(f"Running M2 Ours env-step (v_cmd={v_cmd:.2f}, {n_episodes} ep)...")
    mujoco.mj_step = mj_wrap  # type: ignore[assignment]
    try:
        for ep in range(n_episodes):
            sub_rows.clear()
            obs, _ = env.reset(seed=ep)
            done = False
            steps = 0
            while not done and steps < rss.ENV_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
                steps += 1
            log = rss._accumulate_substep_log(sub_rows)
            sc = rss.episode_scalars(ep, log)
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


def load_m2_row(v_cmd: float, *, source: str) -> Dict[str, Any]:
    if source == "pilot":
        return _load_m2_from_json(v_cmd)
    if source != "envstep":
        raise ValueError(f"unknown m2 source: {source}")

    _, scalars = run_m2_envstep(v_cmd)
    spec = M2_PER_SPEED[v_cmd]
    tag = f"vcmd{v_cmd:.2f}".replace(".", "")
    env_kw = rss.load_env_kwargs(spec["env_kwargs"])
    summary = rss.build_eval_summary(
        "M2_ours",
        str(spec["ckpt"].relative_to(CPG_ROOT)),
        {
            "v_cmd": v_cmd,
            "target_speed": env_kw.get("target_speed", v_cmd),
            "cot_pipeline": "env-step |tau·qdot|/(M·g·d_fwd)",
        },
        scalars,
        notes=f"Per-speed RL ckpt; unified env-step COT @ v_cmd={v_cmd}",
    )
    _save_eval_summary(f"m2_ours/{tag}", summary)

    row = _scalars_to_row(
        v_cmd, "M2_ours", scalars,
        target_mm_s=float(summary["fwd_speed_mm_s"]["mean"]),
        source="sim (env-step COT)",
    )
    row["ckpt"] = str(spec["ckpt"].relative_to(CPG_ROOT))

    out = OUT_ROOT / "m2_ours" / f"{tag}_envstep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def _load_planner_rows_from_csv(v_cmds: List[float]) -> List[Dict[str, Any]]:
    """Reuse M3/M4 rows from existing CSV (numeric fields restored)."""
    if not CSV_PATH.is_file():
        return []
    float_fields = {
        "v_cmd_mps", "target_speed_mm_s", "fwd_speed_mm_s_mean", "fwd_speed_mm_s_std",
        "cot_mean", "cot_std", "distance_mm_mean", "distance_mm_std",
        "eta_locomotion_mean", "eta_locomotion_std", "m3_fre", "m4_f_hz",
    }
    rows: List[Dict[str, Any]] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "M2_ours":
                continue
            if float(r["v_cmd_mps"]) not in v_cmds:
                continue
            for k in float_fields:
                if r.get(k) not in (None, "", "None"):
                    r[k] = float(r[k])
                else:
                    r[k] = None
            rows.append(r)
    return rows


def _m3_fre_candidates(target_mm_s: float) -> List[float]:
    """Coarse + fine grid around linear estimate from single-speed anchor (0.78 Hz @ 97 mm/s)."""
    anchor_f, anchor_v = 0.78, 97.0
    guess = anchor_f * (target_mm_s / anchor_v)
    coarse = [round(guess + d, 2) for d in (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)]
    extra = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90, 1.0]
    out: List[float] = []
    for f in coarse + extra:
        if f > 0.35 and f not in out:
            out.append(f)
    return sorted(out)


def _m4_f_candidates(target_mm_s: float) -> List[float]:
    anchor_f, anchor_v = 2.80, 96.0
    guess = anchor_f * (target_mm_s / anchor_v)
    coarse = [round(guess + d, 2) for d in (-0.20, -0.10, 0.0, 0.10, 0.20)]
    extra = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
    out: List[float] = []
    for f in coarse + extra:
        if f > 1.5 and f not in out:
            out.append(f)
    return sorted(out)


def _calibrate_m3(target_mm_s: float, n_sweep: int) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    print(f"\n=== M3 fre calibration (target {target_mm_s:.1f} mm/s) ===")
    for fre in _m3_fre_candidates(target_mm_s):
        speeds = [rss._m3_quick_episode(fre, ep).fwd_speed_mps * 1000.0 for ep in range(n_sweep)]
        mean = float(np.mean(speeds))
        rows.append({"fre_Hz": fre, "fwd_speed_mm_s_mean": mean, "per_ep_mm_s": speeds})
        print(f"  fre={fre:.2f} Hz: {mean:.1f} mm/s")
    best = min(rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - target_mm_s))
    print(f"  → fre={best['fre_Hz']:.2f} Hz ({best['fwd_speed_mm_s_mean']:.1f} mm/s)")
    return float(best["fre_Hz"]), rows


def _calibrate_m4(target_mm_s: float, n_sweep: int) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    env = rss._make_m4_env()
    print(f"\n=== M4 f calibration (target {target_mm_s:.1f} mm/s) ===")
    try:
        for f in _m4_f_candidates(target_mm_s):
            speeds = [
                rss._m4_single_episode(env, ep, f)[1].fwd_speed_mps * 1000.0
                for ep in range(n_sweep)
            ]
            mean = float(np.mean(speeds))
            rows.append({"f_Hz": f, "fwd_speed_mm_s_mean": mean, "per_ep_mm_s": speeds})
            print(f"  f={f:.2f} Hz: {mean:.1f} mm/s")
    finally:
        env.close()
    best = min(rows, key=lambda r: abs(r["fwd_speed_mm_s_mean"] - target_mm_s))
    print(f"  → f={best['f_Hz']:.2f} Hz ({best['fwd_speed_mm_s_mean']:.1f} mm/s)")
    return float(best["f_Hz"]), rows


def _scalars_to_row(
    v_cmd: float,
    model: str,
    scalars: List[rss.EpisodeScalars],
    *,
    target_mm_s: float,
    m3_fre: float | None = None,
    m4_f: float | None = None,
    source: str = "sim",
) -> Dict[str, Any]:
    speeds = [s.fwd_speed_mps * 1000.0 for s in scalars]
    cots = [s.cot for s in scalars]
    dists = [s.distance_m * 1000.0 for s in scalars]
    etas = [s.eta_locomotion for s in scalars]
    return {
        "v_cmd_mps": v_cmd,
        "model": model,
        "source": source,
        "n_episodes": len(scalars),
        "target_speed_mm_s": target_mm_s,
        "fwd_speed_mm_s_mean": float(np.mean(speeds)),
        "fwd_speed_mm_s_std": float(np.std(speeds, ddof=1)) if len(speeds) > 1 else 0.0,
        "cot_mean": float(np.mean(cots)),
        "cot_std": float(np.std(cots, ddof=1)) if len(cots) > 1 else 0.0,
        "distance_mm_mean": float(np.mean(dists)),
        "distance_mm_std": float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
        "eta_locomotion_mean": float(np.mean(etas)),
        "eta_locomotion_std": float(np.std(etas, ddof=1)) if len(etas) > 1 else 0.0,
        "per_ep_speed_mm_s": speeds,
        "per_ep_cot": cots,
        "m3_fre": m3_fre,
        "m4_f_hz": m4_f,
    }


def _save_eval_summary(subdir: str, summary: Dict[str, Any]) -> None:
    path = OUT_ROOT / subdir / "eval_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_one_speed(
    v_cmd: float,
    *,
    models: List[str],
    m2_source: str,
    n_sweep: int,
    skip_calibration: bool,
    m3_fre_override: float | None,
    m4_f_override: float | None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    m2_row = load_m2_row(v_cmd, source=m2_source) if "m2" in models else _load_m2_from_json(v_cmd)
    target_mm_s = m2_row["fwd_speed_mm_s_mean"]
    print(f"\n{'='*60}\n v_cmd={v_cmd:.2f} m/s — M2 reference speed {target_mm_s:.1f} mm/s\n{'='*60}")

    m3_row = m4_row = None
    calib_log: Dict[str, Any] = {"v_cmd": v_cmd, "target_mm_s": target_mm_s, "m3_sweep": [], "m4_sweep": []}

    tag = f"vcmd{v_cmd:.2f}".replace(".", "")

    if "m3" in models:
        fre = m3_fre_override
        if fre is None and not skip_calibration:
            fre, calib_log["m3_sweep"] = _calibrate_m3(target_mm_s, n_sweep)
        elif fre is None:
            fre = rss.M3_FRE
        _, sc3 = rss.run_m3_planner(fre=fre)
        m3_row = _scalars_to_row(
            v_cmd, "M3_planner", sc3,
            target_mm_s=target_mm_s, m3_fre=fre,
        )
        m3_row["source"] = "sim (env-step COT)"
        _save_eval_summary(
            f"m3_planner/{tag}",
            rss.build_eval_summary(
                "M3_planner",
                str(rss.SCRIPT_M3.relative_to(CPG_ROOT.parent)),
                {"v_cmd": v_cmd, "fre": fre, "speed_matched": True, "target_mm_s": target_mm_s},
                sc3,
                notes=f"Multi-vcmd speed-match to M2 @ v_cmd={v_cmd}",
            ),
        )

    if "m4" in models:
        f_hz = m4_f_override
        if f_hz is None and not skip_calibration:
            f_hz, calib_log["m4_sweep"] = _calibrate_m4(target_mm_s, n_sweep)
        elif f_hz is None:
            f_hz = rss.F_M4
        _, sc4 = rss.run_m4_planner_simplified(f_planner=f_hz)
        m4_row = _scalars_to_row(
            v_cmd, "M4_planner_simplified", sc4,
            target_mm_s=target_mm_s, m4_f=f_hz,
        )
        m4_row["source"] = "sim (env-step COT)"
        _save_eval_summary(
            f"m4_planner_simplified/{tag}",
            rss.build_eval_summary(
                "M4_planner_simplified",
                str(rss.SCRIPT_M4.relative_to(CPG_ROOT)),
                {"v_cmd": v_cmd, "f": f_hz, "a": rss.A_M4, "b": rss.B_M4,
                 "speed_matched": True, "target_mm_s": target_mm_s},
                sc4,
                notes=f"Multi-vcmd speed-match to M2 @ v_cmd={v_cmd}",
            ),
        )

    if m2_source == "pilot" and "m2" in models and m2_row.get("source", "").endswith(".json"):
        m2_out = OUT_ROOT / "m2_ours" / f"{tag}_from_pilot.json"
        m2_out.parent.mkdir(parents=True, exist_ok=True)
        m2_out.write_text(json.dumps(m2_row, indent=2), encoding="utf-8")

    calib_path = OUT_ROOT / f"calibration_{tag}.json"
    calib_path.write_text(json.dumps(calib_log, indent=2), encoding="utf-8")

    return m2_row, m3_row, m4_row


def write_comparison_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "v_cmd_mps", "model", "target_speed_mm_s",
        "fwd_speed_mm_s_mean", "fwd_speed_mm_s_std",
        "cot_mean", "cot_std",
        "distance_mm_mean", "distance_mm_std",
        "eta_locomotion_mean", "eta_locomotion_std",
        "m3_fre", "m4_f_hz", "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_readme(all_rows: List[Dict[str, Any]], *, m2_source: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m2_note = (
        "env-step |τ·q̇|/(M·g·d_fwd), per-speed RL ckpt"
        if m2_source == "envstep"
        else "pilot eval JSON (per-speed training)"
    )
    lines = [
        "# Multi-velocity three-way comparison",
        "",
        f"Generated: {ts}",
        "",
        f"**COT pipeline:** M2 — {m2_note}; M3/M4 — env-step, speed-matched to M2 realized speed.",
        "",
        "| v_cmd | Model | Speed (mm/s) | COT | Calib param |",
        "|------:|-------|-------------:|----:|-------------|",
    ]
    for r in all_rows:
        cal = ""
        if r["model"] == "M3_planner" and r.get("m3_fre"):
            cal = f"fre={float(r['m3_fre']):.2f} Hz"
        elif r["model"] == "M4_planner_simplified" and r.get("m4_f_hz"):
            cal = f"f={float(r['m4_f_hz']):.2f} Hz"
        elif r["model"] == "M2_ours":
            cal = "per-speed RL ckpt"
        lines.append(
            f"| {float(r['v_cmd_mps']):.2f} | {r['model']} | "
            f"{float(r['fwd_speed_mm_s_mean']):.1f} ± {float(r['fwd_speed_mm_s_std']):.2f} | "
            f"{float(r['cot_mean']):.2f} ± {float(r['cot_std']):.2f} | {cal} |"
        )
    lines += ["", f"CSV: `{CSV_PATH.name}`"]
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v-cmds", nargs="+", type=float, default=list(V_CMDS))
    parser.add_argument("--models", nargs="+", default=["m3", "m4"], choices=["m2", "m3", "m4"])
    parser.add_argument("--m2-source", choices=["pilot", "envstep"], default="pilot")
    parser.add_argument(
        "--refresh-m2-only",
        action="store_true",
        help="Re-eval M2 only; merge M3/M4 from existing CSV",
    )
    parser.add_argument("--sweep-episodes", type=int, default=5)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    m2_source = "envstep" if args.refresh_m2_only else args.m2_source
    models = list(args.models)
    if args.refresh_m2_only:
        models = ["m2"]

    all_rows: List[Dict[str, Any]] = []
    planner_rows: List[Dict[str, Any]] = []
    if args.refresh_m2_only:
        planner_rows = _load_planner_rows_from_csv(args.v_cmds)

    for v_cmd in args.v_cmds:
        m2, m3, m4 = run_one_speed(
            v_cmd,
            models=models,
            m2_source=m2_source,
            n_sweep=args.sweep_episodes,
            skip_calibration=args.skip_calibration or args.refresh_m2_only,
            m3_fre_override=None,
            m4_f_override=None,
        )
        if "m2" in models:
            all_rows.append(m2)
        if m3 is not None:
            all_rows.append(m3)
        if m4 is not None:
            all_rows.append(m4)

    if args.refresh_m2_only:
        all_rows.extend(planner_rows)
        _model_order = {"M2_ours": 0, "M3_planner": 1, "M4_planner_simplified": 2}
        all_rows.sort(
            key=lambda r: (float(r["v_cmd_mps"]), _model_order.get(r["model"], 9))
        )

    write_comparison_csv(all_rows, CSV_PATH)
    _write_readme(all_rows, m2_source=m2_source)
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {OUT_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
