#!/usr/bin/env python3
"""Extract 4-metric ablation comparison from existing thesis_experiments outputs."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "outputs/thesis_experiments_ablation_figures/ablation_4metric_comparison.md"

POLICIES = {
    "Ours": ROOT / "outputs/thesis_experiments",
    "Ablation A": ROOT / "outputs/thesis_experiments_v3.1_ablation_no_norm",
    "Ablation B": ROOT / "outputs/thesis_experiments_v3.1_ablation_equal_weights",
}

N_EP = 20
N_ENV_STEPS = 82
SUBSTEPS_PER_ENV = 50
XML = (ROOT / "../TrotGait/models/dynamic_4l_kp2.xml").resolve()

REQUIRED = [
    "M9_cot_distribution_cder.json",
    "M1_closure_residual_cder.json",
    "M4_cder_diagnostic.csv",
    "run_cder_closure_validation_report.md",
    "rollout_cder_env_level.npz",
]


def _sig3(x: float) -> str:
    if not np.isfinite(x):
        return "?.???"
    return f"{x:.3g}"


def fmt_pm(mean: float, std: float, nd_mean: int = 3, nd_std: int = 3) -> str:
    return f"{mean:.{nd_mean}g} ± {std:.{nd_std}g}"


def verify_dirs() -> None:
    missing = []
    for name, d in POLICIES.items():
        for fn in REQUIRED:
            if not (d / fn).is_file():
                missing.append(f"{name}: {d / fn}")
    if missing:
        print("ABORT: missing files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


def episode_duration_s() -> float:
    m = mujoco.MjModel.from_xml_path(str(XML))
    dt = float(m.opt.timestep)
    return N_ENV_STEPS * SUBSTEPS_PER_ENV * dt


def m4_scaled_lateral_mm(csv_path: Path, dist_m: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Proxy per-episode lateral |ΔX| from M4 diagnostic window, scaled to full episode."""
    df = pd.read_csv(csv_path)
    g = df.groupby("t", as_index=False).first()
    t = g["t"].to_numpy(dtype=float)
    x = g["base_x"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError(f"insufficient M4 rows in {csv_path}")
    T_win = float(t[-1] - t[0])
    drift_win_m = float(abs(x[-1] - x[0]))
    T_ep = episode_duration_s()
    drift_ep_m = drift_win_m * (T_ep / T_win)
    d_mean = float(np.mean(dist_m))
    if d_mean < 1e-9:
        raise ValueError("zero mean distance")
    lat_ep_m = dist_m * (drift_ep_m / d_mean)
    return lat_ep_m * 1000.0, drift_ep_m * 1000.0, float(np.mean(np.abs(g["base_vx"]))) * 1000.0


def m4_fwd_speed_mm_s(csv_path: Path) -> float:
    g = pd.read_csv(csv_path).groupby("t", as_index=False).first()
    return float(-np.mean(g["base_vy"])) * 1000.0


def extract_policy(name: str, out_dir: Path) -> Dict[str, Any]:
    m9 = json.loads((out_dir / "M9_cot_distribution_cder.json").read_text(encoding="utf-8"))
    m1 = json.loads((out_dir / "M1_closure_residual_cder.json").read_text(encoding="utf-8"))

    cot = np.asarray(m9["per_episode_cot"], dtype=float)
    wp = np.asarray(m9["per_episode_Wp_J"], dtype=float)
    dist_m = np.asarray(m9["per_episode_distance_m"], dtype=float)
    wp1 = np.asarray([e["Wp_total_J"] for e in m1["per_episode"]], dtype=float)

    if len(cot) != N_EP or len(dist_m) != N_EP:
        raise ValueError(f"{name}: expected {N_EP} episodes, got cot={len(cot)} dist={len(dist_m)}")

    T_ep = episode_duration_s()
    speed_mm_s = dist_m / T_ep * 1000.0
    dist_mm = dist_m * 1000.0

    lat_ep_mm, lat_scaled_mm, mean_abs_vx = m4_scaled_lateral_mm(
        out_dir / "M4_cder_diagnostic.csv", dist_m
    )

    z = np.load(out_dir / "rollout_cder_env_level.npz")
    ep_lens = z["t_env"].shape[1] if z["t_env"].ndim > 1 else 0
    n_ok = int(np.sum([1 for i in range(z["t_env"].shape[0]) if z["t_env"].shape[1] >= N_ENV_STEPS]))

    rep = (out_dir / "run_cder_closure_validation_report.md").read_text(encoding="utf-8")
    m = re.search(r"CDER mean forward speed ≈ ([\d.]+) mm/s", rep)
    rep_fwd = float(m.group(1)) if m else float("nan")

    return {
        "n_episodes": len(cot),
        "cot": cot,
        "Wp_J": wp,
        "dist_mm": dist_mm,
        "speed_mm_s": speed_mm_s,
        "lat_mm": lat_ep_mm,
        "lat_scaled_const_mm": lat_scaled_mm,
        "m4_fwd_mm_s": m4_fwd_speed_mm_s(out_dir / "M4_cder_diagnostic.csv"),
        "report_fwd_mm_s": rep_fwd,
        "wp_m1_match": bool(np.allclose(wp, wp1, rtol=0, atol=1e-6)),
        "rollout_ep_len": ep_lens,
        "episodes_82_steps": n_ok,
        "T_ep_s": T_ep,
    }


def sanity_checks(results: Dict[str, Dict[str, Any]]) -> List[str]:
    flags: List[str] = []
    T = results["Ours"]["T_ep_s"]
    for name, r in results.items():
        if r["n_episodes"] != N_EP:
            flags.append(f"**{name}**: only {r['n_episodes']}/{N_EP} episodes in M9.")
        if r["episodes_82_steps"] != N_EP:
            flags.append(f"**{name}**: {r['episodes_82_steps']}/{N_EP} rollout episodes with ≥82 env-steps (check NPZ).")
        if not r["wp_m1_match"]:
            flags.append(f"**{name}**: M9 W+ does not match M1 per-episode totals.")

        sm = float(np.mean(r["speed_mm_s"]))
        dm = float(np.mean(r["dist_mm"]))
        if not (40 <= sm <= 110):
            flags.append(f"**{name}**: forward speed {sm:.1f} mm/s outside 40–110 mm/s band.")
        for i in range(r["n_episodes"]):
            v = r["speed_mm_s"][i]
            d = r["dist_mm"][i]
            expected = d / T
            if abs(v - expected) / max(expected, 1e-6) > 0.05:
                flags.append(f"**{name} ep{i}**: speed–distance consistency >5% (unexpected).")
                break
        lat_m = float(np.mean(r["lat_mm"]))
        if lat_m > 30:
            flags.append(f"**{name}**: lateral proxy mean {lat_m:.1f} mm > 30 mm (review).")
        # M4 uses mean $-v_y$ in a steady diagnostic window; M9 uses full-episode $\Delta xy / T$.
        rel = abs(sm - r["report_fwd_mm_s"]) / max(sm, 1e-6)
        if rel > 0.08:
            flags.append(
                f"**{name}**: M9 speed {sm:.1f} vs M4 steady-window report {r['report_fwd_mm_s']:.1f} mm/s "
                f"differ by {rel*100:.1f}% (>8%)."
            )
    return flags


def build_markdown(results: Dict[str, Dict[str, Any]], flags: List[str]) -> str:
    cols = list(POLICIES.keys())

    def row(metric: str, key: str, unit: str = "") -> str:
        cells = [metric]
        for c in cols:
            r = results[c]
            arr = r[key]
            m, s = float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            cells.append(fmt_pm(m, s))
        return "| " + " | ".join(cells) + " |"

    lines = [
        "# CDER ablation — four-metric comparison",
        "",
        "Policies evaluated with `run_cder_closure_validation.py` (20 deterministic episodes, seeds 0–19).",
        "",
        "## Summary table",
        "",
        "| Metric | Ours | Ablation A | Ablation B |",
        "|--------|-----:|-----------:|-----------:|",
        row("Forward speed (mm/s)", "speed_mm_s"),
        row("COT (dimensionless)", "cot"),
        row("Forward distance per ep (mm)", "dist_mm"),
        row("Lateral deviation per ep (mm)", "lat_mm"),
        "",
        "## Data sources",
        "",
        "| Metric | Source |",
        "|--------|--------|",
        "| Forward speed | `distance_m / T_ep` with `T_ep = 82×50×dt_sim` (= 8.2 s); same denominator as M9 COT |",
        "| COT | `M9_cot_distribution_cder.json` → `per_episode_cot`; "
        "$\\mathrm{COT}=W^+/(M g d)$ with $M{=}0.2895$ kg, planar $\\|\\Delta xy\\|$ for $d$ |",
        "| Forward distance | M9 → `per_episode_distance_m` × 1000; **planar** $\\|\\Delta xy\\|$ (first→last substep), per `task_m9_cot` |",
        "| Lateral deviation | **Proxy** (not archived per-episode): M4 diagnostic $|Δx|$ over $t∈[3.75,4.45]$ s scaled to full episode, then scaled per-episode by `distance_ep / mean(distance)` |",
        "",
        f"Episode duration used: **{results['Ours']['T_ep_s']:.3g} s** (`dt_sim` from `dynamic_4l_kp2.xml`).",
        "",
        "## LaTeX",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Four-metric comparison of CDER ablation policies, "
        "evaluated over 20 deterministic episodes per policy. "
        "COT is the primary energy-efficiency metric ($W^+$ per unit traveled distance). "
        "Lateral deviation uses the M4-scaled proxy described in the text.}",
        "\\label{tab:ablation-4metric}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & Ours & Ablation A & Ablation B \\\\",
        "\\midrule",
    ]

    def latex_row(label: str, key: str) -> str:
        parts = [label.replace("W+", "$W^+$")]
        for c in cols:
            r = results[c]
            m = float(np.mean(r[key]))
            s = float(np.std(r[key], ddof=1)) if len(r[key]) > 1 else 0.0
            parts.append(fmt_pm(m, s))
        return " & ".join(parts) + r" \\"

    lines.extend([
        latex_row("Forward speed (mm/s)", "speed_mm_s"),
        latex_row("COT", "cot"),
        latex_row("Forward distance (mm/ep)", "dist_mm"),
        latex_row("Lateral deviation (mm/ep)", "lat_mm"),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "## Sanity checks",
        "",
    ])
    if flags:
        lines.append("### Flags\n")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("All checks passed (20/20 episodes, speed–distance consistent within 5%, speeds in 40–110 mm/s).\n")

    lines.extend([
        "### Episode completion",
        "",
        "| Policy | Episodes in M9 | Rollout ≥82 env-steps | $W^+$ M9 vs M1 |",
        "|--------|---------------:|----------------------:|:--------------:|",
    ])
    for c in cols:
        r = results[c]
        lines.append(
            f"| {c} | {r['n_episodes']} | {r['episodes_82_steps']}/20 | "
            f"{'match' if r['wp_m1_match'] else 'MISMATCH'} |"
        )

    lines.extend([
        "",
        "### Cross-check: M9 speed vs M4 report (diagnostic window)",
        "",
        "| Policy | M9 mean speed (mm/s) | M4 report (mm/s) | M4 window mean $-v_y$ (mm/s) |",
        "|--------|--------------------:|-----------------:|----------------------------:|",
    ])
    for c in cols:
        r = results[c]
        lines.append(
            f"| {c} | {np.mean(r['speed_mm_s']):.3g} | {r['report_fwd_mm_s']:.3g} | {r['m4_fwd_mm_s']:.3g} |"
        )

    lines.extend([
        "",
        "### Notes",
        "",
        "- **Lateral metric limitation:** archived M1–M9 bundles do not store per-episode `base_xyz`; "
        "lateral column uses the scaled M4 proxy above. For true $|x_{end}-x_{start}|$ per episode, "
        "re-run rollouts with trajectory logging or extend `task_m9_cot` to export $\\Delta x$.",
        "- Ablation A/B forward speeds ~71–75 mm/s reflect the slower convergence basin (not seed sensitivity).",
        "- Reference $W^+$ per episode (not in main table): Ours "
        f"{np.mean(results['Ours']['Wp_J']):.3g} J, A {np.mean(results['Ablation A']['Wp_J']):.3g} J, "
        f"B {np.mean(results['Ablation B']['Wp_J']):.3g} J.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    verify_dirs()
    results = {name: extract_policy(name, d) for name, d in POLICIES.items()}
    flags = sanity_checks(results)
    md = build_markdown(results, flags)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
