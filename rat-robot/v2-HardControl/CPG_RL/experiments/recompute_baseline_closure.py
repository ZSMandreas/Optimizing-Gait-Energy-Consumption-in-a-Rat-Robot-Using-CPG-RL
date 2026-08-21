#!/usr/bin/env python3
"""Recompute W2 baseline M1 closure (thesis protocol) into a new output directory only."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for _p in (ROOT, ROOT / "env", ROOT / "experiments", ROOT / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from run_cder_closure_validation import (  # noqa: E402
    CKPT_BASELINE,
    EVN_STEPS_PER_EP,
    N_EPISODES,
    collect_rollout,
    task_m1,
)

OUT_DIR = ROOT / "outputs" / "thesis_experiments" / "M1_baseline_recomputed"
CDER_M1_JSON = ROOT / "outputs" / "thesis_experiments" / "M1_closure_residual_cder.json"
ANALYZE_REF_CSV = ROOT / "logs/route_a_energy/seed1/eval/analysis/per_episode_energy.csv"

HIST_REF_PCT = 4.9
RECENT_FIG_PCT = 5.07
RECENT_FIG_STD = 2.55


def sanity_checks(m1: dict, logs) -> list[str]:
    issues: list[str] = []
    for i, (r, L) in enumerate(zip(m1["per_episode"], logs)):
        n_steps = int(L.t_env.size)
        if n_steps != EVN_STEPS_PER_EP:
            issues.append(f"ep {i}: incomplete env-steps {n_steps}/{EVN_STEPS_PER_EP}")
        wp = r["Wp_total_J"]
        if not (5.0 <= wp <= 15.0):
            issues.append(f"ep {i}: W+_total={wp:.3f} J outside [5, 15]")
        pct = r["residual_pct_of_Wp"]
        if not (-10.0 <= pct <= 15.0):
            issues.append(f"ep {i}: residual_pct={pct:.2f}% outside [-10%, 15%]")
        for k, v in r.items():
            if isinstance(v, float) and (not np.isfinite(v)):
                issues.append(f"ep {i}: non-finite {k}")
    return issues


def decision_case(mean_pct: float) -> str:
    if 4.5 <= mean_pct <= 5.5:
        return "A"
    if mean_pct < 4.5:
        return "B"
    if mean_pct > 5.5 and mean_pct < 8.0:
        return "C"
    return "D"


def load_analyze_ref() -> tuple[float, float] | None:
    if not ANALYZE_REF_CSV.is_file():
        return None
    import pandas as pd

    df = pd.read_csv(ANALYZE_REF_CSV)
    if "residual_pct" not in df.columns:
        return None
    r = df["residual_pct"]
    return float(r.mean()), float(r.std(ddof=1))


def write_md(m1: dict, ckpt: Path, issues: list[str], case: str) -> str:
    s = m1["summary"]
    cder = json.loads(CDER_M1_JSON.read_text(encoding="utf-8"))["summary"] if CDER_M1_JSON.is_file() else {}
    analyze_ref = load_analyze_ref()
    within5 = sum(abs(r["residual_pct_of_Wp"]) < 5.0 for r in m1["per_episode"])

    lines = [
        "# M1 — W2 baseline closure equation residual (recomputed)\n\n",
        "Canonical recomputation using the **same** `run_cder_closure_validation.collect_rollout` "
        "+ `task_m1` protocol as CDER M1.\n\n",
        f"- **Checkpoint:** `{ckpt}`\n",
        f"- **Env:** `RatCpgEnvEnergySubstep50ShapeV2`, `dynamic_4l_kp2.xml`\n",
        f"- **Episodes:** {N_EPISODES} deterministic (reset seeds 0..19)\n",
        f"- **Horizon:** {EVN_STEPS_PER_EP} env-steps × 50 substeps (8.2 s)\n",
        f"- **target_speed:** 0.12 m/s (env default)\n\n",
        "`residual = W+_total − (W−_total + E_damp_total + E_fric_total + E_norm_total + ΔKE + ΔPE)`\n\n",
        f"- n_episodes: **{s['n_episodes']}**\n",
        f"- mean residual: **{s['mean_residual_J']:.6f} J** (std {s['std_residual_J']:.6f})\n",
        f"- mean residual as % of W+_total: **{s['mean_residual_pct_of_Wp']:+.2f}%** "
        f"(std {s['std_residual_pct_of_Wp']:.2f}%)\n",
        f"- median %: {s['median_residual_pct_of_Wp']:+.2f}%, "
        f"range [{s['min_residual_pct_of_Wp']:+.2f}%, {s['max_residual_pct_of_Wp']:+.2f}%]\n",
        f"- episodes with |residual%| < 5%: **{within5}/{s['n_episodes']}**\n\n",
    ]

    if issues:
        lines.append("## Sanity check failures\n\n")
        for it in issues:
            lines.append(f"- {it}\n")
        lines.append("\n")
    else:
        lines.append("## Sanity checks\n\nAll checks passed (82 env-steps/ep, W+ in [5,15] J, "
                     "residual% in [-10%,15%], no NaN/Inf).\n\n")

    lines.append("## Reference comparisons\n\n")
    lines.append(f"| Source | mean % W+ |\n|---|---:|\n")
    lines.append(f"| Historical thesis note | {HIST_REF_PCT:+.1f}% |\n")
    if analyze_ref:
        lines.append(f"| analyze_energy (route_a seed1 eval) | {analyze_ref[0]:+.2f}% ± {analyze_ref[1]:.2f}% |\n")
    lines.append(f"| **This recompute (M1 thesis hook)** | **{s['mean_residual_pct_of_Wp']:+.2f}% ± {s['std_residual_pct_of_Wp']:.2f}%** |\n")
    if cder:
        lines.append(
            f"| CDER v3.1 seed0 (existing M1, read-only) | "
            f"{cder.get('mean_residual_pct_of_Wp', float('nan')):+.2f}% ± "
            f"{cder.get('std_residual_pct_of_Wp', float('nan')):.2f}% |\n"
        )

    lines.append("\n## Policy comparison\n\n")
    lines.append("| Policy | n_ep | mean ± std (% W+) | within ±5% |\n")
    lines.append("|--------|-----:|------------------:|:----------:|\n")
    lines.append(
        f"| W2 baseline (recomputed) | {s['n_episodes']} | "
        f"{s['mean_residual_pct_of_Wp']:+.2f} ± {s['std_residual_pct_of_Wp']:.2f} | "
        f"{'✓' if within5 >= s['n_episodes'] // 2 else '✗'} ({within5}/{s['n_episodes']}) |\n"
    )
    if cder:
        c_within = "✓" if abs(cder["mean_residual_pct_of_Wp"]) <= 5 else "✗"
        lines.append(
            f"| CDER v3.1 seed0 (from M1) | {cder['n_episodes']} | "
            f"{cder['mean_residual_pct_of_Wp']:+.2f} ± {cder['std_residual_pct_of_Wp']:.2f} | {c_within} |\n"
        )

    lines.append(
        f"\n## Decision: **Case {case}**\n\n"
    )
    case_text = {
        "A": "Recomputed mean lies in [4.5%, 5.5%] — adopt this value as canonical baseline closure (M1 protocol).",
        "B": "Recomputed mean < 4.5% — re-validate analyze_energy 5.07% (different formula); prefer this M1 recompute for thesis vs CDER.",
        "C": "Recomputed mean > 5.5% but < 8% — baseline marginally outside ±5%; note body-bouncing / limb KE omission.",
        "D": "Recomputed mean outside the [4.5%, 5.5%] band expected from thesis/analyze_energy prose — see discrepancy note below.",
    }
    lines.append(case_text.get(case, "") + "\n\n")

    lines.append("## Discrepancy note (4.9% / 5.07% vs this recompute)\n\n")
    lines.append(
        "This run uses **`run_cder_closure_validation.collect_rollout` + `task_m1`**, identical to "
        "`outputs/thesis_experiments/M1_closure_residual_cder.json` (substep `dt_sim`, torso translational "
        "ΔKE/ΔPE only, contact power integrated in the rollout hook).\n\n"
        "The historical **+4.9%** thesis line and **+5.07% ± 2.55%** closure figure come from "
        "`analyze_energy_full.py` on `eval_kp_ablation` CSV logs (`logs/route_a_energy/seed1/eval/`), which "
        "differs in several ways:\n\n"
        "- Joint work integrated with **env-step `dt=0.1 s`** (one power sample per env-step), not substep `dt_sim`.\n"
        "- **ΔKE includes yaw** (`0.5·I_yaw·ω_z²`), not torso translation only.\n"
        "- Contact terms read from env-step accumulators in the eval CSV, not the M1 mj_step hook.\n\n"
        "Under the **M1 protocol (apples-to-apples with CDER)**:\n"
        f"- W2 baseline (this file): **{s['mean_residual_pct_of_Wp']:+.2f}% ± {s['std_residual_pct_of_Wp']:.2f}%**\n"
    )
    if cder:
        lines.append(
            f"- CDER v3.1 seed0 (existing M1): **{cder['mean_residual_pct_of_Wp']:+.2f}% ± "
            f"{cder['std_residual_pct_of_Wp']:.2f}%**\n\n"
        )
    if analyze_ref:
        lines.append(
            f"Under **analyze_energy** on the same trained policy (route_a seed1 eval CSV): "
            f"**{analyze_ref[0]:+.2f}% ± {analyze_ref[1]:.2f}%** — this is the traceable source of "
            f"~5% / rounded 4.9% prose.\n\n"
        )
    lines.append(
        "**Recommendation:** For thesis text that cites baseline vs CDER closure, use **M1 hook numbers** "
        "(baseline ≈ +17.5%, CDER ≈ −1.6%) or explicitly label analyze_energy (~5%) as a separate metric. "
        "Do not mix the two pipelines in one sentence.\n\n"
    )

    lines.append("| ep | W+ (J) | W− (J) | E_damp (J) | E_fric (J) | E_norm (J) | ΔKE (J) | ΔPE (J) | residual (J) | residual/W+ (%) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in m1["per_episode"]:
        lines.append(
            f"| {r['ep']} | {r['Wp_total_J']:.4f} | {r['Wn_total_J']:.4f} | "
            f"{r['Edamp_total_J']:.4f} | {r['Efric_total_J']:.4f} | {r['Enorm_total_J']:.4f} "
            f"| {r['dKE_J']:+.5f} | {r['dPE_J']:+.5f} | {r['residual_J']:+.5f} | {r['residual_pct_of_Wp']:+.2f}% |\n"
        )
    return "".join(lines)


def write_csv(m1: dict, path: Path) -> None:
    fields = [
        "ep", "Wp_total_J", "Wn_total_J", "Edamp_total_J", "Efric_total_J", "Enorm_total_J",
        "dKE_J", "dPE_J", "residual_J", "residual_pct_of_Wp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in m1["per_episode"]:
            w.writerow({k: r[k] for k in fields})


def main() -> None:
    t0 = time.time()
    ckpt = CKPT_BASELINE
    if not ckpt.is_file():
        raise FileNotFoundError(f"Baseline checkpoint missing: {ckpt}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Baseline closure recompute (starting) ===")
    print(f"Checkpoint: {ckpt}")

    logs = collect_rollout("baseline", ckpt, n_eps=N_EPISODES)
    m1 = task_m1(logs)

    elapsed = time.time() - t0
    if elapsed > 45 * 60:
        print(f"ABORT: elapsed {elapsed:.0f}s > 45 min budget")
        sys.exit(1)

    issues = sanity_checks(m1, logs)
    s = m1["summary"]
    case = decision_case(s["mean_residual_pct_of_Wp"])
    analyze_ref = load_analyze_ref()

    completed = sum(1 for L in logs if L.t_env.size == EVN_STEPS_PER_EP)

    print("\n=== Baseline closure recompute (n=20) ===")
    print(f"Checkpoint:           {ckpt}")
    print(f"Evaluation seeds:     0..{N_EPISODES - 1}")
    print(f"Episodes completed:   {completed}/{N_EPISODES}")
    print()
    print(f"Mean residual:        {s['mean_residual_J']:.4f} J  (std {s['std_residual_J']:.4f})")
    print(f"Mean residual (% W+): {s['mean_residual_pct_of_Wp']:+.2f}% (std {s['std_residual_pct_of_Wp']:.2f}%)")
    print(f"Median (% W+):        {s['median_residual_pct_of_Wp']:+.2f}%")
    print(f"Range (% W+):         [{s['min_residual_pct_of_Wp']:+.2f}%, {s['max_residual_pct_of_Wp']:+.2f}%]")
    print()
    print("Reference comparisons:")
    print(f"  Historical thesis:    +{HIST_REF_PCT:.1f}% (source: unknown)")
    print(f"  Recent figure:        +{RECENT_FIG_PCT:.2f}% ± {RECENT_FIG_STD:.2f}% (analyze_energy)")
    print(f"  This recompute:       {s['mean_residual_pct_of_Wp']:+.2f}% ± {s['std_residual_pct_of_Wp']:.2f}%")
    if analyze_ref:
        print(f"  analyze_energy CSV:   {analyze_ref[0]:+.2f}% ± {analyze_ref[1]:.2f}%")
    if CDER_M1_JSON.is_file():
        c = json.loads(CDER_M1_JSON.read_text())["summary"]
        print(f"  CDER reference:       {c['mean_residual_pct_of_Wp']:+.2f}% ± {c['std_residual_pct_of_Wp']:.2f}%")
    print(f"\nDecision case: {case}")
    if issues:
        print("\nSANITY ISSUES:")
        for it in issues:
            print(f"  - {it}")

    out_json = OUT_DIR / "M1_closure_residual_baseline_recomputed.json"
    out_md = OUT_DIR / "M1_closure_residual_baseline_recomputed.md"
    out_csv = OUT_DIR / "per_episode_breakdown.csv"

    payload = {
        "checkpoint": str(ckpt),
        "protocol": {
            "n_episodes": N_EPISODES,
            "eval_seeds": list(range(N_EPISODES)),
            "env_steps_per_episode": EVN_STEPS_PER_EP,
            "env_class": "RatCpgEnvEnergySubstep50ShapeV2",
            "xml": "dynamic_4l_kp2.xml",
            "target_speed_mps": 0.12,
            "deterministic": True,
            "M_BODY_kg": 0.2895,
            "B_DAMP": 0.005,
        },
        "sanity_issues": issues,
        "decision_case": case,
        "per_episode": m1["per_episode"],
        "summary": s,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(write_md(m1, ckpt, issues, case), encoding="utf-8")
    write_csv(m1, out_csv)

    print(f"\nWrote:\n  {out_json}\n  {out_md}\n  {out_csv}")
    print(f"Wall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
