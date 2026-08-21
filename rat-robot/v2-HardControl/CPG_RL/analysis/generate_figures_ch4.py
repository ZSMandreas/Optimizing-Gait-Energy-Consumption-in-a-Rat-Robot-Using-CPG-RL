#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Internal keys for CSV paths; plot labels are English display names only.
KEY_BASELINE = "baseline"
KEY_OURS = "ours"

DISPLAY_NAME: Dict[str, str] = {
    KEY_BASELINE: "Baseline",
    KEY_OURS: "Ours",
}

METHOD_ORDER: List[str] = [KEY_BASELINE, KEY_OURS]

# High-contrast pair typical in robotics venue figures
COLORS = {
    "Baseline": "#0072B2",
    "Ours": "#D55E00",
}

N_ENV_STEPS = 82
T_HORIZON_S = 8.2
DT = 0.1

# IROS / IEEE-robotics-style matplotlib (sans-serif, clean spines, print-friendly)
IROS_RC = {
    "figure.figsize": (3.6, 2.35),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Nimbus Sans", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Distinct, colorblind-friendly stack (Wong-style inspired)
COMPONENT_COLORS = {
    "W_pos": "#0072B2",
    "W_neg": "#E69F00",
    "E_damp": "#009E73",
    "E_fric": "#CC79A7",
    "E_norm": "#56B4E9",
}

# Short legend text so a below-axes legend does not crowd the plot
COMPONENT_LEGEND = {
    "W_pos": r"$W^+$",
    "W_neg": r"$W^-$",
    "E_damp": "Damping",
    "E_fric": "Friction",
    "E_norm": "Normal",
}


def resolve_output_dir() -> Path:
    out = Path(__file__).resolve().parent / "outputs" / "w3" / "figures_ch4"
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_columns(df: pd.DataFrame, required: List[str], label: str, anomalies: List[str]) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        anomalies.append(f"{label}: missing columns {missing}")
        return False
    return True


def per_leg_neg_frac(df: pd.DataFrame, anomalies: List[str], label: str) -> Dict[str, float]:
    legs = {"FL": (0, 1), "FR": (2, 3), "RL": (4, 5), "RR": (6, 7)}
    out: Dict[str, float] = {}
    for leg, (j1, j2) in legs.items():
        needed = [f"tau_{j1}", f"qvel_{j1}", f"tau_{j2}", f"qvel_{j2}"]
        if not check_columns(df, needed, f"{label} ({leg})", anomalies):
            out[leg] = float("nan")
            continue
        p_leg = df[f"tau_{j1}"] * df[f"qvel_{j1}"] + df[f"tau_{j2}"] * df[f"qvel_{j2}"]
        out[leg] = float((p_leg < 0).mean())
    return out


def extract_trajectory(
    df: pd.DataFrame,
    label: str,
    anomalies: List[str],
    episode: int = 0,
    max_steps: int = N_ENV_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    if not check_columns(df, ["episode", "step", "base_vx", "base_vy"], f"{label} trajectory", anomalies):
        return np.array([]), np.array([])
    d = df[df["episode"] == episode].sort_values("step")
    if d.empty:
        anomalies.append(f"{label}: no rows for episode={episode}, using first available episode")
        first_episode = int(df["episode"].min())
        d = df[df["episode"] == first_episode].sort_values("step")
    if max_steps is not None and len(d) > max_steps:
        d = d.iloc[:max_steps]
    x_lateral = np.cumsum(d["base_vx"].to_numpy()) * DT
    y_forward = -np.cumsum(d["base_vy"].to_numpy()) * DT
    return x_lateral, y_forward


def _savefig_iros(path: Path) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="none")


def main() -> None:
    plt.rcParams.update(IROS_RC)
    root = Path(__file__).resolve().parent
    anomalies: List[str] = []

    methods_82_energy = {
        KEY_BASELINE: root / "logs/w2_baseline/eval_82steps_correct/seed0/analysis/per_episode_energy.csv",
        KEY_OURS: root / "logs/w3_ours_v3.1/seed0/eval_82steps/analysis/per_episode_energy.csv",
    }
    methods_82_step = {
        KEY_BASELINE: root / "logs/w2_baseline/eval_82steps_correct/seed0/kp2_kv0/step_metrics.csv",
        KEY_OURS: root / "logs/w3_ours_v3.1/seed0/eval_82steps/kp2_kv0/step_metrics.csv",
    }

    output_dir = resolve_output_dir()

    energy_frames: Dict[str, pd.DataFrame] = {}
    for label, path in methods_82_energy.items():
        if not path.exists():
            anomalies.append(f"Missing file: {path}")
            continue
        energy_frames[label] = pd.read_csv(path)

    step82_frames: Dict[str, pd.DataFrame] = {}
    for label, path in methods_82_step.items():
        if not path.exists():
            anomalies.append(f"Missing file: {path}")
            continue
        step82_frames[label] = pd.read_csv(path)

    # Figure 1: component breakdown (8.2 s horizon via T_HORIZON_S)
    comp_data: Dict[str, Dict[str, float]] = {}
    comp_cols = ["W_pos_J", "W_neg_J", "E_damp_J", "E_contact_friction_J", "E_contact_normal_J"]
    for label, df in energy_frames.items():
        if not check_columns(df, comp_cols, f"{DISPLAY_NAME.get(label, label)} energy components", anomalies):
            continue
        comp_data[label] = {
            "W_pos": float(df["W_pos_J"].mean() / T_HORIZON_S),
            "W_neg": float(df["W_neg_J"].mean() / T_HORIZON_S),
            "E_damp": float(df["E_damp_J"].mean() / T_HORIZON_S),
            "E_fric": float(df["E_contact_friction_J"].mean() / T_HORIZON_S),
            "E_norm": float(df["E_contact_normal_J"].mean() / T_HORIZON_S),
        }
    if comp_data:
        methods_keys = [m for m in METHOD_ORDER if m in comp_data]
        y_labels = [DISPLAY_NAME[m] for m in methods_keys]
        y_pos = np.arange(len(methods_keys))
        components = ["W_pos", "W_neg", "E_damp", "E_fric", "E_norm"]
        fig, ax = plt.subplots(figsize=(4.0, 2.35))
        left = np.zeros(len(methods_keys))
        for c in components:
            vals = np.array([comp_data[m][c] for m in methods_keys])
            ax.barh(
                y_pos,
                vals,
                left=left,
                height=0.55,
                color=COMPONENT_COLORS[c],
                label=COMPONENT_LEGEND[c],
                edgecolor="#222222",
                linewidth=0.35,
            )
            left += vals
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis="y", pad=6)
        ax.set_xlabel(r"Power (J/s)")
        ax.set_ylabel("Method")
        ax.set_title(f"Energy components ({T_HORIZON_S:g} s, {N_ENV_STEPS} steps)")
        xmax = float(left.max()) if left.size else 1.0
        ax.set_xlim(0.0, xmax * 1.18)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.38),
            ncol=3,
            frameon=True,
            columnspacing=0.9,
            handletextpad=0.45,
            fontsize=7.5,
        )
        if KEY_BASELINE in comp_data and KEY_OURS in comp_data:
            total_baseline = sum(comp_data[KEY_BASELINE].values())
            total_ours = sum(comp_data[KEY_OURS].values())
            if total_baseline > 0:
                reduction = (total_baseline - total_ours) / total_baseline * 100.0
                iy = y_labels.index(DISPLAY_NAME[KEY_OURS])
                y_coord = float(y_pos[iy])
                ax.text(
                    total_ours + 0.02 * xmax,
                    y_coord,
                    f"-{reduction:.0f}% vs Baseline",
                    va="center",
                    fontsize=8,
                )
        fig.subplots_adjust(left=0.20, bottom=0.40, right=0.98, top=0.88)
        _savefig_iros(output_dir / "fig_4_5_component_breakdown.png")
        plt.close()
    else:
        anomalies.append("Figure 1 skipped: insufficient component data.")

    # Figure 2: per-leg symmetry (same 82-step rollouts)
    per_leg = {label: per_leg_neg_frac(df, anomalies, DISPLAY_NAME.get(label, label)) for label, df in step82_frames.items()}
    if per_leg:
        legs = ["FL", "FR", "RL", "RR"]
        methods_keys = [m for m in METHOD_ORDER if m in per_leg]
        x = np.arange(len(legs))
        n_methods = len(methods_keys)
        bar_width = 0.25 if n_methods >= 3 else 0.36
        fig, ax = plt.subplots()
        for i, m in enumerate(methods_keys):
            offset = (i - n_methods / 2 + 0.5) * bar_width
            vals = [per_leg[m].get(leg, np.nan) for leg in legs]
            disp = DISPLAY_NAME[m]
            ax.bar(
                x + offset,
                vals,
                bar_width,
                label=disp,
                color=COLORS.get(disp, "#777777"),
                edgecolor="#222222",
                linewidth=0.4,
            )
        ax.axhline(
            y=0.25,
            linestyle="--",
            color="#666666",
            alpha=0.9,
            linewidth=0.9,
            label="Trot ref. ($\\approx$0.25)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(legs)
        ax.set_ylim(0.0, 0.7)
        ax.set_xlabel("Leg")
        ax.set_ylabel("Frac. timesteps braking")
        ax.set_title(f"Per-leg braking ({T_HORIZON_S:g} s, {N_ENV_STEPS} steps)")
        ax.legend(loc="upper left", frameon=True)
        rr_idx = legs.index("RR")
        ax.text(
            rr_idx,
            0.53,
            "RR anchor\n(W1)",
            ha="center",
            fontsize=8,
            alpha=0.85,
        )
        ax.axvspan(rr_idx - 0.45, rr_idx + 0.45, facecolor="none", edgecolor="#222222", linewidth=0.6, linestyle=":")
        plt.tight_layout()
        _savefig_iros(output_dir / "fig_4_5_per_leg_symmetry.png")
        plt.close()
    else:
        anomalies.append("Figure 2 skipped: no step-level data available.")

    # Figure 3: drift trajectories — same 82-step / 8.2 s data only
    traj_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, df in step82_frames.items():
        x_lateral, y_forward = extract_trajectory(
            df, DISPLAY_NAME.get(label, label), anomalies, episode=0, max_steps=N_ENV_STEPS
        )
        if x_lateral.size and y_forward.size:
            traj_data[label] = (x_lateral, y_forward)
    if traj_data:
        fig, ax = plt.subplots()
        for label in METHOD_ORDER:
            if label not in traj_data:
                continue
            x_lateral, y_forward = traj_data[label]
            disp = DISPLAY_NAME[label]
            color = COLORS.get(disp, "#777777")
            ax.plot(
                y_forward,
                x_lateral,
                label=f"{disp} ({N_ENV_STEPS} steps)",
                color=color,
                solid_capstyle="round",
            )
            ax.scatter([y_forward[0]], [x_lateral[0]], color=color, s=18, zorder=5, edgecolors="#222222", linewidths=0.35)
            if len(y_forward) > 1:
                ax.annotate(
                    "",
                    xy=(y_forward[-1], x_lateral[-1]),
                    xytext=(y_forward[-2], x_lateral[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=8),
                )
        ax.axhline(0.0, linestyle=":", color="#888888", alpha=0.9, linewidth=0.9, label="Straight path")
        ax.set_xlabel("Forward $Y$ (m)")
        ax.set_ylabel("Lateral $X$ (m)")
        ax.set_title(f"Trajectory ({T_HORIZON_S:g} s, {N_ENV_STEPS} steps)")
        ax.legend(loc="upper left", frameon=True)
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        _savefig_iros(output_dir / "fig_4_5_drift_trajectories.png")
        plt.close()
    else:
        anomalies.append("Figure 3 skipped: insufficient 82-step trajectory data.")

    # Figure 4: COT vs speed (per-episode from 8.2 s eval CSV)
    cot_ready = False
    fig, ax = plt.subplots()
    ax.axhspan(5.0, 10.0, color="#dddddd", alpha=0.55, zorder=0, linewidth=0)
    ax.text(
        0.02,
        7.35,
        "Biological COT band\n(Heglund & Taylor, 1988)",
        ha="left",
        fontsize=7.5,
        alpha=0.85,
    )
    for label in METHOD_ORDER:
        if label not in energy_frames:
            continue
        df = energy_frames[label]
        disp = DISPLAY_NAME[label]
        if not check_columns(df, ["distance_m", "COT_mech_true"], f"{disp} COT scatter", anomalies):
            continue
        speeds = df["distance_m"].to_numpy() / T_HORIZON_S
        cots = df["COT_mech_true"].to_numpy()
        color = COLORS.get(disp, "#777777")
        ax.scatter(
            speeds,
            cots,
            label=disp,
            color=color,
            alpha=0.85,
            s=28,
            edgecolors="#222222",
            linewidths=0.35,
            zorder=3,
        )
        ax.scatter(
            [float(np.mean(speeds))],
            [float(np.mean(cots))],
            color=color,
            marker="*",
            s=140,
            edgecolor="#222222",
            linewidth=0.45,
            zorder=5,
        )
        if label == KEY_OURS:
            ax.annotate(
                f"Ours mean: {np.mean(cots):.2f}",
                xy=(float(np.mean(speeds)), float(np.mean(cots))),
                xytext=(0.11, float(np.mean(cots)) + 0.75),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, mutation_scale=7),
                fontsize=8,
            )
        cot_ready = True
    if cot_ready:
        ax.set_xlabel("Forward speed (m/s)")
        ax.set_ylabel(r"$\mathrm{COT}_{\mathrm{mech,true}}$")
        ax.set_title(f"COT vs speed ({T_HORIZON_S:g} s episodes)")
        ax.legend(loc="upper right", frameon=True)
        _savefig_iros(output_dir / "fig_4_5_cot_vs_speed.png")
        plt.close()
    else:
        plt.close()
        anomalies.append("Figure 4 skipped: no COT/speed data available.")

    print(f"Output directory: {output_dir}")
    print("Generated files:")
    for name in [
        "fig_4_5_component_breakdown.png",
        "fig_4_5_per_leg_symmetry.png",
        "fig_4_5_drift_trajectories.png",
        "fig_4_5_cot_vs_speed.png",
    ]:
        p = output_dir / name
        print(f"  - {name}: {'OK' if p.exists() else 'MISSING'}")
    print("\nData anomalies:")
    if anomalies:
        for a in anomalies:
            print(f"  - {a}")
    else:
        print("  - None")


if __name__ == "__main__":
    main()
