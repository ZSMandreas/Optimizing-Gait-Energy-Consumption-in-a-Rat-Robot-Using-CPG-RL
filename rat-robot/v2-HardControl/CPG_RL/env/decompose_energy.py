#!/usr/bin/env python3
"""
Energy decomposition from step_metrics.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JOINT_NAMES = [
    "thigh_fl", "leg_fl",
    "thigh_fr", "leg_fr",
    "thigh_rl", "leg_rl",
    "thigh_rr", "leg_rr",
]
N_JOINTS = len(JOINT_NAMES)
DT = 0.1
TAU_MAX = 0.157
JOINT_DAMPING = 0.005

LEG_IDX = {"fl": [0, 1], "fr": [2, 3], "rl": [4, 5], "rr": [6, 7]}
FRONT_IDX = [0, 1, 2, 3]
REAR_IDX = [4, 5, 6, 7]
LEFT_IDX = [0, 1, 4, 5]
RIGHT_IDX = [2, 3, 6, 7]


def decompose(df: pd.DataFrame, dt: float = DT, tau_max: float = TAU_MAX):
    tau = df[[f"tau_{i}" for i in range(N_JOINTS)]].to_numpy()
    qvel = df[[f"qvel_{i}" for i in range(N_JOINTS)]].to_numpy()
    p = tau * qvel
    return {
        "p": p,
        "w_pos": np.maximum(p, 0.0) * dt,
        "w_neg": np.maximum(-p, 0.0) * dt,
        "iv2": (qvel ** 2) * dt,
        "it2": (tau ** 2) * dt,
        "sat": (np.abs(tau) >= 0.999 * tau_max).astype(float),
    }


def per_episode(df: pd.DataFrame, parts: dict) -> pd.DataFrame:
    rows = []
    for ep, g in df.groupby("episode"):
        idx = g.index.to_numpy()
        w_pos = float(parts["w_pos"][idx].sum())
        w_neg = float(parts["w_neg"][idx].sum())
        iv2 = float(parts["iv2"][idx].sum())
        it2 = float(parts["it2"][idx].sum())
        e_damp = float(JOINT_DAMPING * iv2)
        sat_frac = float(parts["sat"][idx].mean())
        steps = int(len(idx))
        distance = float((g["fwd_vel_mps"].to_numpy() * DT).sum())
        e_total_internal = w_pos + w_neg + e_damp
        rows.append(
            {
                "episode": ep,
                "steps": steps,
                "duration_s": steps * DT,
                "distance_m": distance,
                "mean_speed_mps": float(g["fwd_vel_mps"].mean()),
                "W_pos_J": w_pos,
                "W_neg_J": w_neg,
                "E_damp_J": e_damp,
                "abs_work_J": w_pos + w_neg,
                "I_v2": iv2,
                "I_tau2": it2,
                "sat_frac": sat_frac,
                "neg_frac": w_neg / max(w_pos + w_neg, 1e-9),
                "damp_frac": e_damp / max(e_total_internal, 1e-9),
                "jpm_mech": (w_pos + w_neg) / max(distance, 1e-6),
                "jpm_total": e_total_internal / max(distance, 1e-6),
            }
        )
    return pd.DataFrame(rows).set_index("episode")


def per_joint(parts: dict) -> pd.DataFrame:
    iv2_per = parts["iv2"].sum(axis=0)
    return pd.DataFrame(
        {
            "joint": JOINT_NAMES,
            "W_pos_J": parts["w_pos"].sum(axis=0),
            "W_neg_J": parts["w_neg"].sum(axis=0),
            "E_damp_J": JOINT_DAMPING * iv2_per,
            "I_v2": iv2_per,
            "I_tau2": parts["it2"].sum(axis=0),
            "sat_frac": parts["sat"].mean(axis=0),
        }
    )


def per_leg(parts: dict) -> pd.DataFrame:
    rows = []
    for leg, idx in LEG_IDX.items():
        w_pos = float(parts["w_pos"][:, idx].sum())
        w_neg = float(parts["w_neg"][:, idx].sum())
        rows.append(
            {
                "leg": leg,
                "W_pos_J": w_pos,
                "W_neg_J": w_neg,
                "E_damp_J": float(JOINT_DAMPING * parts["iv2"][:, idx].sum()),
                "neg_frac": w_neg / max(w_pos + w_neg, 1e-9),
            }
        )
    return pd.DataFrame(rows)


def front_rear_lr(parts: dict) -> pd.DataFrame:
    rows = []
    for label, idx in [("front", FRONT_IDX), ("rear", REAR_IDX), ("left", LEFT_IDX), ("right", RIGHT_IDX)]:
        w_pos = float(parts["w_pos"][:, idx].sum())
        w_neg = float(parts["w_neg"][:, idx].sum())
        rows.append(
            {
                "group": label,
                "W_pos_J": w_pos,
                "W_neg_J": w_neg,
                "E_damp_J": float(JOINT_DAMPING * parts["iv2"][:, idx].sum()),
                "neg_frac": w_neg / max(w_pos + w_neg, 1e-9),
            }
        )
    return pd.DataFrame(rows)


def per_phase(df: pd.DataFrame, parts: dict) -> pd.DataFrame:
    rows = []
    for leg, idx in LEG_IDX.items():
        thigh_idx = idx[0]
        thigh_q = df[f"qpos_{thigh_idx}"].to_numpy()
        ep = df["episode"].to_numpy()
        ep_means = pd.Series(thigh_q).groupby(ep).transform("mean").to_numpy()
        is_high = thigh_q > ep_means
        for label, mask in [("phase_high", is_high), ("phase_low", ~is_high)]:
            rows.append(
                {
                    "leg": leg,
                    "phase": label,
                    "frac_steps": float(mask.mean()),
                    "W_pos_J": float(parts["w_pos"][mask][:, idx].sum()),
                    "W_neg_J": float(parts["w_neg"][mask][:, idx].sum()),
                    "E_damp_J": float(JOINT_DAMPING * parts["iv2"][mask][:, idx].sum()),
                }
            )
    return pd.DataFrame(rows)


def fig_episode_decomp(ep_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    eps = ep_df.index.values
    w_pos = ep_df["W_pos_J"].values
    w_neg = ep_df["W_neg_J"].values
    e_damp = ep_df["E_damp_J"].values
    ax.bar(eps, w_pos, label="W+ (driving)", color="#2b8cbe")
    ax.bar(eps, w_neg, bottom=w_pos, label="W- (braking)", color="#e34a33")
    ax.bar(eps, e_damp, bottom=w_pos + w_neg, label="E_damp", color="#fdae61")
    ax.set_xlabel("episode")
    ax.set_ylabel("internal energy budget (J)")
    ax.set_title("Per-episode energy decomposition")
    ax.set_xticks(eps)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_neg_frac(ep_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    eps = ep_df.index.values
    ax.bar(eps, ep_df["neg_frac"].values * 100, color="#e34a33", alpha=0.85)
    ax.axhline(50, color="k", linestyle="--", lw=1, label="50%")
    ax.set_xlabel("episode")
    ax.set_ylabel("W- / (W+ + W-)  (%)")
    ax.set_title("Braking work fraction")
    ax.set_xticks(eps)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_per_joint(joint_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [
        ("W_pos_J", "W+ per joint (J)"),
        ("W_neg_J", "W- per joint (J)"),
        ("E_damp_J", "E_damp per joint (J)"),
        ("I_tau2", "Integral tau^2 dt"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        colors = ["#2b8cbe" if "thigh" in n else "#74a9cf" for n in joint_df["joint"]]
        ax.bar(range(N_JOINTS), joint_df[key].values, color=colors)
        ax.set_xticks(range(N_JOINTS))
        ax.set_xticklabels(joint_df["joint"].values, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-joint decomposition", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_power_timeseries(df: pd.DataFrame, parts: dict, episode: int, out_path: Path):
    g = df[df["episode"] == episode]
    idx = g.index.to_numpy()
    p = parts["p"][idx]
    t = g["time_s"].to_numpy()
    p_total = p.sum(axis=1)
    p_pos_total = np.maximum(p, 0).sum(axis=1)
    p_neg_total = np.maximum(-p, 0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, p_pos_total, label="sum max(tau*qvel,0)", color="#2b8cbe")
    ax.plot(t, -p_neg_total, label="-sum max(-tau*qvel,0)", color="#e34a33")
    ax.plot(t, p_total, label="net sum tau*qvel", color="black", lw=1, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (W)")
    ax.set_title(f"Episode {episode} power")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_saturation_vs_power(df: pd.DataFrame, parts: dict, out_path: Path):
    sat_frac_step = parts["sat"].mean(axis=1)
    abs_p_step = np.abs(parts["p"]).sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sat_frac_step, abs_p_step, s=8, alpha=0.4, color="#2b8cbe")
    ax.set_xlabel("fraction of joints at saturation")
    ax.set_ylabel("sum_j |tau*qvel| (W)")
    ax.set_title("Saturation vs instantaneous power")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_per_leg(leg_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    legs = leg_df["leg"].values
    x = np.arange(len(legs))
    w_pos = leg_df["W_pos_J"].values
    w_neg = leg_df["W_neg_J"].values
    e_damp = leg_df["E_damp_J"].values
    ax.bar(x, w_pos, label="W+", color="#2b8cbe")
    ax.bar(x, w_neg, bottom=w_pos, label="W-", color="#e34a33")
    ax.bar(x, e_damp, bottom=w_pos + w_neg, label="E_damp", color="#fdae61")
    ax.set_xticks(x)
    ax.set_xticklabels(legs)
    ax.set_ylabel("internal energy (J)")
    ax.set_title("Per-leg decomposition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_front_rear_lr(g_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = g_df["group"].values
    x = np.arange(len(labels))
    w_pos = g_df["W_pos_J"].values
    w_neg = g_df["W_neg_J"].values
    e_damp = g_df["E_damp_J"].values
    ax.bar(x, w_pos, label="W+", color="#2b8cbe")
    ax.bar(x, w_neg, bottom=w_pos, label="W-", color="#e34a33")
    ax.bar(x, e_damp, bottom=w_pos + w_neg, label="E_damp", color="#fdae61")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("internal energy (J)")
    ax.set_title("Front/rear and left/right split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_per_phase(phase_df: pd.DataFrame, out_path: Path):
    legs = ["fl", "fr", "rl", "rr"]
    width = 0.38
    x = np.arange(len(legs))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, key, title, color_pair in zip(
        axes,
        ["W_pos_J", "W_neg_J", "E_damp_J"],
        ["W+ by leg-phase", "W- by leg-phase", "E_damp by leg-phase"],
        [("#2b8cbe", "#a6cee3"), ("#e34a33", "#fdcc8a"), ("#fdae61", "#fee0b6")],
    ):
        high_vals = [phase_df[(phase_df["leg"] == l) & (phase_df["phase"] == "phase_high")][key].iloc[0] for l in legs]
        low_vals = [phase_df[(phase_df["leg"] == l) & (phase_df["phase"] == "phase_low")][key].iloc[0] for l in legs]
        ax.bar(x - width / 2, high_vals, width, label="phase_high", color=color_pair[0])
        ax.bar(x + width / 2, low_vals, width, label="phase_low", color=color_pair[1])
        ax.set_xticks(x)
        ax.set_xticklabels(legs)
        ax.set_ylabel("J")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Energy split by heuristic leg phase", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="path to step_metrics.csv")
    ap.add_argument("--out", dest="out_dir", required=True, help="output directory")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path).reset_index(drop=True)
    print(f"{len(df)} rows, {df['episode'].nunique()} episodes")

    parts = decompose(df)
    ep_df = per_episode(df, parts)
    j_df = per_joint(parts)
    leg_df = per_leg(parts)
    grp_df = front_rear_lr(parts)
    phase_df = per_phase(df, parts)

    ep_df.to_csv(out_dir / "episode_decomp.csv")
    j_df.to_csv(out_dir / "per_joint_decomp.csv", index=False)
    leg_df.to_csv(out_dir / "per_leg_decomp.csv", index=False)
    grp_df.to_csv(out_dir / "front_rear_lr_decomp.csv", index=False)
    phase_df.to_csv(out_dir / "per_phase_decomp.csv", index=False)

    fig_episode_decomp(ep_df, fig_dir / "decomp_per_episode.png")
    fig_neg_frac(ep_df, fig_dir / "neg_frac_per_episode.png")
    fig_per_joint(j_df, fig_dir / "decomp_per_joint.png")
    fig_per_leg(leg_df, fig_dir / "decomp_per_leg.png")
    fig_front_rear_lr(grp_df, fig_dir / "decomp_front_rear_lr.png")
    fig_per_phase(phase_df, fig_dir / "decomp_per_phase.png")
    rep_ep = int(ep_df["abs_work_J"].idxmax())
    fig_power_timeseries(df, parts, rep_ep, fig_dir / f"power_timeseries_ep{rep_ep}.png")
    fig_saturation_vs_power(df, parts, fig_dir / "saturation_vs_power.png")

    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
