#!/usr/bin/env python3
"""
Thesis mechanism figures — IROS style.

Fig 1: Per-leg stance F_fwd + body v_fwd (same normalized stance time, twin y-axes).
Fig 2: Stride cycle — per-leg GRF + cumulative W+ (FL CPG period; v_fwd in Fig 1).

Read-only mechanism_v2 parquet → thesis_experiments_planner_baseline/figures/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CPG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPG_ROOT))
sys.path.insert(0, str(CPG_ROOT / "env"))
sys.path.insert(0, str(CPG_ROOT / "experiments"))
sys.path.insert(0, str(CPG_ROOT / "scripts"))

from mechanism_v2_analyze import find_stances, stance_slice, steady_mask  # noqa: E402
from mechanism_v2_grf_corrected import infer_forward_axis  # noqa: E402
from mechanism_v3_analysis import enrich_df  # noqa: E402
from mechanism_v4_stride_internal import IROS_RC  # noqa: E402
from mechanism_v4_unified_stride_cycle import (  # noqa: E402
    N_PHASE,
    mean_stride_profiles_physical,
)
from run_cder_closure_validation import LEG_NAMES  # noqa: E402

DATA_DIR = CPG_ROOT / "outputs/thesis_experiments_planner_baseline/mechanism_v2"
OUT_DIR = CPG_ROOT / "outputs/thesis_experiments_planner_baseline/figures"

# IROS double-column width (in); thesis palette
FIG_W = 7.16
COL_P = "#6B4226"
COL_P_DARK = "#4A2810"
COL_O = "#E08E45"
COL_O_DARK = "#B86820"

N_STANCE = 50
LW_F = 1.6
LW_V = 1.3

IROS_MECH_RC = {
    **IROS_RC,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
}


def mean_stance_profile(df: pd.DataFrame, leg: str, col: str) -> np.ndarray:
    """Mean over steady stances for this leg; x = normalized stance time [0, 1]."""
    sm = steady_mask(df)
    curves: List[np.ndarray] = []
    for ev in find_stances(df, leg):
        seg = stance_slice(df, ev)
        seg = seg.loc[sm[seg.index]]
        if len(seg) < 8:
            continue
        y = seg[col].to_numpy(dtype=np.float64)
        tn = np.linspace(0.0, 1.0, len(y))
        ti = np.linspace(0.0, 1.0, N_STANCE)
        curves.append(np.interp(ti, tn, y))
    return np.mean(np.stack(curves), axis=0) if curves else np.zeros(N_STANCE)


def unified_f_ylim(profiles: List[np.ndarray], *, floor: float = -2.0, ceil: float = 1.0) -> Tuple[float, float]:
    vals = np.concatenate(profiles)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = 0.08 * max(hi - lo, 0.5)
    return max(floor, lo - pad), min(ceil, hi + pad)


def _style_axes_iros(ax: plt.Axes, *, twin: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    if not twin:
        ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=twin)


def fig_grf_4legs_stance_v_merged(dfp: pd.DataFrame, dfo: pd.DataFrame) -> None:
    """
    2×2: $F_\\mathrm{fwd}$ (left) and $v_\\mathrm{body,fwd}$ (right) vs **the same**
    normalized stance time for each leg (touchdown→liftoff of that leg).
    """
    t = np.linspace(0.0, 1.0, N_STANCE)
    f_profiles: List[np.ndarray] = []
    v_profiles: List[np.ndarray] = []

    data: Dict[str, Dict[str, np.ndarray]] = {"p": {}, "o": {}}
    for leg in LEG_NAMES:
        fp = mean_stance_profile(dfp, leg, f"grf_fwd_{leg}")
        fo = mean_stance_profile(dfo, leg, f"grf_fwd_{leg}")
        vp = mean_stance_profile(dfp, leg, "body_v_forward") * 1000.0
        vo = mean_stance_profile(dfo, leg, "body_v_forward") * 1000.0
        data["p"][f"f_{leg}"] = fp
        data["o"][f"f_{leg}"] = fo
        data["p"][f"v_{leg}"] = vp
        data["o"][f"v_{leg}"] = vo
        f_profiles.extend([fp, fo])
        v_profiles.extend([vp, vo])

    yf0, yf1 = unified_f_ylim(f_profiles)
    v_all = np.concatenate(v_profiles)
    yv0, yv1 = 0.0, min(150.0, float(np.max(v_all)) * 1.08 + 5.0)

    fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 5.0), sharex=True)
    ax_v_ref = None

    for ax, leg in zip(axs.ravel(), LEG_NAMES):
        fp, fo = data["p"][f"f_{leg}"], data["o"][f"f_{leg}"]
        vp, vo = data["p"][f"v_{leg}"], data["o"][f"v_{leg}"]

        ax.plot(t, fp, color=COL_P, lw=LW_F, ls="-", label=r"$F_\mathrm{fwd}$ Planner")
        ax.plot(t, fo, color=COL_O, lw=LW_F, ls="--", label=r"$F_\mathrm{fwd}$ Ours")
        ax.axhline(0, color="0.35", lw=0.5, zorder=1)
        ax.set_ylim(yf0, yf1)
        _style_axes_iros(ax)

        if leg in ("RL", "RR"):
            mask = (fp < -0.05) & (vo > fp)
            if np.any(mask):
                ax.fill_between(t, fp, fo, where=mask, color="#C44E52", alpha=0.18, zorder=0)
            if leg == "RL":
                ax.text(
                    0.58,
                    0.08,
                    "Planner: braking\nOurs: propulsion",
                    transform=ax.transAxes,
                    fontsize=7.5,
                    va="bottom",
                    ha="left",
                    color="0.25",
                )

        ax2 = ax.twinx()
        if ax_v_ref is None:
            ax_v_ref = ax2
        else:
            ax2.sharey(ax_v_ref)
        ax2.plot(t, vp, color=COL_P, lw=LW_V, ls="-.", label=r"$v_\mathrm{body}$ Planner")
        ax2.plot(t, vo, color=COL_O, lw=LW_V, ls=":", label=r"$v_\mathrm{body}$ Ours")
        cruise_p, cruise_o = float(np.mean(vp)), float(np.mean(vo))
        ax2.axhline(cruise_p, color=COL_P, ls="--", lw=0.9, alpha=0.65)
        ax2.axhline(cruise_o, color=COL_O, ls="--", lw=0.9, alpha=0.65)
        ax2.set_ylim(yv0, yv1)
        _style_axes_iros(ax2, twin=True)

        ax.set_title(leg, fontweight="bold", fontsize=10, pad=4)
        ax.set_ylabel(r"$F_{\mathrm{fwd}}$ (N)", fontsize=8.5)
        if ax is axs[0, 1] or ax is axs[1, 1]:
            ax2.set_ylabel(r"$v_{\mathrm{body,fwd}}$ (mm/s)", fontsize=8.5)

    axs[1, 0].set_xlabel(
        r"Normalized stance time ($0$ = touchdown, $1$ = liftoff)"
    )
    axs[1, 1].set_xlabel(
        r"Normalized stance time ($0$ = touchdown, $1$ = liftoff)"
    )

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], color=COL_P, lw=LW_F, ls="-", label=r"$F_\mathrm{fwd}$ Planner"),
        Line2D([0], [0], color=COL_O, lw=LW_F, ls="--", label=r"$F_\mathrm{fwd}$ Ours"),
        Line2D([0], [0], color=COL_P, lw=LW_V, ls="-.", label=r"$v_\mathrm{body}$ Planner"),
        Line2D([0], [0], color=COL_O, lw=LW_V, ls=":", label=r"$v_\mathrm{body}$ Ours"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="upper center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "Stance-phase forward GRF and body velocity (per-leg alignment)",
        fontsize=10,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    stem = OUT_DIR / "fig_grf_4legs_stance_unified"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def fig_stride_cycle_two_panel(dfp: pd.DataFrame, dfo: pd.DataFrame, ax_p) -> None:
    """IROS stride cycle: per-leg GRF + cumulative $W_+$ (FL period); $v$ in Fig 1."""
    pp = mean_stride_profiles_physical(dfp, ax_p)
    po = mean_stride_profiles_physical(dfo, infer_forward_axis(dfo))
    t = np.linspace(0.0, 1.0, N_PHASE)

    w_stride_p = float(pp["Wp_cum_J"][-1])
    w_stride_o = float(po["Wp_cum_J"][-1])

    grf_all = np.concatenate(
        [pp[f"grf_fwd_{leg}"] for leg in LEG_NAMES]
        + [po[f"grf_fwd_{leg}"] for leg in LEG_NAMES]
    )
    grf_pad = 0.08 * max(float(np.ptp(grf_all)), 0.5)
    grf_ylim = (float(np.min(grf_all)) - grf_pad, float(np.max(grf_all)) + grf_pad)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 4.6), sharex=True)

    ax = axes[0]
    styles_p = {
        "FL": (COL_P, "-"),
        "FR": (COL_P, "--"),
        "RL": (COL_P_DARK, "-"),
        "RR": (COL_P_DARK, "--"),
    }
    styles_o = {
        "FL": (COL_O, "-"),
        "FR": (COL_O, "--"),
        "RL": (COL_O_DARK, "-"),
        "RR": (COL_O_DARK, "--"),
    }
    for leg in LEG_NAMES:
        c, ls = styles_p[leg]
        ax.plot(t, pp[f"grf_fwd_{leg}"], color=c, lw=1.3, ls=ls, label=f"P {leg}")
    for leg in LEG_NAMES:
        c, ls = styles_o[leg]
        ax.plot(t, po[f"grf_fwd_{leg}"], color=c, lw=1.3, ls=ls, label=f"O {leg}")
    ax.axhline(0, color="0.35", lw=0.4)
    ax.set_ylabel(r"$F_{\mathrm{fwd}}$ (N)")
    ax.set_ylim(grf_ylim)
    ax.set_title(
        r"(a) Per-leg forward GRF (swing $\rightarrow 0$)",
        fontsize=9,
        loc="left",
        fontweight="bold",
    )
    ax.legend(ncol=4, fontsize=5.5, loc="upper right", frameon=False)
    _style_axes_iros(ax)

    ax = axes[1]
    ax.plot(t, pp["Wp_cum_J"], color=COL_P, lw=LW_F, label="Planner")
    ax.plot(t, po["Wp_cum_J"], color=COL_O, lw=LW_F, ls="--", label="Ours")
    ax.set_ylim(0, 0.2)
    ax.set_ylabel(r"Cumulative $W_+$ (J)")
    ax.set_title(
        r"(b) Cumulative motor positive work",
        fontsize=9,
        loc="left",
        fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    _style_axes_iros(ax)
    ax.text(
        0.99,
        0.35,
        f"Planner: {w_stride_p:.3f} J/stride\nOurs: {w_stride_o:.3f} J/stride",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.2",
    )
    ax.set_xlabel(
        r"Normalized time: one FL CPG period (same physical instants)"
    )

    fig.suptitle(
        "Stride cycle (synchronized)",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    stem = OUT_DIR / "fig_stride_cycle_three_panel"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  W+ per stride (J): Planner {w_stride_p:.3f}, Ours {w_stride_o:.3f}")


def main() -> None:
    plt.rcParams.update(IROS_MECH_RC)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_p = pd.read_parquet(DATA_DIR / "mechanism_v2_planner_data.parquet")
    df_o = pd.read_parquet(DATA_DIR / "mechanism_v2_ours_data.parquet")
    ax_p = infer_forward_axis(df_p)
    dfp = enrich_df(df_p, ax_p)
    dfo = enrich_df(df_o, infer_forward_axis(df_o))

    fig_grf_4legs_stance_v_merged(dfp, dfo)
    fig_stride_cycle_two_panel(dfp, dfo, ax_p)

    print(f"Wrote {OUT_DIR}/fig_grf_4legs_stance_unified.pdf|.png")
    print(f"Wrote {OUT_DIR}/fig_stride_cycle_three_panel.pdf|.png (2 panels, IROS)")


if __name__ == "__main__":
    main()
