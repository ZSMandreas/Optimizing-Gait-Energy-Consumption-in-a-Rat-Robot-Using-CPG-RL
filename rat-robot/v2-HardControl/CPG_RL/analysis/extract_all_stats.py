from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon
from scipy.stats import ttest_1samp


BASE = "/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/rat-robot/v2-HardControl/CPG_RL"
EP_CSV = os.path.join(BASE, "logs/paper_coupling_analysis/episode_metrics.csv")
STEP_CSV = os.path.join(BASE, "logs/paper_coupling_analysis/step_metrics.csv")
NPZ_PATH = os.path.join(BASE, "logs/paper_coupling_analysis/substep_signals.npz")
META_JSON = os.path.join(BASE, "logs/paper_coupling_analysis/fix_result_table_meta.json")
OUT_TXT = os.path.join(BASE, "logs/paper_coupling_analysis/stats_summary.txt")


col_map = {
    "jpm": ["J/m", "jpm", "j_per_m", "energy_per_dist", "J_per_m"],
    "dist": ["dist", "ep_distance", "forward_dist"],
    "vel": ["vel", "fwd_vel", "mean_fwd_vel", "mean_vel", "fwd_vel_mean"],
    "x_offset": ["x_offset_m", "x_offset", "lateral_offset"],
    "swing_ratio": ["swing_ratio", "swing_work_ratio"],
    "r_sync": ["r_sync", "r_sync_mean", "ep_r_sync"],
    "f4": ["action_f4_mean", "f4_mean", "mean_f4"],
    "stance_work": ["stance_abs_work", "stance_work", "stance_abs_work_total", "stance_abs_work_per_step"],
    "swing_work": ["swing_abs_work", "swing_work", "swing_abs_work_total", "swing_abs_work_per_step"],
}


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return "na"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def fmt_num(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}g}" if np.isfinite(v) else "N/A"


def fmt_mean_std(x: np.ndarray) -> Tuple[str, float, float]:
    xx = x[np.isfinite(x)]
    if xx.size == 0:
        return "N/A", float("nan"), float("nan")
    m = float(np.mean(xx))
    s = float(np.std(xx, ddof=1 if xx.size > 1 else 0))
    return f"{m:.3f}±{s:.3f}", m, s


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "N/A"
    if p == 0:
        return "p < 1.0×10^-300"
    e = int(math.floor(math.log10(abs(p))))
    a = p / (10 ** e)
    return f"p = {a:.3g}×10^{e}"


def pair_test(a: np.ndarray, b: np.ndarray) -> Tuple[str, float]:
    aa = a[np.isfinite(a)]
    bb = b[np.isfinite(b)]
    n = min(len(aa), len(bb))
    if n == 0:
        return "none", float("nan")
    if len(aa) == len(bb):
        try:
            return "wilcoxon", float(wilcoxon(aa, bb, zero_method="pratt").pvalue)
        except Exception:
            pass
    return "mannwhitneyu", float(mannwhitneyu(aa, bb, alternative="two-sided").pvalue)


def bootstrap_ci_pearson(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    xv = x[m]
    yv = y[m]
    n = len(xv)
    if n < 3:
        return float("nan"), float("nan")
    rs = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            rs[i] = pearsonr(xv[idx], yv[idx]).statistic
        except Exception:
            rs[i] = np.nan
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _paired_from_episode(df_wt: pd.DataFrame, df_x: pd.DataFrame, col: str):
    a = pd.to_numeric(df_wt[col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(df_x[col], errors="coerce").to_numpy(dtype=float)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _range_str(x: np.ndarray) -> str:
    xx = x[np.isfinite(x)]
    if xx.size == 0:
        return "[N/A, N/A]"
    return f"[{np.min(xx)*100:.1f}%, {np.max(xx)*100:.1f}%]"


def task_A_swing_vs_stance_reduction(df_wt, df_ab, df_cp, col_stance, col_swing) -> List[str]:
    if not isinstance(col_stance, str):
        col_stance = "stance_abs_work_per_step"
    if not isinstance(col_swing, str):
        col_swing = "swing_abs_work_per_step"
    out = []
    out.append("=== Task A：swing vs stance 削减比较（论断 1 量化）===")
    out.append("")

    def one_block(name: str, df_x: pd.DataFrame):
        sw_w, sw_x = _paired_from_episode(df_wt, df_x, col_swing)
        st_w, st_x = _paired_from_episode(df_wt, df_x, col_stance)
        n = min(len(sw_w), len(st_w), len(sw_x), len(st_x))
        sw_w, sw_x, st_w, st_x = sw_w[:n], sw_x[:n], st_w[:n], st_x[:n]
        eps = 1e-12
        swing_drop = (sw_w - sw_x) / np.maximum(sw_w, eps)
        stance_drop = (st_w - st_x) / np.maximum(st_w, eps)
        diff = swing_drop - stance_drop

        mw, sw = float(np.mean(swing_drop)), float(np.std(swing_drop, ddof=1 if n > 1 else 0))
        ms, ss = float(np.mean(stance_drop)), float(np.std(stance_drop, ddof=1 if n > 1 else 0))
        md, sd = float(np.mean(diff)), float(np.std(diff, ddof=1 if n > 1 else 0))

        try:
            pw = float(wilcoxon(diff, alternative="greater", zero_method="pratt").pvalue)
        except Exception:
            pw = float("nan")
        try:
            pt = float(ttest_1samp(diff, popmean=0.0, alternative="greater", nan_policy="omit").pvalue)
        except Exception:
            pt = float("nan")

        ratio = mw / max(ms, 1e-12)
        out.append(f"--- withoutET → {name} ---")
        out.append(f"per-episode swing 下降%：mean={mw*100:.1f}±{sw*100:.1f}, range={_range_str(swing_drop)}")
        out.append(f"per-episode stance下降%：mean={ms*100:.1f}±{ss*100:.1f}, range={_range_str(stance_drop)}")
        out.append(f"per-episode 差值（swing-stance）：mean={md*100:.1f}±{sd*100:.1f}")
        out.append(f"Wilcoxon（H1: swing_drop > stance_drop）: {fmt_p(pw)}, 显著性={sig_stars(pw)}")
        out.append(f"t-test  （H1: swing_drop > stance_drop）: {fmt_p(pt)}, 显著性={sig_stars(pt)}")
        out.append(f"结论：swing 削减{'显著' if np.isfinite(pw) and pw<0.05 else '不显著'}大于 stance 削减（ratio={ratio:.2f}×）")
        out.append("")
        return {
            "swing_mean_pct": mw * 100.0,
            "stance_mean_pct": ms * 100.0,
            "ratio": ratio,
            "p_w": pw,
            "p_t": pt,
        }

    res_ab = one_block("ablation（IDER threshold 贡献）", df_ab)
    res_cp = one_block("cpl12（完整方法）", df_cp)
    return out + ["__TASKA_JSON__" + json.dumps({"ablation": res_ab, "cpl12": res_cp}, ensure_ascii=False)]


def task_B_stance_efficiency(df_wt, df_ab, df_cp, col_stance, col_dist, ep_steps=82) -> List[str]:
    if not isinstance(col_stance, str):
        col_stance = "stance_abs_work_per_step"
    if not isinstance(col_dist, str):
        col_dist = "dist"
    out = []
    out.append("=== Task B：支撑相协调效率 η_stance（论断 2 量化）===")
    out.append(f"η_stance = dist / (stance_work_per_step × {ep_steps})   单位：m/J")
    out.append("")

    def eta(df):
        d = pd.to_numeric(df[col_dist], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(df[col_stance], errors="coerce").to_numpy(dtype=float)
        n = min(len(d), len(s))
        d = d[:n]
        s = s[:n]
        m = np.isfinite(d) & np.isfinite(s) & (np.abs(s) > 1e-12)
        return d[m] / (s[m] * float(ep_steps))

    eta_w = eta(df_wt)
    eta_a = eta(df_ab)
    eta_c = eta(df_cp)
    sw, mw, sdw = fmt_mean_std(eta_w)
    sa, ma, sda = fmt_mean_std(eta_a)
    sc, mc, sdc = fmt_mean_std(eta_c)

    pa = pair_test(eta_w, eta_a)
    pc = pair_test(eta_w, eta_c)
    pac = pair_test(eta_a, eta_c)
    # one-sided "greater" for claimed improvement
    try:
        p_w_a_g = float(wilcoxon(eta_a[: min(len(eta_w), len(eta_a))] - eta_w[: min(len(eta_w), len(eta_a))], alternative="greater", zero_method="pratt").pvalue)
    except Exception:
        p_w_a_g = float("nan")
    try:
        p_w_c_g = float(wilcoxon(eta_c[: min(len(eta_w), len(eta_c))] - eta_w[: min(len(eta_w), len(eta_c))], alternative="greater", zero_method="pratt").pvalue)
    except Exception:
        p_w_c_g = float("nan")
    try:
        p_a_c_g = float(wilcoxon(eta_c[: min(len(eta_a), len(eta_c))] - eta_a[: min(len(eta_a), len(eta_c))], alternative="greater", zero_method="pratt").pvalue)
    except Exception:
        p_a_c_g = float("nan")

    pct_ab = (ma - mw) / max(mw, 1e-12) * 100.0
    pct_cp = (mc - mw) / max(mw, 1e-12) * 100.0
    pct_cp_ab = (mc - ma) / max(ma, 1e-12) * 100.0

    out.append(f"withoutET:            η = {sw}  m/J")
    out.append(f"ablation_no_coupling: η = {sa}  m/J  (vs withoutET: +{pct_ab:.1f}%)")
    out.append(f"cpl12:                η = {sc}  m/J  (vs withoutET: +{pct_cp:.1f}%, vs ablation: +{pct_cp_ab:.1f}%)")
    out.append("")
    out.append("=== Wilcoxon 检验（η_stance）===")
    out.append(f"withoutET vs ablation:  {pa[0]}, {fmt_p(pa[1])}, 显著性={sig_stars(pa[1])}")
    out.append(f"withoutET vs cpl12:     {pc[0]}, {fmt_p(pc[1])}, 显著性={sig_stars(pc[1])}")
    out.append(f"ablation  vs cpl12:     {pac[0]}, {fmt_p(pac[1])}, 显著性={sig_stars(pac[1])}")
    out.append("")
    out.append("结论：")
    out.append(f"  IDER threshold 贡献（withoutET→ablation）：η_stance 提升 {pct_ab:.1f}%（one-sided {fmt_p(p_w_a_g)}）")
    out.append(f"  耦合图额外贡献（ablation→cpl12）：η_stance 再提升 {pct_cp_ab:.1f}%（one-sided {fmt_p(p_a_c_g)}）")
    out.append(f"  完整方法提升（withoutET→cpl12）：η_stance 提升 {pct_cp:.1f}%（one-sided {fmt_p(p_w_c_g)}）")
    out.append("")

    tpl1 = (
        'The IDER threshold structure reduces swing-phase mechanical work by '
        f'{taskA_json_holder.get("ablation",{}).get("swing_mean_pct", float("nan")):.1f}% and stance-phase work by '
        f'{taskA_json_holder.get("ablation",{}).get("stance_mean_pct", float("nan")):.1f}% '
        "relative to Baseline (swing/stance ratio "
        f'{taskA_json_holder.get("ablation",{}).get("ratio", float("nan")):.2f}×, '
        f'Wilcoxon {fmt_p(taskA_json_holder.get("ablation",{}).get("p_w", float("nan")))}), demonstrating that the inverse-dynamics '
        "threshold preferentially suppresses non-propulsive expenditure."
    )
    tpl2 = (
        "Propulsive work efficiency η_stance increases from "
        f"{mw:.3f}±{sdw:.3f} m/J (Baseline) to {ma:.3f}±{sda:.3f} m/J "
        f"(w/o coupling, +{pct_ab:.1f}%, {fmt_p(p_w_a_g)}) and further to "
        f"{mc:.3f}±{sdc:.3f} m/J (Ours, +{pct_cp_ab:.1f}% vs w/o coupling, "
        f"{fmt_p(p_a_c_g)}), quantifying the stance-phase coordination improvement "
        "introduced by the coupling graph."
    )
    out.append("=== 论文英文模板（已填实数）===")
    out.append(tpl1)
    out.append("")
    out.append(tpl2)
    return out


taskA_json_holder: Dict[str, Dict[str, float]] = {}


def load_meta():
    if os.path.exists(META_JSON):
        with open(META_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    ep = pd.read_csv(EP_CSV)
    step = pd.read_csv(STEP_CSV)
    npz = np.load(NPZ_PATH)
    meta = load_meta()

    models = ["withoutET", "ablation_no_coupling", "cpl12"]
    labels = {m: m for m in models}

    c_jpm = find_col(ep, col_map["jpm"])
    c_dist = find_col(ep, col_map["dist"])
    c_vel = find_col(ep, col_map["vel"])
    c_x = find_col(ep, col_map["x_offset"])
    c_swr = find_col(ep, col_map["swing_ratio"])
    c_sync = find_col(ep, col_map["r_sync"])
    c_f4 = find_col(ep, col_map["f4"])
    c_st = find_col(ep, col_map["stance_work"])
    c_sw = find_col(ep, col_map["swing_work"])

    lines: List[str] = []
    lines.append("=== Data Sources ===")
    lines.append(f"episode_metrics: {EP_CSV}")
    lines.append(f"step_metrics: {STEP_CSV}")
    lines.append(f"substep_signals: {NPZ_PATH}")
    lines.append("")
    lines.append("=== Column Mapping ===")
    lines.append(f"jpm={c_jpm}, dist={c_dist}, vel={c_vel}, x_offset={c_x}, swing_ratio={c_swr}, r_sync={c_sync}, f4={c_f4}")
    lines.append(f"stance_work={c_st}, swing_work={c_sw}")
    lines.append("")

    # Task 1
    lines.append("=== 主结果表 (n=20, mean±std) ===")
    lines.append("模型                  J/m            dist           vel            x_offset       swing_ratio    r_sync")
    per_model = {}
    x_meta_stats = {}
    for m in models:
        dm = ep[ep["model"] == m]
        arrs = {
            "jpm": pd.to_numeric(dm[c_jpm], errors="coerce").to_numpy(dtype=float) if c_jpm else np.array([]),
            "dist": pd.to_numeric(dm[c_dist], errors="coerce").to_numpy(dtype=float) if c_dist else np.array([]),
            "vel": pd.to_numeric(dm[c_vel], errors="coerce").to_numpy(dtype=float) if c_vel else np.array([]),
            "x": pd.to_numeric(dm[c_x], errors="coerce").to_numpy(dtype=float) if c_x else np.array([]),
            "swr": pd.to_numeric(dm[c_swr], errors="coerce").to_numpy(dtype=float) if c_swr else np.array([]),
            "sync": pd.to_numeric(dm[c_sync], errors="coerce").to_numpy(dtype=float) if c_sync else np.array([]),
        }
        # fallback x_offset from meta
        if arrs["x"].size == 0 or np.all(~np.isfinite(arrs["x"])):
            key = "baseline_x_offset_m_mean_std" if m == "withoutET" else ("ablation_x_offset_m_mean_std" if m == "ablation_no_coupling" else "ours_x_offset_m_mean_std")
            if key in meta:
                x_meta_stats[m] = (float(meta[key][0]), float(meta[key][1]))
        per_model[m] = arrs
        s_j, _, _ = fmt_mean_std(arrs["jpm"])
        s_d, _, _ = fmt_mean_std(arrs["dist"])
        s_v, _, _ = fmt_mean_std(arrs["vel"])
        if m in x_meta_stats:
            s_x = f"{x_meta_stats[m][0]:.3f}±{x_meta_stats[m][1]:.3f}"
        else:
            s_x, _, _ = fmt_mean_std(arrs["x"])
        s_sw, _, _ = fmt_mean_std(arrs["swr"])
        s_sy, _, _ = fmt_mean_std(arrs["sync"])
        lines.append(f"{labels[m]:22s} {s_j:14s} {s_d:14s} {s_v:14s} {s_x:14s} {s_sw:14s} {s_sy:14s}")
    lines.append("  [论文论断] 主性能对比、稳定性与同步性量化")
    lines.append("")

    pairs = [("withoutET", "ablation_no_coupling"), ("withoutET", "cpl12"), ("ablation_no_coupling", "cpl12")]
    for key, title in [("jpm", "J/m"), ("dist", "dist"), ("vel", "vel")]:
        lines.append(f"=== Wilcoxon/MWU 检验（{title}）===")
        for a, b in pairs:
            m, p = pair_test(per_model[a][key], per_model[b][key])
            lines.append(f"{a} vs {b}: {m}, {fmt_p(p)}, 显著性={sig_stars(p)}")
        lines.append("")

    # Task 2
    st_stats = {}
    sw_stats = {}
    for m in models:
        st_stats[m] = fmt_mean_std(per_model[m]["dist"] * 0 + pd.to_numeric(ep[ep["model"] == m][c_st], errors="coerce").to_numpy(dtype=float) if c_st else np.array([]))
        sw_stats[m] = fmt_mean_std(per_model[m]["dist"] * 0 + pd.to_numeric(ep[ep["model"] == m][c_sw], errors="coerce").to_numpy(dtype=float) if c_sw else np.array([]))
    st_w = st_stats["withoutET"][1]
    sw_w = sw_stats["withoutET"][1]
    st_a = st_stats["ablation_no_coupling"][1]
    sw_a = sw_stats["ablation_no_coupling"][1]
    st_c = st_stats["cpl12"][1]
    sw_c = sw_stats["cpl12"][1]
    stance_drop_ablation = (st_w - st_a) / st_w * 100.0
    swing_drop_ablation = (sw_w - sw_a) / sw_w * 100.0
    stance_drop_cpl12 = (st_w - st_c) / st_w * 100.0
    swing_drop_cpl12 = (sw_w - sw_c) / sw_w * 100.0
    stance_drop_coupling = (st_a - st_c) / st_a * 100.0
    swing_drop_coupling = (sw_a - sw_c) / sw_a * 100.0

    lines.append("=== stance/swing 做功下降比例（以 withoutET 为基准）===")
    lines.append("               stance 下降%    swing 下降%    swing/stance 比值")
    lines.append(f"ablation       {stance_drop_ablation:8.3f}%      {swing_drop_ablation:8.3f}%      {swing_drop_ablation/max(stance_drop_ablation,1e-12):8.3f}x")
    lines.append(f"cpl12          {stance_drop_cpl12:8.3f}%      {swing_drop_cpl12:8.3f}%      {swing_drop_cpl12/max(stance_drop_cpl12,1e-12):8.3f}x")
    lines.append("")
    lines.append("=== 耦合图额外贡献（ablation→cpl12）===")
    lines.append("               stance 下降%    swing 下降%")
    lines.append(f"coupling       {stance_drop_coupling:8.3f}%      {swing_drop_coupling:8.3f}%")
    lines.append("")
    lines.append("=== 绝对值（J/step）===")
    lines.append("               stance          swing          total")
    for m in models:
        st_s, st_m, st_std = st_stats[m]
        sw_s, sw_m, sw_std = sw_stats[m]
        total = fmt_mean_std(pd.to_numeric(ep[ep["model"] == m][c_st], errors="coerce").to_numpy(dtype=float) + pd.to_numeric(ep[ep["model"] == m][c_sw], errors="coerce").to_numpy(dtype=float))[0] if c_st and c_sw else "N/A"
        lines.append(f"{m:14s} {st_s:14s} {sw_s:14s} {total:14s}")
    lines.append("  [论文论断] IDER 对做功分配的抑制优先级")
    lines.append("")

    lines.append("=== Wilcoxon/MWU 检验（stance_abs_work）===")
    for a, b in pairs:
        va = pd.to_numeric(ep[ep["model"] == a][c_st], errors="coerce").to_numpy(dtype=float) if c_st else np.array([])
        vb = pd.to_numeric(ep[ep["model"] == b][c_st], errors="coerce").to_numpy(dtype=float) if c_st else np.array([])
        m, p = pair_test(va, vb)
        lines.append(f"{a} vs {b}: {m}, {fmt_p(p)}, 显著性={sig_stars(p)}")
    lines.append("")
    lines.append("=== Wilcoxon/MWU 检验（swing_abs_work）===")
    for a, b in pairs:
        va = pd.to_numeric(ep[ep["model"] == a][c_sw], errors="coerce").to_numpy(dtype=float) if c_sw else np.array([])
        vb = pd.to_numeric(ep[ep["model"] == b][c_sw], errors="coerce").to_numpy(dtype=float) if c_sw else np.array([])
        m, p = pair_test(va, vb)
        lines.append(f"{a} vs {b}: {m}, {fmt_p(p)}, 显著性={sig_stars(p)}")
    lines.append("")

    # Task 3
    lines.append("=== r_sync 统计 ===")
    for m in models:
        v = per_model[m]["sync"]
        vv = v[np.isfinite(v)]
        s, mm, ss = fmt_mean_std(v)
        rng = f"[{np.min(vv):.3f}, {np.max(vv):.3f}]" if vv.size else "[N/A, N/A]"
        lines.append(f"{m:22s} mean={s}, range={rng}")
    lines.append("")
    lines.append("=== Wilcoxon/MWU 检验（r_sync）===")
    for a, b in pairs:
        mth, p = pair_test(per_model[a]["sync"], per_model[b]["sync"])
        lines.append(f"{a} vs {b}: {mth}, {fmt_p(p)}, 显著性={sig_stars(p)}")
    all_sync = pd.to_numeric(ep[c_sync], errors="coerce").to_numpy(dtype=float) if c_sync else np.array([])
    all_jpm = pd.to_numeric(ep[c_jpm], errors="coerce").to_numpy(dtype=float) if c_jpm else np.array([])
    m = np.isfinite(all_sync) & np.isfinite(all_jpm)
    if np.sum(m) >= 3:
        r, p = pearsonr(all_sync[m], all_jpm[m])
        lo, hi = bootstrap_ci_pearson(all_sync[m], all_jpm[m], n_boot=5000, seed=42)
    else:
        r = p = lo = hi = float("nan")
    lines.append("")
    lines.append("=== r_sync 与 J/m 的 Pearson 相关 ===")
    lines.append(f"r = {r:.6f}, {fmt_p(p)}")
    lines.append(f"95% bootstrap CI = [{lo:.6f}, {hi:.6f}]")
    lines.append("  [论文论断] 同步性与能耗之间的相关关系")
    lines.append("")

    # Task 4
    lines.append("=== 步频 f4 统计 ===")
    f4 = {}
    for m in models:
        v = pd.to_numeric(ep[ep["model"] == m][c_f4], errors="coerce").to_numpy(dtype=float) if c_f4 else np.array([])
        f4[m] = v
        s, _, _ = fmt_mean_std(v)
        lines.append(f"{m:22s} f4 = {s}")
    lines.append("")
    lines.append("=== Wilcoxon/MWU 检验（f4）===")
    for a, b in pairs:
        method, p = pair_test(f4[a], f4[b])
        ma = np.nanmean(f4[a]) if len(f4[a]) else np.nan
        mb = np.nanmean(f4[b]) if len(f4[b]) else np.nan
        drop = (ma - mb) / ma * 100.0 if np.isfinite(ma) and ma != 0 and np.isfinite(mb) else np.nan
        lines.append(f"{a} vs {b}: {method}, {fmt_p(p)}, 下降={drop:.3f}%")
    lines.append("  [论文论断] 频率策略变化是否与节能一致")
    lines.append("")

    # Task 5 (from npz)
    lines.append("=== hip 关节 |dq| 均值（rad/s）===")
    hips = [0, 2, 4, 6]
    hip_names = ["FL-hip", "FR-hip", "RL-hip", "RR-hip"]
    lines.append("关节          withoutET-st   withoutET-sw   ablation-st    ablation-sw    cpl12-st      cpl12-sw")

    # build helpers for episode-level means
    def ep_means(ep_arr, qv, ct, joint, stance):
        out = []
        for e in np.unique(ep_arr):
            m_ep = ep_arr == e
            leg = joint // 2
            m_st = (ct[:, leg] > 0.5) if stance else (ct[:, leg] <= 0.5)
            m = m_ep & m_st
            vals = np.abs(qv[m, joint])
            out.append(float(np.mean(vals)) if vals.size else np.nan)
        return np.asarray(out, dtype=float)

    ep_w = npz["without_episode"]
    qv_w = npz["without_qvel"]
    ct_w = npz["without_contacts"]
    ep_a = npz["ablation_episode"]
    qv_a = npz["ablation_qvel"]
    ct_a = npz["ablation_contacts"]
    ep_c = npz["cpl12_episode"]
    qv_c = npz["cpl12_qvel"]
    ct_c = npz["cpl12_contacts"]

    fl_tests = {}
    for j, jname in zip(hips, hip_names):
        w_st = ep_means(ep_w, qv_w, ct_w, j, True)
        w_sw = ep_means(ep_w, qv_w, ct_w, j, False)
        a_st = ep_means(ep_a, qv_a, ct_a, j, True)
        a_sw = ep_means(ep_a, qv_a, ct_a, j, False)
        c_st = ep_means(ep_c, qv_c, ct_c, j, True)
        c_sw = ep_means(ep_c, qv_c, ct_c, j, False)

        lines.append(
            f"{jname:12s} {fmt_mean_std(w_st)[0]:14s} {fmt_mean_std(w_sw)[0]:14s} {fmt_mean_std(a_st)[0]:14s} "
            f"{fmt_mean_std(a_sw)[0]:14s} {fmt_mean_std(c_st)[0]:14s} {fmt_mean_std(c_sw)[0]:14s}"
        )
        if j == 0:
            fl_tests["st_wa"] = pair_test(w_st, a_st)
            fl_tests["st_wc"] = pair_test(w_st, c_st)
            fl_tests["sw_wa"] = pair_test(w_sw, a_sw)
            fl_tests["sw_wc"] = pair_test(w_sw, c_sw)

    lines.append("")
    lines.append("=== Wilcoxon 检验（FL-hip |dq| 均值，across episodes）===")
    lines.append(f"withoutET vs ablation（stance）: {fl_tests['st_wa'][0]}, {fmt_p(fl_tests['st_wa'][1])}")
    lines.append(f"withoutET vs cpl12 （stance）:  {fl_tests['st_wc'][0]}, {fmt_p(fl_tests['st_wc'][1])}")
    lines.append(f"withoutET vs ablation（swing）:  {fl_tests['sw_wa'][0]}, {fmt_p(fl_tests['sw_wa'][1])}")
    lines.append(f"withoutET vs cpl12 （swing）:   {fl_tests['sw_wc'][0]}, {fmt_p(fl_tests['sw_wc'][1])}")
    lines.append("  [论文论断] 关节速度抑制是否在支撑/摆动相成立")
    lines.append("")

    # New Task A/B: ONLY episode_metrics.csv
    df_wt = ep[ep["model"] == "withoutET"].reset_index(drop=True)
    df_ab = ep[ep["model"] == "ablation_no_coupling"].reset_index(drop=True)
    df_cp = ep[ep["model"] == "cpl12"].reset_index(drop=True)
    task_a_lines = task_A_swing_vs_stance_reduction(df_wt, df_ab, df_cp, c_st, c_sw)
    # unpack json carrier
    filtered = []
    for ln in task_a_lines:
        if ln.startswith("__TASKA_JSON__"):
            global taskA_json_holder
            taskA_json_holder = json.loads(ln[len("__TASKA_JSON__") :])
        else:
            filtered.append(ln)
    lines.extend(filtered)
    lines.append("")
    lines.extend(task_B_stance_efficiency(df_wt, df_ab, df_cp, c_st, c_dist, ep_steps=82))
    lines.append("")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
