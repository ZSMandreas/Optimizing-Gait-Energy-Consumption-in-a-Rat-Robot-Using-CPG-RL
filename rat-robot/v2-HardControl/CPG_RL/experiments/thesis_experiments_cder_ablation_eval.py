#!/usr/bin/env python3
"""
M1–M9 run_cder_closure_validation protocol for CDER weight ablations (seed 0).

Runs baseline + ablation policy per output directory, reusing
`run_cder_closure_validation.py` task implementations.

Outputs (new dirs only):
  outputs/thesis_experiments_v3.1_ablation_no_norm/
  outputs/thesis_experiments_v3.1_ablation_equal_weights/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for _p in (ROOT, ROOT / "env", ROOT / "experiments", ROOT / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import run_cder_closure_validation as M  # noqa: E402
from ours_cder_v31_env import RatCpgEnvEnergySubstep50ShapeV3  # noqa: E402

V31_BASELINE_OUT = ROOT / "outputs" / "thesis_experiments"
V31_CDER_CKPT = ROOT / "logs/ours/seed0/rat_cpg_ppo_route_a.zip"

ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_norm": {
        "title": "Ablation A (alpha_E_norm = 0)",
        "log_dir": ROOT / "logs/ours_ablation_no_norm/seed0",
        "env_kwargs_file": ROOT / "configs/ablation_no_norm_env_kwargs.json",
        "out_dir": ROOT / "outputs/thesis_experiments_v3.1_ablation_no_norm",
    },
    "equal_weights": {
        "title": "Ablation B (equal CDER weights 0.50)",
        "log_dir": ROOT / "logs/ours_ablation_equal_weights/seed0",
        "env_kwargs_file": ROOT / "configs/ablation_equal_weights_env_kwargs.json",
        "out_dir": ROOT / "outputs/thesis_experiments_v3.1_ablation_equal_weights",
    },
}


def resolve_ckpt(log_dir: Path) -> Path:
    wrapped = log_dir / "rat_cpg_ppo_route_a.zip"
    if wrapped.is_file():
        return wrapped
    cks = sorted(log_dir.glob("checkpoints/rat_cpg_ppo_route_a_*_steps.zip"))
    if cks:
        return cks[-1]
    raise FileNotFoundError(f"No checkpoint under {log_dir}")


def resolve_tb_dir(log_dir: Path) -> Path:
    p = log_dir / "tb_logs" / "PPO_1"
    if p.is_dir():
        return p
    alts = sorted((log_dir / "tb_logs").glob("PPO_*"))
    if alts:
        return alts[0]
    return p


def load_env_kwargs(path: Path) -> dict:
    kw = json.loads(path.read_text(encoding="utf-8"))
    mp = kw.get("model_path", M.XML_REL)
    if not Path(mp).is_absolute():
        kw["model_path"] = str((ROOT / mp).resolve())
    kw.setdefault("render_mode", None)
    kw.setdefault("enable_csv_log", False)
    return kw


def reuse_baseline_artifacts(out_dir: Path, src_dir: Path = V31_BASELINE_OUT) -> None:
    names = [
        "M3_baseline_contact_pattern.csv",
        "M3_baseline_gait_diagram.png",
        "M4_baseline_diagnostic.csv",
        "M4_baseline_diagnostic.png",
        "M5_reward_components_baseline.csv",
        "M6_per_leg_per_component_baseline.csv",
        "M8_per_stride_baseline.csv",
        "M9_cot_distribution_baseline.json",
        "rollout_baseline_env_level.npz",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in names:
        src = src_dir / fn
        if src.is_file():
            shutil.copy2(src, out_dir / fn)


def run_one_ablation(key: str, reuse_baseline: bool) -> None:
    spec = ABLATIONS[key]
    out_dir: Path = spec["out_dir"]
    log_dir: Path = spec["log_dir"]
    ckpt = resolve_ckpt(log_dir)
    env_kw_path: Path = spec["env_kwargs_file"]
    env_kw = load_env_kwargs(env_kw_path)

    print(f"\n{'='*60}\n{spec['title']}\n{'='*60}")
    print(f"  ckpt:    {ckpt}")
    print(f"  out_dir: {out_dir}")
    print(f"  env:     {env_kw_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env(_policy_kind: str):
        if _policy_kind == "cder":
            return RatCpgEnvEnergySubstep50ShapeV3(**env_kw)
        return M.RatCpgEnvEnergySubstep50ShapeV2(
            model_path=M.MODEL_PATH,
            max_episode_steps=4100,
            render_mode=None,
            enable_csv_log=False,
        )

    M.OUT_DIR = out_dir
    M.CKPT_CDER = ckpt
    M.CKPT_BASELINE = M.CKPT_BASELINE  # unchanged W2 path
    M.TB_LOGDIR_CDER = resolve_tb_dir(log_dir)
    M.make_env = make_env

    if reuse_baseline and V31_BASELINE_OUT.is_dir():
        print(f"  reusing baseline artifacts from {V31_BASELINE_OUT}")
        reuse_baseline_artifacts(out_dir)
        _orig = M.collect_rollout
        _cache: dict = {}

        def collect_rollout_cached(policy_kind: str, ckpt_path, n_eps=M.N_EPISODES):
            if policy_kind == "baseline":
                if "logs" not in _cache:
                    print("\n=== Rollout: baseline (cached from v3.1 thesis_experiments) ===")
                    _cache["logs"] = _orig(policy_kind, ckpt_path, n_eps)
                return _cache["logs"]
            return _orig(policy_kind, ckpt_path, n_eps)

        M.collect_rollout = collect_rollout_cached

    M.main()

    prov = out_dir / "eval_provenance.md"
    prov.write_text(
        f"# {spec['title']} — M1–M9 eval\n\n"
        f"- Driver: `{Path(__file__).name}`\n"
        f"- Checkpoint: `{ckpt}`\n"
        f"- Env kwargs: `{env_kw_path}`\n"
        f"- `ours_weights`: `{env_kw.get('ours_weights', {})}`\n"
        f"- Baseline: `{M.CKPT_BASELINE}` (W2 seed1)\n"
        f"- Reuse baseline rollout: {reuse_baseline}\n",
        encoding="utf-8",
    )


def _sanity_fwd_mm_s(log_dir: Path) -> float:
    sc = log_dir / "sanity_check.md"
    if not sc.is_file():
        return float("nan")
    for ln in sc.read_text(encoding="utf-8").splitlines():
        if "Forward speed" in ln and "|" in ln:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    pass
    return float("nan")


def write_comparison_md() -> None:
    entries = [
        ("v3.1 CDER (ref)", V31_BASELINE_OUT, ROOT / "logs/ours/seed0"),
        ("Ablation A no_norm", ABLATIONS["no_norm"]["out_dir"], ABLATIONS["no_norm"]["log_dir"]),
        ("Ablation B equal_w", ABLATIONS["equal_weights"]["out_dir"], ABLATIONS["equal_weights"]["log_dir"]),
    ]
    lines = [
        "# CDER weight ablations — headline comparison\n",
        "| Policy | COT mean | COT std | M9 file | Sanity fwd (mm/s) |",
        "|--------|---------:|--------:|---------|------------------:|",
    ]
    for name, od, log_dir in entries:
        cot_m = cot_s = float("nan")
        m9p = od / "M9_cot_distribution_cder.json"
        if m9p.is_file():
            j = json.loads(m9p.read_text(encoding="utf-8"))
            s = j.get("summary", j)
            cot_m = float(s.get("mean", float("nan")))
            cot_s = float(s.get("std", float("nan")))
        fwd = _sanity_fwd_mm_s(log_dir)
        lines.append(f"| {name} | {cot_m:.3f} | {cot_s:.3f} | `{m9p.name if m9p.is_file() else '—'}` | {fwd:.1f} |")
    lines.append("\nSee per-run `run_cder_closure_validation_report.md` in each output directory.\n")
    out = ROOT / "outputs/thesis_experiments_cder_ablations_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation",
        choices=["no_norm", "equal_weights", "both"],
        default="both",
    )
    ap.add_argument(
        "--reuse-baseline",
        action="store_true",
        default=True,
        help="Reuse baseline rollout from outputs/thesis_experiments (default: on).",
    )
    ap.add_argument(
        "--no-reuse-baseline",
        action="store_false",
        dest="reuse_baseline",
        help="Re-run W2 baseline rollout for each ablation output dir.",
    )
    args = ap.parse_args()

    keys = ["no_norm", "equal_weights"] if args.ablation == "both" else [args.ablation]
    for k in keys:
        spec = ABLATIONS[k]
        if spec["out_dir"].exists() and any(spec["out_dir"].glob("M9_*")):
            print(f"[skip] {k}: outputs already present in {spec['out_dir']}")
            continue
        run_one_ablation(k, reuse_baseline=args.reuse_baseline)

    write_comparison_md()


if __name__ == "__main__":
    main()
