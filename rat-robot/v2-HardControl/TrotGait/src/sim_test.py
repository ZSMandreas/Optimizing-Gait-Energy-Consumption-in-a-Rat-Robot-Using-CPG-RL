# sim_test.py
# -*- coding: utf-8 -*-

import argparse
import os
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ToSim import SimModel
from Controller import MouseController
from sim_closure import compute_closure_from_sim, save_closure_outputs

SRC_DIR = Path(__file__).resolve().parent
# Same MuJoCo model as W2 / Ours RL eval (thesis_experiments_master, eval_kp_ablation)
MODEL_XML = (SRC_DIR / "../models/dynamic_4l_kp2.xml").resolve()

LEG_ACTUATOR_NAMES = [
    "thigh_joint_fl", "leg_joint_fl",
    "thigh_joint_fr", "leg_joint_fr",
    "thigh_joint_rl", "leg_joint_rl",
    "thigh_joint_rr", "leg_joint_rr",
]

MEASURE_TIME_LENGTH = 8.0   # closure统计窗口 (s)，对齐 thesis 单 episode 8.2s
WARMUP_TIME_LENGTH = 2.0    # CPG 稳定预热，不计入 closure


def override_kp_kv(model: mujoco.MjModel, kp: float = 2.0, kv: float = 0.0) -> int:
    """Match eval_kp_ablation.py / thesis eval (kp2_kv0)."""
    n = 0
    for name in LEG_ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            print(f"[warn] actuator '{name}' not found, skipping")
            continue
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kv
        n += 1
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Open-loop CPG sim + closure validation")
    ap.add_argument("--render", action="store_true", help="Enable MuJoCo viewer (default: headless)")
    ap.add_argument("--plot", action="store_true", help="Show matplotlib figures (needs display)")
    ap.add_argument(
        "--measure-time",
        type=float,
        default=MEASURE_TIME_LENGTH,
        help="Closure measurement window after warmup (s)",
    )
    ap.add_argument(
        "--warmup-time",
        type=float,
        default=WARMUP_TIME_LENGTH,
        help="CPG warmup before closure stats (s), not included in balance",
    )
    ap.add_argument(
        "--run-time",
        type=float,
        default=None,
        help="Deprecated alias for --measure-time",
    )
    ap.add_argument("--out-dir", type=str, default="./sim_logs_kp2_cpg", help="Log output directory")
    ap.add_argument("--kp", type=float, default=2.0, help="Leg actuator kp (default 2, same as thesis eval)")
    ap.add_argument("--kv", type=float, default=0.0, help="Leg actuator kv (default 0)")
    args = ap.parse_args()

    if not args.plot:
        matplotlib.use("Agg")

    fre = 0.5
    time_step = 0.002
    spine_angle = 20
    measure_time = float(args.measure_time if args.run_time is None else args.run_time)
    warmup_time = float(args.warmup_time)
    measure_steps = int(measure_time / time_step)
    warmup_steps = int(warmup_time / time_step)
    ideal_contacts_measure = []

    model_xml = str(MODEL_XML)
    if not Path(model_xml).is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_xml}")

    theMouse = SimModel(model_xml)
    n_ovr = override_kp_kv(theMouse.model, kp=float(args.kp), kv=float(args.kv))
    print(f"[model] {model_xml}")
    print(f"[actuators] overridden kp={args.kp}, kv={args.kv} on {n_ovr} leg actuators")

    ENABLE_RENDER = bool(args.render)
    theMouse.enable_render(ENABLE_RENDER, camera=(0.7, -20.0, 90.0))

    theController = MouseController(fre, time_step, spine_angle)

    theMouse.print_action()
    print("[Motor DOF idx parsed] ->", theMouse.motor_dof_idx)
    print("[Motor joints] ->", theMouse.motor_joint_names)

    # 初始姿态稳定（固定 ctrl，不计入 closure）
    settle_steps = int(2.0 / time_step)
    for _ in range(settle_steps):
        ctrlData = [0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, 0, 0, 0, 0]
        theMouse.runStep(ctrlData, time_step, realtime=ENABLE_RENDER)

    theMouse.print_action()

    theMouse.initializing()

    # CPG 预热 2s：让步态/接触进入周期稳态，丢弃此段日志
    print(f"[warmup] CPG {warmup_time:.1f} s ({warmup_steps} macro-steps), excluded from closure")
    for _ in range(warmup_steps):
        tCtrlData, _ = theController.runStep()
        theMouse.runStep(tCtrlData, time_step, realtime=ENABLE_RENDER)

    # 清零后才开始 closure 统计（能量/接触/轨迹日志仅含测量窗）
    theMouse.initializing()
    theController.trgXList = [[], [], [], []]
    theController.trgYList = [[], [], [], []]
    theMouse.snapshot_closure_start()

    print(f"[measure] closure window {measure_time:.1f} s ({measure_steps} macro-steps)")
    start = time.time()
    for _ in range(measure_steps):
        tCtrlData, ideal_contact = theController.runStep()
        ideal_contacts_measure.append(ideal_contact)
        theMouse.runStep(tCtrlData, time_step, realtime=ENABLE_RENDER)

    end = time.time()
    timeCost = end - start
    print("Measure wall time -> ", timeCost)

    theMouse.snapshot_closure_end()
    closure = compute_closure_from_sim(theMouse)
    print(
        f"\n=== Closure validation (open-loop CPG, dynamic_4l_kp2.xml, kp2_kv0) "
        f"[after {warmup_time:.1f}s warmup] ==="
    )
    print(f"  W+_total     = {closure['Wp_J']:.4f} J")
    print(f"  W-_total     = {closure['Wn_J']:.4f} J")
    print(f"  E_damp       = {closure['Edamp_J']:.4f} J")
    print(f"  E_fric       = {closure['Efric_J']:.4f} J")
    print(f"  E_norm       = {closure['Enorm_J']:.4f} J")
    print(f"  ΔKE          = {closure['delta_KE_J']:+.6f} J")
    print(f"  ΔPE          = {closure['delta_PE_J']:+.6f} J")
    print(f"  residual     = {closure['residual_J']:+.4f} J  ({closure['residual_pct_of_Wp']:+.2f}% of W+)")
    print(f"  distance     = {closure['distance_m']:.6f} m")
    print(f"  COT (W+/mgd) = {closure['COT_mech']:.4f}")
    print(f"  |r| < 5%     = {'yes' if abs(closure['residual_pct_of_Wp']) < 5 else 'no'}")

    ideal_contacts = np.array(ideal_contacts_measure)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ideal_contacts.npy"), ideal_contacts)

    dis = theMouse.drawPath(
        show=args.plot,
        save_fig_path=None if args.plot else os.path.join(out_dir, "body_path_xy.png"),
    )
    print("py_v --> ", dis / timeCost)
    print("sim_v --> ", dis / (measure_steps * time_step))
    theMouse.savePath("hang_fixed")

    cot_val = theMouse.cot()
    print(f"[COT] Cost of Transport (EnergyMeter) = {cot_val:.6f}")

    with open(os.path.join(out_dir, "cot.txt"), "w") as f:
        f.write(f"{cot_val:.10f}\n")

    theMouse.save_numpy(out_dir)
    save_closure_outputs(closure, out_dir, title="sim_test open-loop CPG closure (dynamic_4l_kp2)")

    meta = {
        "xml": model_xml,
        "warmup_time_s": warmup_time,
        "warmup_steps": warmup_steps,
        "measure_time_s": measure_time,
        "measure_steps": measure_steps,
        "time_step_arg": time_step,
        "model_dt": theMouse.dt,
        "motor_joint_names": theMouse.motor_joint_names,
        "motor_dof_idx": [int(x) for x in theMouse.motor_dof_idx],
        "nu": int(theMouse.model.nu),
        "nv": int(theMouse.model.nv),
        "nq": int(theMouse.model.nq),
        "render_enabled": ENABLE_RENDER,
        "kp": float(args.kp),
        "kv": float(args.kv),
        "closure": closure,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    subTitle = ["Fore Left Leg", "Fore Right Leg", "Hind Left Leg", "Hind Right Leg"]
    for i in range(4):
        pos_1 = int(i / 2)
        pos_2 = int(i % 2)
        axs[pos_1, pos_2].set_title(subTitle[i])
        axs[pos_1, pos_2].plot(theController.trgXList[i], theController.trgYList[i], label="Target")
        axs[pos_1, pos_2].plot(theMouse.legRealPoint_x[i], theMouse.legRealPoint_y[i], label="Real")
        axs[pos_1, pos_2].legend()
        axs[pos_1, pos_2].grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(out_dir, "leg_trajectories_4panel.png"), dpi=120)
        plt.close()

    plt.figure()
    plt.plot(theController.trgXList[0], theController.trgYList[0], label='Target trajectory')
    plt.plot(theMouse.legRealPoint_x[0], theMouse.legRealPoint_y[0], label='Real trajectory')
    plt.legend()
    plt.xlabel('y-coordinate (m)')
    plt.ylabel('z-coordinate (m)')
    plt.grid()
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(out_dir, "leg0_trajectory.png"), dpi=120)
        plt.close()

    vy = np.array(theMouse.body_velocity[0:measure_steps])
    plt.figure()
    plt.plot(np.arange(vy.shape[0]) * time_step, vy, label='body vy (approx)')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('m/s')
    plt.grid()
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(out_dir, "body_vy.png"), dpi=120)
        plt.close()

    print(f"\nLogs + closure report → {out_dir}/")
