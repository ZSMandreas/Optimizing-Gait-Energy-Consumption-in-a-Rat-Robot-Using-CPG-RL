# eval_rat_mapping.py
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")

from Rat_Env_cpg import Go2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import mujoco as mj


def find_latest_checkpoint(root: str, prefix: str):
    zips = sorted(glob.glob(os.path.join(root, f"{prefix}_*_steps.zip")))
    if not zips:
        return None, None
    latest_zip = zips[-1]
    step = latest_zip.split("_")[-2]
    vn_path = os.path.join(root, f"{prefix}_{step}_vecnormalize.pkl")
    if not os.path.exists(vn_path):
        vn_path = None
    return latest_zip, vn_path


def resolve_model_data(raw_env):
    model, data = None, None
    if hasattr(raw_env, "model") and hasattr(raw_env, "data"):
        model, data = raw_env.model, raw_env.data
    elif hasattr(raw_env, "sim"):
        sim = raw_env.sim
        if hasattr(sim, "model") and hasattr(sim, "data"):
            model, data = sim.model, sim.data
    return model, data


def name2id(model, obj_type, name):
    try:
        return mj.mj_name2id(model, obj_type, name)
    except Exception:
        return -1


def try_fetch_target_lists(raw_env):
    candidates = []
    if hasattr(raw_env, "controller"):
        candidates.append(raw_env.controller)
    if hasattr(raw_env, "state"):
        st = raw_env.state
        candidates.append(st)
        if hasattr(st, "controller"):
            candidates.append(st.controller)

    for obj in candidates:
        if obj is None:
            continue
        tx = getattr(obj, "trgXList", None)
        ty = getattr(obj, "trgYList", None)
        if tx is not None and ty is not None and len(tx) == 4 and len(ty) == 4:
            return tx, ty
    return None, None


def plot_and_save(out_dir, t, base_xy, vy, feet_zabs, feet_yrel, feet_zrel,
                  trgXList=None, trgYList=None):
    os.makedirs(out_dir, exist_ok=True)

    if t is not None:
        np.save(os.path.join(out_dir, "t.npy"), t)
    if base_xy is not None:
        np.save(os.path.join(out_dir, "base_xy.npy"), base_xy)
    if vy is not None:
        np.save(os.path.join(out_dir, "vy.npy"), vy)
    if feet_zabs is not None:
        np.save(os.path.join(out_dir, "feet_zabs.npy"), feet_zabs)
    if feet_yrel is not None:
        np.save(os.path.join(out_dir, "feet_yrel.npy"), feet_yrel)
    if feet_zrel is not None:
        np.save(os.path.join(out_dir, "feet_zrel.npy"), feet_zrel)
    if trgXList is not None and trgYList is not None:
        np.save(os.path.join(out_dir, "trgXList.npy"), np.array([np.asarray(x) for x in trgXList], dtype=object))
        np.save(os.path.join(out_dir, "trgYList.npy"), np.array([np.asarray(y) for y in trgYList], dtype=object))

    # Base path
    if base_xy is not None and len(base_xy) > 1:
        plt.figure(figsize=(5, 5))
        plt.plot(base_xy[:, 0], base_xy[:, 1], lw=2)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Base XY Path")
        plt.axis("equal")
        plt.grid(True, ls="--", alpha=0.5)
        plt.savefig(os.path.join(out_dir, "base_xy.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Forward velocity
    if (vy is not None) and (t is not None):
        plt.figure(figsize=(7, 3))
        plt.plot(t[:len(vy)], vy, lw=1.8)
        plt.xlabel("time (s)")
        plt.ylabel("vy (m/s)")
        plt.title("Forward Velocity")
        plt.grid(True, ls="--", alpha=0.5)
        plt.savefig(os.path.join(out_dir, "vy.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Feet z_abs
    if feet_zabs is not None and t is not None:
        names = ["FL", "FR", "RL", "RR"]
        fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            axs[r][c].plot(t[:feet_zabs.shape[1]], feet_zabs[i], lw=1.6)
            axs[r][c].set_title(f"{names[i]} | z_abs")
            axs[r][c].set_ylabel("z (m)")
            axs[r][c].grid(True, ls="--", alpha=0.4)
        axs[1][0].set_xlabel("time (s)")
        axs[1][1].set_xlabel("time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feet_zabs.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 相对轨迹 vs 目标
    if feet_yrel is not None and feet_zrel is not None:
        names = ["Fore Left", "Fore Right", "Hind Left", "Hind Right"]
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            axs[r, c].set_title(names[i])
            axs[r, c].plot(feet_yrel[i], feet_zrel[i], label="Real")
            if trgXList is not None and trgYList is not None:
                tx = np.asarray(trgXList[i]).squeeze()
                ty = np.asarray(trgYList[i]).squeeze()
                L = min(len(tx), len(ty), feet_yrel.shape[1])
                if L > 0:
                    axs[r, c].plot(tx[:L], ty[:L], label="Target")
            axs[r, c].legend()
            axs[r, c].grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feet_rel_compare.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[EVAL] Saved plots & npy to: {out_dir}")


def main(args):
    # 模型 & VecNormalize
    model_path = args.model
    vn_path = args.vecnorm
    if model_path is None:
        model_path, vn_path_auto = find_latest_checkpoint(args.ckpt_dir, args.prefix)
        if model_path is None:
            model_path = os.path.join(args.ckpt_dir, f"{args.prefix}_final.zip")
            vn_path_auto = os.path.join(args.ckpt_dir, f"{args.prefix}_final_vecnormalize.pkl")
            print("[WARN] No step checkpoint found, fallback to FINAL.")
        if vn_path is None:
            vn_path = vn_path_auto

    test_env = make_vec_env(lambda: Go2Env(render_mode="window"), n_envs=1, seed=args.seed)
    if vn_path is not None and os.path.exists(vn_path):
        test_env = VecNormalize.load(vn_path, test_env)
        print(f"[INFO] Loaded VecNormalize from: {vn_path}")
    else:
        test_env = VecNormalize(test_env, norm_obs=True)
        print("[INFO] No VecNormalize file found. Created a fresh one.")
    test_env.training = False
    test_env.norm_reward = False

    model = PPO.load(model_path, env=test_env, device=args.device)
    print(f"[INFO] Loaded model: {model_path}")

    raw_env = test_env.envs[0]
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env

    model_mj, data_mj = resolve_model_data(raw_env)
    if model_mj is None or data_mj is None:
        print("[WARN] Cannot access mujoco model/data.")

    if model_mj is not None:
        sim_dt = float(model_mj.opt.timestep)
    else:
        sim_dt = 0.002

    # time_slice_steps 仅用于 step 级 dt_step（画 base_xy / vy）
    if hasattr(raw_env, "time_slice_steps"):
        substeps = int(getattr(raw_env, "time_slice_steps"))
    else:
        substeps = 1
    substeps = max(substeps, 1)
    dt_step = sim_dt * substeps
    print(f"[INFO] sim_dt={sim_dt}, substeps_per_step={substeps}, dt_step={dt_step}")

    # Site
    body_fix_name = args.body_fix
    thighs = args.thigh_sites.split(",") if args.thigh_sites else ["thigh_link_fl","thigh_link_fr","thigh_link_rl","thigh_link_rr"]
    ankles = args.ankle_sites.split(",") if args.ankle_sites else ["ankle_fl","ankle_fr","ankle_rl","ankle_rr"]

    if model_mj is not None:
        body_fix_id = name2id(model_mj, mj.mjtObj.mjOBJ_SITE, body_fix_name)
        thigh_ids = [name2id(model_mj, mj.mjtObj.mjOBJ_SITE, n) for n in thighs]
        ankle_ids = [name2id(model_mj, mj.mjtObj.mjOBJ_SITE, n) for n in ankles]
    else:
        body_fix_id = -1
        thigh_ids = [-1, -1, -1, -1]
        ankle_ids = [-1, -1, -1, -1]

    trgXList, trgYList = try_fetch_target_lists(raw_env)
    if trgXList is not None:
        print("[INFO] Found target relative trajectories.")
    else:
        print("[INFO] No target trajectories found.")

    obs = test_env.reset()
    steps = 0

    base_xy = []
    vy_list = []
    prev_fix = None

    feet_zabs = [[] for _ in range(4)]
    feet_yrel = [[] for _ in range(4)]
    feet_zrel = [[] for _ in range(4)]

    # --------- substep 级日志 ---------
    cot_sub_list = []           # [N_sub]
    motor_sub_list = []         # [N_sub, 8]
    q_err_sub_list = []         # [N_sub, 8]
    q_act_sub_list = []         # [N_sub, 8]
    q_cmd_sub_list = []         # [N_sub, 8]
    action_sub_list = []        # [N_sub, 8]

    while steps < args.eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = test_env.step(action)

        info0 = infos[0] if isinstance(infos, (list, tuple, np.ndarray)) else infos

        # -------- substep 数据从 info 里展开 --------
        sub_cot = info0.get("sub_cot", None)
        if sub_cot is not None:
            sub_cot = np.asarray(sub_cot, dtype=float).reshape(-1)
            cot_sub_list.extend(sub_cot.tolist())

        sub_motor = info0.get("sub_motor_avg_current", None)
        if sub_motor is not None:
            sub_motor = np.asarray(sub_motor, dtype=float)
            for row in sub_motor:
                motor_sub_list.append(row.copy())

        sub_q_err = info0.get("sub_q_err", None)
        if sub_q_err is not None:
            sub_q_err = np.asarray(sub_q_err, dtype=float)
            for row in sub_q_err:
                q_err_sub_list.append(row.copy())

        sub_q_act = info0.get("sub_q_act_leg", None)
        if sub_q_act is not None:
            sub_q_act = np.asarray(sub_q_act, dtype=float)
            for row in sub_q_act:
                q_act_sub_list.append(row.copy())

        sub_q_cmd = info0.get("sub_q_cmd_leg", None)
        if sub_q_cmd is not None:
            sub_q_cmd = np.asarray(sub_q_cmd, dtype=float)
            for row in sub_q_cmd:
                q_cmd_sub_list.append(row.copy())

        sub_act = info0.get("sub_action_filtered", None)
        if sub_act is not None:
            sub_act = np.asarray(sub_act, dtype=float)
            for row in sub_act:
                action_sub_list.append(row.copy())

        # -------- step 级 base / feet --------
        if (model_mj is not None) and (body_fix_id >= 0):
            p = data_mj.site_xpos[body_fix_id].copy()
            base_xy.append([p[0], p[1]])
            if prev_fix is not None:
                vy = (p[1] - prev_fix[1]) / dt_step
            else:
                vy = 0.0
            vy_list.append(vy)
            prev_fix = p.copy()

        if model_mj is not None:
            for i in range(4):
                sid_th = thigh_ids[i] if i < len(thigh_ids) else -1
                sid_ak = ankle_ids[i] if i < len(ankles) else -1

                if sid_ak >= 0:
                    zabs = data_mj.site_xpos[sid_ak][2]
                    feet_zabs[i].append(zabs)
                else:
                    feet_zabs[i].append(np.nan)

                if sid_th >= 0 and sid_ak >= 0:
                    rel_y = data_mj.site_xpos[sid_ak][1] - data_mj.site_xpos[sid_th][1]
                    rel_z = data_mj.site_xpos[sid_ak][2] - data_mj.site_xpos[sid_th][2]
                else:
                    rel_y, rel_z = np.nan, np.nan
                feet_yrel[i].append(rel_y)
                feet_zrel[i].append(rel_z)

        steps += 1
        if isinstance(done, (list, tuple, np.ndarray)):
            if np.any(done):
                obs = test_env.reset()
        else:
            if done:
                obs = test_env.reset()

    # -------- step 级整理用于画 base_xy / vy --------
    t_axis = np.arange(steps) * dt_step
    base_xy = np.asarray(base_xy) if len(base_xy) > 0 else None
    vy_arr = np.asarray(vy_list) if len(vy_list) > 0 else None

    def stack_or_none(lst):
        if len(lst[0]) == 0:
            return None
        return np.stack([np.asarray(x) for x in lst], axis=0)

    zabs_arr = stack_or_none(feet_zabs)
    yrel_arr = stack_or_none(feet_yrel)
    zrel_arr = stack_or_none(feet_zrel)

    out_dir = os.path.join(args.out_dir, f"steps_{steps}_eval")
    plot_and_save(out_dir, t_axis, base_xy, vy_arr, zabs_arr, yrel_arr, zrel_arr,
                  trgXList=trgXList, trgYList=trgYList)

    # -------- 下面是 substep CSV，第一列恒为 time --------
    # 1) substep CoT
    if len(cot_sub_list) > 0:
        cot_arr = np.array(cot_sub_list, dtype=float).reshape(-1, 1)
        N = cot_arr.shape[0]
        t_sub = np.arange(N) * sim_dt
        data = np.column_stack([t_sub, cot_arr])
        csv_path = os.path.join(out_dir, "cot_substep.csv")
        np.savetxt(csv_path, data, delimiter=",", header="time,cot_substep", comments="")
        print(f"[EVAL] Saved substep cot CSV to: {csv_path}")
    else:
        print("[EVAL] No substep cot found; CSV not saved.")

    # 2) substep motor
    if len(motor_sub_list) > 0:
        motor_mat = np.vstack(motor_sub_list)
        N = motor_mat.shape[0]
        t_sub = np.arange(N) * sim_dt
        data = np.column_stack([t_sub, motor_mat])
        if hasattr(raw_env, "motor_labels"):
            header_curr = "time," + ",".join(raw_env.motor_labels)
        else:
            header_curr = "time," + ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_curr = os.path.join(out_dir, "motor_avg_current_substep.csv")
        np.savetxt(csv_path_curr, data, delimiter=",", header=header_curr, comments="")
        print(f"[EVAL] Saved substep motor CSV to: {csv_path_curr}")
    else:
        print("[EVAL] No substep motor found; CSV not saved.")

    # 3) substep q_act
    if len(q_act_sub_list) > 0:
        q_act_mat = np.vstack(q_act_sub_list)
        N = q_act_mat.shape[0]
        t_sub = np.arange(N) * sim_dt
        data = np.column_stack([t_sub, q_act_mat])
        if hasattr(raw_env, "motor_labels"):
            header_q_act = "time," + ",".join(raw_env.motor_labels)
        else:
            header_q_act = "time," + ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_q_act = os.path.join(out_dir, "q_act_leg_substep.csv")
        np.savetxt(csv_path_q_act, data, delimiter=",", header=header_q_act, comments="")
        print(f"[EVAL] Saved substep q_act CSV to: {csv_path_q_act}")
    else:
        print("[EVAL] No substep q_act found; CSV not saved.")

    # 4) substep q_err
    if len(q_err_sub_list) > 0:
        q_err_mat = np.vstack(q_err_sub_list)
        N = q_err_mat.shape[0]
        t_sub = np.arange(N) * sim_dt
        data = np.column_stack([t_sub, q_err_mat])
        if hasattr(raw_env, "motor_labels"):
            header_q_err = "time," + ",".join(raw_env.motor_labels)
        else:
            header_q_err = "time," + ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_q_err = os.path.join(out_dir, "q_err_substep.csv")
        np.savetxt(csv_path_q_err, data, delimiter=",", header=header_q_err, comments="")
        print(f"[EVAL] Saved substep q_err CSV to: {csv_path_q_err}")
    else:
        print("[EVAL] No substep q_err found; CSV not saved.")

    # 5) substep q_cmd
    if len(q_cmd_sub_list) > 0:
        q_cmd_mat = np.vstack(q_cmd_sub_list)
        N = q_cmd_mat.shape[0]
        t_sub = np.arange(N) * sim_dt
        data = np.column_stack([t_sub, q_cmd_mat])
        if hasattr(raw_env, "motor_labels"):
            header_q_cmd = "time," + ",".join(raw_env.motor_labels)
        else:
            header_q_cmd = "time," + ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_q_cmd = os.path.join(out_dir, "q_cmd_leg_substep.csv")
        np.savetxt(csv_path_q_cmd, data, delimiter=",", header=header_q_cmd, comments="")
        print(f"[EVAL] Saved substep q_cmd CSV to: {csv_path_q_cmd}")
    else:
        print("[EVAL] No substep q_cmd found; CSV not saved.")

    # 6) substep action
    if len(action_sub_list) > 0:
        action_mat = np.vstack(action_sub_list)
        N = action_mat.shape[0]
        t_sub = np.arange(N) * sim_dt
        data = np.column_stack([t_sub, action_mat])
        header_action = "time," + ",".join([f"a{i}" for i in range(action_mat.shape[1])])
        csv_path_action = os.path.join(out_dir, "action_filtered_substep.csv")
        np.savetxt(csv_path_action, data, delimiter=",", header=header_action, comments="")
        print(f"[EVAL] Saved substep action CSV to: {csv_path_action}")
    else:
        print("[EVAL] No substep action found; CSV not saved.")

    test_env.close()
    print("[DONE] Evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--prefix", type=str, default="rat_mapping")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=14)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--out_dir", type=str, default="./eval_vis")
    parser.add_argument("--body_fix", type=str, default="body_ss")
    parser.add_argument("--thigh_sites", type=str, default="")
    parser.add_argument("--ankle_sites", type=str, default="")

    args = parser.parse_args()
    main(args)
