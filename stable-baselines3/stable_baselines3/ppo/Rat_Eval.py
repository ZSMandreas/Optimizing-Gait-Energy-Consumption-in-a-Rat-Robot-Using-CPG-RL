# eval_rat_mapping.py
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# 工程路径
os.chdir("/home/simiao/rat_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")

from Rat_Env_cpg import Go2Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 官方 mujoco
import mujoco as mj


def find_latest_checkpoint(root: str, prefix: str):
    zips = sorted(glob.glob(os.path.join(root, f"{prefix}_*_steps.zip")))
    if not zips:
        return None, None
    latest_zip = zips[-1]
    step = latest_zip.split("_")[-2]  # rat_mapping_3000000_steps.zip → "3000000"
    vn_path = os.path.join(root, f"{prefix}_{step}_vecnormalize.pkl")
    if not os.path.exists(vn_path):
        vn_path = None
    return latest_zip, vn_path


# =============== 低耦合工具：解析 env 的 mujoco model/data、目标轨迹 ===============
def resolve_model_data(raw_env):
    """尽量从 raw_env 拿到 mujoco model/data（不同封装字段名可能不同）"""
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
    """
    尝试像 sim_test.py 一样拿目标相对轨迹：
      - 优先查 raw_env.controller.trgXList / trgYList
      - 再查 raw_env.state.controller / raw_env.state
      - 找不到就返回 (None, None)
    返回：
      (trgXList, trgYList)，其中每个应为长度=4 的 list，每个元素为 1D 数组/列表（时间序列）
    """
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
        if tx is not None and ty is not None:
            if len(tx) == 4 and len(ty) == 4:
                return tx, ty
    return None, None


# ================= 保存与作图 =================
def plot_and_save(out_dir, t, base_xy, vy, feet_zabs, feet_yrel, feet_zrel,
                  trgXList=None, trgYList=None):
    os.makedirs(out_dir, exist_ok=True)

    # 保存 numpy
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
        np.save(os.path.join(out_dir, "trgXList.npy"),
                np.array([np.asarray(x) for x in trgXList], dtype=object))
        np.save(os.path.join(out_dir, "trgYList.npy"),
                np.array([np.asarray(y) for y in trgYList], dtype=object))

    # 1) Base XY Path
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

    # 2) Forward velocity
    if (vy is not None) and (t is not None):
        plt.figure(figsize=(7, 3))
        plt.plot(t[:len(vy)], vy, lw=1.8)
        plt.xlabel("time (s)")
        plt.ylabel("vy (m/s)")
        plt.title("Forward Velocity")
        plt.grid(True, ls="--", alpha=0.5)
        plt.savefig(os.path.join(out_dir, "vy.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 3) Feet z_abs（四条腿）
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

    # 4) 相对足端轨迹 vs 目标
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


# ================= 主逻辑 =================
def main(args):
    # 解析模型和 VecNormalize
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

    # 构建评测环境（window 渲染）
    test_env = make_vec_env(lambda: Go2Env(render_mode="window"), n_envs=1, seed=args.seed)
    if vn_path is not None and os.path.exists(vn_path):
        test_env = VecNormalize.load(vn_path, test_env)
        print(f"[INFO] Loaded VecNormalize from: {vn_path}")
    else:
        test_env = VecNormalize(test_env, norm_obs=True)
        print("[INFO] No VecNormalize file found. Created a fresh one (not recommended for fair eval).")
    test_env.training = False
    test_env.norm_reward = False
    
    import stable_baselines3.common.utils as sb3_utils

    # # 兼容旧模型里引用的 FloatSchedule
    # if not hasattr(sb3_utils, "FloatSchedule"):
    #     class FloatSchedule:
    #         # pickle 反序列化时一般不会调用 __init__，但这里给个默认值更保险
    #         def __init__(self, value=0.0):
    #             self.value = float(value)

    #         def __call__(self, progress_remaining: float):
    #             # progress_remaining 在 SB3 里通常是 [1 -> 0]
    #             return float(getattr(self, "value", 0.0))

    #     sb3_utils.FloatSchedule = FloatSchedule

    # 加载模型
    model = PPO.load(model_path, env=test_env, device=args.device)
    print(f"[INFO] Loaded model: {model_path}")

    # 取 raw_env
    raw_env = test_env.envs[0]
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env

    # 尝试解析 mujoco model/data
    model_mj, data_mj = resolve_model_data(raw_env)
    if model_mj is None or data_mj is None:
        print("[WARN] Cannot access mujoco model/data. Visualization data will be limited.")

    # 解析 time 信息
    if model_mj is not None:
        sim_dt = float(model_mj.opt.timestep)   # substep dt
    else:
        sim_dt = 0.002
    dt = sim_dt  # 这里 dt 用来做 t_axis 和 vy，仍然按 substep 时间刻度

    # 解析 site id
    body_fix_name = args.body_fix
    thighs = args.thigh_sites.split(",") if args.thigh_sites else [
        "thigh_link_fl", "thigh_link_fr", "thigh_link_rl", "thigh_link_rr"
    ]
    ankles = args.ankle_sites.split(",") if args.ankle_sites else [
        "ankle_fl", "ankle_fr", "ankle_rl", "ankle_rr"
    ]

    if model_mj is not None:
        body_fix_id = name2id(model_mj, mj.mjtObj.mjOBJ_SITE, body_fix_name)
        thigh_ids = [name2id(model_mj, mj.mjtObj.mjOBJ_SITE, n) for n in thighs]
        ankle_ids = [name2id(model_mj, mj.mjtObj.mjOBJ_SITE, n) for n in ankles]
    else:
        body_fix_id = -1
        thigh_ids = [-1, -1, -1, -1]
        ankle_ids = [-1, -1, -1, -1]

    # 试图获取“目标相对轨迹”（若环境有 controller）
    trgXList, trgYList = try_fetch_target_lists(raw_env)
    if trgXList is not None:
        print("[INFO] Found target relative trajectories from env (trgXList/trgYList).")
    else:
        print("[INFO] No target trajectories found in env; will only plot real relative tracks.")

    # 评测循环 + 采集（step 级）
    obs = test_env.reset()
    steps = 0

    base_xy = []
    vy_list = []
    prev_fix = None

    feet_zabs = [[] for _ in range(4)]
    feet_yrel = [[] for _ in range(4)]
    feet_zrel = [[] for _ in range(4)]

    cot_step_list = []
    motor_current_steps = []
    q_err_steps = []
    q_act_leg_steps = []
    q_cmd_leg_steps = []
    action_steps = []

    # ===== 新增：substep 级日志容器 =====
    sub_q_cmd_all = []          # list of [K,8]
    sub_q_act_all = []          # list of [K,8]
    sub_motor_tau_all = []      # list of [K,8]

    # 跑 eval_steps
    while steps < args.eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = test_env.step(action)

        try:
            test_env.render()
        except Exception:
            pass

        info0 = infos[0] if isinstance(infos, (list, tuple, np.ndarray)) else infos

        cot_step = info0.get("cot_step", None)
        if cot_step is not None:
            cot_step_list.append(float(cot_step))

        motor_curr = info0.get("motor_avg_current", None)
        if motor_curr is not None:
            motor_curr = np.asarray(motor_curr, dtype=float)
            motor_current_steps.append(motor_curr)

        q_err = info0.get("q_err", None)
        if q_err is not None:
            q_err = np.asarray(q_err, dtype=float)
            q_err_steps.append(q_err)

        q_act_leg = info0.get("q_act_leg", None)
        if q_act_leg is not None:
            q_act_leg = np.asarray(q_act_leg, dtype=float)
            q_act_leg_steps.append(q_act_leg)

        q_cmd_leg = info0.get("q_cmd_leg", None)
        if q_cmd_leg is not None:
            q_cmd_leg = np.asarray(q_cmd_leg, dtype=float)
            q_cmd_leg_steps.append(q_cmd_leg)

        act_info = info0.get("action", None)
        if act_info is not None:
            act_info = np.asarray(act_info, dtype=float)
            action_steps.append(act_info)

        # ===== 关键：读取 substep 级数据 =====
        sub_q_cmd = info0.get("sub_q_cmd_leg", None)
        if sub_q_cmd is not None:
            sub_q_cmd_all.append(np.asarray(sub_q_cmd, dtype=float))

        sub_q_act = info0.get("sub_q_act_leg", None)
        if sub_q_act is not None:
            sub_q_act_all.append(np.asarray(sub_q_act, dtype=float))

        # 这里用 env 里记录的 substep 力矩数组 key：sub_motor_avg_current
        sub_motor = info0.get("sub_motor_avg_current", None)
        if sub_motor is not None:
            sub_motor_tau_all.append(np.asarray(sub_motor, dtype=float))

        # 采集 base_xy / vy（step 级）
        if (model_mj is not None) and (body_fix_id >= 0):
            p = data_mj.site_xpos[body_fix_id].copy()
            base_xy.append([p[0], p[1]])
            if prev_fix is not None:
                vy = (p[1] - prev_fix[1]) / dt
            else:
                vy = 0.0
            vy_list.append(vy)
            prev_fix = p.copy()

        # 采集四足数据：相对 y/z + 绝对 z
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

    # 整理数组（step 级）
    t_axis = np.arange(steps) * dt
    base_xy = np.asarray(base_xy) if len(base_xy) > 0 else None
    vy_arr = np.asarray(vy_list) if len(vy_list) > 0 else None

    def stack_or_none(lst):
        if len(lst[0]) == 0:
            return None
        return np.stack([np.asarray(x) for x in lst], axis=0)

    zabs_arr = stack_or_none(feet_zabs)
    yrel_arr = stack_or_none(feet_yrel)
    zrel_arr = stack_or_none(feet_zrel)

    # 输出目录
    out_dir = os.path.join(args.out_dir, f"steps_{steps}_eval")
    os.makedirs(out_dir, exist_ok=True)

    # 画图 + 保存（带相对轨迹对比）
    plot_and_save(out_dir, t_axis, base_xy, vy_arr, zabs_arr, yrel_arr, zrel_arr,
                  trgXList=trgXList, trgYList=trgYList)

    # ===== step 级 CSV（与之前一致） =====
    if len(cot_step_list) > 0:
        cot_arr = np.array(cot_step_list, dtype=float)
        csv_path = os.path.join(out_dir, "cot_step.csv")
        np.savetxt(csv_path, cot_arr, delimiter=",", header="cot_step", comments="")
        print(f"[EVAL] Saved cot_step CSV to: {csv_path}")
    else:
        print("[EVAL] No cot_step found in infos; CSV not saved.")

    if len(motor_current_steps) > 0:
        motor_mat = np.vstack(motor_current_steps)  # [T,8]
        if hasattr(raw_env, "motor_labels"):
            header_curr = ",".join(raw_env.motor_labels)
        else:
            header_curr = ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_curr = os.path.join(out_dir, "motor_avg_current.csv")
        np.savetxt(csv_path_curr, motor_mat, delimiter=",", header=header_curr, comments="")
        print(f"[EVAL] Saved motor_avg_current CSV to: {csv_path_curr}")
    else:
        print("[EVAL] No motor_avg_current found in infos; CSV not saved.")

    if len(q_act_leg_steps) > 0:
        q_act_leg_mat = np.vstack(q_act_leg_steps)
        if hasattr(raw_env, "motor_labels"):
            header_q_act_leg = ",".join(raw_env.motor_labels)
        else:
            header_q_act_leg = ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_q_act_leg = os.path.join(out_dir, "q_act_leg_eval.csv")
        np.savetxt(csv_path_q_act_leg, q_act_leg_mat, delimiter=",",
                   header=header_q_act_leg, comments="")
        print(f"[EVAL] Saved q_act_leg CSV to: {csv_path_q_act_leg}")
    else:
        print("[EVAL] No q_act_leg found in infos; CSV not saved.")

    if len(q_err_steps) > 0:
        q_err_mat = np.vstack(q_err_steps)
        if hasattr(raw_env, "motor_labels"):
            header_qerr = ",".join(raw_env.motor_labels)
        else:
            header_qerr = ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_qerr = os.path.join(out_dir, "q_err_eval.csv")
        np.savetxt(csv_path_qerr, q_err_mat, delimiter=",",
                   header=header_qerr, comments="")
        print(f"[EVAL] Saved q_err CSV to: {csv_path_qerr}")
    else:
        print("[EVAL] No q_err found in infos; CSV not saved.")

    if len(q_cmd_leg_steps) > 0:
        q_cmd_leg_mat = np.vstack(q_cmd_leg_steps)
        if hasattr(raw_env, "motor_labels"):
            header_q_cmd_leg = ",".join(raw_env.motor_labels)
        else:
            header_q_cmd_leg = ",".join([
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ])
        csv_path_q_cmd_leg = os.path.join(out_dir, "q_cmd_leg_eval.csv")
        np.savetxt(csv_path_q_cmd_leg, q_cmd_leg_mat, delimiter=",",
                   header=header_q_cmd_leg, comments="")
        print(f"[EVAL] Saved q_cmd_leg CSV to: {csv_path_q_cmd_leg}")
    else:
        print("[EVAL] No q_cmd_leg found in infos; CSV not saved.")

    if len(action_steps) > 0:
        action_mat = np.vstack(action_steps)
        if hasattr(raw_env, "motor_labels") and action_mat.shape[1] == len(raw_env.motor_labels):
            header_action = ",".join(raw_env.motor_labels)
        else:
            header_action = ",".join([f"a{i}" for i in range(action_mat.shape[1])])
        csv_path_action = os.path.join(out_dir, "action_eval.csv")
        np.savetxt(csv_path_action, action_mat, delimiter=",",
                   header=header_action, comments="")
        print(f"[EVAL] Saved action CSV to: {csv_path_action}")
    else:
        print("[EVAL] No action found in infos; CSV not saved.")

    # ===== 新增：substep 级 CSV =====
    if len(sub_q_cmd_all) > 0 and len(sub_q_act_all) > 0 and len(sub_motor_tau_all) > 0:
        # [N_step, K, 8] → [N_total_substeps, 8]
        q_cmd_sub = np.concatenate(sub_q_cmd_all, axis=0)        # [Tsub,8]
        q_act_sub = np.concatenate(sub_q_act_all, axis=0)        # [Tsub,8]
        tau_sub = np.concatenate(sub_motor_tau_all, axis=0)      # [Tsub,8]

        Tsub = q_cmd_sub.shape[0]
        t_sub = np.arange(Tsub) * sim_dt

        if hasattr(raw_env, "motor_labels"):
            labels = list(raw_env.motor_labels)
        else:
            labels = [
                "FL_thigh", "FL_leg",
                "FR_thigh", "FR_leg",
                "RL_thigh", "RL_leg",
                "RR_thigh", "RR_leg",
            ]

        header_cols = (
            ["t"]
            + [f"cmd_{name}" for name in labels]
            + [f"act_{name}" for name in labels]
            + [f"tau_{name}" for name in labels]
        )

        mat = np.column_stack([t_sub, q_cmd_sub, q_act_sub, tau_sub])
        csv_path_sub = os.path.join(out_dir, "substep_leg_data.csv")
        np.savetxt(csv_path_sub, mat, delimiter=",",
                   header=",".join(header_cols), comments="")
        print(f"[EVAL] Saved substep leg data CSV to: {csv_path_sub}")
    else:
        print("[EVAL] No substep data (sub_q_cmd_leg / sub_q_act_leg / sub_motor_avg_current) found in infos; substep CSV not saved.")

    test_env.close()
    print("[DONE] Evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--prefix", type=str, default="rat_mapping", help="Checkpoint file prefix")
    parser.add_argument("--model", type=str, default=None, help="(Optional) path to a specific model .zip")
    parser.add_argument("--vecnorm", type=str, default=None, help="(Optional) path to a specific VecNormalize .pkl")
    parser.add_argument("--eval_steps", type=int, default=32, help="Number of evaluation steps")
    parser.add_argument("--seed", type=int, default=999, help="Seed for evaluation env")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")

    parser.add_argument("--out_dir", type=str, default="./eval_vis", help="Where to save plots and npy")
    parser.add_argument("--body_fix", type=str, default="body_ss", help="Site name of body reference (for path/vel)")
    parser.add_argument("--thigh_sites", type=str, default="", help="Comma list for thigh sites (4 names)")
    parser.add_argument("--ankle_sites", type=str, default="", help="Comma list for ankle sites (4 names)")

    args = parser.parse_args()
    main(args)
