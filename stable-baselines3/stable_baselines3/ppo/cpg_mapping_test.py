# polar_cpg_ik_test.py
# -*- coding: utf-8 -*-
"""
测试目标：
- 不用 StarMapper，不用 RL。
- 直接用极坐标 CPG 生成足底轨迹 (Fy, Fz)。
- 用 LegModel.pos_2_angle 做 IK 得到关节角。
- 下发到 MuJoCo，看看小鼠能不能自己 trotting 往前走。

使用方式：
    python polar_cpg_ik_test.py
"""

import os
import math
import numpy as np
import mujoco
import mujoco_viewer

# 修改为你工程里的 env 路径
from Rat_Env_cpg import Go2Env

# 如果你平时训练有 chdir，可以照抄一份（其实 xml 是绝对路径的话不强制需要）
os.chdir("/home/geriatronics/ratmujoco_ws/Pathological-gait-generation-for-rat-robot-with-spine-based-damage-control/rat-robot/v2-HardControl/TrotGait/models")


def polar_to_foot(theta, rho, center_y, center_z, r_min, r_max):
    """
    极坐标 → 足底轨迹（在腿的局部坐标系里）：
        r = r_min + (r_max - r_min) * rho
        Fy = center_y + r * cos(theta)
        Fz = center_z + r * sin(theta)

    说明：
    - theta：相位角 [rad]
    - rho  ：[0,1] 的“半径系数”，这里只是把转圈变成略微变形的椭圆 / 花瓣形
    - center_y, center_z：椭圆中心（注意 Fz 要保持 < 0）
    """
    r = r_min + (r_max - r_min) * rho
    Fy = center_y + r * math.cos(theta)
    Fz = center_z + r * math.sin(theta)
    return Fy, Fz


def main():
    # 1) 建一个带渲染的环境（但我们不用 env.step，只用其中的 model/data/legs）
    env = Go2Env(render_mode="window", max_steps=10_000)
    # 调一次 reset，完成站立初始化
    obs, _ = env.reset()

    model = env.model
    data = env.data
    legs = env.legs           # [FL, FR, RL, RR] 的 LegModel
    n_legs = env.n_legs       # 4
    leg_action_dim = env.leg_action_dim  # 8
    sim_dt = env.sim_dt       # mujoco timestep

    # 2) CPG 参数（可以先用 env 里已有的频率）
    f_cpg = getattr(env, "cpg_freq", 2.0)  # Hz
    omega = 2.0 * math.pi * f_cpg         # rad/s

    # trot 相位差（和你 env 里的相位差一致）
    phase_offsets = np.array([0.0, math.pi, math.pi, 0.0], dtype=float)  # FL, FR, RL, RR

    # 3) 足底椭圆轨迹的几何参数（在腿的局部坐标系）
    # 这些值只是一个“保守猜测”，如果 IK 报错/关节打到底，可以适当缩小 r_max 或调整 center_z.
    center_y = 0.025   # m，足端平均前后位置（正方向：前）
    center_z = -0.035  # m，足端平均高度（必须为负，向下）
    r_min   = 0.005    # m，最小半径，避免轨迹太靠中心
    r_max   = 0.02     # m，最大半径，控制步长和抬脚幅度

    # 4) 仿真总时间
    T_total = 8.0  # 秒
    total_steps = int(T_total / sim_dt)

    # 记录一下初始和结束的身体 y 位置，看看到底有没有走
    y_start = float(data.qpos[1])

    # 初始化 viewer（也可以只调 env.render()，这里直接用 env.render() 更简单）
    print("开始仿真，看看老鼠能不能跑起来...")

    for step in range(total_steps):
        t = step * sim_dt

        # 当前 step 要下发的 8 维关节角
        ctrl = np.zeros(leg_action_dim, dtype=float)

        for leg_idx in range(n_legs):
            # 4 条腿的相位：ωt + 相位偏置
            theta = omega * t + phase_offsets[leg_idx]

            # 这里给一个简单的 rho(t)：0.4 ~ 1.0，随着相位变化
            #   - 你可以理解为“某种意义上的抬脚程度”
            #   - sin(theta) > 0 时足端稍远（swing），<0 时稍近（stance）
            rho = 0.7 + 0.3 * math.sin(theta)  # ∈ [0.4, 1.0]

            # 极坐标 → 足底轨迹（不经过 StarMapper）
            Fy, Fz = polar_to_foot(
                theta=theta,
                rho=rho,
                center_y=center_y,
                center_z=center_z,
                r_min=r_min,
                r_max=r_max,
            )

            # IK: 足底位置 → 该腿的关节角 (q1, q2)
            qVal = legs[leg_idx].pos_2_angle(
                np.array([Fy], dtype=float),
                np.array([Fz], dtype=float),
            )
            qVal = np.asarray(qVal).reshape(-1)
            if qVal.shape[0] < 2:
                raise RuntimeError(f"pos_2_angle 返回形状异常：{qVal.shape}, 内容={qVal}")

            # 按 [FL_thigh, FL_leg, FR_thigh, FR_leg, RL_thigh, RL_leg, RR_thigh, RR_leg] 顺序写入
            ctrl[2 * leg_idx]     = qVal[0]
            ctrl[2 * leg_idx + 1] = qVal[1]

        # 把关节角当作 position target 下发到前 8 个控制量
        data.ctrl[:leg_action_dim] = ctrl

        # 迈一步
        mujoco.mj_step(model, data)

        # 渲染
        env.render()

        # 简单防炸：如果摔倒就停
        base_z = data.sensordata[18] if data.sensordata.size >= 19 else data.qpos[2]
        if base_z < 0.02:
            print(f"[WARN] Robot seems to have fallen at step {step}, z={base_z:.3f}.")
            break

    # 结束后看看前进距离
    y_end = float(data.qpos[1])
    dist = y_end - y_start
    avg_v = dist / (step * sim_dt) if step > 0 else 0.0

    print(f"\n仿真结束: 总时间 ~ {step * sim_dt:.2f} s")
    print(f"起始 y = {y_start:.3f} m, 结束 y = {y_end:.3f} m, 前进距离 = {dist:.3f} m")
    print(f"平均前向速度 ≈ {avg_v:.3f} m/s")

    env.close()


if __name__ == "__main__":
    main()
