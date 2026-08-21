# Controller.py
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import math

from LegModel.forPath import LegPath
from LegModel.legs import LegModel


class MouseController(object):
    """docstring for MouseController"""
    def __init__(self, fre, time_step, spine_angle):
        super(MouseController, self).__init__()
        PI = np.pi
        self.curStep = 0

        # turn angles（沿用你原来的）
        self.turn_F = 0 * PI / 180
        self.turn_H = 12 * PI / 180

        # gait: desired phase offsets (trot) —— 现在用于“期望锁相”，而不是直接加 offset
        # leg order: 0 FL, 1 FR, 2 HL, 3 HR
        self.phaseDiff = [0, PI, PI, 0]      # Trot

        self.period = 2 / 2                  # 1.0
        self.fre_cyc = float(fre)
        self.time_step = float(time_step)

        # 仍保留 SteNum/stepDiff（不再用于 contact_states，但保留不破坏你其他潜在逻辑）
        self.SteNum = int(1 / (self.time_step * self.fre_cyc))
        print("----> ", self.SteNum)
        self.spinePhase = self.phaseDiff[3]

        self.spine_A = 2 * spine_angle
        print("angle --> ", spine_angle)
        print("cpg has been created...")
        self.spine_A = self.spine_A * PI / 180

        # 方案B：4腿耦合振荡器网络 CPG（核心替换）
        # desired_offsets = self.phaseDiff 用于锁相目标
        self.pathStore = LegPath(fre_cyc=self.fre_cyc, dt=self.time_step, desired_offsets=self.phaseDiff)

        # LegModel（沿用）
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
        self.fl_left = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)

        # 仍保留 stepDiff（不再用于 contact_states，但保留）
        self.stepDiff = [0, 0, 0, 0]
        for i in range(4):
            self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i] / (2 * PI))
        self.stepDiff.append(int(self.SteNum * self.spinePhase / (2 * PI)))

        # 用于画 target
        self.trgXList = [[], [], [], []]
        self.trgYList = [[], [], [], []]

        # ---------------------------
        # CPG logs（amp & phase）
        # ---------------------------
        self._cpg_time = 0.0
        self.leg_names = ["FL", "FR", "HL", "HR"]
        self.cpg_phase_log = []  # [T,4]
        self.cpg_amp_log = []    # [T,4]
        self.cpg_time_log = []   # [T]

    def _log_cpg(self):
        """记录当前时刻每条腿的相位和幅值（不改变控制）"""
        phases = []
        amps = []
        for i in range(4):
            phases.append(self.pathStore.get_leg_phase(i))  # rad
            amps.append(self.pathStore.get_leg_amp(i))      # r_i
        self.cpg_time_log.append(self._cpg_time)
        self.cpg_phase_log.append(phases)
        self.cpg_amp_log.append(amps)

    def save_cpg_csv(self, out_dir: str):
        """输出 cpg_amps.csv 和 cpg_phases.csv"""
        os.makedirs(out_dir, exist_ok=True)

        # phases
        phase_path = os.path.join(out_dir, "cpg_phases.csv")
        with open(phase_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + [f"theta_{n}" for n in self.leg_names])
            for t, row in zip(self.cpg_time_log, self.cpg_phase_log):
                w.writerow([f"{t:.6f}"] + [f"{v:.10f}" for v in row])

        # amps
        amp_path = os.path.join(out_dir, "cpg_amps.csv")
        with open(amp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + [f"r_{n}" for n in self.leg_names])
            for t, row in zip(self.cpg_time_log, self.cpg_amp_log):
                w.writerow([f"{t:.6f}"] + [f"{v:.10f}" for v in row])

        print(f"[MouseController] Saved CPG logs → {out_dir} (cpg_phases.csv, cpg_amps.csv)")

    def getLegCtrl(self, leg_M, leg_ID):
        # 前/后腿 turnAngle（沿用）
        turnAngle = self.turn_F
        leg_flag = "F"
        if leg_ID > 1:
            leg_flag = "H"
            turnAngle = self.turn_H

        # 方案B：每条腿的相位由 CPG 网络内部维护，因此这里传 leg_id
        currentPos = self.pathStore.getCPGOvalPathPoint(
            leg_flag=leg_flag,
            leg_id=leg_ID,
            halfPeriod=self.period
        )

        trg_x = currentPos[0]
        trg_y = currentPos[1]
        self.trgXList[leg_ID].append(trg_x)
        self.trgYList[leg_ID].append(trg_y)

        # turnAngle 旋转（不变）
        tX = math.cos(turnAngle) * trg_x - math.sin(turnAngle) * trg_y
        tY = math.cos(turnAngle) * trg_y + math.sin(turnAngle) * trg_x

        # IK（不变）
        qVal = leg_M.pos_2_angle(tX, tY)
        return qVal

    def runStep(self):
        # 方案B：耦合振荡器网络推进（不变接口）
        self.pathStore.step()

        # 记录 CPG 相位/幅值（不影响控制）
        self._log_cpg()

        foreLeg_left_q = self.getLegCtrl(self.fl_left, 0)
        foreLeg_right_q = self.getLegCtrl(self.fl_right, 1)
        hindLeg_left_q = self.getLegCtrl(self.hl_left, 2)
        hindLeg_right_q = self.getLegCtrl(self.hl_right, 3)

        # curStep 仍更新（保留，不破坏你原来潜在依赖）
        self.curStep = (self.curStep + 1) % self.SteNum

        # 与 TrotGait/src 一致：模型 actuator 顺序为 [leg_joint, thigh_joint]，q1→leg、q2→thigh，故每条腿写入 [q1, q2]
        # pos_2_angle 返回 (q1, q2)，直接按顺序写
        ctrlData = []
        for q in [foreLeg_left_q, foreLeg_right_q, hindLeg_left_q, hindLeg_right_q]:
            ctrlData.extend([q[0], q[1]])  # q1→leg_joint, q2→thigh_joint

        # 其余 4 个 actuator 仍置 0（不变）
        for _ in range(4):
            ctrlData.append(0)

        # ✅ 推荐：contact_states 直接用当前 CPG 相位判断（与轨迹生成严格同步）
        contact_states = []
        for i in range(4):
            phi = self.pathStore.get_leg_phase(i)  # [0, 2pi)
            is_contact = 1 if phi >= self.period * np.pi else 0
            contact_states.append(is_contact)

        # 更新时间（用于 CSV）
        self._cpg_time += self.time_step

        return ctrlData, contact_states
