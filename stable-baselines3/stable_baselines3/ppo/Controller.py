import numpy as np
import math

from LegModel.forPath import LegPath
# -----------------------------------------------------------
from LegModel.legs import LegModel
from CPGcontroller import OscillatorLeg

class MouseController(object):
    """docstring for MouseController"""
    def __init__(self, fre, time_step):
        super(MouseController, self).__init__()
        PI = np.pi
        self.curStep = 0# Spine
        
        # Spine A = 0
        #self.turn_F = 0*PI/180
        #self.turn_H = 8*PI/180
        # Spine A = 20
        self.turn_F = 0*PI/180
        self.turn_H = 12*PI/180
        self.pathStore = LegPath()
        # [LF, RF, LH, RH]
        # --------------------------------------------------------------------- #
        #self.phaseDiff = [0, PI, PI*1/2, PI*3/2]	# Walk
        #self.period = 3/2
        #self.SteNum = 36							#32 # Devide 2*PI to multiple steps
        #self.spinePhase = self.phaseDiff[3]
        # --------------------------------------------------------------------- #
        self.phaseDiff = [0, PI, PI, 0]			# Trot
        self.period = 2/2
        self.fre_cyc = fre#1.25#0.80
        self.SteNum = int(1/(time_step*self.fre_cyc))
        print("----> ", self.SteNum)
        # self.spinePhase = self.phaseDiff[3]
        # --------------------------------------------------------------------- #
        # self.spine_A =2*spine_angle#10 a_s = 2theta_s
        # print("angle --> ", spine_angle)#self.spine_A)
        # self.spine_A = self.spine_A*PI/180
        # --------------------------------------------------------------------- #
        leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
        self.fl_left = LegModel(leg_params)
        self.fl_right = LegModel(leg_params)
        self.hl_left = LegModel(leg_params)
        self.hl_right = LegModel(leg_params)
        # --------------------------------------------------------------------- #
        self.stepDiff = [0,0,0,0]
        for i in range(4):
            self.stepDiff[i] = int(self.SteNum * self.phaseDiff[i]/(2*PI))
        # self.stepDiff.append(int(self.SteNum * self.spinePhase/(2*PI)))
        self.trgXList = [[],[],[],[]]
        self.trgYList = [[],[],[],[]]

    def getLegCtrl(self, leg_M, curStep, leg_ID, leg_action):
        curStep = curStep % self.SteNum
        leg_name_list = ["LF", "RF", "LH", "RH"]
        leg_name = leg_name_list[leg_ID]
        turnAngle = self.turn_F if leg_ID < 2 else self.turn_H
        radian = 2 * np.pi * curStep / self.SteNum
    
        # 使用 RL 输出的参数
        currentPos = self.pathStore.getOvalPathPoint(radian, leg_name, self.period, leg_action)
    
        trg_x, trg_y = currentPos
        self.trgXList[leg_ID].append(trg_x)
        self.trgYList[leg_ID].append(trg_y)
    
        tX = math.cos(turnAngle) * trg_x - math.sin(turnAngle) * trg_y
        tY = math.cos(turnAngle) * trg_y + math.sin(turnAngle) * trg_x
        qVal = leg_M.pos_2_angle(tX, tY)
        return qVal

    # def getSpineVal(self, spineStep):
    # 	temp_step = int(spineStep)# / 5)
    # 	radian = 2*np.pi * temp_step/self.SteNum
    # 	return self.spine_A*math.cos(radian-self.spinePhase)
        #spinePhase = 2*np.pi*spineStep/self.SteNum
        #return self.spine_A*math.sin(spinePhase)

    def runStep(self, action):
        """
        接收 RL 动作向量，分别应用到每条腿的轨迹控制。
        action: 长度 16，4 条腿的椭圆轨迹参数 [ox, oy, rx, ry] * 4
        """
        leg_names = ["LF", "RF", "LH", "RH"]
    
        # 1. 为每条腿提取 4 维 action 子段
        leg_actions = {
            leg: action[i*4:(i+1)*4]
            for i, leg in enumerate(leg_names)
        }
    
        # 2. 控制每条腿
        foreLeg_left_q = self.getLegCtrl(self.fl_left, self.curStep + self.stepDiff[0], 0, leg_actions["LF"])
        foreLeg_right_q = self.getLegCtrl(self.fl_right, self.curStep + self.stepDiff[1], 1, leg_actions["RF"])
        hindLeg_left_q = self.getLegCtrl(self.hl_left, self.curStep + self.stepDiff[2], 2, leg_actions["LH"])
        hindLeg_right_q = self.getLegCtrl(self.hl_right, self.curStep + self.stepDiff[3], 3, leg_actions["RH"])
    
        self.curStep = (self.curStep + 1) % self.SteNum
    
        ctrlData = []
        ctrlData.extend(foreLeg_left_q)
        ctrlData.extend(foreLeg_right_q)
        ctrlData.extend(hindLeg_left_q)
        ctrlData.extend(hindLeg_right_q)
        # ctrlData.extend([0, 0, 0, 0])  # 保留与原结构一致（spine未用）
        # print(ctrlData)

    
        return ctrlData