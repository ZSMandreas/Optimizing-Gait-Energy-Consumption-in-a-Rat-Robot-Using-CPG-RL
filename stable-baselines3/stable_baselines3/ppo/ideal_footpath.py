# phase_radius_star_mapper_test.py
# -*- coding: utf-8 -*-
"""
测试目标：
1. 用 WorkspaceDetection 采样你的腿的足底可达区域。
2. 用 alpha-shape 得到 reachable area 的多边形 alpha_poly / safe_poly。
3. 构造 StarMapper（使用 r_max(theta) + 面积保持半径）。
4. 在 action 空间里用三维参数化：
      a_phase1 ≈ cos(phi)
      a_phase2 ≈ sin(phi)
      a_rho_raw ∈ [-1, 1]  映射到 ρ ∈ [0,1]
   调用:
      Fy, Fz = mapper.map_phase_radius_area_uniform(a_phase1, a_phase2, a_rho_raw)
   生成一整周期的足底轨迹。
5. 绘制：
   - reachable area + 轨迹
   - 三个 action 分量随时间
   - action 相位子空间轨迹 (a_phase1, a_phase2)
   - 每步位置误差（因为是前向生成，这里误差几乎为 0）
"""

import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay

# =========================================
# 1. 一些工具函数
# =========================================

def pick_incircle_center(poly, n=100):
    """
    粗略在多边形内部采样一个“内切圆圆心”近似位置，用于可视化或当射线中心。
    当前 demo 里我们还是用 (0,0) 做 origin，这个函数可以不一定用到。
    """
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    best, best_d = (poly.representative_point().x, poly.representative_point().y), -1.0
    for y in ys:
        for x in xs:
            p = Point(x, y)
            if poly.contains(p):
                d = p.distance(poly.boundary)
                if d > best_d:
                    best, best_d = (x, y), d
    return best


def law_of_cosines_angle(la, lb, lc):
    """
    余弦定理反解角度，返回角度（弧度）。
    若 cos 值数值误差超过 [-1,1]，返回 -10 作为非法标记。
    """
    cos_val = (la**2 + lb**2 - lc**2) / (2 * la * lb)
    if abs(cos_val) > 1:
        return -10
    return math.acos(cos_val)


def check_cross(line1, line2):
    """
    线段相交检查 + 交点坐标
    line1: [[Cx, Cz], [Dx, Dz]]
    line2: [[Ax, Az], [Ex, Ez]]
    若不相交，返回 []；相交返回 [x, z]
    """
    C, D = line1
    A, E = line2
    area_CDA = (C[0]-A[0])*(D[1]-A[1]) - (C[1]-A[1])*(D[0]-A[0])
    area_CDE = (C[0]-E[0])*(D[1]-E[1]) - (C[1]-E[1])*(D[0]-E[0])
    area_AEC = (A[0]-C[0])*(E[1]-C[1]) - (A[1]-C[1])*(E[0]-C[0])
    area_AED = (A[0]-D[0])*(E[1]-D[1]) - (A[1]-D[1])*(E[0]-D[0])
    if (area_CDA * area_CDE) >= 0 or (area_AEC * area_AED) >= 0:
        return []
    tmp = area_AEC / (area_CDE - area_CDA)
    dx = tmp * (D[0] - C[0])
    dz = tmp * (D[1] - C[1])
    return [C[0] + dx, C[1] + dz]


def alpha_shape(pts, alpha):
    """
    经典 alpha-shape 实现：
    - 输入: pts (N,2) 点云
    - alpha: 控制“凹凸程度”的参数
    返回: shapely Polygon 或 MultiPolygon
    """
    if len(pts) < 4:
        return MultiPoint(list(pts)).convex_hull

    tri = Delaunay(pts)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = pts[ia], pts[ib], pts[ic]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        area = 0.5 * abs(np.cross(pb - pa, pc - pa))
        R = a * b * c / (4.0 * area + 1e-12)  # 外接圆半径
        if R < 1.0 / alpha:                   # alpha 判据
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
    m = unary_union(edge_segments)
    return unary_union(list(polygonize(m)))


# =========================================
# 2. 腿模型 & 工作空间采样
# =========================================

class WorkspaceDetection:
    """
    你原来的 WorkspaceDetection.angel_2_pos，略去注释。
    """
    def __init__(self, leg_params):
        self.len = leg_params
        self.By = 0.0
        self.Bz = self.len[1]
        self.limit_CBz = law_of_cosines_angle(self.len[2], self.len[1], 0.0075)
        self.limit_DCB = law_of_cosines_angle(0.012735, self.len[2], 0.002)
        self.limit_AEF = law_of_cosines_angle(0.01025, 0.01025, 0.0042)

    def angel_2_pos(self, q1, q2):
        PI = math.pi
        Ey = self.len[0] * math.cos(q1)
        Ez = self.len[0] * math.sin(q1)

        Cy = -self.len[2] * math.sin(q2)
        Cz = self.len[1] + self.len[2] * math.cos(q2)

        CE = math.hypot(Ey - Cy, Ez - Cz)
        if CE == 0:
            return []

        a_ECz = math.acos((Cz - Ez) / CE) * np.sign(Ey - Cy)
        a_ECD = law_of_cosines_angle(CE, self.len[3], self.len[4])
        if a_ECD == -10:
            return []
        a_DCz = a_ECD + a_ECz

        Dy = Cy + self.len[3] * math.sin(a_DCz)
        Dz = Cz - self.len[3] * math.cos(a_DCz)

        DEy, DEz = Dy - Ey, Dz - Ez
        Fy = Ey - (self.len[5] / self.len[4]) * DEy
        Fz = Ez - (self.len[5] / self.len[4]) * DEz

        BD = math.hypot(self.By - Dy, self.Bz - Dz)

        # 交叉检查
        cross = check_cross([[Cy, Cz], [Dy, Dz]], [[0, 0], [Ey, Ez]])
        if cross:
            dist = math.hypot(cross[0] - Ey, cross[1] - Ez)
            if dist < self.len[0] * 1.5:
                return []

        AF = math.hypot(Fy, Fz)
        a_AEF = law_of_cosines_angle(self.len[0], self.len[5], AF)
        a_BCD = law_of_cosines_angle(self.len[2], self.len[3], BD)
        if (a_AEF < PI / 6) or (a_AEF > PI * 5 / 6) or (a_BCD < PI / 6) or (a_BCD > PI * 5 / 6):
            return []

        return Fy, Fz


# ====== 采样工作空间点云 ======
print("Sampling workspace point cloud...")
leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
model = WorkspaceDetection(leg_params)

grid = 400
q_vals = np.linspace(-3.0, 3.0, grid)
Fy_col, Fz_col = [], []

for q1 in q_vals:
    for q2 in q_vals:
        res = model.angel_2_pos(q1, q2)
        if res and res[1] < 0:   # 只保留落地半空间
            Fy_col.append(res[0])
            Fz_col.append(res[1])

Fy_arr = np.array(Fy_col)
Fz_arr = np.array(Fz_col)
points = np.vstack((Fy_arr, Fz_arr)).T
print("  workspace samples:", points.shape[0])

# ====== alpha-shape 近似 reachable area 多边形 ======
alpha = 100.0
print("Computing alpha-shape polygon...")
alpha_poly = alpha_shape(points, alpha)   # shapely Polygon or MultiPolygon
alpha_poly = alpha_poly.buffer(0)         # 修复几何
safe_poly = alpha_poly.buffer(1e-6)       # 略扩一圈，防止数值误差

def is_valid(Fy, Fz):
    return safe_poly.covers(Point(Fy, Fz))


# =========================================
# 3. 射线 r_max(theta) 计算
# =========================================

def raycast_rmax_poly(alpha_poly, thetas, origin=(0.0, 0.0), R=None):
    """
    从 origin 沿各个角度 thetas，和多边形“面”相交，
    取沿射线最远的交点到 origin 的距离为 r_max(theta)。
    """
    alpha_poly = alpha_poly.buffer(0)
    cx, cz = origin

    if R is None:
        r_pts = np.hypot(Fy_arr - cx, Fz_arr - cz)
        R = 1.5 * (np.max(r_pts) + 1e-6)

    rmax = np.zeros_like(thetas, dtype=float)

    def _collect_coords(g):
        gt = g.geom_type
        if gt == "Point":
            return [(g.x, g.y)]
        if gt == "MultiPoint":
            return [(p.x, p.y) for p in g.geoms]
        if gt == "LineString":
            return list(g.coords)
        if gt == "MultiLineString":
            coords = []
            for seg in g.geoms:
                coords += list(seg.coords)
            return coords
        if gt == "GeometryCollection":
            coords = []
            for gg in g.geoms:
                coords += _collect_coords(gg)
            return coords
        return []

    for i, th in enumerate(thetas):
        tip = (cx + R * math.cos(th), cz + R * math.sin(th))
        ray = LineString([origin, tip])
        inter = alpha_poly.intersection(ray)
        if inter.is_empty:
            rmax[i] = 0.0
            continue
        pts = _collect_coords(inter)
        if not pts:
            rmax[i] = 0.0
            continue
        dists = [math.hypot(x - cx, z - cz) for (x, z) in pts]
        rmax[i] = float(max(dists))

    return rmax


# =========================================
# 4. StarMapper（含原 box 映射 + 新的 phase+ρ 映射）
# =========================================

class StarMapper:
    """
    原来的 StarMapper：
      Box(a1,a2)∈[-1,1]^2 → (Fy,Fz) 单射映射（面积保持）
    新增：
      phase+ρ 版本：
        (a_phase1, a_phase2) ≈ (cosφ, sinφ)
        a_rho_raw ∈ [-1,1] → ρ ∈ [0,1]
      调用:
        Fy,Fz = map_phase_radius_area_uniform(a_phase1,a_phase2,a_rho_raw)
    """
    def __init__(self, alpha_poly, origin,
                 theta_min=-np.pi, theta_max=np.pi, num=720,
                 alpha_in=0.10, beta=0.05, r_thr=0.002):

        self.alpha_poly = alpha_poly.buffer(0)  # 多边形边界用于投影/检查
        self.origin = origin

        self.alpha_in = float(alpha_in)
        self.beta = float(beta)
        self.theta_min, self.theta_max = theta_min, theta_max
        self.thetas = np.linspace(theta_min, theta_max, num, endpoint=False)

        # --- rmax(θ) ---
        self.rmax = raycast_rmax_poly(alpha_poly, self.thetas, origin=self.origin, R=None)

        # --- r_in / r_out ---
        self.r_in = self.alpha_in * self.rmax
        self.r_out = (1.0 - self.beta) * self.rmax

        # --- 面积权重 S(θ) 与 CDF ---
        S = 0.5 * (self.r_out**2 - self.r_in**2)
        S = np.clip(S, 0.0, None)
        S_sum = S.sum() + 1e-12
        self.cdf = np.cumsum(S) / S_sum

        # --- 自动裁掉 rmax 很小的角段 ---
        valid = self.rmax > r_thr
        if np.any(valid) and not np.all(valid):
            valid2 = np.r_[valid, valid]
            best_len, best_i0 = 0, 0
            i = 0
            n = len(valid2)
            while i < n:
                if valid2[i]:
                    j = i
                    while j < n and valid2[j]:
                        j += 1
                    seg_len = j - i
                    if seg_len > best_len:
                        best_len, best_i0 = seg_len, i
                    i = j
                else:
                    i += 1
            best_i0 %= len(valid)
            best_i1 = (best_i0 + best_len - 1) % len(valid)
            if best_len >= 3:
                if best_i1 >= best_i0:
                    sl = slice(best_i0, best_i1 + 1)
                    self.thetas = self.thetas[sl]
                    self.rmax = self.rmax[sl]
                else:
                    self.thetas = np.r_[self.thetas[best_i0:], self.thetas[:best_i1 + 1]]
                    self.rmax = np.r_[self.rmax[best_i0:], self.rmax[:best_i1 + 1]]
                self.theta_min = float(self.thetas[0])
                self.theta_max = float(self.thetas[-1])

    def rmax_interp(self, theta):
        """
        对 θ 做线性插值，得到 rmax(θ)。
        注意 theta_min/theta_max 可能不是完整 2π。
        """
        L = len(self.thetas)
        idx = (theta - self.theta_min) / (self.theta_max - self.theta_min) * (L - 1)
        idx = np.clip(idx, 0.0, L - 1 - 1e-9)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, L - 1)
        t = float(idx - i0)
        return (1.0 - t) * self.rmax[i0] + t * self.rmax[i1]

    # ===== 原来的 box 映射接口（保留） =====
    def map_box_area_uniform(self, a1, a2):
        """
        原有接口：a1,a2∈[-1,1]，通过面积均匀角度 CDF + 面积保持半径映射到 (Fy,Fz)。
        这里保留原逻辑（简化版），方便对比 / 向后兼容。
        """
        u = 0.5 * (a1 + 1.0)  # [0,1]
        rho = 0.5 * (a2 + 1.0)

        # θ 来自 CDF（简单用 self.cdf 与 self.thetas 插值）
        u = float(np.clip(u, 0.0, 1.0 - 1e-12))
        idx = np.searchsorted(self.cdf, u, side="right")
        i1 = int(np.clip(idx, 1, len(self.thetas) - 1))
        i0 = i1 - 1
        c0, c1 = self.cdf[i0], self.cdf[i1]
        if c1 == c0:
            t = 0.0
        else:
            t = (u - c0) / (c1 - c0)
        theta = (1.0 - t) * self.thetas[i0] + t * self.thetas[i1]
        r_in = (1.0 - t) * self.r_in[i0] + t * self.r_in[i1]
        r_out = (1.0 - t) * self.r_out[i0] + t * self.r_out[i1]

        r2 = r_in * r_in + rho * (r_out * r_out - r_in * r_in)
        r2 = max(0.0, r2)
        r = math.sqrt(r2)

        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)
        return Fy, Fz

    # ===== 新增：phase + ρ 的参数化 =====
    def _phase_to_theta(self, a_phase1, a_phase2):
        """
        (a_phase1, a_phase2) ≈ (cosφ, sinφ) → θ
        1) 归一化到单位圆 → (cosφ, sinφ)
        2) φ = atan2(sinφ, cosφ) ∈ [-π, π]
        3) 再线性映射到 [theta_min, theta_max]:
             t = (φ + π)/ (2π) ∈ [0,1]
             θ = theta_min + t*(theta_max - theta_min)
        """
        a_phase1 = float(a_phase1)
        a_phase2 = float(a_phase2)

        r_phase = math.hypot(a_phase1, a_phase2)
        if r_phase < 1e-8:
            # 如果 RL 偶尔给 (0,0)，就取中间相位
            t = 0.5
        else:
            cos_phi = a_phase1 / r_phase
            sin_phi = a_phase2 / r_phase
            phi = math.atan2(sin_phi, cos_phi)  # [-π, π]
            t = (phi + math.pi) / (2.0 * math.pi)  # [0,1]

        theta = self.theta_min + t * (self.theta_max - self.theta_min)
        return theta

    def map_phase_radius_area_uniform(self, a_phase1, a_phase2, a_rho_raw):
        """
        新接口：给 RL 用的 3D action → (Fy,Fz)
        - a_phase1, a_phase2: 相位编码 ≈ (cosφ, sinφ)
        - a_rho_raw         : ∈ [-1,1]，映射到 ρ∈[0,1]

        内部：
        1) θ = _phase_to_theta(...)
        2) r_max(θ)，再乘 alpha_in/beta 得到 r_in, r_out
        3) 面积保持半径：
             r^2 = r_in^2 + ρ*(r_out^2 - r_in^2)
        4) (Fy,Fz) = origin + r*(cosθ, sinθ)
        """
        theta = self._phase_to_theta(a_phase1, a_phase2)
        rmax = max(0.0, self.rmax_interp(theta))

        r_in = self.alpha_in * rmax
        r_out = (1.0 - self.beta) * rmax

        rho = 0.5 * (float(a_rho_raw) + 1.0)  # [-1,1]→[0,1]
        if rho < 0.0:
            rho = 0.0
        elif rho > 1.0:
            rho = 1.0

        r2 = r_in * r_in + rho * (r_out * r_out - r_in * r_in)
        if r2 < 0.0:
            r2 = 0.0
        r = math.sqrt(r2)

        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)
        return Fy, Fz


# =========================================
# 5. 用 phase+ρ action 生成一条测试轨迹
# =========================================

def build_phase_radius_action_trajectory(num_steps=400, rho_min=0.3, rho_max=0.9):
    """
    生成一条“理想 action 轨迹”：
      - 相位 φ(t) 从 0~2π 均匀扫一圈
      - ρ(t) 在 [rho_min, rho_max] 内做平滑正弦起伏
    返回:
      actions: (N,3) 数组，每一行是 [a_phase1, a_phase2, a_rho_raw]
    """
    ts = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)
    phi = ts

    a_phase1 = np.cos(phi)
    a_phase2 = np.sin(phi)

    rho_mid = 0.5 * (rho_min + rho_max)
    rho_amp = 0.5 * (rho_max - rho_min)
    rho = rho_mid + rho_amp * np.sin(2.0 * phi)   # 2 倍频率起伏
    rho = np.clip(rho, 0.0, 1.0)

    a_rho_raw = 2.0 * rho - 1.0   # [0,1] → [-1,1]

    actions = np.stack([a_phase1, a_phase2, a_rho_raw], axis=1)
    return actions


def analyze_action_smoothness(actions, jump_threshold=0.05):
    diff = np.diff(actions, axis=0)
    step_norm = np.linalg.norm(diff, axis=1)
    max_jump = float(np.max(step_norm))
    jump_idx = np.where(step_norm > jump_threshold)[0]
    return {
        "max_jump": max_jump,
        "jump_indices": jump_idx,
        "step_norm": step_norm,
    }


# =========================================
# 6. 主测试流程
# =========================================

def main():
    origin = (0.0, 0.0)  # 射线起点/映射中心，和你原来一致

    print("Initializing StarMapper...")
    mapper = StarMapper(alpha_poly, origin,
                        theta_min=-np.pi, theta_max=np.pi,
                        num=720,
                        alpha_in=0.10,  # 内半径比例
                        beta=0.05,      # 外边界预留裕度
                        r_thr=0.002)    # 小 rmax 角度裁剪阈值

    print("Building phase+rho action trajectory...")
    actions = build_phase_radius_action_trajectory(
        num_steps=400,
        rho_min=0.3,
        rho_max=0.9
    )

    # 用新的 map_phase_radius_area_uniform 生成足底轨迹
    Fy_list, Fz_list = [], []
    for a_phase1, a_phase2, a_rho in actions:
        Fy, Fz = mapper.map_phase_radius_area_uniform(a_phase1, a_phase2, a_rho)
        Fy_list.append(Fy)
        Fz_list.append(Fz)
    Fy_list = np.array(Fy_list)
    Fz_list = np.array(Fz_list)
    path_xy = np.stack([Fy_list, Fz_list], axis=1)

    # 检查是否都在 safe_poly 内
    inside_flags = np.array([is_valid(Fy, Fz) for Fy, Fz in path_xy])
    print(f"Path points inside reachable area: {inside_flags.mean()*100:.2f}%")
    if not np.all(inside_flags):
        print("WARNING: Some points are outside safe_poly (可能是 alpha/beta 参数太激进)。")

    # 因为轨迹就是正向生成的“理想轨迹”，所以重构误差可以直接设为 0；
    # 这里我们可以当成“self-consistency 检查”，所以误差就是 0。
    # 为了可视化，我们就认为 ideal_path = path_xy
    ideal_path = path_xy.copy()
    err = np.linalg.norm(path_xy - ideal_path, axis=1)

    info = analyze_action_smoothness(actions, jump_threshold=0.05)
    print(f"max |Δa| (over 3 dims) = {info['max_jump']:.6f}")
    print(f"jumps (>0.05) count = {len(info['jump_indices'])}")

    # 保存 action 轨迹
    np.savetxt(
        "phase_radius_actions_star_mapper.csv",
        actions,
        delimiter=",",
        header="a_phase1_cos_phi,a_phase2_sin_phi,a_rho_raw",
        comments=""
    )
    print("Saved actions to phase_radius_actions_star_mapper.csv")

    # ================== 画图 ==================
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    # ---- 图1：工作空间点云 + alpha_poly + 足底轨迹 ----
    ax = axs[0, 0]
    ax.scatter(Fy_arr, Fz_arr, s=1, alpha=0.3, label="workspace samples")
    if isinstance(alpha_poly, Polygon):
        x_poly, z_poly = alpha_poly.exterior.xy
        ax.plot(x_poly, z_poly, "k--", label="alpha-shape reachable area")
    else:
        # MultiPolygon
        for g in alpha_poly.geoms:
            x_poly, z_poly = g.exterior.xy
            ax.plot(x_poly, z_poly, "k--", linewidth=1)
    ax.plot(path_xy[:, 0], path_xy[:, 1], "r", linewidth=2, label="phase+rho foot path")
    ax.set_aspect("equal", "box")
    ax.set_title("Workspace & Foot Path (phase+ρ mapping)")
    ax.legend(loc="best")

    # ---- 图2：三个 action 分量 vs step ----
    ax = axs[0, 1]
    t = np.arange(len(actions))
    ax.plot(t, actions[:, 0], label="a_phase1 = cos φ")
    ax.plot(t, actions[:, 1], label="a_phase2 = sin φ")
    ax.plot(t, actions[:, 2], label="a_rho_raw ∈ [-1,1]")
    ax.set_xlabel("step")
    ax.set_ylabel("action value")
    ax.set_title("Actions over Steps (phase + ρ)")
    ax.legend(loc="best")

    # ---- 图3：位置误差 ----
    ax = axs[1, 0]
    ax.plot(t, err)
    ax.set_xlabel("step")
    ax.set_ylabel("||reconstructed - ideal|| (m)")
    ax.set_title("Reconstruction Error per Step (here ≈0)")

    # ---- 图4：相位子空间轨迹 (a_phase1,a_phase2) ----
    ax = axs[1, 1]
    ax.plot(actions[:, 0], actions[:, 1])
    ax.set_xlabel("a_phase1")
    ax.set_ylabel("a_phase2")
    ax.set_title("Phase Subspace Trajectory (cos φ vs sin φ)")
    ax.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
