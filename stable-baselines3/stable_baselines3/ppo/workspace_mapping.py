
# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize, nearest_points
from scipy.spatial import Delaunay
from shapely.geometry import Point
import matplotlib.colors as mcolors

def pick_incircle_center(poly, n=100):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    best, best_d = (poly.representative_point().x, poly.representative_point().y), -1.0
    for y in ys:
        for x in xs:
            p = Point(x,y)
            if poly.contains(p):
                d = p.distance(poly.boundary)
                if d > best_d:
                    best, best_d = (x,y), d
    return best

# ---------------- utility functions ----------------
def law_of_cosines_angle(la, lb, lc):
    cos_val = (la**2 + lb**2 - lc**2) / (2 * la * lb)
    if abs(cos_val) > 1:
        return -10
    return math.acos(cos_val)

def check_cross(line1, line2):
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
    dy = tmp * (D[1] - C[1])
    return [C[0] + dx, C[1] + dy]

def alpha_shape(pts, alpha):
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
        R = a*b*c / (4.0*area + 1e-12)       # 外接圆半径
        if R < 1.0 / alpha:                  # alpha 判据
            edges.update([(ia, ib), (ib, ic), (ic, ia)])
    edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
    m = unary_union(edge_segments)
    return unary_union(list(polygonize(m)))

# ---------------- leg model ----------------
class WorkspaceDetection:
    def __init__(self, leg_params):
        self.len = leg_params
        self.By = 0.0
        self.Bz = self.len[1]
        self.limit_CBz = law_of_cosines_angle(self.len[2], self.len[1], 0.0075)
        self.limit_DCB = law_of_cosines_angle(0.012735, self.len[2], 0.002)
        self.limit_AEF = law_of_cosines_angle(0.01025, 0.01025, 0.0042)

    def angel_2_pos(self, q1, q2):
        PI = math.pi
        Ey = self.len[0]*math.cos(q1)
        Ez = self.len[0]*math.sin(q1)

        Cy = -self.len[2]*math.sin(q2)
        Cz = self.len[1] + self.len[2]*math.cos(q2)

        CE = math.hypot(Ey - Cy, Ez - Cz)
        if CE == 0:
            return []

        a_ECz = math.acos((Cz - Ez)/CE) * np.sign(Ey - Cy)
        a_ECD = law_of_cosines_angle(CE, self.len[3], self.len[4])
        if a_ECD == -10:
            return []
        a_DCz = a_ECD + a_ECz

        Dy = Cy + self.len[3]*math.sin(a_DCz)
        Dz = Cz - self.len[3]*math.cos(a_DCz)

        DEy, DEz = Dy - Ey, Dz - Ez
        Fy = Ey - (self.len[5]/self.len[4]) * DEy
        Fz = Ez - (self.len[5]/self.len[4]) * DEz

        BD = math.hypot(self.By - Dy, self.Bz - Dz)

        # 交叉检查
        cross = check_cross([[Cy, Cz], [Dy, Dz]], [[0, 0], [Ey, Ez]])
        if cross:
            dist = math.hypot(cross[0]-Ey, cross[1]-Ez)
            if dist < self.len[0]*1.5:
                return []

        AF = math.hypot(Fy, Fz)
        a_AEF = law_of_cosines_angle(self.len[0], self.len[5], AF)
        a_BCD = law_of_cosines_angle(self.len[2], self.len[3], BD)
        if (a_AEF < PI/6) or (a_AEF > PI*5/6) or (a_BCD < PI/6) or (a_BCD > PI*5/6):
            return []

        return Fy, Fz

# ----------- sampling the workspace ------------
leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
model = WorkspaceDetection(leg_params)

grid = 400
q_vals = np.linspace(-3.0, 3.0, grid)
Fy_col, Fz_col = [], []

for q1 in q_vals:
    for q2 in q_vals:
        res = model.angel_2_pos(q1, q2)
        if res and res[1] < 0:  # 只保留落地半空间
            Fy_col.append(res[0])
            Fz_col.append(res[1])

Fy_arr = np.array(Fy_col)
Fz_arr = np.array(Fz_col)
points = np.vstack((Fy_arr, Fz_arr)).T

alpha = 100
alpha_poly = alpha_shape(points, alpha)        # shapely Polygon or MultiPolygon
alpha_poly = alpha_poly.buffer(0)              # 修复几何
safe_poly = alpha_poly.buffer(1e-6)

def is_valid(Fy, Fz):
    return safe_poly.covers(Point(Fy, Fz))

# --------- 射线 r_max(theta)（与多边形相交，更鲁棒） ---------
def raycast_rmax_poly(alpha_poly, thetas, origin=(0.0,0.0), R=None):
    """
    从 origin 沿 theta 方向，用长度 R 的线段与“多边形面”相交；
    取相交处沿射线“最远点”的距离作为 r_max(theta)。
    R 若为 None，则自动取点云最大半径的 1.5 倍。
    """
    alpha_poly = alpha_poly.buffer(0)
    cx, cy = origin

    if R is None:
        # 点云半径的 1.5 倍，避免 1e3 这种巨值导致数值病态与绘图不直观
        r_pts = np.hypot(Fy_arr - cx, Fz_arr - cy)
        R = 1.5 * (np.max(r_pts) + 1e-6)

    rmax = np.zeros_like(thetas, dtype=float)

    def _collect_coords(g):
        gt = g.geom_type
        if gt == "Point":
            return [(g.x, g.y)]
        if gt == "MultiPoint":
            return [(p.x, p.y) for p in g.geoms]
        if gt == "LineString":
            return list(g.coords)  # 整段在多边形内部
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
        tip = (cx + R*np.cos(th), cy + R*np.sin(th))
        ray = LineString([origin, tip])
        inter = alpha_poly.intersection(ray)  # 与“面”相交（更稳）
        if inter.is_empty:
            rmax[i] = 0.0
            continue
        pts = _collect_coords(inter)
        if not pts:
            rmax[i] = 0.0
            continue
        dists = [np.hypot(x - cx, y - cy) for (x, y) in pts]
        rmax[i] = float(max(dists))
       

    return rmax

# --------- 单射映射（带内半径/外裕度 + 有效角裁剪） ---------
class StarMapper:
    """
    Box(a1,a2)∈[-1,1]^2 → (Fy,Fz) 单射映射
    - alpha_in:   内半径比例（避免中心奇异） 0.0~0.3
    - beta:       外边界裕度（避免贴边抖动） 0.0~0.2
    - r_thr:      有效角阈值（米），裁掉 rmax 很小的角段
    """
    def __init__(self, alpha_poly, origin,
                 theta_min=-np.pi, theta_max=np.pi, num=720,
                 alpha_in=0.10, beta=0.05, r_thr=0.002):
        
        self.alpha_poly = alpha_poly.buffer(0)  # ✅ 保存多边形边界用于奖励计算

        self.alpha_in = float(alpha_in)
        self.beta     = float(beta)
        self.theta_min, self.theta_max = theta_min, theta_max
        self.thetas = np.linspace(theta_min, theta_max, num, endpoint=False)
        self.origin = origin

        # 计算 rmax(θ)
        self.rmax = raycast_rmax_poly(alpha_poly, self.thetas, origin=self.origin, R=None)
        # 先算 r_in/out
        self.r_in  = self.alpha_in * self.rmax
        self.r_out = (1.0 - self.beta) * self.rmax

        # 面积权重 S(θ) 与累计 CDF
        S = 0.5 * (self.r_out**2 - self.r_in**2)
        S = np.clip(S, 0.0, None)
        S_sum = S.sum() + 1e-12
        self.cdf = np.cumsum(S) / S_sum    # 递增到1
        # 自动裁剪可用角段：找最长连续 rmax>r_thr 的区间
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
                    sl = slice(best_i0, best_i1+1)
                    self.thetas = self.thetas[sl]
                    self.rmax   = self.rmax[sl]
                else:
                    self.thetas = np.r_[self.thetas[best_i0:], self.thetas[:best_i1+1]]
                    self.rmax   = np.r_[self.rmax[best_i0:], self.rmax[:best_i1+1]]
                self.theta_min = float(self.thetas[0])
                self.theta_max = float(self.thetas[-1])

    def rmax_interp(self, theta):
        L = len(self.thetas)
        # 线性插值（注意当前 thetas 可能不是完整整圈）
        idx = (theta - self.theta_min) / (self.theta_max - self.theta_min) * (L - 1)
        idx = np.clip(idx, 0.0, L - 1 - 1e-9)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, L - 1)
        t = float(idx - i0)
        return (1.0 - t) * self.rmax[i0] + t * self.rmax[i1]

    def map_box(self, a1, a2):
        # a1,a2 ∈ [-1,1]
        theta = self.theta_min + 0.5*(a1 + 1.0)*(self.theta_max - self.theta_min)
        rho   = 0.5*(a2 + 1.0)  # [0,1]
        rmax  = max(0.0, self.rmax_interp(theta))

        # 内/外边界处理
        r_in = self.alpha_in * rmax
        r    = r_in + (1.0 - self.beta) * rho * (rmax - r_in)

        Fy   = self.origin[0] + r * np.cos(theta)
        Fz   = self.origin[1] + r * np.sin(theta)
        return Fy, Fz
    def map_box_with_rho(self, a1, a2):
        theta = self.theta_min + 0.5*(a1 + 1.0)*(self.theta_max - self.theta_min)
        rho_raw = 0.5*(a2 + 1.0)  # [0,1]
        rmax = max(0.0, self.rmax_interp(theta))

        r_in  = self.alpha_in * rmax
        r_out = (1.0 - self.beta) * rmax
        r     = r_in + rho_raw * (r_out - r_in)

        Fy = self.origin[0] + r * np.cos(theta)
        Fz = self.origin[1] + r * np.sin(theta)

        denom = max(r_out - r_in, 1e-9)
        rho = (r - r_in) / denom
        # 数值安全夹取
        rho = 0.0 if rho < 0.0 else (1.0 if rho > 1.0 else rho)
        return Fy, Fz, rho
    def map_box_area_preserving(self, a1, a2):
        theta = self.theta_min + 0.5*(a1 + 1.0)*(self.theta_max - self.theta_min)
        rho   = 0.5*(a2 + 1.0)  # [0,1]
        rmax  = max(0.0, self.rmax_interp(theta))

        r_in  = self.alpha_in * rmax
        r_out = (1.0 - self.beta) * rmax

        # ★ 面积保持的径向映射（替代线性半径）
        r = math.sqrt(r_in*r_in + rho * (r_out*r_out - r_in*r_in))

        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)
        return Fy, Fz
    def theta_from_u_area_uniform(self, u):
        # u∈[0,1] → 按面积均匀的 θ（用CDF反查 + 线性插值）
        u = np.clip(u, 0.0, 1.0 - 1e-12)
        idx = np.searchsorted(self.cdf, u, side="right")
        i1 = int(np.clip(idx, 1, len(self.thetas)-1))
        i0 = i1 - 1
        # 片内线性插值
        c0, c1 = self.cdf[i0], self.cdf[i1]
        t = 0.0 if c1==c0 else (u - c0) / (c1 - c0)
        theta = (1.0 - t) * self.thetas[i0] + t * self.thetas[i1]
        r_in  = (1.0 - t) * (self.alpha_in * self.rmax[i0]) + t * (self.alpha_in * self.rmax[i1])
        r_out = (1.0 - t) * ((1.0 - self.beta) * self.rmax[i0]) + t * ((1.0 - self.beta) * self.rmax[i1])
        return theta, r_in, r_out

    def map_box_area_uniform(self, a1, a2):
        u   = 0.5 * (a1 + 1.0)  # → [0,1]
        rho = 0.5 * (a2 + 1.0)  # → [0,1]
        theta, r_in, r_out = self.theta_from_u_area_uniform(u)
        # 面积保持的半径
        r = math.sqrt(max(0.0, r_in*r_in + rho * (r_out*r_out - r_in*r_in)))
        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)
        if not is_valid(Fy, Fz):
            # 投影回最近边界点
            p_proj = alpha_poly.boundary.interpolate(alpha_poly.boundary.project(Point(Fy,Fz)))
            Fy, Fz = p_proj.x, p_proj.y
        return Fy, Fz


# ========= Append this block at the END of your file =========

import numpy as np
import pandas as pd

def make_trot_actions(T=8.0, dt=0.01, freq=2.0,
                      A_theta=0.85,     # 方向角通道幅度（对应 a1）
                      A_rho=0.70,      # 半径通道幅度（对应 a2）
                      bias_rho=-0.10,  # 半径通道偏置（把平均半径往内收一点）
                      phi_rho=0.0):    # 半径与角度相位差（如需抬脚-摆动错开可调）
    """
    生成理想的 trot 动作：FR~RL 同相，FL~RR 反相；输出在 [-1,1]。
    """
    t = np.arange(0.0, T, dt)
    w = 2.0 * np.pi * freq
    phase = {"FR": 0.0, "RL": 0.0, "FL": np.pi, "RR": np.pi}

    def curve(leg):
        a1 = A_theta * np.sin(w * t + phase[leg])  # 角度通道
        a2 = np.clip(bias_rho + A_rho * np.sin(w * t + phase[leg] + phi_rho), -1.0, 1.0)  # 半径通道
        return a1, a2

    a1_FR, a2_FR = curve("FR")
    a1_FL, a2_FL = curve("FL")
    a1_RR, a2_RR = curve("RR")
    a1_RL, a2_RL = curve("RL")
    return t, (a1_FR, a2_FR, a1_FL, a2_FL, a1_RR, a2_RR, a1_RL, a2_RL)

def map_actions_to_feet_with_star(mapper, t, actions):
    """
    用你实现的 StarMapper.map_box_area_uniform 把 (a1,a2) → (Fy,Fz)。
    """
    a1_FR, a2_FR, a1_FL, a2_FL, a1_RR, a2_RR, a1_RL, a2_RL = actions
    n = len(t)
    Fy_FR = np.zeros(n); Fz_FR = np.zeros(n)
    Fy_FL = np.zeros(n); Fz_FL = np.zeros(n)
    Fy_RR = np.zeros(n); Fz_RR = np.zeros(n)
    Fy_RL = np.zeros(n); Fz_RL = np.zeros(n)
    for i in range(n):
        Fy_FR[i], Fz_FR[i] = mapper.map_box_area_uniform(a1_FR[i], a2_FR[i])
        Fy_FL[i], Fz_FL[i] = mapper.map_box_area_uniform(a1_FL[i], a2_FL[i])
        Fy_RR[i], Fz_RR[i] = mapper.map_box_area_uniform(a1_RR[i], a2_RR[i])
        Fy_RL[i], Fz_RL[i] = mapper.map_box_area_uniform(a1_RL[i], a2_RL[i])
    return (Fy_FR, Fz_FR, Fy_FL, Fz_FL, Fy_RR, Fz_RR, Fy_RL, Fz_RL)

# if __name__ == "__main__":
#     # 1) 从你的 alpha_shape 工作空间挑原点（已在文件前面构造了 alpha_poly/safe_poly/is_valid）
#     origin = pick_incircle_center(alpha_poly)  # 也可用 alpha_poly.representative_point()
#     origin = (origin[0]-0.02, origin[1])

#     # 2) 构建 StarMapper（使用你的实现与参数）
#     mapper = StarMapper(
#         alpha_poly,
#         origin=origin,
#         num=1440,        # 高分辨率插值（0.25°）
#         alpha_in=0.10,   # 内裕度
#         beta=0.05,       # 外裕度
#         r_thr=0.002      # 角段阈值（去掉极短扇区）
#     )

# # ----------------- 可视化 -----------------
# # 1) α-shape 工作空间
# plt.figure(figsize=(6, 6))
# plt.scatter(Fy_arr, Fz_arr, s=2, alpha=0.35, label='valid samples')
# if alpha_poly.geom_type == 'Polygon':
#     x, y = alpha_poly.exterior.xy
#     plt.fill(x, y, alpha=0.2)
#     plt.plot(x, y, label='alpha-shape boundary')
# else:
#     for poly in alpha_poly.geoms:
#         x, y = poly.exterior.xy
#         plt.fill(x, y, alpha=0.2)
#         plt.plot(x, y, label='alpha-shape boundary')
# plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
# plt.axis('equal'); plt.grid(True); plt.legend()
# plt.title("Workspace via Alpha-Shape")
# plt.show()

# # 2) Box→Workspace 单射映射（用网格演示）
# grid_lin = np.linspace(-1, 1, 720)  # 可适当调密度
# U1, U2 = np.meshgrid(grid_lin, grid_lin)
# Fy_map = np.zeros_like(U1, dtype=float)
# Fz_map = np.zeros_like(U2, dtype=float)
# ok = 0; total = U1.size
# for i in range(U1.shape[0]):
#     for j in range(U1.shape[1]):
#         Fy_map[i, j], Fz_map[i, j] = mapper.map_box_area_uniform(U1[i, j], U2[i, j])
#         if is_valid(Fy_map[i, j], Fz_map[i, j]):
#             ok += 1

# plt.figure(figsize=(6, 6))
# plt.plot(x, y, linewidth=2, label='alpha-shape boundary')
# plt.scatter(Fy_map.flatten(), Fz_map.flatten(), s=10, alpha=0.85, label='mapped points from Box')
# plt.scatter([origin[0]], [origin[1]], s=50, marker='x', label='origin')
# plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
# plt.axis('equal'); plt.grid(True); plt.legend()
# plt.title(f"Box→Workspace Mapping (inside: {ok}/{total})\n"
#           f"theta range [{mapper.theta_min*180/np.pi:.1f}°, {mapper.theta_max*180/np.pi:.1f}°], "
#           f"alpha_in={mapper.alpha_in}, beta={mapper.beta}")
# plt.show()
# # ========= 10k actions in [-1,1]^2 -> map -> compare with reachable area =========
# np.random.seed(42)
# N = 10_000
# actions = np.random.uniform(-1.0, 1.0, size=(N, 2))  # (a1, a2)

# Fy_s = np.empty(N, dtype=float)
# Fz_s = np.empty(N, dtype=float)
# inside = np.empty(N, dtype=bool)

# for i in range(N):
#     a1, a2 = actions[i]
#     Fy_i, Fz_i = mapper.map_box_area_uniform(a1, a2)
#     Fy_s[i] = Fy_i
#     Fz_s[i] = Fz_i
#     inside[i] = is_valid(Fy_i, Fz_i)
    

# inside_ratio = inside.mean()
# print(f"[Sampling] N={N}, inside={inside.sum()} ({inside_ratio*100:.2f}%), outside={(~inside).sum()}")

# # --- Plot: reachable area boundary + mapped samples ---
# plt.figure(figsize=(7, 7))

# # 1) reachable area boundary
# if alpha_poly.geom_type == 'Polygon':
#     bx, by = alpha_poly.exterior.xy
#     plt.fill(bx, by, alpha=0.15, label='reachable area (α-shape)')
#     plt.plot(bx, by, linewidth=2, color='k')
# else:
#     # MultiPolygon
#     for poly in alpha_poly.geoms:
#         bx, by = poly.exterior.xy
#         plt.fill(bx, by, alpha=0.15, label='reachable area (α-shape)')
#         plt.plot(bx, by, linewidth=2, color='k')

# # 2) mapped points
# plt.scatter(Fy_s[inside],  Fz_s[inside],  s=6, alpha=0.7, label='mapped (inside)')
# plt.scatter(Fy_s[~inside], Fz_s[~inside], s=6, alpha=0.7, label='mapped (outside)')

# # 3) origin marker
# plt.scatter([origin[0]], [origin[1]], s=60, marker='x', label='origin')

# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(True)
# plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
# plt.title(f"Box [-1,1]^2 -> Workspace mapping (N={N})\n"
#           f"inside={inside.sum()} ({inside_ratio*100:.2f}%), "
#           f"θ∈[{mapper.theta_min*180/np.pi:.1f}°, {mapper.theta_max*180/np.pi:.1f}°], "
#           f"α_in={mapper.alpha_in}, β={mapper.beta}")
# plt.legend(loc='best')
# plt.show()

# # ===================== tanh-高斯采样并映射 =====================
# np.random.seed(42)
# N = 50000
# mu, log_std = 0.0, 0.0
# sigma = np.exp(log_std)
# eps = np.random.randn(N,2)
# x = mu + sigma*eps
# actions = np.tanh(x)  # tanh-高斯动作

# # ==== 绘图 ====
# fig, axs = plt.subplots(1, 3, figsize=(15,4))

# # 1) x 分布（原始高斯）
# axs[0].hist(x[:,0], bins=100, color='tab:blue', alpha=0.7)
# axs[0].set_title(f"Raw Gaussian x ~ N(0, σ={sigma:.3f})")
# axs[0].set_xlabel("x")
# axs[0].set_ylabel("count")
# axs[0].grid(True)

# # 2) a = tanh(x) 分布（单维）
# axs[1].hist(actions[:,0], bins=100, color='tab:orange', alpha=0.7)
# axs[1].set_title("Tanh-Normal Action Distribution (1D)")
# axs[1].set_xlabel("action a = tanh(x)")
# axs[1].set_ylabel("count")
# axs[1].grid(True)
# axs[1].set_xlim(-1, 1)

# # 3) a₁ vs a₂ 2D 密度
# H, xedges, yedges = np.histogram2d(actions[:,0], actions[:,1], bins=200)
# axs[2].imshow(H.T, origin='lower', extent=[-1,1,-1,1],
#               cmap='viridis', aspect='auto')
# axs[2].set_title("2D Action Density (tanh-Normal)")
# axs[2].set_xlabel("a₁")
# axs[2].set_ylabel("a₂")

# plt.tight_layout()
# plt.show()

# # ==== 映射到 (θ, ρ) 并绘制分布 ====
# # a → (θ, ρ)
# a1, a2 = actions[:,0], actions[:,1]
# u_theta = 0.5*(a1 + 1.0)
# rho     = 0.5*(a2 + 1.0)

# theta = np.empty(N)
# for i, u in enumerate(u_theta):
#     th, _, _ = mapper.theta_from_u_area_uniform(u)
#     theta[i] = th

# # 直方图：θ
# plt.figure(figsize=(6,4))
# plt.hist(theta, bins=180)
# plt.xlabel("theta (rad)")
# plt.ylabel("count")
# plt.title("θ distribution from tanh-Gaussian → (uθ→θ)")
# plt.grid(True)
# plt.show()

# # 直方图：ρ
# plt.figure(figsize=(6,4))
# plt.hist(rho, bins=100)
# plt.xlabel("rho in [0,1]")
# plt.ylabel("count")
# plt.title("ρ distribution from tanh-Gaussian → linear [0,1]")
# plt.grid(True)
# plt.show()

# # 二维密度：θ vs ρ
# plt.figure(figsize=(6,5))
# plt.hist2d(theta, rho, bins=120)
# plt.xlabel("theta (rad)")
# plt.ylabel("rho")
# plt.title("2D density of (theta, rho) from tanh-Gaussian")
# plt.colorbar()
# plt.show()




# # ---- θ(a1) ----
# a1_grid = np.linspace(-1, 1, 2000)
# u_theta = 0.5*(a1_grid + 1.0)            # [-1,1] → [0,1]
# theta = np.empty_like(a1_grid)
# r_in = np.empty_like(a1_grid)
# r_out = np.empty_like(a1_grid)
# for i, u in enumerate(u_theta):
#     th, rin, rout = mapper.theta_from_u_area_uniform(u)
#     theta[i], r_in[i], r_out[i] = th, rin, rout

# plt.figure(figsize=(6,4))
# plt.plot(a1_grid, u_theta)
# plt.xlabel("a1 in [-1,1]")
# plt.ylabel("theta (rad)")
# plt.title("Mapping Function: θ(a1) via area-CDF inverse")
# plt.grid(True)
# plt.show()

# # ---- ρ(a2) ----
# a2_grid = np.linspace(-1, 1, 2000)
# rho = 0.5*(a2_grid + 1.0)                 # [-1,1] → [0,1]
# plt.figure(figsize=(6,4))
# plt.plot(a2_grid, rho)
# plt.xlabel("a2 in [-1,1]")
# plt.ylabel("rho in [0,1]")
# plt.title("Mapping Function: ρ(a2) (linear to [0,1])")
# plt.grid(True)
# plt.show()

# # ==== 映射到工作空间并绘制密度图 ====
# Fy_s, Fz_s = np.zeros(N), np.zeros(N)
# for i in range(N):
#     Fy_s[i], Fz_s[i] = mapper.map_box_area_uniform(actions[i,0], actions[i,1])
#     if not is_valid(Fy_s[i], Fz_s[i]):
#         # 投影回最近边界点
#         p_proj = alpha_poly.boundary.interpolate(alpha_poly.boundary.project(Point(Fy_s[i],Fz_s[i])))
#         Fy_s[i], Fz_s[i] = p_proj.x, p_proj.y
# # 2) Box→Workspace 单射映射（用网格演示）
# grid_lin = np.linspace(-1, 1, 720)  # 可适当调密度
# U1, U2 = np.meshgrid(grid_lin, grid_lin)
# Fy_s = np.zeros_like(U1, dtype=float)
# Fz_s = np.zeros_like(U2, dtype=float)
# ok = 0; total = U1.size
# for i in range(U1.shape[0]):
#     for j in range(U1.shape[1]):
#         Fy_s[i, j], Fz_s[i, j] = mapper.map_box_area_uniform(U1[i, j], U2[i, j])
#         if is_valid(Fy_s[i, j], Fz_s[i, j]):
#             ok += 1

# # 生成二维密度直方图
# bins = 200
# H, xedges, yedges = np.histogram2d(Fy_s.ravel(), Fz_s.ravel(), bins=bins)
# H = H.T
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# # 避免 LogNorm 对 0 取对数报错（可选）
# # H[H == 0] = np.nan

# plt.figure(figsize=(7,7))
# if alpha_poly.geom_type == 'Polygon':
#     bx, by = alpha_poly.exterior.xy
#     plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')
# else:
#     for poly in alpha_poly.geoms:
#         bx, by = poly.exterior.xy
#         plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')

# import matplotlib.colors as mcolors
# plt.imshow(H, extent=extent, origin='lower', cmap='viridis',
#            norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(H)))
# plt.scatter([origin[0]], [origin[1]], c='r', s=40, marker='x', label='origin')
# plt.xlabel("Fy (m)")
# plt.ylabel("Fz (m)")
# plt.title("Uniform Sampling + Area-Preserving + θ-CDF")
# plt.legend()
# plt.axis('equal'); plt.grid(True)
# plt.show()



# plt.figure(); plt.hist(u_theta.ravel(), bins=180)
# plt.xlabel("θ (rad)"); plt.ylabel("count"); plt.title("Angular density"); plt.show()


# plt.figure(); plt.hist(rho.ravel(), bins=100)
# plt.xlabel("r²"); plt.ylabel("count"); plt.title("Area density (should be flat if area-preserving)"); plt.show()

# H2, _, _ = np.histogram2d(Fy_s.ravel(), Fz_s.ravel(), bins=200)
# p = H2 / H2.sum()
# entropy = -np.sum(p[p>0] * np.log(p[p>0]))
# print("采样熵(越高越均匀):", entropy)

# # ===================== 绘制密度图 =====================
# # 生成二维密度直方图
# bins = 200
# H, xedges, yedges = np.histogram2d(Fy_s, Fz_s, bins=bins)
# H = H.T
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# plt.figure(figsize=(7,7))
# # 工作空间边界
# if alpha_poly.geom_type == 'Polygon':
#     bx, by = alpha_poly.exterior.xy
#     plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')
# else:
#     for poly in alpha_poly.geoms:
#         bx, by = poly.exterior.xy
#         plt.plot(bx, by, 'k', lw=1.5, label='α-shape boundary')

# # 密度热图
# plt.imshow(H, extent=extent, origin='lower', cmap='viridis',
#            norm=mcolors.LogNorm(vmin=1, vmax=H.max()))
# plt.scatter([origin[0]], [origin[1]], c='r', s=40, marker='x', label='origin')
# plt.xlabel("Fy (m)")
# plt.ylabel("Fz (m)")
# # plt.title(f"Tanh-Normal Sampling + Area-Preserving + θ-CDF Mapping\n(log_std_init=0, σ={sigma:.2f})")
# plt.title(f"Uniform Sampling + Area-Preserving + θ-CDF")
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.show()

# r = np.hypot(Fy_s - origin[0], Fz_s - origin[1])
# theta = np.arctan2(Fz_s - origin[1], Fy_s - origin[0])
# plt.figure()
# plt.hist(theta, bins=180)
# plt.xlabel("θ (rad)")
# plt.ylabel("count")
# plt.title("Angular density")
# plt.show()

# r2 = r**2
# plt.figure()
# plt.hist(r2, bins=100)
# plt.xlabel("r²")
# plt.ylabel("count")
# plt.title("Area density (should be flat if area-preserving)")
# plt.show()

# H, _, _ = np.histogram2d(Fy_s, Fz_s, bins=200)
# p = H / np.sum(H)
# entropy = -np.sum(p[p>0] * np.log(p[p>0]))
# print("采样熵(越高越均匀):", entropy)




# # ---------- 1) 函数曲线：theta(a1) ----------
# a1_grid = np.linspace(-1, 1, 2000)
# u_grid  = 0.5*(a1_grid + 1.0)      # [-1,1]→[0,1]
# theta_curve = np.empty_like(a1_grid)

# for i,u in enumerate(u_grid):
#     th, _, _ = mapper.theta_from_u_area_uniform(u)
#     theta_curve[i] = th

# plt.figure(figsize=(6,4))
# plt.plot(a1_grid, theta_curve)
# plt.xlabel("a1  (policy action dim-1)")
# plt.ylabel("theta (rad)")
# plt.title("Mapping function  θ(a1)  via area-CDF inverse")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ---------- 2) 函数曲线：在固定θ下的 r(a2) ----------
# def r_from_theta_a2(theta, a2):
#     """给定 θ 和 a2，按面积保持计算 r."""
#     rmax = mapper.rmax_interp(theta)
#     r_in  = mapper.alpha_in * rmax
#     r_out = (1.0 - mapper.beta) * rmax
#     rho   = 0.5*(a2 + 1.0)                 # [-1,1]→[0,1]
#     r     = np.sqrt(r_in*r_in + rho*(r_out*r_out - r_in*r_in))
#     return r

# # 选三种代表性角度：CDF 上的 10%, 50%, 90% 分位
# u_thetas = [0.10, 0.50, 0.90]
# thetas_fixed = [mapper.theta_from_u_area_uniform(u)[0] for u in u_thetas]
# labels = [f"θ @ CDF {int(p*100)}%" for p in u_thetas]

# a2_grid = np.linspace(-1, 1, 2000)
# plt.figure(figsize=(6,4))
# for th, lab in zip(thetas_fixed, labels):
#     r_curve = r_from_theta_a2(th, a2_grid)
#     plt.plot(a2_grid, r_curve, label=lab)
# plt.xlabel("a2  (policy action dim-2)")
# plt.ylabel("r (m)")
# plt.title("Mapping function  r(a2)  at fixed θ (area-preserving)")
# plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# # ---------- 3) 采样对比：tanh高斯 vs 均匀 ----------
# def theta_r_from_actions(actions):
#     """actions: (N,2) in [-1,1]; 返回对应的 (theta, r)."""
#     a1, a2 = actions[:,0], actions[:,1]
#     u = 0.5*(a1 + 1.0)
#     theta = np.empty_like(a1)
#     r     = np.empty_like(a1)
#     for i, ui in enumerate(u):
#         th, r_in, r_out = mapper.theta_from_u_area_uniform(ui)
#         rho  = 0.5*(a2[i] + 1.0)
#         r[i] = np.sqrt(r_in*r_in + rho*(r_out*r_out - r_in*r_in))
#         theta[i] = th
#     return theta, r

# N = 50000
# # a) tanh(N(0,1))
# np.random.seed(42)
# x = np.random.randn(N,2)
# a_tanh = np.tanh(x)
# theta_tanh, r_tanh = theta_r_from_actions(a_tanh)

# # b) Uniform[-1,1]
# a_uni = np.random.uniform(-1,1,size=(N,2))
# theta_uni, r_uni = theta_r_from_actions(a_uni)

# # ---- 直方图对比：theta ----
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.hist(theta_tanh, bins=180, alpha=0.75, label='tanh-Gaussian', density=True)
# plt.hist(theta_uni,  bins=180, alpha=0.55, label='Uniform',       density=True)
# plt.xlabel("theta (rad)"); plt.ylabel("density"); plt.title("θ distribution"); plt.legend(); plt.grid(True)

# # ---- 直方图对比：r ----
# plt.subplot(1,2,2)
# plt.hist(r_tanh, bins=180, alpha=0.75, label='tanh-Gaussian', density=True)
# plt.hist(r_uni,  bins=180, alpha=0.55, label='Uniform',       density=True)
# plt.xlabel("r (m)"); plt.ylabel("density"); plt.title("r distribution"); plt.legend(); plt.grid(True)
# plt.tight_layout(); plt.show()

# # ---- 二维密度：theta vs r（可选）
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.hist2d(theta_tanh, r_tanh, bins=140, cmap='viridis')
# plt.xlabel("theta (rad)"); plt.ylabel("r (m)"); plt.title("tanh-Gaussian → (theta, r)")
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.hist2d(theta_uni, r_uni, bins=140, cmap='viridis')
# plt.xlabel("theta (rad)"); plt.ylabel("r (m)"); plt.title("Uniform → (theta, r)")
# plt.colorbar()
# plt.tight_layout(); plt.show()
