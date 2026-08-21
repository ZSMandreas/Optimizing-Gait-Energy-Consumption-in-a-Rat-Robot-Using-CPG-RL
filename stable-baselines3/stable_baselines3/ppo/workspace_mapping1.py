# # -*- coding: utf-8 -*-
# import numpy as np
# import math
# import matplotlib.pyplot as plt

# from shapely.geometry import Point, MultiPoint, Polygon, LineString
# from shapely.ops import unary_union, polygonize, nearest_points
# from scipy.spatial import Delaunay
# import matplotlib.colors as mcolors

# # ==========================
# # 1. 工具函数 & 几何构造
# # ==========================

# def pick_incircle_center(poly, n=100):
#     """
#     简单网格搜索近似内切圆圆心：
#     在 polygon 内部找到“离边界最远”的一点。
#     """
#     minx, miny, maxx, maxy = poly.bounds
#     xs = np.linspace(minx, maxx, n)
#     ys = np.linspace(miny, maxy, n)
#     best, best_d = (poly.representative_point().x, poly.representative_point().y), -1.0
#     for y in ys:
#         for x in xs:
#             p = Point(x, y)
#             if poly.contains(p):
#                 d = p.distance(poly.boundary)
#                 if d > best_d:
#                     best, best_d = (x, y), d
#     return best

# def law_of_cosines_angle(la, lb, lc):
#     cos_val = (la**2 + lb**2 - lc**2) / (2 * la * lb)
#     if abs(cos_val) > 1:
#         return -10
#     return math.acos(cos_val)

# def check_cross(line1, line2):
#     C, D = line1
#     A, E = line2
#     area_CDA = (C[0]-A[0])*(D[1]-A[1]) - (C[1]-A[1])*(D[0]-A[0])
#     area_CDE = (C[0]-E[0])*(D[1]-E[1]) - (C[1]-E[1])*(D[0]-E[0])
#     area_AEC = (A[0]-C[0])*(E[1]-C[1]) - (A[1]-C[1])*(E[0]-C[0])
#     area_AED = (A[0]-D[0])*(E[1]-D[1]) - (A[1]-D[1])*(E[0]-D[0])
#     if (area_CDA * area_CDE) >= 0 or (area_AEC * area_AED) >= 0:
#         return []
#     tmp = area_AEC / (area_CDE - area_CDA)
#     dx = tmp * (D[0] - C[0])
#     dy = tmp * (D[1] - C[1])
#     return [C[0] + dx, C[1] + dy]

# def alpha_shape(pts, alpha):
#     """
#     alpha shape 生成不规则工作空间的“外壳”多边形
#     """
#     if len(pts) < 4:
#         return MultiPoint(list(pts)).convex_hull
#     tri = Delaunay(pts)
#     edges = set()
#     for ia, ib, ic in tri.simplices:
#         pa, pb, pc = pts[ia], pts[ib], pts[ic]
#         a = np.linalg.norm(pb - pc)
#         b = np.linalg.norm(pa - pc)
#         c = np.linalg.norm(pa - pb)
#         area = 0.5 * abs(np.cross(pb - pa, pc - pa))
#         R = a * b * c / (4.0 * area + 1e-12)       # 外接圆半径
#         if R < 1.0 / alpha:                        # alpha 判据
#             edges.update([(ia, ib), (ib, ic), (ic, ia)])
#     edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
#     m = unary_union(edge_segments)
#     return unary_union(list(polygonize(m)))

# # ==========================
# # 2. 腿模型 & 工作空间采样
# # ==========================

# class WorkspaceDetection:
#     def __init__(self, leg_params):
#         self.len = leg_params
#         self.By = 0.0
#         self.Bz = self.len[1]
#         self.limit_CBz = law_of_cosines_angle(self.len[2], self.len[1], 0.0075)
#         self.limit_DCB = law_of_cosines_angle(0.012735, self.len[2], 0.002)
#         self.limit_AEF = law_of_cosines_angle(0.01025, 0.01025, 0.0042)

#     def angel_2_pos(self, q1, q2):
#         PI = math.pi
#         Ey = self.len[0] * math.cos(q1)
#         Ez = self.len[0] * math.sin(q1)

#         Cy = -self.len[2] * math.sin(q2)
#         Cz = self.len[1] + self.len[2] * math.cos(q2)

#         CE = math.hypot(Ey - Cy, Ez - Cz)
#         if CE == 0:
#             return []

#         a_ECz = math.acos((Cz - Ez) / CE) * np.sign(Ey - Cy)
#         a_ECD = law_of_cosines_angle(CE, self.len[3], self.len[4])
#         if a_ECD == -10:
#             return []
#         a_DCz = a_ECD + a_ECz

#         Dy = Cy + self.len[3] * math.sin(a_DCz)
#         Dz = Cz - self.len[3] * math.cos(a_DCz)

#         DEy, DEz = Dy - Ey, Dz - Ez
#         Fy = Ey - (self.len[5] / self.len[4]) * DEy
#         Fz = Ez - (self.len[5] / self.len[4]) * DEz

#         BD = math.hypot(self.By - Dy, self.Bz - Dz)

#         # 交叉检查
#         cross = check_cross([[Cy, Cz], [Dy, Dz]], [[0, 0], [Ey, Ez]])
#         if cross:
#             dist = math.hypot(cross[0] - Ey, cross[1] - Ez)
#             if dist < self.len[0] * 1.5:
#                 return []

#         AF = math.hypot(Fy, Fz)
#         a_AEF = law_of_cosines_angle(self.len[0], self.len[5], AF)
#         a_BCD = law_of_cosines_angle(self.len[2], self.len[3], BD)
#         if (a_AEF < PI / 6) or (a_AEF > PI * 5 / 6) or (a_BCD < PI / 6) or (a_BCD > PI * 5 / 6):
#             return []

#         return Fy, Fz

# # --------- 全局采样得到点云 & alpha shape ---------
# leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
# model = WorkspaceDetection(leg_params)

# grid = 400
# q_vals = np.linspace(-3.0, 3.0, grid)
# Fy_col, Fz_col = [], []

# for q1 in q_vals:
#     for q2 in q_vals:
#         res = model.angel_2_pos(q1, q2)
#         if res and res[1] < 0:  # 只保留落地半空间
#             Fy_col.append(res[0])
#             Fz_col.append(res[1])

# Fy_arr = np.array(Fy_col)
# Fz_arr = np.array(Fz_col)
# points = np.vstack((Fy_arr, Fz_arr)).T

# alpha = 100
# alpha_poly = alpha_shape(points, alpha)        # Polygon or MultiPolygon
# alpha_poly = alpha_poly.buffer(0)              # 修复几何
# safe_poly = alpha_poly.buffer(1e-6)            # 略扩一点，避免边界数值问题

# def is_valid(Fy, Fz):
#     return safe_poly.covers(Point(Fy, Fz))

# # ==========================
# # 3. 射线 r_max(theta) 计算
# # ==========================

# def raycast_rmax_poly(poly, thetas, origin=(0.0, 0.0), R=None):
#     """
#     从 origin 沿 theta 方向，用长度 R 的线段与“多边形面”相交；
#     取相交处沿射线“最远点”的距离作为 r_max(theta)。
#     poly: shapely Polygon（使用 safe_poly/alpha_poly）
#     """
#     poly = poly.buffer(0)
#     cx, cy = origin

#     if R is None:
#         # 点云半径的 1.5 倍，避免过大
#         r_pts = np.hypot(Fy_arr - cx, Fz_arr - cy)
#         R = 1.5 * (np.max(r_pts) + 1e-6)

#     rmax = np.zeros_like(thetas, dtype=float)

#     def _collect_coords(g):
#         gt = g.geom_type
#         if gt == "Point":
#             return [(g.x, g.y)]
#         if gt == "MultiPoint":
#             return [(p.x, p.y) for p in g.geoms]
#         if gt == "LineString":
#             return list(g.coords)  # 线段在多边形内部
#         if gt == "MultiLineString":
#             coords = []
#             for seg in g.geoms:
#                 coords += list(seg.coords)
#             return coords
#         if gt == "GeometryCollection":
#             coords = []
#             for gg in g.geoms:
#                 coords += _collect_coords(gg)
#             return coords
#         return []

#     for i, th in enumerate(thetas):
#         tip = (cx + R * np.cos(th), cy + R * np.sin(th))
#         ray = LineString([origin, tip])
#         inter = poly.intersection(ray)  # 与“面”相交
#         if inter.is_empty:
#             rmax[i] = 0.0
#             continue
#         pts = _collect_coords(inter)
#         if not pts:
#             rmax[i] = 0.0
#             continue
#         dists = [np.hypot(x - cx, y - cy) for (x, y) in pts]
#         rmax[i] = float(max(dists))

#     return rmax

# # ==========================
# # 4. StarMapper：Box → 工作空间
# # ==========================

# class StarMapper:
#     """
#     Box(a1,a2)∈[-1,1]^2 → (Fy,Fz) 单射映射（带内半径/外裕度 + 面积均匀角度）
#     - alpha_in:   内半径比例（避免中心奇异） 0.0~0.3
#     - beta:       外边界裕度（避免贴边抖动） 0.0~0.2
#     - r_thr:      有效角阈值（米），裁掉 rmax 很小的角段
#     """
#     def __init__(self, alpha_poly, origin,
#                  theta_min=-np.pi, theta_max=np.pi, num=720,
#                  alpha_in=0.10, beta=0.05, r_thr=0.002):

#         # 保存原始 polygon（用于投影）
#         self.alpha_poly = alpha_poly.buffer(0)

#         self.alpha_in = float(alpha_in)
#         self.beta     = float(beta)
#         self.origin   = origin

#         # ===== 4.1 整圈角度采样 & rmax =====
#         thetas_full = np.linspace(theta_min, theta_max, num, endpoint=False)
#         rmax_full   = raycast_rmax_poly(alpha_poly, thetas_full, origin=self.origin, R=None)

#         # ===== 4.2 根据 r_thr 找最长连续有效角段 =====
#         valid = rmax_full > r_thr

#         if np.any(valid):
#             if np.all(valid):
#                 # 全部有效，直接用整圈
#                 thetas_use = thetas_full.copy()
#                 rmax_use   = rmax_full.copy()
#             else:
#                 # 周期展开后找最长连续 True 段
#                 valid2 = np.concatenate([valid, valid])
#                 n2 = len(valid2)
#                 best_len, best_i0 = 0, 0
#                 i = 0
#                 while i < n2:
#                     if valid2[i]:
#                         j = i
#                         while j < n2 and valid2[j]:
#                             j += 1
#                         seg_len = j - i
#                         if seg_len > best_len:
#                             best_len = seg_len
#                             best_i0 = i
#                         i = j
#                     else:
#                         i += 1
#                 best_i0 = best_i0 % len(valid)
#                 best_i1 = (best_i0 + best_len - 1) % len(valid)

#                 if best_len < 3:
#                     # 太短就干脆不用裁剪
#                     thetas_use = thetas_full.copy()
#                     rmax_use   = rmax_full.copy()
#                 else:
#                     if best_i1 >= best_i0:
#                         # 不跨 -π/π 边界
#                         thetas_use = thetas_full[best_i0:best_i1+1]
#                         rmax_use   = rmax_full[best_i0:best_i1+1]
#                     else:
#                         # 跨越边界：尾部 + (头部 + 2π) 拼接
#                         tail_theta = thetas_full[best_i0:]
#                         tail_r     = rmax_full[best_i0:]
#                         head_theta = thetas_full[:best_i1+1] + 2.0 * np.pi
#                         head_r     = rmax_full[:best_i1+1]
#                         thetas_use = np.concatenate([tail_theta, head_theta])
#                         rmax_use   = np.concatenate([tail_r, head_r])
#         else:
#             # 全无效（理论上不该出现），回退整圈
#             thetas_use = thetas_full.copy()
#             rmax_use   = rmax_full.copy()

#         # 角度保证单调（再保险排序一次）
#         sort_idx = np.argsort(thetas_use)
#         self.thetas = thetas_use[sort_idx]
#         self.rmax   = rmax_use[sort_idx]

#         self.theta_min = float(self.thetas[0])
#         self.theta_max = float(self.thetas[-1])

#         # ===== 4.3 基于裁剪后的角度/半径，重新计算 r_in/out 与 CDF =====
#         self.r_in  = self.alpha_in * self.rmax
#         self.r_out = (1.0 - self.beta) * self.rmax

#         S = 0.5 * (self.r_out**2 - self.r_in**2)
#         S = np.clip(S, 0.0, None)
#         S_cum = np.cumsum(S)
#         S_tot = S_cum[-1] + 1e-12
#         cdf = S_cum / S_tot
#         # 强制单调非减，避免数值抖动
#         self.cdf = np.maximum.accumulate(cdf)

#     # ---------- 半径插值 ----------
#     def rmax_interp(self, theta):
#         """
#         线性插值 rmax(theta)，假定 self.thetas 单调递增。
#         """
#         L = len(self.thetas)
#         idx = (theta - self.theta_min) / (self.theta_max - self.theta_min) * (L - 1)
#         idx = np.clip(idx, 0.0, L - 1 - 1e-9)
#         i0 = int(np.floor(idx))
#         i1 = min(i0 + 1, L - 1)
#         t  = float(idx - i0)
#         return (1.0 - t) * self.rmax[i0] + t * self.rmax[i1]

#     # ---------- 简单 Box → 星形区域 ----------
#     def map_box(self, a1, a2):
#         """
#         最简单：角度线性、半径线性（非面积保持）
#         """
#         theta = self.theta_min + 0.5 * (a1 + 1.0) * (self.theta_max - self.theta_min)
#         rho   = 0.5 * (a2 + 1.0)  # [0,1]
#         rmax  = max(0.0, self.rmax_interp(theta))

#         r_in = self.alpha_in * rmax
#         r    = r_in + (1.0 - self.beta) * rho * (rmax - r_in)

#         Fy   = self.origin[0] + r * np.cos(theta)
#         Fz   = self.origin[1] + r * np.sin(theta)
#         return Fy, Fz

#     def map_box_with_rho(self, a1, a2):
#         """
#         返回 (Fy,Fz,rho) 方便你调试半径映射。
#         """
#         theta   = self.theta_min + 0.5 * (a1 + 1.0) * (self.theta_max - self.theta_min)
#         rho_raw = 0.5 * (a2 + 1.0)  # [0,1]
#         rmax    = max(0.0, self.rmax_interp(theta))

#         r_in  = self.alpha_in * rmax
#         r_out = (1.0 - self.beta) * rmax
#         r     = r_in + rho_raw * (r_out - r_in)

#         Fy = self.origin[0] + r * np.cos(theta)
#         Fz = self.origin[1] + r * np.sin(theta)

#         denom = max(r_out - r_in, 1e-9)
#         rho   = (r - r_in) / denom
#         rho   = 0.0 if rho < 0.0 else (1.0 if rho > 1.0 else rho)
#         return Fy, Fz, rho

#     def map_box_area_preserving(self, a1, a2):
#         """
#         面积保持的径向映射（但角度仍是线性）
#         """
#         theta = self.theta_min + 0.5 * (a1 + 1.0) * (self.theta_max - self.theta_min)
#         rho   = 0.5 * (a2 + 1.0)  # [0,1]
#         rmax  = max(0.0, self.rmax_interp(theta))

#         r_in  = self.alpha_in * rmax
#         r_out = (1.0 - self.beta) * rmax

#         r = math.sqrt(r_in * r_in + rho * (r_out * r_out - r_in * r_in))

#         Fy = self.origin[0] + r * math.cos(theta)
#         Fz = self.origin[1] + r * math.sin(theta)
#         return Fy, Fz

#     # ---------- CDF 反查 u → θ（按面积均匀） ----------
#     def theta_from_u_area_uniform(self, u):
#         """
#         u∈[0,1] → 按面积均匀的 θ，
#         使用裁剪后 self.cdf / self.thetas / self.rmax。
#         """
#         u = np.clip(u, 0.0, 1.0 - 1e-12)
#         idx = np.searchsorted(self.cdf, u, side="right")
#         i1 = int(np.clip(idx, 1, len(self.thetas) - 1))
#         i0 = i1 - 1

#         c0, c1 = self.cdf[i0], self.cdf[i1]
#         t = 0.0 if c1 == c0 else (u - c0) / (c1 - c0)

#         theta = (1.0 - t) * self.thetas[i0] + t * self.thetas[i1]
#         r_in  = (1.0 - t) * (self.alpha_in * self.rmax[i0]) + t * (self.alpha_in * self.rmax[i1])
#         r_out = (1.0 - t) * ((1.0 - self.beta) * self.rmax[i0]) + t * ((1.0 - self.beta) * self.rmax[i1])
#         return theta, r_in, r_out

#     def map_box_area_uniform(self, a1, a2):
#         """
#         你现在在用的方法：
#         a1 → u（沿面积均匀选角度），a2 → ρ（面积保持半径）
#         """
#         u   = 0.5 * (a1 + 1.0)  # → [0,1]
#         rho = 0.5 * (a2 + 1.0)  # → [0,1]

#         theta, r_in, r_out = self.theta_from_u_area_uniform(u)

#         # 面积保持的半径
#         r_sq = r_in * r_in + rho * (r_out * r_out - r_in * r_in)
#         r_sq = max(r_sq, 0.0)
#         r = math.sqrt(r_sq)

#         Fy = self.origin[0] + r * math.cos(theta)
#         Fz = self.origin[1] + r * math.sin(theta)

#         # 越界保险：若数值抖动导致落在 poly 外，吸到最近边界
#         if not is_valid(Fy, Fz):
#             p_proj = self.alpha_poly.boundary.interpolate(
#                 self.alpha_poly.boundary.project(Point(Fy, Fz))
#             )
#             Fy, Fz = p_proj.x, p_proj.y

#         return Fy, Fz

# # ==========================
# # 5. 测试代码
# # ==========================

# if __name__ == "__main__":
#     # ---------- 5.1 选 origin ----------
#     origin = pick_incircle_center(safe_poly)
#     print("origin (approx incircle center):", origin)

#     mapper = StarMapper(
#         alpha_poly=safe_poly,
#         origin=origin,
#         theta_min=-np.pi,
#         theta_max=np.pi,
#         num=720,
#         alpha_in=0.10,
#         beta=0.05,
#         r_thr=0.002
#     )

#     # ---------- 5.2 可视化 rmax(theta) ----------
#     fig1, ax1 = plt.subplots(figsize=(6, 4))
#     ax1.plot(mapper.thetas, mapper.rmax, '-')
#     ax1.set_title("r_max(theta) after trimming")
#     ax1.set_xlabel("theta [rad]")
#     ax1.set_ylabel("r_max [m]")
#     ax1.grid(True)

#     # ---------- 5.3 在 Fy-Fz 平面画 polygon + 映射的 action 网格 ----------
#     fig2, ax2 = plt.subplots(figsize=(6, 6))
#     # 原始工作空间点云
#     ax2.scatter(Fy_arr, Fz_arr, s=1, alpha=0.2, label="sampled workspace")
#     # safe_poly 边界
#     x_poly, y_poly = safe_poly.exterior.xy
#     ax2.plot(x_poly, y_poly, 'k-', linewidth=1.5, label="safe_poly")

#     # StarMapper 映射 action 网格
#     N = 40
#     a1s = np.linspace(-1.0, 1.0, N)
#     a2s = np.linspace(-1.0, 1.0, N)
#     Fy_map, Fz_map = [], []
#     for a1 in a1s:
#         for a2 in a2s:
#             Fy, Fz = mapper.map_box_area_uniform(a1, a2)
#             Fy_map.append(Fy)
#             Fz_map.append(Fz)
#     ax2.scatter(Fy_map, Fz_map, s=5, c='r', alpha=0.5, label="mapped from Box")

#     ax2.scatter([origin[0]], [origin[1]], c='g', s=40, label="origin")
#     ax2.set_aspect('equal', 'box')
#     ax2.set_title("Workspace & Box-mapped points (area-uniform)")
#     ax2.legend()
#     ax2.grid(True)

#     # ---------- 5.4 测试：a1 一圈、a2 固定，看轨迹是否闭合、无跳变 ----------
#     fig3, ax3 = plt.subplots(figsize=(6, 6))
#     # 在 action 空间画 a1-a2 轨迹
#     a2_fixed = 0.0
#     Ts = 400
#     a1_circle = np.linspace(-1.0, 1.0, Ts, endpoint=False)
#     A1, A2 = [], []
#     Fy_traj, Fz_traj = [], []
#     for a1 in a1_circle:
#         Fy, Fz = mapper.map_box_area_uniform(a1, a2_fixed)
#         Fy_traj.append(Fy)
#         Fz_traj.append(Fz)
#         A1.append(a1)
#         A2.append(a2_fixed)

#     # 足底轨迹
#     ax3.plot(Fy_traj, Fz_traj, '-b', label="foot traj (a1 sweep, a2=0)")
#     ax3.scatter(Fy_traj[0], Fz_traj[0], c='g', s=50, label="start")
#     ax3.scatter(Fy_traj[-1], Fz_traj[-1], c='r', s=50, label="end")
#     # 工作空间边界
#     ax3.plot(x_poly, y_poly, 'k-', linewidth=1.0, alpha=0.7, label="safe_poly")
#     ax3.set_aspect('equal', 'box')
#     ax3.set_title("Foot trajectory for a1 sweep (check closure & smoothness)")
#     ax3.legend()
#     ax3.grid(True)

#     plt.tight_layout()
#     plt.show()


# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay

# ==========================
# 1. 工具函数 & 几何构造
# ==========================

def pick_incircle_center(poly, n=100):
    """
    简单网格搜索近似内切圆圆心：
    在 polygon 内部找到“离边界最远”的一点。
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
    """
    alpha shape 生成不规则工作空间的“外壳”多边形
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
        R = a*b*c / (4.0*area + 1e-12)       # 外接圆半径
        if R < 1.0 / alpha:                  # alpha 判据
            edges.update([(ia, ib), (ib, ic), (ic, ia)])
    edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
    m = unary_union(edge_segments)
    return unary_union(list(polygonize(m)))

# ==========================
# 2. 腿模型 & 工作空间采样
# ==========================

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

# ---------- 采样工作空间 ----------
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
alpha_poly = alpha_shape(points, alpha)
alpha_poly = alpha_poly.buffer(0)
safe_poly = alpha_poly.buffer(1e-6)

def is_valid(Fy, Fz):
    return safe_poly.covers(Point(Fy, Fz))

# ==========================
# 3. 射线 r_max(theta)
# ==========================

def raycast_rmax_poly(alpha_poly, thetas, origin=(0.0,0.0), R=None):
    """
    从 origin 沿 theta 方向，用长度 R 的线段与“多边形面”相交；
    取相交处沿射线“最远点”的距离作为 r_max(theta)。
    """
    alpha_poly = alpha_poly.buffer(0)
    cx, cy = origin

    if R is None:
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
        tip = (cx + R*np.cos(th), cy + R*np.sin(th))
        ray = LineString([origin, tip])
        inter = alpha_poly.intersection(ray)
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

# ==========================
# 4. 老版 StarMapper
# ==========================

class StarMapperOld:
    """
    你之前的老版本 StarMapper，未修 CDF / 角度裁剪对齐
    """
    def __init__(self, alpha_poly, origin,
                 theta_min=-np.pi, theta_max=np.pi, num=720,
                 alpha_in=0.10, beta=0.05, r_thr=0.002):

        self.alpha_poly = alpha_poly.buffer(0)  # 保存多边形边界用于投影

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

        # 面积权重 S(θ) 与累计 CDF —— 注意：这里是“老逻辑”
        S = 0.5 * (self.r_out**2 - self.r_in**2)
        S = np.clip(S, 0.0, None)
        S_sum = S.sum() + 1e-12
        self.cdf = np.cumsum(S) / S_sum

        # 自动裁剪可用角段：找最长连续 rmax>r_thr 的区间（老逻辑）
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
        idx = (theta - self.theta_min) / (self.theta_max - self.theta_min) * (L - 1)
        idx = np.clip(idx, 0.0, L - 1 - 1e-9)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, L - 1)
        t = float(idx - i0)
        return (1.0 - t) * self.rmax[i0] + t * self.rmax[i1]

    def map_box(self, a1, a2):
        theta = self.theta_min + 0.5*(a1 + 1.0)*(self.theta_max - self.theta_min)
        rho   = 0.5*(a2 + 1.0)  # [0,1]
        rmax  = max(0.0, self.rmax_interp(theta))
        r_in  = self.alpha_in * rmax
        r     = r_in + (1.0 - self.beta) * rho * (rmax - r_in)
        Fy    = self.origin[0] + r * np.cos(theta)
        Fz    = self.origin[1] + r * np.sin(theta)
        return Fy, Fz

    def map_box_area_preserving(self, a1, a2):
        theta = self.theta_min + 0.5*(a1 + 1.0)*(self.theta_max - self.theta_min)
        rho   = 0.5*(a2 + 1.0)  # [0,1]
        rmax  = max(0.0, self.rmax_interp(theta))

        r_in  = self.alpha_in * rmax
        r_out = (1.0 - self.beta) * rmax
        r = math.sqrt(r_in*r_in + rho * (r_out*r_out - r_in*r_in))

        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)
        return Fy, Fz

    def theta_from_u_area_uniform(self, u):
        u = np.clip(u, 0.0, 1.0 - 1e-12)
        idx = np.searchsorted(self.cdf, u, side="right")
        i1 = int(np.clip(idx, 1, len(self.thetas)-1))
        i0 = i1 - 1
        c0, c1 = self.cdf[i0], self.cdf[i1]
        t = 0.0 if c1==c0 else (u - c0) / (c1 - c0)
        theta = (1.0 - t) * self.thetas[i0] + t * self.thetas[i1]
        r_in  = (1.0 - t) * (self.alpha_in * self.rmax[i0]) + t * (self.alpha_in * self.rmax[i1])
        r_out = (1.0 - t) * ((1.0 - self.beta) * self.rmax[i0]) + t * ((1.0 - self.beta) * self.rmax[i1])
        return theta, r_in, r_out

    def map_box_area_uniform(self, a1, a2):
        u   = 0.5 * (a1 + 1.0)  # [0,1]
        rho = 0.5 * (a2 + 1.0)  # [0,1]
        theta, r_in, r_out = self.theta_from_u_area_uniform(u)
        r = math.sqrt(max(0.0, r_in*r_in + rho * (r_out*r_out - r_in*r_in)))
        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)
        if not is_valid(Fy, Fz):
            p_proj = alpha_poly.boundary.interpolate(alpha_poly.boundary.project(Point(Fy,Fz)))
            Fy, Fz = p_proj.x, p_proj.y
        return Fy, Fz

# ==========================
# 5. 测试：老版 StarMapperOld
# ==========================

if __name__ == "__main__":
    # 选一个 origin（内切圆近似中心）
    origin = pick_incircle_center(safe_poly)
    print("origin (approx incircle center):", origin)

    mapper_old = StarMapperOld(
        alpha_poly=safe_poly,
        origin=origin,
        theta_min=-np.pi,
        theta_max=np.pi,
        num=720,
        alpha_in=0.10,
        beta=0.05,
        r_thr=0.002
    )

    # ---- Figure 1: r_max(theta) ----
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(mapper_old.thetas, mapper_old.rmax, '-')
    ax1.set_title("OLD: r_max(theta) after trimming")
    ax1.set_xlabel("theta [rad]")
    ax1.set_ylabel("r_max [m]")
    ax1.grid(True)

    # ---- Figure 2: 工作空间 + Box 映射点 ----
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(Fy_arr, Fz_arr, s=1, alpha=0.2, label="sampled workspace")
    x_poly, y_poly = safe_poly.exterior.xy
    ax2.plot(x_poly, y_poly, 'k-', linewidth=1.5, label="safe_poly")

    N = 40
    a1s = np.linspace(-1.0, 1.0, N)
    a2s = np.linspace(-1.0, 1.0, N)
    Fy_map, Fz_map = [], []
    for a1 in a1s:
        for a2 in a2s:
            Fy, Fz = mapper_old.map_box_area_uniform(a1, a2)
            Fy_map.append(Fy)
            Fz_map.append(Fz)
    ax2.scatter(Fy_map, Fz_map, s=5, c='r', alpha=0.5, label="mapped from Box")

    ax2.scatter([origin[0]], [origin[1]], c='g', s=40, label="origin")
    ax2.set_aspect('equal', 'box')
    ax2.set_title("OLD: Workspace & Box-mapped points (area-uniform)")
    ax2.legend()
    ax2.grid(True)

    # ---- Figure 3: a1 扫一圈, a2 固定 ----
    fig3, ax3 = plt.subplots(figsize=(6,6))
    a2_fixed = 0.0
    Ts = 400
    a1_circle = np.linspace(-1.0, 1.0, Ts, endpoint=False)
    Fy_traj, Fz_traj = [], []
    for a1 in a1_circle:
        Fy, Fz = mapper_old.map_box_area_uniform(a1, a2_fixed)
        Fy_traj.append(Fy)
        Fz_traj.append(Fz)

    ax3.plot(Fy_traj, Fz_traj, '-b', label="foot traj (a1 sweep, a2=0)")
    ax3.scatter(Fy_traj[0],  Fz_traj[0],  c='g', s=50, label="start")
    ax3.scatter(Fy_traj[-1], Fz_traj[-1], c='r', s=50, label="end")
    ax3.plot(x_poly, y_poly, 'k-', linewidth=1.0, alpha=0.7, label="safe_poly")
    ax3.set_aspect('equal', 'box')
    ax3.set_title("OLD: Foot trajectory for a1 sweep (check closure & smoothness)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
