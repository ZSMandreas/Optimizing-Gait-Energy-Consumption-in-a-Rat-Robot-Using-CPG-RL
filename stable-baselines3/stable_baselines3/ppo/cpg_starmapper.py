# -*- coding: utf-8 -*-
"""
CPG + StarMapper 轨迹测试脚本

功能：
1. 用 WorkspaceDetection 采样单腿足端工作空间；
2. 用 alpha-shape 拟合不规则可达域多边形；
3. 构造 StarMapper，把 Box([-1,1]^2) 映射到不规则可达域；
4. 用 CPG 在 (u_theta, rho) 极坐标参数空间里生成一段轨迹；
5. 通过 StarMapper 得到可达域中的足底轨迹，并与可达域一起画图。

运行：
    python cpg_starmapper_test.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay


# ========== 一些几何小工具 ==========

def pick_incircle_center(poly: Polygon, n: int = 100):
    """
    粗略找一个“内切圆圆心”（最大内接圆中心的近似），
    用网格暴力搜索。只是为了选一个比较“居中”的 origin。
    """
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    best, best_d = (poly.representative_point().x,
                    poly.representative_point().y), -1.0
    for y in ys:
        for x in xs:
            p = Point(x, y)
            if poly.contains(p):
                d = p.distance(poly.boundary)
                if d > best_d:
                    best, best_d = (x, y), d
    return best  # (x,y)


def law_of_cosines_angle(la, lb, lc):
    """余弦定理求角，返回弧度；若数值超界则返回 -10 作为无效标志。"""
    cos_val = (la ** 2 + lb ** 2 - lc ** 2) / (2 * la * lb)
    if abs(cos_val) > 1:
        return -10
    return math.acos(cos_val)


def check_cross(line1, line2):
    """
    线段相交判断 + 求交点，参考你原来的代码。
    line1 = [[Cx,Cy],[Dx,Dy]]
    line2 = [[Ax,Ay],[Ex,Ey]]
    """
    C, D = line1
    A, E = line2
    area_CDA = (C[0] - A[0]) * (D[1] - A[1]) - (C[1] - A[1]) * (D[0] - A[0])
    area_CDE = (C[0] - E[0]) * (D[1] - E[1]) - (C[1] - E[1]) * (D[0] - E[0])
    area_AEC = (A[0] - C[0]) * (E[1] - C[1]) - (A[1] - C[1]) * (E[0] - C[0])
    area_AED = (A[0] - D[0]) * (E[1] - D[1]) - (A[1] - D[1]) * (E[0] - D[0])

    if (area_CDA * area_CDE) >= 0 or (area_AEC * area_AED) >= 0:
        return []
    tmp = area_AEC / (area_CDE - area_CDA)
    dx = tmp * (D[0] - C[0])
    dy = tmp * (D[1] - C[1])
    return [C[0] + dx, C[1] + dy]


def alpha_shape(pts, alpha):
    """
    用 alpha-shape 把散点拟合成不规则多边形。
    pts: (N,2) numpy 数组
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
        if R < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])
    edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
    m = unary_union(edge_segments)
    return unary_union(list(polygonize(m)))


# ========== 单腿工作空间模型 ==========

class WorkspaceDetection:
    """
    你的腿几何模型（简化版），从关节角 (q1,q2) → 足底位置 (Fy,Fz)。
    直接用你之前贴的那套几何约束。
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
        if (a_AEF < PI / 6) or (a_AEF > PI * 5 / 6) or \
           (a_BCD < PI / 6) or (a_BCD > PI * 5 / 6):
            return []

        return Fy, Fz


# ========== 射线 r_max(theta)（与多边形相交） ==========

def raycast_rmax_poly(alpha_poly, thetas, origin=(0.0, 0.0), R=None):
    """
    从 origin 沿 theta 方向，用长度 R 的线段与多边形相交；
    取相交处沿射线最远点的距离作为 r_max(theta)。
    """
    alpha_poly = alpha_poly.buffer(0)
    cx, cy = origin

    if R is None:
        # 根据点云半径估个合理 R
        r_pts = np.hypot(FY_ARR - cx, FZ_ARR - cy)
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
        tip = (cx + R * np.cos(th), cy + R * np.sin(th))
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


# ========== StarMapper：Box → 不规则可达域（带面积均匀） ==========

class StarMapper:
    """
    Box(a1,a2)∈[-1,1]^2 → (Fy,Fz) 的单射映射。
    - a1 控制角度（通过面积 CDF 均匀在有效角域上铺开）
    - a2 控制径向 rho（通过面积保持映射 r(rho)）
    """
    def __init__(
        self,
        alpha_poly,
        origin,
        theta_min=-np.pi,
        theta_max=np.pi,
        num=720,
        alpha_in=0.10,
        beta=0.05,
        r_thr=0.002,
    ):
        self.alpha_poly = alpha_poly.buffer(0)
        self.origin = origin
        self.alpha_in = float(alpha_in)
        self.beta = float(beta)

        self.theta_min = theta_min
        self.theta_max = theta_max
        self.thetas = np.linspace(theta_min, theta_max, num, endpoint=False)

        # 计算 rmax(θ)
        self.rmax = raycast_rmax_poly(alpha_poly, self.thetas, origin=self.origin, R=None)

        # 先算 r_in/out
        self.r_in = self.alpha_in * self.rmax
        self.r_out = (1.0 - self.beta) * self.rmax

        # 面积权重 S(θ) 与累计 CDF（用于角度面积均匀）
        S = 0.5 * (self.r_out ** 2 - self.r_in ** 2)
        S = np.clip(S, 0.0, None)
        S_sum = S.sum() + 1e-12
        self.cdf = np.cumsum(S) / S_sum

        # 自动裁剪有效角段：找最长连续 rmax>r_thr 的区间
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

        # 裁剪后重新算一次 r_in/out 与 CDF
        self.r_in = self.alpha_in * self.rmax
        self.r_out = (1.0 - self.beta) * self.rmax
        S = 0.5 * (self.r_out ** 2 - self.r_in ** 2)
        S = np.clip(S, 0.0, None)
        S_sum = S.sum() + 1e-12
        self.cdf = np.cumsum(S) / S_sum

    def theta_from_u_area_uniform(self, u):
        """
        u∈[0,1] → 按面积均匀选择 θ，并插值得到对应 r_in/out。
        """
        u = np.clip(u, 0.0, 1.0 - 1e-12)
        idx = np.searchsorted(self.cdf, u, side="right")
        i1 = int(np.clip(idx, 1, len(self.thetas) - 1))
        i0 = i1 - 1
        c0, c1 = self.cdf[i0], self.cdf[i1]
        t = 0.0 if c1 == c0 else (u - c0) / (c1 - c0)
        theta = (1.0 - t) * self.thetas[i0] + t * self.thetas[i1]
        r_in = (1.0 - t) * self.r_in[i0] + t * self.r_in[i1]
        r_out = (1.0 - t) * self.r_out[i0] + t * self.r_out[i1]
        return theta, r_in, r_out

    def map_box_area_uniform(self, a1, a2):
        """
        a1,a2 ∈ [-1,1] 映射到不规则可达域中的足底点 (Fy,Fz)。
        - a1: 通过 u = (a1+1)/2 控制角度 CDF；
        - a2: 通过 rho = (a2+1)/2 控制径向面积。
        """
        # 角度参数
        u = 0.5 * (a1 + 1.0)  # → [0,1]
        rho = 0.5 * (a2 + 1.0)  # → [0,1]

        theta, r_in, r_out = self.theta_from_u_area_uniform(u)
        r = math.sqrt(max(0.0, r_in * r_in + rho * (r_out * r_out - r_in * r_in)))

        Fy = self.origin[0] + r * math.cos(theta)
        Fz = self.origin[1] + r * math.sin(theta)

        # 若数值误差导致点略偏出多边形，可选择投影回边界
        p = Point(Fy, Fz)
        if not self.alpha_poly.buffer(1e-9).covers(p):
            proj = self.alpha_poly.boundary.interpolate(self.alpha_poly.boundary.project(p))
            Fy, Fz = proj.x, proj.y

        return Fy, Fz


# ========== 全局变量占位（raycast_rmax_poly 里用到） ==========

FY_ARR = None
FZ_ARR = None


# ========== CPG + StarMapper 轨迹测试 ==========

def main():
    global FY_ARR, FZ_ARR

    # 1) 采样单腿工作空间点云
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

    FY_ARR = np.array(Fy_col)
    FZ_ARR = np.array(Fz_col)
    points = np.vstack((FY_ARR, FZ_ARR)).T

    # 2) alpha-shape 拟合不规则可达域多边形
    alpha = 100.0
    alpha_poly = alpha_shape(points, alpha)
    alpha_poly = alpha_poly.buffer(0)      # 修正几何
    safe_poly = alpha_poly.buffer(1e-6)    # 轻微膨胀，数值安全

    # 3) 选择 origin（大致内切圆中心）
    origin = pick_incircle_center(safe_poly, n=100)
    print("Origin (incircle approx):", origin)

    # 4) 构造 StarMapper（参数可调）
    mapper = StarMapper(
        alpha_poly=safe_poly,
        origin=origin,
        theta_min=-np.pi,
        theta_max=np.pi,
        num=720,
        alpha_in=0.10,
        beta=0.05,
        r_thr=0.002,
    )
    print("Theta range after cropping: ", mapper.theta_min, mapper.theta_max)

    # 5) 用 CPG 在 (u_theta, rho) 空间生成一段轨迹，再映射到可达域
    T = 1.0      # 轨迹总时长 1 秒
    dt = 0.002   # 时间步长
    f_cpg = 2.0  # CPG 频率 Hz

    omega = 2.0 * np.pi * f_cpg
    times = np.arange(0.0, T, dt)

    # 可以自己调的两个参数（相当于当前“这条腿”的策略输出）
    amp = 0.8      # 振幅 [0,1]
    offset = 0.5   # 偏置 [0,1]

    phi0 = 0.0     # 初始相位

    Fy_path, Fz_path = [], []
    u_theta_list, rho_list = [], []

    for t in times:
        phi = phi0 + omega * t

        # 角度：归一化到 [0,1]，对应 StarMapper 里的 u
        u_theta = (phi % (2.0 * np.pi)) / (2.0 * np.pi)

        # CPG 波形：0~1
        base = 0.5 * (1.0 + math.sin(phi))

        # 把策略参数编码进 rho(t)
        rho = offset + (base - 0.5) * amp
        rho = np.clip(rho, 0.0, 1.0)

        # 转回 Box 坐标
        a1 = 2.0 * u_theta - 1.0  # [-1,1]
        a2 = 2.0 * rho     - 1.0  # [-1,1]

        Fy, Fz = mapper.map_box_area_uniform(a1, a2)

        Fy_path.append(Fy)
        Fz_path.append(Fz)
        u_theta_list.append(u_theta)
        rho_list.append(rho)

    Fy_path = np.array(Fy_path)
    Fz_path = np.array(Fz_path)

    # 6) 画图：背景 = 点云 + 多边形；前景 = CPG 轨迹
    fig, ax = plt.subplots(figsize=(6, 6))

    # 点云
    ax.scatter(FY_ARR, FZ_ARR, s=1, alpha=0.15, label="Workspace samples")

    # alpha-shape 多边形边界
    x_poly, y_poly = safe_poly.exterior.xy
    ax.plot(x_poly, y_poly, "k-", linewidth=1.0, label="Alpha shape boundary")

    # CPG 映射后的足底轨迹
    ax.plot(Fy_path, Fz_path, "r-", linewidth=2.0, label="CPG + StarMapper trajectory")

    # origin
    ax.scatter([origin[0]], [origin[1]], c="g", s=30, label="Origin")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("Fy [m]")
    ax.set_ylabel("Fz [m]")
    ax.set_title("CPG trajectory mapped by StarMapper into reachable workspace")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # 额外：可以打印一下 u_theta / rho 的范围看看
    print("u_theta in [%.3f, %.3f]" % (min(u_theta_list), max(u_theta_list)))
    print("rho     in [%.3f, %.3f]" % (min(rho_list), max(rho_list)))


if __name__ == "__main__":
    main()


# # -*- coding: utf-8 -*-
# """
# CPG + StarMapper 轨迹测试脚本（多条轨迹 & 平滑 reachable area）

# 功能：
# 1. 用 WorkspaceDetection 采样单腿足端工作空间；
# 2. 用 alpha-shape 拟合不规则可达域多边形，并用 simplify 平滑边界；
# 3. 构造 StarMapper，把 Box([-1,1]^2) 映射到不规则可达域；
# 4. 定义多组 (amp, offset)，用 CPG 在 (u_theta, rho) 极坐标参数空间里生成多条轨迹；
# 5. 通过 StarMapper 映射到可达域中，并在同一张图里画出。

# 运行：
#     python cpg_starmapper_test_multi.py
# """

# import math
# import numpy as np
# import matplotlib.pyplot as plt

# from shapely.geometry import Point, MultiPoint, Polygon, LineString
# from shapely.ops import unary_union, polygonize
# from scipy.spatial import Delaunay


# # ========== 一些几何小工具 ==========

# def pick_incircle_center(poly: Polygon, n: int = 100):
#     """
#     粗略找一个“内切圆圆心”（最大内接圆中心的近似），
#     用网格暴力搜索。只是为了选一个比较“居中”的 origin。
#     """
#     minx, miny, maxx, maxy = poly.bounds
#     xs = np.linspace(minx, maxx, n)
#     ys = np.linspace(miny, maxy, n)
#     best, best_d = (poly.representative_point().x,
#                     poly.representative_point().y), -1.0
#     for y in ys:
#         for x in xs:
#             p = Point(x, y)
#             if poly.contains(p):
#                 d = p.distance(poly.boundary)
#                 if d > best_d:
#                     best, best_d = (x, y), d
#     return best  # (x,y)


# def law_of_cosines_angle(la, lb, lc):
#     """余弦定理求角，返回弧度；若数值超界则返回 -10 作为无效标志。"""
#     cos_val = (la ** 2 + lb ** 2 - lc ** 2) / (2 * la * lb)
#     if abs(cos_val) > 1:
#         return -10
#     return math.acos(cos_val)


# def check_cross(line1, line2):
#     """
#     线段相交判断 + 求交点，参考你原来的代码。
#     line1 = [[Cx,Cy],[Dx,Dy]]
#     line2 = [[Ax,Ay],[Ex,Ey]]
#     """
#     C, D = line1
#     A, E = line2
#     area_CDA = (C[0] - A[0]) * (D[1] - A[1]) - (C[1] - A[1]) * (D[0] - A[0])
#     area_CDE = (C[0] - E[0]) * (D[1] - E[1]) - (C[1] - E[1]) * (D[0] - E[0])
#     area_AEC = (A[0] - C[0]) * (E[1] - C[1]) - (A[1] - C[1]) * (E[0] - C[0])
#     area_AED = (A[0] - D[0]) * (E[1] - D[1]) - (A[1] - D[1]) * (E[0] - D[0])

#     if (area_CDA * area_CDE) >= 0 or (area_AEC * area_AED) >= 0:
#         return []
#     tmp = area_AEC / (area_CDE - area_CDA)
#     dx = tmp * (D[0] - C[0])
#     dy = tmp * (D[1] - C[1])
#     return [C[0] + dx, C[1] + dy]


# def alpha_shape(pts, alpha):
#     """
#     用 alpha-shape 把散点拟合成不规则多边形。
#     pts: (N,2) numpy 数组
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
#         R = a * b * c / (4.0 * area + 1e-12)  # 外接圆半径
#         if R < 1.0 / alpha:
#             edges.update([(ia, ib), (ib, ic), (ic, ia)])
#     edge_segments = [LineString([pts[i], pts[j]]) for i, j in edges]
#     m = unary_union(edge_segments)
#     return unary_union(list(polygonize(m)))


# # ========== 单腿工作空间模型 ==========

# class WorkspaceDetection:
#     """
#     腿几何模型（简化版），从关节角 (q1,q2) → 足底位置 (Fy,Fz)。
#     直接用你之前贴的那套几何约束。
#     """
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
#         if (a_AEF < PI / 6) or (a_AEF > PI * 5 / 6) or \
#            (a_BCD < PI / 6) or (a_BCD > PI * 5 / 6):
#             return []

#         return Fy, Fz


# # ========== 射线 r_max(theta)（与多边形相交） ==========

# def raycast_rmax_poly(alpha_poly, thetas, origin=(0.0, 0.0), R=None):
#     """
#     从 origin 沿 theta 方向，用长度 R 的线段与多边形相交；
#     取相交处沿射线最远点的距离作为 r_max(theta)。
#     """
#     alpha_poly = alpha_poly.buffer(0)
#     cx, cy = origin

#     if R is None:
#         # 根据点云半径估个合理 R
#         r_pts = np.hypot(FY_ARR - cx, FZ_ARR - cy)
#         R = 1.5 * (np.max(r_pts) + 1e-6)

#     rmax = np.zeros_like(thetas, dtype=float)

#     def _collect_coords(g):
#         gt = g.geom_type
#         if gt == "Point":
#             return [(g.x, g.y)]
#         if gt == "MultiPoint":
#             return [(p.x, p.y) for p in g.geoms]
#         if gt == "LineString":
#             return list(g.coords)
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
#         inter = alpha_poly.intersection(ray)
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


# # ========== StarMapper：Box → 不规则可达域（带面积均匀） ==========

# class StarMapper:
#     """
#     Box(a1,a2)∈[-1,1]^2 → (Fy,Fz) 的单射映射。
#     - a1 控制角度（通过面积 CDF 均匀在有效角域上铺开）
#     - a2 控制径向 rho（通过面积保持映射 r(rho)）
#     """
#     def __init__(
#         self,
#         alpha_poly,
#         origin,
#         theta_min=-np.pi,
#         theta_max=np.pi,
#         num=720,
#         alpha_in=0.10,
#         beta=0.05,
#         r_thr=0.002,
#     ):
#         self.alpha_poly = alpha_poly.buffer(0)
#         self.origin = origin
#         self.alpha_in = float(alpha_in)
#         self.beta = float(beta)

#         self.theta_min = theta_min
#         self.theta_max = theta_max
#         self.thetas = np.linspace(theta_min, theta_max, num, endpoint=False)

#         # 计算 rmax(θ)
#         self.rmax = raycast_rmax_poly(alpha_poly, self.thetas, origin=self.origin, R=None)

#         # 先算 r_in/out
#         self.r_in = self.alpha_in * self.rmax
#         self.r_out = (1.0 - self.beta) * self.rmax

#         # 面积权重 S(θ) 与累计 CDF（用于角度面积均匀）
#         S = 0.5 * (self.r_out ** 2 - self.r_in ** 2)
#         S = np.clip(S, 0.0, None)
#         S_sum = S.sum() + 1e-12
#         self.cdf = np.cumsum(S) / S_sum

#         # 自动裁剪有效角段：找最长连续 rmax>r_thr 的区间
#         valid = self.rmax > r_thr
#         if np.any(valid) and not np.all(valid):
#             valid2 = np.r_[valid, valid]
#             best_len, best_i0 = 0, 0
#             i = 0
#             n = len(valid2)
#             while i < n:
#                 if valid2[i]:
#                     j = i
#                     while j < n and valid2[j]:
#                         j += 1
#                     seg_len = j - i
#                     if seg_len > best_len:
#                         best_len, best_i0 = seg_len, i
#                     i = j
#                 else:
#                     i += 1
#             best_i0 %= len(valid)
#             best_i1 = (best_i0 + best_len - 1) % len(valid)
#             if best_len >= 3:
#                 if best_i1 >= best_i0:
#                     sl = slice(best_i0, best_i1 + 1)
#                     self.thetas = self.thetas[sl]
#                     self.rmax = self.rmax[sl]
#                 else:
#                     self.thetas = np.r_[self.thetas[best_i0:], self.thetas[:best_i1 + 1]]
#                     self.rmax = np.r_[self.rmax[best_i0:], self.rmax[:best_i1 + 1]]
#                 self.theta_min = float(self.thetas[0])
#                 self.theta_max = float(self.thetas[-1])

#         # 裁剪后重新算一次 r_in/out 与 CDF
#         self.r_in = self.alpha_in * self.rmax
#         self.r_out = (1.0 - self.beta) * self.rmax
#         S = 0.5 * (self.r_out ** 2 - self.r_in ** 2)
#         S = np.clip(S, 0.0, None)
#         S_sum = S.sum() + 1e-12
#         self.cdf = np.cumsum(S) / S_sum

#     def theta_from_u_area_uniform(self, u):
#         """
#         u∈[0,1] → 按面积均匀选择 θ，并插值得到对应 r_in/out。
#         """
#         u = np.clip(u, 0.0, 1.0 - 1e-12)
#         idx = np.searchsorted(self.cdf, u, side="right")
#         i1 = int(np.clip(idx, 1, len(self.thetas) - 1))
#         i0 = i1 - 1
#         c0, c1 = self.cdf[i0], self.cdf[i1]
#         t = 0.0 if c1 == c0 else (u - c0) / (c1 - c0)
#         theta = (1.0 - t) * self.thetas[i0] + t * self.thetas[i1]
#         r_in = (1.0 - t) * self.r_in[i0] + t * self.r_in[i1]
#         r_out = (1.0 - t) * self.r_out[i0] + t * self.r_out[i1]
#         return theta, r_in, r_out

#     def map_box_area_uniform(self, a1, a2):
#         """
#         a1,a2 ∈ [-1,1] 映射到不规则可达域中的足底点 (Fy,Fz)。
#         - a1: 通过 u = (a1+1)/2 控制角度 CDF；
#         - a2: 通过 rho = (a2+1)/2 控制径向面积。
#         """
#         # 角度参数
#         u = 0.5 * (a1 + 1.0)  # → [0,1]
#         rho = 0.5 * (a2 + 1.0)  # → [0,1]

#         theta, r_in, r_out = self.theta_from_u_area_uniform(u)
#         r = math.sqrt(max(0.0, r_in * r_in + rho * (r_out * r_out - r_in * r_in)))

#         Fy = self.origin[0] + r * math.cos(theta)
#         Fz = self.origin[1] + r * math.sin(theta)

#         # 若数值误差导致点略偏出多边形，可选择投影回边界
#         p = Point(Fy, Fz)
#         if not self.alpha_poly.buffer(1e-9).covers(p):
#             proj = self.alpha_poly.boundary.interpolate(self.alpha_poly.boundary.project(p))
#             Fy, Fz = proj.x, proj.y

#         return Fy, Fz


# # ========== 全局变量占位（raycast_rmax_poly 里用到） ==========

# FY_ARR = None
# FZ_ARR = None


# # ========== CPG + StarMapper 轨迹测试 ==========

# def main():
#     global FY_ARR, FZ_ARR

#     # 1) 采样单腿工作空间点云
#     leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
#     model = WorkspaceDetection(leg_params)

#     grid = 400
#     q_vals = np.linspace(-3.0, 3.0, grid)
#     Fy_col, Fz_col = [], []

#     for q1 in q_vals:
#         for q2 in q_vals:
#             res = model.angel_2_pos(q1, q2)
#             if res and res[1] < 0:  # 只保留落地半空间
#                 Fy_col.append(res[0])
#                 Fz_col.append(res[1])

#     FY_ARR = np.array(Fy_col)
#     FZ_ARR = np.array(Fz_col)
#     points = np.vstack((FY_ARR, FZ_ARR)).T

#     # 2) alpha-shape 拟合不规则可达域多边形 + 平滑
#     alpha = 100.0
#     raw_poly = alpha_shape(points, alpha)
#     raw_poly = raw_poly.buffer(0)  # 修复几何

#     # 平滑：简化边界，tolerance 可以微调（比如 0.0005~0.002）
#     smooth_tol = 0.001
#     smooth_poly = raw_poly.simplify(smooth_tol, preserve_topology=True)
#     smooth_poly = smooth_poly.buffer(0)  # 再修复一次几何

#     safe_poly = smooth_poly.buffer(1e-6)  # 轻微膨胀，数值安全

#     # 3) 选择 origin（大致内切圆中心）
#     origin = pick_incircle_center(safe_poly, n=120)
#     print("Origin (incircle approx):", origin)

#     # 4) 构造 StarMapper（参数可调）
#     mapper = StarMapper(
#         alpha_poly=safe_poly,
#         origin=origin,
#         theta_min=-np.pi,
#         theta_max=np.pi,
#         num=720,
#         alpha_in=0.10,
#         beta=0.05,
#         r_thr=0.002,
#     )
#     print("Theta range after cropping: ", mapper.theta_min, mapper.theta_max)

#     # 5) 定义多组 (amp, offset)，生成多条 CPG 轨迹
#     T = 1.0      # 每条轨迹总时长 1 秒
#     dt = 0.002   # 时间步长
#     f_cpg = 2.0  # CPG 频率 Hz

#     omega = 2.0 * np.pi * f_cpg
#     times = np.arange(0.0, T, dt)

#     phi0 = 0.0   # 初始相位

#     # 多组 (amp, offset)，可以根据需要调整/增加
#     traj_params = [
#         (0.8, 0.5),
#         (0.6, 0.6),
#         (0.4, 0.4),
#         (0.3, 0.8),
#     ]

#     # 为了颜色好看，用 matplotlib 的 colormap
#     cmap = plt.get_cmap("tab10")

#     # 6) 画图：背景 = 点云 + 平滑后的多边形；前景 = 多条 CPG 轨迹
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # 点云
#     ax.scatter(FY_ARR, FZ_ARR, s=1, alpha=0.1, label="Workspace samples")

#     # 平滑后的 alpha-shape 多边形边界
#     x_poly, y_poly = safe_poly.exterior.xy
#     ax.plot(x_poly, y_poly, "k-", linewidth=1.0, label="Smoothed alpha-shape boundary")

#     # origin
#     ax.scatter([origin[0]], [origin[1]], c="g", s=30, label="Origin")

#     # 对每一组 (amp, offset) 画一条轨迹
#     for idx, (amp, offset) in enumerate(traj_params):
#         Fy_path, Fz_path = [], []
#         u_theta_list, rho_list = [], []

#         for t in times:
#             phi = phi0 + omega * t

#             # 角度：归一化到 [0,1]，对应 StarMapper 里的 u
#             u_theta = (phi % (2.0 * np.pi)) / (2.0 * np.pi)

#             # CPG 波形：0~1
#             base = 0.5 * (1.0 + math.sin(phi))

#             # 把 (amp, offset) 编码进 rho(t)
#             rho = offset + (base - 0.5) * amp
#             rho = np.clip(rho, 0.0, 1.0)

#             # 转回 Box 坐标
#             a1 = 2.0 * u_theta - 1.0  # [-1,1]
#             a2 = 2.0 * rho     - 1.0  # [-1,1]

#             Fy, Fz = mapper.map_box_area_uniform(a1, a2)

#             Fy_path.append(Fy)
#             Fz_path.append(Fz)
#             u_theta_list.append(u_theta)
#             rho_list.append(rho)

#         Fy_path = np.array(Fy_path)
#         Fz_path = np.array(Fz_path)

#         color = cmap(idx % 10)
#         ax.plot(
#             Fy_path,
#             Fz_path,
#             "-",
#             color=color,
#             linewidth=2.0,
#             label=f"CPG traj {idx+1}: amp={amp}, offset={offset}",
#         )

#         print(
#             f"Traj {idx+1}: u_theta in [{min(u_theta_list):.3f}, {max(u_theta_list):.3f}], "
#             f"rho in [{min(rho_list):.3f}, {max(rho_list):.3f}]"
#         )

#     ax.set_aspect("equal", "box")
#     ax.set_xlabel("Fy [m]")
#     ax.set_ylabel("Fz [m]")
#     ax.set_title("Smoothed workspace & multiple CPG trajectories via StarMapper")
#     ax.legend(loc="best", fontsize=8)
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()
