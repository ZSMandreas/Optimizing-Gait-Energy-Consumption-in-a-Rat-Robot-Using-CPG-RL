# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import matplotlib.colors as mcolors

def pick_incircle_center(poly, n=100):
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

# ---------------- utility functions ----------------
def law_of_cosines_angle(la, lb, lc):
    cos_val = (la**2 + lb**2 - lc**2) / (2 * la * lb + 1e-12)
    if cos_val < -1 or cos_val > 1:
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
    tmp = area_AEC / (area_CDE - area_CDA + 1e-12)
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
        R = a*b*c / (4.0*area + 1e-12)  # 外接圆半径
        if R < 1.0 / alpha:
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

        a_ECz = math.acos((Cz - Ez)/CE) * np.sign(Ey - Cy + 1e-12)
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

# --------- 新：r_max 为“第一次击中边界”的半径 ---------
def _collect_coords(g):
    gt = g.geom_type
    if gt == "Point":
        return [(g.x, g.y)]
    if gt == "MultiPoint":
        return [(p.x, p.y) for p in g.geoms]
    if gt == "LineString":
        # 边界和射线重合的极少数情况：取段的两个端点
        coords = list(g.coords)
        if len(coords) >= 2:
            return [coords[0], coords[-1]]
        return coords
    if gt == "MultiLineString":
        coords = []
        for seg in g.geoms:
            cc = list(seg.coords)
            if len(cc) >= 2:
                coords += [cc[0], cc[-1]]
            else:
                coords += cc
        return coords
    if gt == "GeometryCollection":
        coords = []
        for gg in g.geoms:
            coords += _collect_coords(gg)
        return coords
    return []

def raycast_rmax_poly_first(alpha_poly, thetas, origin=(0.0, 0.0), R=None):
    """
    与 alpha_poly 的“边界”求交，取从 origin 出发沿 θ 方向遇到的
    **第一个相交点**的距离作为 r_max(θ)。
    若无交点，则 r_max=0。
    """
    alpha_poly = alpha_poly.buffer(0)
    boundary = alpha_poly.boundary  # 使用边界而不是面
    cx, cy = origin

    if R is None:
        r_pts = np.hypot(Fy_arr - cx, Fz_arr - cy)
        R = 1.5 * (np.max(r_pts) + 1e-6)

    rmax = np.zeros_like(thetas, dtype=float)

    for i, th in enumerate(thetas):
        tip = (cx + R*np.cos(th), cy + R*np.sin(th))
        ray = LineString([origin, tip])
        inter = boundary.intersection(ray)  # 与边界求交 → 一组点/线段
        if inter.is_empty:
            rmax[i] = 0.0
            continue
        pts = _collect_coords(inter)
        if not pts:
            rmax[i] = 0.0
            continue
        # 取最小的正向距离（第一个击中）
        dists = [np.hypot(x - cx, y - cy) for (x, y) in pts]
        # 去除 0 距离（origin 在边界上）带来的奇异
        dists = [d for d in dists if d > 1e-9]
        rmax[i] = float(min(dists)) if len(dists) > 0 else 0.0
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
        self.alpha_poly = alpha_poly.buffer(0)  # 保存边界用于校验/投影
        self.alpha_in = float(alpha_in)
        self.beta     = float(beta)
        self.theta_min, self.theta_max = theta_min, theta_max
        self.thetas = np.linspace(theta_min, theta_max, num, endpoint=False)
        self.origin = origin

        # ★ 用“第一次击中边界”的半径做 rmax
        self.rmax = raycast_rmax_poly_first(alpha_poly, self.thetas, origin=self.origin, R=None)

        # 先算 r_in/out
        self.r_in  = self.alpha_in * self.rmax
        self.r_out = (1.0 - self.beta) * self.rmax

        # 面积权重 S(θ) 与累计 CDF
        S = 0.5 * (self.r_out**2 - self.r_in**2)
        S = np.clip(S, 0.0, None)
        S_sum = S.sum() + 1e-12
        self.cdf = np.cumsum(S) / S_sum

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
        if L <= 1:
            return 0.0
        idx = (theta - self.theta_min) / (self.theta_max - self.theta_min + 1e-12) * (L - 1)
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
        rho = 0.0 if rho < 0.0 else (1.0 if rho > 1.0 else rho)
        return Fy, Fz, rho

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
        t = 0.0 if c1 == c0 else (u - c0) / (c1 - c0)
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
            p_proj = alpha_poly.boundary.interpolate(alpha_poly.boundary.project(Point(Fy, Fz)))
            Fy, Fz = p_proj.x, p_proj.y
        return Fy, Fz

# 选择一个位于多边形内部的原点作为射线中心
origin = pick_incircle_center(alpha_poly)

# 构建 Mapper
mapper = StarMapper(
    alpha_poly,
    origin=origin,
    num=1440,         # 每 0.5 度一个采样
    alpha_in=0.10,    # 内半径比例
    beta=0.05,        # 外边界裕度
    r_thr=0.002       # 有效角阈值（2~3 mm）
)

# ----------------- 可视化 -----------------
# 1) α-shape 工作空间
plt.figure(figsize=(6, 6))
plt.scatter(Fy_arr, Fz_arr, s=2, alpha=0.35, label='valid samples')
if alpha_poly.geom_type == 'Polygon':
    x, y = alpha_poly.exterior.xy
    plt.fill(x, y, alpha=0.2)
    plt.plot(x, y, label='alpha-shape boundary')
else:
    for poly in alpha_poly.geoms:
        x, y = poly.exterior.xy
        plt.fill(x, y, alpha=0.2)
        plt.plot(x, y, label='alpha-shape boundary')
plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
plt.axis('equal'); plt.grid(True); plt.legend()
plt.title("Workspace via Alpha-Shape")
plt.show()

# 2) Box→Workspace 单射映射（用网格演示）
grid_lin = np.linspace(-1, 1, 120)
U1, U2 = np.meshgrid(grid_lin, grid_lin)
Fy_map = np.zeros_like(U1, dtype=float)
Fz_map = np.zeros_like(U2, dtype=float)
ok = 0; total = U1.size
for i in range(U1.shape[0]):
    for j in range(U1.shape[1]):
        Fy_map[i, j], Fz_map[i, j] = mapper.map_box_area_uniform(U1[i, j], U2[i, j])
        if is_valid(Fy_map[i, j], Fz_map[i, j]):
            ok += 1

plt.figure(figsize=(6, 6))
plt.plot(x, y, linewidth=2, label='alpha-shape boundary')
plt.scatter(Fy_map.flatten(), Fz_map.flatten(), s=6, alpha=0.85, label='mapped points from Box')
plt.scatter([origin[0]], [origin[1]], s=50, marker='x', label='origin')
plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
plt.axis('equal'); plt.grid(True); plt.legend()
plt.title(f"Box→Workspace Mapping (inside: {ok}/{total})\n"
          f"theta range [{mapper.theta_min*180/np.pi:.1f}°, {mapper.theta_max*180/np.pi:.1f}°], "
          f"alpha_in={mapper.alpha_in}, beta={mapper.beta}")
plt.show()

# ========= 10k actions in [-1,1]^2 -> map -> compare with reachable area =========
np.random.seed(42)
N = 10_000
actions = np.random.uniform(-1.0, 1.0, size=(N, 2))  # (a1, a2)

Fy_s = np.empty(N, dtype=float)
Fz_s = np.empty(N, dtype=float)
inside = np.empty(N, dtype=bool)

for i in range(N):
    a1, a2 = actions[i]
    Fy_i, Fz_i = mapper.map_box_area_uniform(a1, a2)
    Fy_s[i] = Fy_i
    Fz_s[i] = Fz_i
    inside[i] = is_valid(Fy_i, Fz_i)

inside_ratio = inside.mean()
print(f"[Sampling] N={N}, inside={inside.sum()} ({inside_ratio*100:.2f}%), outside={(~inside).sum()}")

# --- Plot: reachable area boundary + mapped samples ---
plt.figure(figsize=(7, 7))
if alpha_poly.geom_type == 'Polygon':
    bx, by = alpha_poly.exterior.xy
    plt.fill(bx, by, alpha=0.15, label='reachable area (α-shape)')
    plt.plot(bx, by, linewidth=2, color='k')
else:
    for poly in alpha_poly.geoms:
        bx, by = poly.exterior.xy
        plt.fill(bx, by, alpha=0.15, label='reachable area (α-shape)')
        plt.plot(bx, by, linewidth=2, color='k')

plt.scatter(Fy_s[inside],  Fz_s[inside],  s=6, alpha=0.7, label='mapped (inside)')
plt.scatter(Fy_s[~inside], Fz_s[~inside], s=6, alpha=0.7, label='mapped (outside)')
plt.scatter([origin[0]], [origin[1]], s=60, marker='x', label='origin')

plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.xlabel("Fy (m)"); plt.ylabel("Fz (m)")
plt.title(f"Box [-1,1]^2 -> Workspace mapping (N={N})\n"
          f"inside={inside.sum()} ({inside_ratio*100:.2f}%), "
          f"θ∈[{mapper.theta_min*180/np.pi:.1f}°, {mapper.theta_max*180/np.pi:.1f}°], "
          f"α_in={mapper.alpha_in}, β={mapper.beta}")
plt.legend(loc='best')
plt.show()
