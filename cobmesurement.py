import os
import math
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree

# =======================
# CONFIG
# =======================
INPUT_DIR = r"maindata"
OUTPUT_DIR = r"results"

UNIT_SCALE_TO_CM = 5.63

N_RINGS = 55
MANUAL_RING_THICKNESS_CM = 1.0

REMOVE_OUTLIERS = True
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.5

# full-length centerline
LENGTH_SAMPLES = 140
LENGTH_SLICE_FRAC = 0.012
LENGTH_CENTERLINE_SMOOTH_WIN = 7
LENGTH_CENTERLINE_RESAMPLE_N = 220
LENGTH_CENTERLINE_FINAL_SMOOTH_WIN = 5

# ring/body region
PROFILE_SAMPLES = 240
PROFILE_SLICE_FRAC = 0.010
PROFILE_SMOOTH_WIN = 11
BODY_THRESH_FRAC = 0.22
BODY_GAP_FILL = 5

SKIP_END_FRAC_FOR_RINGS = 0.03
BOTTOM_POS_FRAC = 0.15
MIDDLE_POS_FRAC = 0.50
TOP_POS_FRAC = 0.85
OUTER_SMOOTH_WIN = 7

# concave boundary
ALPHA_MULT = 2.2  # larger = looser boundary, smaller = tighter
MAX_HOLE_EDGE_MULT = 3.0  # filter long noisy boundary edges

# small bounded end correction for length
END_EXT_FRAC_OF_RADIUS = 0.35
END_EXT_MAX_CM = 0.8

RING_TUBE_FRAC = 0.018
POINT_SIZE = 2.5
BACKGROUND = np.array([0.05, 0.05, 0.05])

SHOW_MATPLOTLIB_3D = True
SHOW_OPEN3D = True
SAVE_CSV = True


# -------------------------
# Utilities
# -------------------------
def moving_average_1d(x, win):
    x = np.asarray(x, float)
    win = int(win)
    if win <= 1 or len(x) < 2:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win) / win
    return np.convolve(xp, k, mode="valid")


def moving_average_vec(x, win):
    x = np.asarray(x, float)
    win = int(win)
    if win <= 1 or len(x) < 2:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(win) / win
    y = np.empty_like(x)
    for d in range(x.shape[1]):
        y[:, d] = np.convolve(xp[:, d], k, mode="valid")
    return y


def fill_small_gaps(mask, max_gap):
    mask = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and mask[i - 1] and mask[j] and gap_len <= max_gap:
            mask[i:j] = True
        i = j
    return mask


def largest_true_segment(mask):
    best = (None, None, 0)
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        length = j - i
        if length > best[2]:
            best = (i, j, length)
        i = j
    return best[0], best[1]


def infer_p_level_from_name(sample_name: str):
    s = sample_name.lower()
    if "ada0" in s:
        return 0
    if "ada25" in s:
        return 25
    if "ada85" in s:
        return 85
    return np.nan


# -------------------------
# Geometry helpers
# -------------------------
def pca_axis(points: np.ndarray):
    center = points.mean(axis=0)
    X = points - center
    C = (X.T @ X) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    return center, axis


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def points_near_plane(points, plane_point, plane_normal, half_thickness):
    d = (points - plane_point) @ plane_normal
    return np.abs(d) <= half_thickness


def resample_polyline_by_arclength(P, n):
    P = np.asarray(P, float)
    if len(P) == 0:
        return P
    if len(P) == 1:
        return np.repeat(P[:1], n, axis=0)

    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return np.repeat(P[:1], n, axis=0)

    s_new = np.linspace(0.0, s[-1], n)
    out = np.zeros((n, 3), float)
    for k in range(3):
        out[:, k] = np.interp(s_new, s, P[:, k])
    return out


def centerline_arclength(cl):
    cl = np.asarray(cl, float)
    if len(cl) < 2:
        return np.array([0.0])
    d = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def sample_centerline_by_arclength(cl, s, s_query):
    s_query = np.clip(np.asarray(s_query, float), s[0], s[-1])
    idx = np.searchsorted(s, s_query, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    t = (s_query - s[idx]) / np.maximum(s[idx + 1] - s[idx], 1e-12)
    p = (1 - t)[:, None] * cl[idx] + t[:, None] * cl[idx + 1]
    return p, idx, t


def tangent_from_centerline(cl, i):
    if len(cl) < 2:
        return np.array([0.0, 0.0, 1.0])
    if i <= 0:
        v = cl[1] - cl[0]
    elif i >= len(cl) - 1:
        v = cl[-1] - cl[-2]
    else:
        v = cl[i + 1] - cl[i - 1]
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


# -------------------------
# Circle fitting
# -------------------------
def fit_circle_center_2d(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if len(x) < 8:
        return float(np.median(x)), float(np.median(y))

    cx0 = np.median(x)
    cy0 = np.median(y)
    d0 = np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2)
    keep = d0 <= np.quantile(d0, 0.95)
    x = x[keep]
    y = y[keep]

    if len(x) < 8:
        return float(cx0), float(cy0)

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, bcoef, _ = sol
        cx = -0.5 * a
        cy = -0.5 * bcoef
        return float(cx), float(cy)
    except Exception:
        return float(np.median(x)), float(np.median(y))


def slice_center_from_points(points_slice, plane_center, normal):
    u, v = plane_basis(normal)
    rel = points_slice - plane_center
    x0 = rel @ u
    y0 = rel @ v
    cx, cy = fit_circle_center_2d(x0, y0)
    return plane_center + cx * u + cy * v


def slice_radius_for_draw(points_slice, plane_center, normal):
    u, v = plane_basis(normal)
    rel = points_slice - plane_center
    x0 = rel @ u
    y0 = rel @ v
    cx, cy = fit_circle_center_2d(x0, y0)
    x = x0 - cx
    y = y0 - cy
    r = np.sqrt(x * x + y * y)

    if len(r) < 20:
        return np.nan, plane_center + cx * u + cy * v

    r = r[r <= np.quantile(r, 0.98)]
    if len(r) < 10:
        return np.nan, plane_center + cx * u + cy * v

    radius_raw = float(np.quantile(r, 0.90))
    adjusted_center = plane_center + cx * u + cy * v
    return radius_raw, adjusted_center


# -------------------------
# Concave boundary perimeter
# -------------------------
def robust_slice_center_xy(x, y):
    cx = np.median(x)
    cy = np.median(y)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    keep = d <= np.quantile(d, 0.92)
    if np.sum(keep) < 20:
        return float(cx), float(cy)
    return float(np.mean(x[keep])), float(np.mean(y[keep]))


def triangle_circumradius(a, b, c):
    la = np.linalg.norm(b - c)
    lb = np.linalg.norm(a - c)
    lc = np.linalg.norm(a - b)
    s = 0.5 * (la + lb + lc)
    area_sq = max(s * (s - la) * (s - lb) * (s - lc), 1e-20)
    area = math.sqrt(area_sq)
    return (la * lb * lc) / (4.0 * area)


def ordered_boundary_perimeter(boundary_edges, pts2d):
    if not boundary_edges:
        return np.nan

    adj = {}
    for i, j in boundary_edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # choose largest loop
    visited_edges = set()
    loops = []

    for start in adj:
        for nxt in adj[start]:
            e = tuple(sorted((start, nxt)))
            if e in visited_edges:
                continue

            loop = [start, nxt]
            visited_edges.add(e)
            prev, cur = start, nxt

            while True:
                nbrs = adj.get(cur, [])
                candidates = [k for k in nbrs if k != prev]
                if not candidates:
                    break

                # choose next by turning continuity
                if len(candidates) == 1:
                    nxt2 = candidates[0]
                else:
                    v_prev = pts2d[cur] - pts2d[prev]
                    v_prev_n = v_prev / max(np.linalg.norm(v_prev), 1e-12)
                    best = None
                    best_score = -1e18
                    for cand in candidates:
                        v = pts2d[cand] - pts2d[cur]
                        v_n = v / max(np.linalg.norm(v), 1e-12)
                        score = np.dot(v_prev_n, v_n)
                        if score > best_score:
                            best_score = score
                            best = cand
                    nxt2 = best

                e2 = tuple(sorted((cur, nxt2)))
                if e2 in visited_edges:
                    if nxt2 == loop[0]:
                        loop.append(nxt2)
                    break

                loop.append(nxt2)
                visited_edges.add(e2)
                prev, cur = cur, nxt2

                if cur == loop[0]:
                    break

            if len(loop) >= 4 and loop[0] == loop[-1]:
                loops.append(loop)

    if not loops:
        return np.nan

    best_loop = max(loops, key=len)
    coords = pts2d[np.array(best_loop)]
    seg = np.diff(coords, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def concave_perimeter_2d(pts2d, alpha_mult=2.2, max_hole_edge_mult=3.0):
    pts2d = np.asarray(pts2d, float)
    if len(pts2d) < 10:
        return np.nan

    # local spacing estimate
    tree = cKDTree(pts2d)
    dists, _ = tree.query(pts2d, k=min(6, len(pts2d)))

    # skip self-distance
    local_spacing = np.median(dists[:, 1:]) if dists.shape[1] > 1 else np.median(dists)
    if not np.isfinite(local_spacing) or local_spacing <= 1e-12:
        return np.nan

    alpha_radius = alpha_mult * local_spacing
    max_edge = max_hole_edge_mult * local_spacing

    try:
        tri = Delaunay(pts2d)
    except Exception:
        return np.nan

    edge_count = {}
    for simplex in tri.simplices:
        a, b, c = simplex
        pa, pb, pc = pts2d[a], pts2d[b], pts2d[c]

        # filter extremely long/noisy triangles
        edges_len = [
            np.linalg.norm(pa - pb),
            np.linalg.norm(pb - pc),
            np.linalg.norm(pc - pa),
        ]
        if max(edges_len) > 4.0 * max_edge:
            continue

        R = triangle_circumradius(pa, pb, pc)
        if R > alpha_radius:
            continue

        edges = [(a, b), (b, c), (c, a)]
        for i, j in edges:
            e = tuple(sorted((int(i), int(j))))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return np.nan

    # remove very long boundary edges
    boundary_edges = [
        e
        for e in boundary_edges
        if np.linalg.norm(pts2d[e[0]] - pts2d[e[1]]) <= max_edge
    ]
    if not boundary_edges:
        return np.nan

    perim = ordered_boundary_perimeter(boundary_edges, pts2d)
    if np.isfinite(perim):
        return perim

    # fallback: convex perimeter
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts2d)
        hull_pts = pts2d[hull.vertices]
        closed = np.vstack([hull_pts, hull_pts[0]])
        seg = np.diff(closed, axis=0)
        return float(np.sum(np.linalg.norm(seg, axis=1)))
    except Exception:
        return np.nan


def boundary_width_from_perimeter(points_slice, plane_center, normal):
    u, v = plane_basis(normal)
    rel = points_slice - plane_center
    x0 = rel @ u
    y0 = rel @ v

    cx, cy = robust_slice_center_xy(x0, y0)
    x = x0 - cx
    y = y0 - cy

    pts2d = np.column_stack([x, y])
    if len(pts2d) < 20:
        return np.nan

    r = np.sqrt(x * x + y * y)
    keep = r <= np.quantile(r, 0.985)
    pts2d = pts2d[keep]
    if len(pts2d) < 10:
        return np.nan

    perimeter_raw = concave_perimeter_2d(
        pts2d,
        alpha_mult=ALPHA_MULT,
        max_hole_edge_mult=MAX_HOLE_EDGE_MULT,
    )
    if not np.isfinite(perimeter_raw):
        return np.nan

    width_cm = (perimeter_raw * UNIT_SCALE_TO_CM) / np.pi
    return float(width_cm)


# -------------------------
# Body interval for ring placement
# -------------------------
def radius_quantile(points_slice, center, axis, q=0.90):
    axis = axis / np.linalg.norm(axis)
    u, v = plane_basis(axis)
    t = (points_slice - center) @ axis
    t0 = float(np.median(t))
    plane_center = center + t0 * axis

    rel = points_slice - plane_center
    x = rel @ u
    y = rel @ v
    r = np.sqrt(x * x + y * y)
    if len(r) < 30:
        return np.nan
    rcap = np.quantile(r, 0.98)
    r = r[r <= rcap]
    return float(np.quantile(r, q))


def detect_body_interval(pts, center, axis, t_full_min, t_full_max):
    L_full = t_full_max - t_full_min
    half_th = 0.5 * (PROFILE_SLICE_FRAC * L_full)

    ts = np.linspace(t_full_min, t_full_max, PROFILE_SAMPLES)
    radii = np.full_like(ts, np.nan, dtype=float)

    t = (pts - center) @ axis
    for i, t0 in enumerate(ts):
        m = (t >= (t0 - half_th)) & (t <= (t0 + half_th))
        if m.sum() < 120:
            continue
        radii[i] = radius_quantile(pts[m], center, axis, q=0.90)

    valid = np.isfinite(radii)
    if valid.sum() < 10:
        return np.quantile(ts, 0.03), np.quantile(ts, 0.97)

    rid = radii.copy()

    last = None
    for i in range(len(rid)):
        if np.isfinite(rid[i]):
            last = rid[i]
        elif last is not None:
            rid[i] = last

    last = None
    for i in range(len(rid) - 1, -1, -1):
        if np.isfinite(rid[i]):
            last = rid[i]
        elif last is not None:
            rid[i] = last

    rid = moving_average_1d(rid, PROFILE_SMOOTH_WIN)

    rmax = float(np.nanmax(rid))
    thr = BODY_THRESH_FRAC * rmax
    body = rid >= thr
    body = fill_small_gaps(body, BODY_GAP_FILL)

    s, e = largest_true_segment(body)
    if s is None:
        return np.quantile(ts, 0.03), np.quantile(ts, 0.97)

    return float(ts[s]), float(ts[e - 1])


# -------------------------
# Stable full-length centerline
# -------------------------
def build_full_length_centerline(pts, center, axis, t_full_min, t_full_max):
    L_full = t_full_max - t_full_min
    half_th = 0.5 * (LENGTH_SLICE_FRAC * L_full)
    ts = np.linspace(t_full_min, t_full_max, LENGTH_SAMPLES)
    t = (pts - center) @ axis

    centers = []
    for t0 in ts:
        m = (t >= (t0 - half_th)) & (t <= (t0 + half_th))
        if m.sum() < 80:
            continue
        plane_center = center + t0 * axis
        slab = pts[m]
        c = slice_center_from_points(slab, plane_center, axis)
        centers.append(c)

    centers = np.asarray(centers, float)
    if len(centers) < 6:
        return np.vstack([center + t_full_min * axis, center + t_full_max * axis])

    centers = moving_average_vec(centers, LENGTH_CENTERLINE_SMOOTH_WIN)
    centers = resample_polyline_by_arclength(
        centers, max(LENGTH_CENTERLINE_RESAMPLE_N, len(centers))
    )
    centers = moving_average_vec(centers, LENGTH_CENTERLINE_FINAL_SMOOTH_WIN)
    return centers


def bounded_endpoint_extension(points, endpoint, tangent, local_radius_raw):
    if local_radius_raw <= 1e-8:
        return 0.0

    rel = points - endpoint
    axial = rel @ tangent
    perp = np.linalg.norm(rel - axial[:, None] * tangent[None, :], axis=1)

    mask = perp <= (1.15 * local_radius_raw)
    if np.sum(mask) < 20:
        return 0.0

    axial_m = axial[mask]
    pos = axial_m[axial_m > 0]
    if len(pos) < 5:
        return 0.0

    ext_raw = float(np.quantile(pos, 0.90))
    ext_raw = min(ext_raw, END_EXT_FRAC_OF_RADIUS * local_radius_raw)
    ext_raw = min(ext_raw, END_EXT_MAX_CM / UNIT_SCALE_TO_CM)
    return max(0.0, ext_raw)


# -------------------------
# Area / volume
# -------------------------
def compute_area_volume_from_profile(s_ring_cm, width_ring_cm):
    s_ring_cm = np.asarray(s_ring_cm, float)
    width_ring_cm = np.asarray(width_ring_cm, float)

    valid = np.isfinite(width_ring_cm)
    if valid.sum() < 2:
        return np.nan, np.nan

    s_cm = s_ring_cm[valid]
    d_cm = width_ring_cm[valid]
    r_cm = d_cm / 2.0

    ds = np.diff(s_cm)
    r_avg = 0.5 * (r_cm[:-1] + r_cm[1:])

    area = np.sum(2.0 * np.pi * r_avg * ds)
    vol = np.sum(np.pi * (r_avg**2) * ds)
    return float(area), float(vol)


def pick_ring_index_by_fraction(valid_idx, frac, total_n):
    target = frac * (total_n - 1)
    return valid_idx[np.argmin(np.abs(valid_idx.astype(float) - target))]


# -------------------------
# Visualization helpers
# -------------------------
def set_axes_equal_3d(ax, X, Y, Z, pad=0.10):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    x_min, x_max = float(np.min(X)), float(np.max(X))
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    z_min, z_max = float(np.min(Z)), float(np.max(Z))

    x_rng = x_max - x_min
    y_rng = y_max - y_min
    z_rng = z_max - z_min
    max_rng = max(x_rng, y_rng, z_rng) * (1.0 + pad)

    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0

    ax.set_xlim(x_mid - max_rng / 2, x_mid + max_rng / 2)
    ax.set_ylim(y_mid - max_rng / 2, y_mid + max_rng / 2)
    ax.set_zlim(z_mid - max_rng / 2, z_mid + max_rng / 2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def rotation_from_z_to_vec(v):
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(v, z):
        return np.eye(3)
    if np.allclose(v, -z):
        return np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=float,
        )
    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(z, v), -1.0, 1.0))
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        dtype=float,
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_torus_ring(center, normal, ring_radius, tube_frac, color_rgb):
    tube = max(ring_radius * tube_frac, 1e-6)
    torus = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=ring_radius,
        tube_radius=tube,
    )
    torus.compute_vertex_normals()
    torus.paint_uniform_color(color_rgb)
    torus.rotate(rotation_from_z_to_vec(normal), center=(0, 0, 0))
    torus.translate(center)
    return torus


def make_polyline(points, color=(1, 0, 0)):
    pts = np.asarray(points, float)
    if len(pts) < 2:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        return ls
    lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, float), (len(lines), 1))
    )
    return ls


# -------------------------
# Main per-sample processing
# -------------------------
def process_one_sample(in_path: str):
    sample_name = os.path.splitext(os.path.basename(in_path))[0]
    p_level_mg = infer_p_level_from_name(sample_name)

    pcd = o3d.io.read_point_cloud(in_path)
    if len(pcd.points) == 0:
        raise RuntimeError(f"No points found in input PLY: {in_path}")

    if REMOVE_OUTLIERS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO,
        )

    pts = np.asarray(pcd.points)

    center, coarse_axis = pca_axis(pts)
    t_all = (pts - center) @ coarse_axis
    t_full_min = float(np.quantile(t_all, 0.001))
    t_full_max = float(np.quantile(t_all, 0.999))

    # stable full-cob centerline for length
    length_cl = build_full_length_centerline(
        pts, center, coarse_axis, t_full_min, t_full_max
    )
    length_s = centerline_arclength(length_cl)

    # body region only for rings
    t_body_min, t_body_max = detect_body_interval(
        pts, center, coarse_axis, t_full_min, t_full_max
    )

    frac_min = (t_body_min - t_full_min) / max(t_full_max - t_full_min, 1e-12)
    frac_max = (t_body_max - t_full_min) / max(t_full_max - t_full_min, 1e-12)
    frac_min = np.clip(frac_min, 0.0, 1.0)
    frac_max = np.clip(frac_max, 0.0, 1.0)

    s_body_min = frac_min * length_s[-1]
    s_body_max = frac_max * length_s[-1]

    body_start = s_body_min + SKIP_END_FRAC_FOR_RINGS * (s_body_max - s_body_min)
    body_end = s_body_max - SKIP_END_FRAC_FOR_RINGS * (s_body_max - s_body_min)
    if body_end <= body_start:
        body_start, body_end = s_body_min, s_body_max

    ring_s = np.linspace(body_start, body_end, N_RINGS)
    ring_centers, idxs, _ = sample_centerline_by_arclength(length_cl, length_s, ring_s)
    ring_normals = np.array([tangent_from_centerline(length_cl, int(i)) for i in idxs])

    band_half_raw = 0.5 * (MANUAL_RING_THICKNESS_CM / UNIT_SCALE_TO_CM)

    ring_widths_cm = np.full(N_RINGS, np.nan, float)
    ring_radii_raw = np.full(N_RINGS, np.nan, float)
    ring_draw_centers = ring_centers.copy()

    for i in range(N_RINGS):
        m = points_near_plane(pts, ring_centers[i], ring_normals[i], band_half_raw)
        if int(m.sum()) < 250:
            continue

        width_i = boundary_width_from_perimeter(
            pts[m],
            plane_center=ring_centers[i],
            normal=ring_normals[i],
        )
        radius_i, draw_center_i = slice_radius_for_draw(
            pts[m],
            plane_center=ring_centers[i],
            normal=ring_normals[i],
        )

        ring_widths_cm[i] = width_i
        ring_radii_raw[i] = radius_i
        ring_draw_centers[i] = draw_center_i

    valid = np.isfinite(ring_widths_cm)
    if OUTER_SMOOTH_WIN > 1 and valid.sum() >= OUTER_SMOOTH_WIN:
        ww = ring_widths_cm.copy()
        idx = np.arange(len(ww))
        ww[~valid] = np.interp(idx[~valid], idx[valid], ww[valid])
        ww = moving_average_1d(ww, OUTER_SMOOTH_WIN)
        ring_widths_cm[valid] = ww[valid]

    # small bounded endpoint correction
    valid_draw = np.isfinite(ring_radii_raw)
    if np.any(valid_draw):
        rv = ring_radii_raw[valid_draw]
        start_r = float(np.nanmedian(rv[: min(5, len(rv))]))
        end_r = float(np.nanmedian(rv[-min(5, len(rv)) :]))
    else:
        start_r = end_r = 0.0

    start_tan = tangent_from_centerline(length_cl, 0)
    end_tan = tangent_from_centerline(length_cl, len(length_cl) - 1)

    ext_start = bounded_endpoint_extension(pts, length_cl[0], -start_tan, start_r)
    ext_end = bounded_endpoint_extension(pts, length_cl[-1], end_tan, end_r)

    length_cm = float((length_s[-1] + ext_start + ext_end) * UNIT_SCALE_TO_CM)

    surface_area_cm2, volume_cm3 = compute_area_volume_from_profile(
        ring_s * UNIT_SCALE_TO_CM, ring_widths_cm
    )

    valid_idx = np.where(np.isfinite(ring_widths_cm))[0]
    if len(valid_idx) > 0:
        i_a = pick_ring_index_by_fraction(valid_idx, BOTTOM_POS_FRAC, len(ring_widths_cm))
        i_m = pick_ring_index_by_fraction(valid_idx, MIDDLE_POS_FRAC, len(ring_widths_cm))
        i_b = pick_ring_index_by_fraction(valid_idx, TOP_POS_FRAC, len(ring_widths_cm))

        if ring_widths_cm[i_a] >= ring_widths_cm[i_b]:
            i_bottom = i_a
            i_top = i_b
        else:
            i_bottom = i_b
            i_top = i_a

        i_middle = i_m
    else:
        i_bottom = i_middle = i_top = -1

    bottom_width_cm = ring_widths_cm[i_bottom] if i_bottom >= 0 else np.nan
    middle_width_cm = ring_widths_cm[i_middle] if i_middle >= 0 else np.nan
    top_width_cm = ring_widths_cm[i_top] if i_top >= 0 else np.nan

    if SHOW_MATPLOTLIB_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(length_cl[:, 0], length_cl[:, 1], length_cl[:, 2], linewidth=2)

        vm = np.isfinite(ring_radii_raw)
        if vm.any():
            sizes = (ring_radii_raw[vm] / np.nanmax(ring_radii_raw[vm])) * 80.0 + 10.0
            ax.scatter(
                ring_draw_centers[vm, 0],
                ring_draw_centers[vm, 1],
                ring_draw_centers[vm, 2],
                s=sizes,
            )

        set_axes_equal_3d(
            ax,
            length_cl[:, 0],
            length_cl[:, 1],
            length_cl[:, 2],
            pad=0.10,
        )
        ax.set_title(f"{sample_name} - full length centerline + rings")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    if SHOW_OPEN3D:
        palette = [
            (0.0, 1.0, 0.0),
            (0.0, 0.7, 1.0),
            (1.0, 0.2, 1.0),
            (1.0, 0.8, 0.0),
            (1.0, 0.4, 0.2),
        ]

        ring_geoms = []
        for i in range(N_RINGS):
            if not np.isfinite(ring_radii_raw[i]):
                continue
            ring_geoms.append(
                make_torus_ring(
                    center=ring_draw_centers[i],
                    normal=ring_normals[i],
                    ring_radius=float(ring_radii_raw[i]),
                    tube_frac=RING_TUBE_FRAC,
                    color_rgb=palette[i % len(palette)],
                )
            )

        length_line = make_polyline(length_cl, color=(1.0, 0.0, 0.0))
        geoms = [pcd, length_line] + ring_geoms

        vis = o3d.visualization.Visualizer()
        vis.create_window(f"{sample_name} - cob rings", 1280, 720)
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = float(POINT_SIZE)
        opt.background_color = BACKGROUND
        vis.run()
        vis.destroy_window()

    return {
        "sample_name": sample_name,
        "p_level_mg": p_level_mg,
        "surface_area_cm2": surface_area_cm2,
        "volume_cm3": volume_cm3,
        "length_cm": length_cm,
        "bottom_width_cm": bottom_width_cm,
        "middle_width_cm": middle_width_cm,
        "top_width_cm": top_width_cm,
    }


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ply_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".ply")
    ]
    ply_files.sort()

    if not ply_files:
        raise RuntimeError(f"No .ply files found in folder: {INPUT_DIR}")

    rows = []
    for in_path in ply_files:
        sample_name = os.path.splitext(os.path.basename(in_path))[0]
        print(f"Processing: {sample_name}")
        try:
            rows.append(process_one_sample(in_path))
        except Exception as e:
            print(f"Failed for {sample_name}: {e}")

    if not rows:
        raise RuntimeError("All samples failed.")

    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values(
        by=["p_level_mg", "sample_name"],
        na_position="last",
    ).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "final_measurements.csv")
    final_df.to_csv(out_path, index=False)

    print("\n=== FINAL TABLE ===")
    print(final_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
