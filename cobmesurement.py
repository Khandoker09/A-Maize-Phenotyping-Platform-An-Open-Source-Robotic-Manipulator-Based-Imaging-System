import os
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

# =======================
# CONFIG
# =======================
IN_PATH = "ada85s2main.ply"

# raw-unit -> cm
UNIT_SCALE_TO_CM = 5.63

# Manual reference values for this sample
MANUAL_LENGTH_CM = 17.0
MANUAL_BOTTOM_RING_CM = 14.0
MANUAL_MIDDLE_RING_CM = 13.5
MANUAL_TOP_RING_CM = 10.0

# Rings along cob
N_RINGS = 55

# Thin-band slice used for original ring profile
SLICE_THICKNESS_FRAC = 0.012

# Manual-style tape band thickness
MANUAL_RING_THICKNESS_CM = 1.0

# Outlier removal
REMOVE_OUTLIERS = True
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.5

# Body detection
PROFILE_SAMPLES = 240
PROFILE_SLICE_FRAC = 0.010
PROFILE_SMOOTH_WIN = 11
BODY_THRESH_FRAC = 0.28
BODY_GAP_FILL = 5

# Skip messy ends
SKIP_END_FRAC = 0.03

# Ring label positions
BOTTOM_POS_FRAC = 0.15
MIDDLE_POS_FRAC = 0.50
TOP_POS_FRAC = 0.85

# Outer radius method
ANGLE_BINS = 240
RADIAL_CAP_Q = 0.995
BIN_RADIUS_Q = 0.94
OUTER_SMOOTH_WIN = 7

# Ring thickness (visualization)
RING_TUBE_FRAC = 0.018

# Curved centerline extraction
CENTERLINE_SAMPLES = 90
CENTERLINE_SLICE_FRAC = 0.010
CENTERLINE_SMOOTH_WIN = 15

# Length stabilization
CENTERLINE_RESAMPLE_N = 160
CENTERLINE_LENGTH_SMOOTH_WIN = 11

# Diameter stats region
AVG_MODE = "all"   # "all" or "mid"
MID_FRAC_LOW = 0.10
MID_FRAC_HIGH = 0.90

# Visualization
POINT_SIZE = 2.5
BACKGROUND = np.array([0.05, 0.05, 0.05])
SHOW_MATPLOTLIB_3D = True
SHOW_OPEN3D = True

# Save outputs
SAVE_CSV = True


# -------------------------
# Small utilities
# -------------------------
def moving_average_1d(x, win):
    win = int(win)
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win) / win
    return np.convolve(xp, k, mode="valid")


def moving_average_vec(x, win):
    win = int(win)
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(win) / win
    y = np.empty_like(xp[pad:-pad])
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


def pct_error(pred, true):
    if not np.isfinite(pred) or not np.isfinite(true) or abs(true) < 1e-12:
        return np.nan
    return 100.0 * (pred - true) / true


def abs_pct_error(pred, true):
    pe = pct_error(pred, true)
    return np.abs(pe) if np.isfinite(pe) else np.nan


# -------------------------
# PCA axis & slicing helpers
# -------------------------
def pca_axis(points: np.ndarray):
    center = points.mean(axis=0)
    X = points - center
    C = (X.T @ X) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    return center, axis


def slice_mask(points, center, axis, t0, half_thickness):
    t = (points - center) @ axis
    return (t >= (t0 - half_thickness)) & (t <= (t0 + half_thickness))


def plane_basis(normal):
    normal = normal / np.linalg.norm(normal)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


# -------------------------
# Body interval detection
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

    for i, t0 in enumerate(ts):
        m = slice_mask(pts, center, axis, t0, half_th)
        if m.sum() < 150:
            continue
        radii[i] = radius_quantile(pts[m], center, axis, q=0.90)

    valid = np.isfinite(radii)
    if valid.sum() < 10:
        return np.quantile(ts, 0.05), np.quantile(ts, 0.95)

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
        return np.quantile(ts, 0.05), np.quantile(ts, 0.95)

    return float(ts[s]), float(ts[e - 1])


# -------------------------
# Curved centerline
# -------------------------
def build_centerline(pts, center, axis, t_body_min, t_body_max):
    L_body = t_body_max - t_body_min
    half_th = 0.5 * (CENTERLINE_SLICE_FRAC * L_body)
    ts = np.linspace(t_body_min, t_body_max, CENTERLINE_SAMPLES)

    u_ax, v_ax = plane_basis(axis)

    cl = []
    for t0 in ts:
        m = slice_mask(pts, center, axis, t0, half_th)
        if m.sum() < 250:
            continue

        slab = pts[m]
        plane_center = center + t0 * axis
        rel = slab - plane_center
        x = rel @ u_ax
        y = rel @ v_ax
        r = np.sqrt(x * x + y * y)

        rcap = np.quantile(r, 0.98)
        keep = r <= rcap
        slab = slab[keep]
        if len(slab) < 180:
            continue

        c = slab.mean(axis=0)
        d = np.linalg.norm(slab - c, axis=1)
        keep2 = d <= np.quantile(d, 0.90)
        slab2 = slab[keep2]

        if len(slab2) < 50:
            continue

        cl.append(slab2.mean(axis=0))

    cl = np.array(cl, float)
    if len(cl) < 10:
        return np.vstack([center + t_body_min * axis, center + t_body_max * axis])

    cl = moving_average_vec(cl, CENTERLINE_SMOOTH_WIN)
    return cl


def centerline_arclength(cl):
    d = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def resample_polyline_by_arclength(P, n):
    P = np.asarray(P, float)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return np.repeat(P[:1], n, axis=0)
    s_new = np.linspace(0.0, s[-1], n)
    out = np.zeros((n, 3), float)
    for k in range(3):
        out[:, k] = np.interp(s_new, s, P[:, k])
    return out


def smooth_centerline_for_length(cl):
    cl2 = resample_polyline_by_arclength(cl, max(CENTERLINE_RESAMPLE_N, len(cl)))
    cl2 = moving_average_vec(cl2, CENTERLINE_LENGTH_SMOOTH_WIN)
    return cl2


def sample_centerline_by_arclength(cl, s, s_query):
    s_query = np.clip(s_query, s[0], s[-1])
    idx = np.searchsorted(s, s_query, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    t = (s_query - s[idx]) / np.maximum(s[idx + 1] - s[idx], 1e-12)
    p = (1 - t)[:, None] * cl[idx] + t[:, None] * cl[idx + 1]
    return p, idx, t


def tangent_from_centerline(cl, i):
    if i <= 0:
        v = cl[1] - cl[0]
    elif i >= len(cl) - 1:
        v = cl[-1] - cl[-2]
    else:
        v = cl[i + 1] - cl[i - 1]
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


# -------------------------
# Ring radius helpers
# -------------------------
def robust_slice_center_xy(x, y):
    cx = np.median(x)
    cy = np.median(y)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    keep = d <= np.quantile(d, 0.90)
    if np.sum(keep) < 20:
        return float(cx), float(cy)
    return float(np.mean(x[keep])), float(np.mean(y[keep]))


def outer_radius_angle_binned(points_slice, plane_center, normal, bins=240, cap_q=0.995, bin_q=0.94):
    u, v = plane_basis(normal)

    rel = points_slice - plane_center
    x0 = rel @ u
    y0 = rel @ v

    cx, cy = robust_slice_center_xy(x0, y0)
    x = x0 - cx
    y = y0 - cy

    ang = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)

    if len(r) < 80:
        return np.nan

    rcap = np.quantile(r, cap_q)
    keep = r <= rcap
    ang = ang[keep]
    r = r[keep]

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    vals = []
    for i in range(bins):
        m = (ang >= edges[i]) & (ang < edges[i + 1])
        if np.any(m):
            vals.append(np.quantile(r[m], bin_q))

    if len(vals) < max(20, bins * 0.15):
        return float(np.quantile(r, bin_q))

    return float(np.median(np.array(vals)))


def circumference_from_band(points_band, plane_center, normal):
    r = outer_radius_angle_binned(
        points_band,
        plane_center=plane_center,
        normal=normal,
        bins=ANGLE_BINS,
        cap_q=RADIAL_CAP_Q,
        bin_q=BIN_RADIUS_Q
    )
    if not np.isfinite(r):
        return np.nan, np.nan
    diameter_cm = 2.0 * r * UNIT_SCALE_TO_CM
    circumference_cm = 2.0 * np.pi * r * UNIT_SCALE_TO_CM
    return diameter_cm, circumference_cm


# -------------------------
# Geometry drawing
# -------------------------
def rotation_from_z_to_vec(v):
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(v, z):
        return np.eye(3)
    if np.allclose(v, -z):
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=float)
    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(z, v), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def make_torus_ring(center, normal, ring_radius, tube_frac, color_rgb):
    tube = max(ring_radius * tube_frac, 1e-6)
    torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=ring_radius, tube_radius=tube)
    torus.compute_vertex_normals()
    torus.paint_uniform_color(color_rgb)
    R = rotation_from_z_to_vec(normal)
    torus.rotate(R, center=(0, 0, 0))
    torus.translate(center)
    return torus


def make_polyline(points, color=(1, 0, 0)):
    pts = np.asarray(points, float)
    lines = np.array([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, float), (len(lines), 1)))
    return ls


# -------------------------
# Metrics
# -------------------------
def compute_area_volume_from_profile(s_ring, r_ring):
    s_ring = np.asarray(s_ring, float)
    r_ring = np.asarray(r_ring, float)
    valid = np.isfinite(r_ring)
    if valid.sum() < 2:
        return np.nan, np.nan

    s = s_ring[valid]
    r = r_ring[valid]

    ds = np.diff(s)
    r_avg = 0.5 * (r[:-1] + r[1:])
    area = np.sum(2.0 * np.pi * r_avg * ds)
    vol = np.sum(np.pi * (r_avg ** 2) * ds)
    return float(area), float(vol)


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


def pick_ring_index_by_fraction(valid_idx, frac, total_n):
    target = frac * (total_n - 1)
    return valid_idx[np.argmin(np.abs(valid_idx.astype(float) - target))]


# =======================
# MAIN
# =======================
def main():
    pcd = o3d.io.read_point_cloud(IN_PATH)
    if len(pcd.points) == 0:
        raise RuntimeError("No points found in the input PLY.")

    if REMOVE_OUTLIERS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO
        )

    pts = np.asarray(pcd.points)

    center, coarse_axis = pca_axis(pts)
    t_all = (pts - center) @ coarse_axis
    t_full_min = float(np.quantile(t_all, 0.001))
    t_full_max = float(np.quantile(t_all, 0.999))

    t_body_min, t_body_max = detect_body_interval(pts, center, coarse_axis, t_full_min, t_full_max)
    L_body = t_body_max - t_body_min

    cl_raw = build_centerline(pts, center, coarse_axis, t_body_min, t_body_max)
    cl = smooth_centerline_for_length(cl_raw)
    s_cl = centerline_arclength(cl)

    L_curved_raw = float(s_cl[-1])
    L_curved_cm = L_curved_raw * UNIT_SCALE_TO_CM

    L_straight_raw = float(np.linalg.norm(cl[-1] - cl[0]))
    L_straight_cm = L_straight_raw * UNIT_SCALE_TO_CM

    s0 = s_cl[0] + SKIP_END_FRAC * (s_cl[-1] - s_cl[0])
    s1 = s_cl[-1] - SKIP_END_FRAC * (s_cl[-1] - s_cl[0])
    ring_s = np.linspace(s0, s1, N_RINGS)

    ring_centers, idxs, _ = sample_centerline_by_arclength(cl, s_cl, ring_s)
    ring_normals = np.array([tangent_from_centerline(cl, int(i)) for i in idxs])

    # thin-band ring profile
    half_thick_thin = 0.5 * (SLICE_THICKNESS_FRAC * L_body)
    t_pts = (pts - center) @ coarse_axis
    t_ring = (ring_centers - center) @ coarse_axis

    ring_radii = np.full(N_RINGS, np.nan, float)
    slice_counts_thin = np.zeros(N_RINGS, int)

    for i in range(N_RINGS):
        m = (t_pts >= (t_ring[i] - half_thick_thin)) & (t_pts <= (t_ring[i] + half_thick_thin))
        n_in = int(m.sum())
        slice_counts_thin[i] = n_in
        if n_in < 250:
            continue

        ring_radii[i] = outer_radius_angle_binned(
            pts[m],
            plane_center=ring_centers[i],
            normal=ring_normals[i],
            bins=ANGLE_BINS,
            cap_q=RADIAL_CAP_Q,
            bin_q=BIN_RADIUS_Q
        )

    valid = np.isfinite(ring_radii)
    if OUTER_SMOOTH_WIN > 1 and valid.sum() >= OUTER_SMOOTH_WIN:
        rr = ring_radii.copy()
        idx = np.arange(len(rr))
        rr[~valid] = np.interp(idx[~valid], idx[valid], rr[valid])
        rr = moving_average_1d(rr, OUTER_SMOOTH_WIN)
        ring_radii[valid] = rr[valid]

    diameters_cm = 2.0 * ring_radii * UNIT_SCALE_TO_CM
    circumferences_cm = 2.0 * np.pi * ring_radii * UNIT_SCALE_TO_CM

    valid_w = np.isfinite(diameters_cm)
    if valid_w.any():
        if AVG_MODE.lower() == "mid":
            wv = diameters_cm[valid_w]
            lo = int(np.floor(MID_FRAC_LOW * len(wv)))
            hi = int(np.ceil(MID_FRAC_HIGH * len(wv)))
            w_use = wv[lo:hi] if hi > lo else wv
        else:
            w_use = diameters_cm[valid_w]

        avg_width_cm = float(np.mean(w_use))
        max_width_cm = float(np.max(diameters_cm[valid_w]))
        min_width_cm = float(np.min(diameters_cm[valid_w]))
    else:
        avg_width_cm = max_width_cm = min_width_cm = float("nan")

    area_raw, vol_raw = compute_area_volume_from_profile(ring_s, ring_radii)
    area_cm2 = area_raw * (UNIT_SCALE_TO_CM ** 2) if np.isfinite(area_raw) else float("nan")
    vol_cm3 = vol_raw * (UNIT_SCALE_TO_CM ** 3) if np.isfinite(vol_raw) else float("nan")

    # select named ring positions from thin-band valid rings
    valid_idx = np.where(np.isfinite(circumferences_cm))[0]
    if len(valid_idx) > 0:
        i_a = pick_ring_index_by_fraction(valid_idx, BOTTOM_POS_FRAC, len(circumferences_cm))
        i_m = pick_ring_index_by_fraction(valid_idx, MIDDLE_POS_FRAC, len(circumferences_cm))
        i_b = pick_ring_index_by_fraction(valid_idx, TOP_POS_FRAC, len(circumferences_cm))

        if diameters_cm[i_a] >= diameters_cm[i_b]:
            i_bottom = i_a
            i_top = i_b
        else:
            i_bottom = i_b
            i_top = i_a

        i_middle = i_m
    else:
        i_bottom = i_middle = i_top = -1

    # thin-band named outputs
    if i_bottom >= 0:
        bottom_circ_cm = float(circumferences_cm[i_bottom])
        middle_circ_cm = float(circumferences_cm[i_middle])
        top_circ_cm = float(circumferences_cm[i_top])

        bottom_diam_cm = float(diameters_cm[i_bottom])
        middle_diam_cm = float(diameters_cm[i_middle])
        top_diam_cm = float(diameters_cm[i_top])
    else:
        bottom_circ_cm = middle_circ_cm = top_circ_cm = float("nan")
        bottom_diam_cm = middle_diam_cm = top_diam_cm = float("nan")

    # 1 cm manual-style band outputs
    manual_half_thick_raw = 0.5 * (MANUAL_RING_THICKNESS_CM / UNIT_SCALE_TO_CM)

    def manual_band_measure(i_ring):
        if i_ring < 0:
            return np.nan, np.nan, 0
        m = (t_pts >= (t_ring[i_ring] - manual_half_thick_raw)) & (t_pts <= (t_ring[i_ring] + manual_half_thick_raw))
        n_in = int(m.sum())
        if n_in < 250:
            return np.nan, np.nan, n_in
        dcm, ccm = circumference_from_band(
            pts[m],
            plane_center=ring_centers[i_ring],
            normal=ring_normals[i_ring]
        )
        return dcm, ccm, n_in

    bottom_diam_cm_1cm, bottom_circ_cm_1cm, bottom_pts_1cm = manual_band_measure(i_bottom)
    middle_diam_cm_1cm, middle_circ_cm_1cm, middle_pts_1cm = manual_band_measure(i_middle)
    top_diam_cm_1cm, top_circ_cm_1cm, top_pts_1cm = manual_band_measure(i_top)

    # errors
    length_err_pct = pct_error(L_curved_cm, MANUAL_LENGTH_CM)
    length_abs_err_pct = abs_pct_error(L_curved_cm, MANUAL_LENGTH_CM)

    bottom_err_pct_thin = pct_error(bottom_circ_cm, MANUAL_BOTTOM_RING_CM)
    middle_err_pct_thin = pct_error(middle_circ_cm, MANUAL_MIDDLE_RING_CM)
    top_err_pct_thin = pct_error(top_circ_cm, MANUAL_TOP_RING_CM)

    bottom_abs_err_pct_thin = abs_pct_error(bottom_circ_cm, MANUAL_BOTTOM_RING_CM)
    middle_abs_err_pct_thin = abs_pct_error(middle_circ_cm, MANUAL_MIDDLE_RING_CM)
    top_abs_err_pct_thin = abs_pct_error(top_circ_cm, MANUAL_TOP_RING_CM)

    bottom_err_pct_1cm = pct_error(bottom_circ_cm_1cm, MANUAL_BOTTOM_RING_CM)
    middle_err_pct_1cm = pct_error(middle_circ_cm_1cm, MANUAL_MIDDLE_RING_CM)
    top_err_pct_1cm = pct_error(top_circ_cm_1cm, MANUAL_TOP_RING_CM)

    bottom_abs_err_pct_1cm = abs_pct_error(bottom_circ_cm_1cm, MANUAL_BOTTOM_RING_CM)
    middle_abs_err_pct_1cm = abs_pct_error(middle_circ_cm_1cm, MANUAL_MIDDLE_RING_CM)
    top_abs_err_pct_1cm = abs_pct_error(top_circ_cm_1cm, MANUAL_TOP_RING_CM)

    ring_mape_thin = np.nanmean([bottom_abs_err_pct_thin, middle_abs_err_pct_thin, top_abs_err_pct_thin])
    ring_mape_1cm = np.nanmean([bottom_abs_err_pct_1cm, middle_abs_err_pct_1cm, top_abs_err_pct_1cm])
    overall_mape_thin = np.nanmean([length_abs_err_pct, bottom_abs_err_pct_thin, middle_abs_err_pct_thin, top_abs_err_pct_thin])
    overall_mape_1cm = np.nanmean([length_abs_err_pct, bottom_abs_err_pct_1cm, middle_abs_err_pct_1cm, top_abs_err_pct_1cm])

    summary = pd.DataFrame([{
        "straight_length_cm": L_straight_cm,
        "curved_length_cm": L_curved_cm,
        "surface_area_cm2": area_cm2,
        "volume_cm3": vol_cm3,

        "avg_diameter_cm_thin": avg_width_cm,
        "max_diameter_cm_thin": max_width_cm,
        "min_diameter_cm_thin": min_width_cm,

        "bottom_circumference_cm_thin": bottom_circ_cm,
        "middle_circumference_cm_thin": middle_circ_cm,
        "top_circumference_cm_thin": top_circ_cm,

        "bottom_diameter_cm_thin": bottom_diam_cm,
        "middle_diameter_cm_thin": middle_diam_cm,
        "top_diameter_cm_thin": top_diam_cm,

        "bottom_circumference_cm_1cm": bottom_circ_cm_1cm,
        "middle_circumference_cm_1cm": middle_circ_cm_1cm,
        "top_circumference_cm_1cm": top_circ_cm_1cm,

        "bottom_diameter_cm_1cm": bottom_diam_cm_1cm,
        "middle_diameter_cm_1cm": middle_diam_cm_1cm,
        "top_diameter_cm_1cm": top_diam_cm_1cm,

        "manual_length_cm": MANUAL_LENGTH_CM,
        "manual_bottom_ring_cm": MANUAL_BOTTOM_RING_CM,
        "manual_middle_ring_cm": MANUAL_MIDDLE_RING_CM,
        "manual_top_ring_cm": MANUAL_TOP_RING_CM,

        "length_error_pct": length_err_pct,
        "length_abs_error_pct": length_abs_err_pct,

        "bottom_error_pct_thin": bottom_err_pct_thin,
        "middle_error_pct_thin": middle_err_pct_thin,
        "top_error_pct_thin": top_err_pct_thin,

        "bottom_abs_error_pct_thin": bottom_abs_err_pct_thin,
        "middle_abs_error_pct_thin": middle_abs_err_pct_thin,
        "top_abs_error_pct_thin": top_abs_err_pct_thin,

        "bottom_error_pct_1cm": bottom_err_pct_1cm,
        "middle_error_pct_1cm": middle_err_pct_1cm,
        "top_error_pct_1cm": top_err_pct_1cm,

        "bottom_abs_error_pct_1cm": bottom_abs_err_pct_1cm,
        "middle_abs_error_pct_1cm": middle_abs_err_pct_1cm,
        "top_abs_error_pct_1cm": top_abs_err_pct_1cm,

        "ring_mape_thin": ring_mape_thin,
        "ring_mape_1cm": ring_mape_1cm,
        "overall_mape_thin": overall_mape_thin,
        "overall_mape_1cm": overall_mape_1cm,

        "bottom_ring_id": int(i_bottom + 1) if i_bottom >= 0 else np.nan,
        "middle_ring_id": int(i_middle + 1) if i_middle >= 0 else np.nan,
        "top_ring_id": int(i_top + 1) if i_top >= 0 else np.nan,

        "bottom_band_points_1cm": bottom_pts_1cm,
        "middle_band_points_1cm": middle_pts_1cm,
        "top_band_points_1cm": top_pts_1cm,

        "rings_total": int(N_RINGS),
        "rings_valid": int(np.isfinite(ring_radii).sum()),
        "unit_scale_to_cm": UNIT_SCALE_TO_CM,
        "manual_ring_thickness_cm": MANUAL_RING_THICKNESS_CM,
    }])

    comparison_rows = [
        {
            "trait": "length",
            "manual_cm": MANUAL_LENGTH_CM,
            "thin_band_cm": np.nan,
            "band_1cm_cm": L_curved_cm,
            "thin_error_pct": np.nan,
            "band_1cm_error_pct": length_err_pct,
            "thin_abs_error_pct": np.nan,
            "band_1cm_abs_error_pct": length_abs_err_pct,
        },
        {
            "trait": "bottom_ring",
            "manual_cm": MANUAL_BOTTOM_RING_CM,
            "thin_band_cm": bottom_circ_cm,
            "band_1cm_cm": bottom_circ_cm_1cm,
            "thin_error_pct": bottom_err_pct_thin,
            "band_1cm_error_pct": bottom_err_pct_1cm,
            "thin_abs_error_pct": bottom_abs_err_pct_thin,
            "band_1cm_abs_error_pct": bottom_abs_err_pct_1cm,
        },
        {
            "trait": "middle_ring",
            "manual_cm": MANUAL_MIDDLE_RING_CM,
            "thin_band_cm": middle_circ_cm,
            "band_1cm_cm": middle_circ_cm_1cm,
            "thin_error_pct": middle_err_pct_thin,
            "band_1cm_error_pct": middle_err_pct_1cm,
            "thin_abs_error_pct": middle_abs_err_pct_thin,
            "band_1cm_abs_error_pct": middle_abs_err_pct_1cm,
        },
        {
            "trait": "top_ring",
            "manual_cm": MANUAL_TOP_RING_CM,
            "thin_band_cm": top_circ_cm,
            "band_1cm_cm": top_circ_cm_1cm,
            "thin_error_pct": top_err_pct_thin,
            "band_1cm_error_pct": top_err_pct_1cm,
            "thin_abs_error_pct": top_abs_err_pct_thin,
            "band_1cm_abs_error_pct": top_abs_err_pct_1cm,
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    print("\n=== SUMMARY TABLE ===")
    print(summary.to_string(index=False))

    print("\n=== MANUAL COMPARISON GUIDE ===")
    print(f"Cob length: {L_curved_cm:.2f} cm")
    print("\nThin-band ring circumferences:")
    print(f"  Bottom: {bottom_circ_cm:.2f} cm")
    print(f"  Middle: {middle_circ_cm:.2f} cm")
    print(f"  Top   : {top_circ_cm:.2f} cm")
    print("\n1 cm band ring circumferences:")
    print(f"  Bottom: {bottom_circ_cm_1cm:.2f} cm")
    print(f"  Middle: {middle_circ_cm_1cm:.2f} cm")
    print(f"  Top   : {top_circ_cm_1cm:.2f} cm")

    print("\n=== ERROR SUMMARY ===")
    print(f"Length abs error: {length_abs_err_pct:.2f}%")
    print(f"Thin-band ring MAPE: {ring_mape_thin:.2f}%")
    print(f"1 cm band ring MAPE: {ring_mape_1cm:.2f}%")
    print(f"Overall MAPE (thin): {overall_mape_thin:.2f}%")
    print(f"Overall MAPE (1 cm): {overall_mape_1cm:.2f}%")

    if SAVE_CSV:
        base = os.path.splitext(IN_PATH)[0]

        rings_df = pd.DataFrame({
            "ring_id": np.arange(1, N_RINGS + 1),
            "s_along_centerline_raw": ring_s,
            "radius_raw_thin": ring_radii,
            "radius_cm_thin": ring_radii * UNIT_SCALE_TO_CM,
            "diameter_cm_thin": diameters_cm,
            "circumference_cm_thin": circumferences_cm,
            "slice_points_thin": slice_counts_thin
        })

        rings_df.to_csv(base + "_rings_profile.csv", index=False)
        summary.to_csv(base + "_summary_metrics.csv", index=False)
        comparison_df.to_csv(base + "_manual_comparison.csv", index=False)

        print(f"\nSaved: {base + '_rings_profile.csv'}")
        print(f"Saved: {base + '_summary_metrics.csv'}")
        print(f"Saved: {base + '_manual_comparison.csv'}")

    if SHOW_MATPLOTLIB_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2])

        vm = np.isfinite(ring_radii)
        if vm.any():
            sizes = (ring_radii[vm] / np.nanmax(ring_radii[vm])) * 80.0 + 10.0
            ax.scatter(ring_centers[vm, 0], ring_centers[vm, 1], ring_centers[vm, 2], s=sizes)

        set_axes_equal_3d(ax, cl[:, 0], cl[:, 1], cl[:, 2], pad=0.10)
        ax.set_title("Cob centerline + ring centers")
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
            if not np.isfinite(ring_radii[i]):
                continue
            col = palette[i % len(palette)]
            ring_geoms.append(
                make_torus_ring(
                    center=ring_centers[i],
                    normal=ring_normals[i],
                    ring_radius=float(ring_radii[i]),
                    tube_frac=RING_TUBE_FRAC,
                    color_rgb=col
                )
            )

        length_line = make_polyline(cl, color=(1.0, 0.0, 0.0))

        geoms = [pcd, length_line] + ring_geoms
        vis = o3d.visualization.Visualizer()
        vis.create_window("Cob rings (thin-band profile)", 1280, 720)
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = float(POINT_SIZE)
        opt.background_color = BACKGROUND
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
