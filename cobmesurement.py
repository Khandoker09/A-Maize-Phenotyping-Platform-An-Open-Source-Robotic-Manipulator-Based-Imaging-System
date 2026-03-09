import os
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

# =======================
# CONFIG
# =======================
INPUT_DIR = r"maindata"
OUTPUT_DIR = r"results"

UNIT_SCALE_TO_CM = 5.63

# manual tape width = 1 cm
SLICE_THICKNESS_CM = 1.2

# width tuning
WIDTH_OUTER_Q = 0.985
WIDTH_CAP_QUANTILE = 0.998
WIDTH_CORRECTION_FACTOR = 1.08

REMOVE_OUTLIERS = True
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.2

# ---------- LENGTH / CENTERLINE ----------
LENGTH_SAMPLES = 180
LENGTH_SLICE_FRAC = 0.008
LENGTH_CENTERLINE_SMOOTH_WIN = 5
LENGTH_CENTERLINE_RESAMPLE_N = 320
LENGTH_CENTERLINE_FINAL_SMOOTH_WIN = 3

AXIS_REFINE_ITERS = 3
MIN_POINTS_PER_SLICE_MIDDLE = 80
MIN_POINTS_PER_SLICE_ENDS = 25
TIP_FRAC = 0.006

# ---------- BODY / PROFILE ----------
PROFILE_SAMPLES = 260
PROFILE_SLICE_FRAC = 0.009
PROFILE_SMOOTH_WIN = 9
BODY_THRESH_FRAC = 0.22
BODY_GAP_FILL = 5

# ---------- WIDTH ----------
WIDTH_MIN_POINTS = 120
WIDTH_POLAR_BINS = 180
WIDTH_POLAR_SMOOTH_WIN = 9

# manual-like positions inside body
END_MEASURE_FRAC = 0.15
MID_MEASURE_FRAC = 0.50

# ---------- VIS ----------
N_VIS_RINGS = 45
VIS_SKIP_END_FRAC = 0.06
VIS_WIDTH_SMOOTH_WIN = 7
VIS_MIN_WIDTH_CM = 1.2
VIS_CIRCLE_POINTS = 120

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
        return x.copy()
    if win % 2 == 0:
        win += 1
    if len(x) < win:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win) / win
    return np.convolve(xp, k, mode="valid")


def moving_average_vec(x, win):
    x = np.asarray(x, float)
    win = int(win)
    if win <= 1 or len(x) < 2:
        return x.copy()
    if win % 2 == 0:
        win += 1
    if len(x) < win:
        return x.copy()
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


# -------------------------
# Slice center
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


# -------------------------
# Width from polar envelope
# -------------------------
def width_from_polar_envelope(points_slice, plane_center, normal, unit_scale_to_cm,
                              n_bins=180, outer_q=0.985, smooth_win=9):
    u, v = plane_basis(normal)
    rel = points_slice - plane_center
    x0 = rel @ u
    y0 = rel @ v

    cx, cy = fit_circle_center_2d(x0, y0)
    x = x0 - cx
    y = y0 - cy

    r = np.sqrt(x * x + y * y)
    if len(r) < WIDTH_MIN_POINTS:
        return np.nan

    theta = np.arctan2(y, x)
    theta[theta < 0] += 2 * np.pi

    edges = np.linspace(0.0, 2 * np.pi, n_bins + 1)
    r_env = np.full(n_bins, np.nan, float)

    for i in range(n_bins):
        if i < n_bins - 1:
            m = (theta >= edges[i]) & (theta < edges[i + 1])
        else:
            m = (theta >= edges[i]) & (theta <= edges[i + 1])

        if np.sum(m) < 3:
            continue

        rr = r[m]
        rr = rr[rr <= np.quantile(rr, WIDTH_CAP_QUANTILE)]
        if len(rr) < 2:
            continue

        r_env[i] = np.quantile(rr, outer_q)

    valid = np.isfinite(r_env)
    if valid.sum() < max(30, n_bins // 3):
        return np.nan

    idx = np.arange(n_bins)
    r_env[~valid] = np.interp(idx[~valid], idx[valid], r_env[valid])
    r_env = moving_average_1d(r_env, smooth_win)

    ang = 0.5 * (edges[:-1] + edges[1:])
    px = r_env * np.cos(ang)
    py = r_env * np.sin(ang)
    poly = np.column_stack([px, py])

    poly_closed = np.vstack([poly, poly[0]])
    seg = np.diff(poly_closed, axis=0)
    perimeter_raw = np.sum(np.linalg.norm(seg, axis=1))

    width_cm = (perimeter_raw * unit_scale_to_cm) / np.pi
    return float(width_cm)


def raw_radius_from_width_cm(width_cm):
    if not np.isfinite(width_cm):
        return np.nan
    return 0.5 * (width_cm / UNIT_SCALE_TO_CM)


def slab_points(points, plane_point, plane_normal, thickness_cm):
    half_th_raw = 0.5 * (thickness_cm / UNIT_SCALE_TO_CM)
    d = (points - plane_point) @ plane_normal
    m = np.abs(d) <= half_th_raw
    return points[m]


# -------------------------
# Body interval
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
        if m.sum() < 80:
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
# Length estimation
# -------------------------
def robust_tip_center(points, center, axis, side="min", frac=TIP_FRAC):
    t = (points - center) @ axis
    if side == "min":
        thr = np.quantile(t, frac)
        sel = points[t <= thr]
    else:
        thr = np.quantile(t, 1.0 - frac)
        sel = points[t >= thr]

    if len(sel) == 0:
        idx = np.argmin(t) if side == "min" else np.argmax(t)
        return points[idx]

    return np.median(sel, axis=0)


def min_points_for_slice(alpha):
    dist_to_end = min(alpha, 1.0 - alpha)
    if dist_to_end < 0.08:
        return MIN_POINTS_PER_SLICE_ENDS
    return MIN_POINTS_PER_SLICE_MIDDLE


def build_full_length_centerline(pts, center, axis, t_full_min, t_full_max):
    L_full = t_full_max - t_full_min
    half_th = 0.5 * (LENGTH_SLICE_FRAC * L_full)
    ts = np.linspace(t_full_min, t_full_max, LENGTH_SAMPLES)
    t = (pts - center) @ axis

    centers = []

    tip_min = robust_tip_center(pts, center, axis, side="min", frac=TIP_FRAC)
    tip_max = robust_tip_center(pts, center, axis, side="max", frac=TIP_FRAC)
    centers.append(tip_min)

    for i, t0 in enumerate(ts):
        alpha = i / max(len(ts) - 1, 1)
        min_pts = min_points_for_slice(alpha)

        m = (t >= (t0 - half_th)) & (t <= (t0 + half_th))
        if m.sum() < min_pts:
            continue

        plane_center = center + t0 * axis
        slab = pts[m]
        c = slice_center_from_points(slab, plane_center, axis)
        centers.append(c)

    centers.append(tip_max)
    centers = np.asarray(centers, float)

    if len(centers) < 6:
        return np.vstack([tip_min, tip_max])

    tc = (centers - center) @ axis
    order = np.argsort(tc)
    centers = centers[order]

    keep = [0]
    for i in range(1, len(centers)):
        if np.linalg.norm(centers[i] - centers[keep[-1]]) > 1e-5:
            keep.append(i)
    centers = centers[keep]

    if len(centers) < 4:
        return np.vstack([tip_min, tip_max])

    centers = moving_average_vec(centers, LENGTH_CENTERLINE_SMOOTH_WIN)
    centers = resample_polyline_by_arclength(
        centers,
        max(LENGTH_CENTERLINE_RESAMPLE_N, len(centers))
    )
    centers = moving_average_vec(centers, LENGTH_CENTERLINE_FINAL_SMOOTH_WIN)

    centers[0] = tip_min
    centers[-1] = tip_max

    return centers


def refine_axis_from_centerline(cl, fallback_axis):
    if len(cl) < 2:
        return fallback_axis / np.linalg.norm(fallback_axis)
    v = cl[-1] - cl[0]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return fallback_axis / np.linalg.norm(fallback_axis)
    return v / n


def estimate_length_centerline_and_axis(pts):
    center, axis = pca_axis(pts)
    t_all = (pts - center) @ axis
    t_full_min = float(np.quantile(t_all, 0.001))
    t_full_max = float(np.quantile(t_all, 0.999))

    cl = None
    for _ in range(AXIS_REFINE_ITERS):
        cl = build_full_length_centerline(pts, center, axis, t_full_min, t_full_max)
        axis = refine_axis_from_centerline(cl, axis)

        if np.dot(axis, cl[-1] - cl[0]) < 0:
            axis = -axis

        t_all = (pts - center) @ axis
        t_full_min = float(np.quantile(t_all, 0.001))
        t_full_max = float(np.quantile(t_all, 0.999))

    s = centerline_arclength(cl)
    length_cm = float(s[-1]) * UNIT_SCALE_TO_CM
    return center, axis, cl, s, t_full_min, t_full_max, length_cm


# -------------------------
# Frustum metrics
# -------------------------
def frustum_volume_cm3(h, d1, d2):
    r1 = d1 / 2.0
    r2 = d2 / 2.0
    return (np.pi * h / 3.0) * (r1 * r1 + r1 * r2 + r2 * r2)


def frustum_lateral_area_cm2(h, d1, d2):
    r1 = d1 / 2.0
    r2 = d2 / 2.0
    s = np.sqrt(h * h + (r1 - r2) ** 2)
    return np.pi * (r1 + r2) * s


def compute_manual_style_volume_area(length_cm, bottom_w, middle_w, top_w):
    vals = np.array([length_cm, bottom_w, middle_w, top_w], float)
    if not np.all(np.isfinite(vals)):
        return np.nan, np.nan

    h = length_cm / 2.0
    vol = frustum_volume_cm3(h, bottom_w, middle_w) + frustum_volume_cm3(h, middle_w, top_w)
    area = frustum_lateral_area_cm2(h, bottom_w, middle_w) + frustum_lateral_area_cm2(h, middle_w, top_w)
    return float(area), float(vol)


# -------------------------
# Width measurement
# -------------------------
def measure_width_at_t(pts, cl, s, center, axis, t_target):
    t_cl = (cl - center) @ axis
    order = np.argsort(t_cl)
    t_sorted = t_cl[order]
    s_sorted = s[order]
    s_target = np.interp(t_target, t_sorted, s_sorted)

    center_pt, _, _ = sample_centerline_by_arclength(cl, s, [s_target])
    center_pt = center_pt[0]

    normal = axis.copy()
    slab = slab_points(pts, center_pt, normal, SLICE_THICKNESS_CM)
    if len(slab) < WIDTH_MIN_POINTS:
        return np.nan

    return width_from_polar_envelope(
        slab,
        center_pt,
        normal,
        unit_scale_to_cm=UNIT_SCALE_TO_CM,
        n_bins=WIDTH_POLAR_BINS,
        outer_q=WIDTH_OUTER_Q,
        smooth_win=WIDTH_POLAR_SMOOTH_WIN
    )


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


def make_circle_polyline(center, normal, radius, color=(0, 1, 0), n=120):
    center = np.asarray(center, float)
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)

    u, v = plane_basis(normal)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.array(
        [center + radius * np.cos(a) * u + radius * np.sin(a) * v for a in ang],
        dtype=float
    )

    lines = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, float), (len(lines), 1))
    )
    return ls


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
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, float), (len(lines), 1)))
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
            std_ratio=OUTLIER_STD_RATIO
        )

    pts = np.asarray(pcd.points)
    if len(pts) < 300:
        raise RuntimeError(f"Too few points after cleaning: {len(pts)}")

    center, refined_axis, length_cl, length_s, t_full_min, t_full_max, length_cm = \
        estimate_length_centerline_and_axis(pts)

    if length_cm < 6.0:
        raise RuntimeError(f"Reconstruction likely failed: very short length ({length_cm:.2f} cm)")

    t_body_min, t_body_max = detect_body_interval(
        pts, center, refined_axis, t_full_min, t_full_max
    )

    t_end_a = t_body_min + END_MEASURE_FRAC * (t_body_max - t_body_min)
    t_mid = t_body_min + MID_MEASURE_FRAC * (t_body_max - t_body_min)
    t_end_b = t_body_min + (1.0 - END_MEASURE_FRAC) * (t_body_max - t_body_min)

    end_a_width_cm = measure_width_at_t(pts, length_cl, length_s, center, refined_axis, t_end_a)
    middle_width_cm = measure_width_at_t(pts, length_cl, length_s, center, refined_axis, t_mid)
    end_b_width_cm = measure_width_at_t(pts, length_cl, length_s, center, refined_axis, t_end_b)

    if np.isfinite(end_a_width_cm) and np.isfinite(end_b_width_cm):
        bottom_width_cm = max(end_a_width_cm, end_b_width_cm)
        top_width_cm = min(end_a_width_cm, end_b_width_cm)
    else:
        bottom_width_cm = np.nan
        top_width_cm = np.nan

    # empirical width calibration
    if np.isfinite(bottom_width_cm):
        bottom_width_cm *= WIDTH_CORRECTION_FACTOR
    if np.isfinite(middle_width_cm):
        middle_width_cm *= WIDTH_CORRECTION_FACTOR
    if np.isfinite(top_width_cm):
        top_width_cm *= WIDTH_CORRECTION_FACTOR

    surface_area_cm2, volume_cm3 = compute_manual_style_volume_area(
        length_cm, bottom_width_cm, middle_width_cm, top_width_cm
    )

    # stable rings for visualization only
    vis_t = np.linspace(
        t_body_min + VIS_SKIP_END_FRAC * (t_body_max - t_body_min),
        t_body_max - VIS_SKIP_END_FRAC * (t_body_max - t_body_min),
        N_VIS_RINGS
    )

    t_cl = (length_cl - center) @ refined_axis
    order = np.argsort(t_cl)
    t_sorted = t_cl[order]
    s_sorted = length_s[order]
    vis_s = np.interp(vis_t, t_sorted, s_sorted)
    vis_centers, _, _ = sample_centerline_by_arclength(length_cl, length_s, vis_s)

    vis_widths_cm = []
    for t0 in vis_t:
        w = measure_width_at_t(pts, length_cl, length_s, center, refined_axis, t0)
        if np.isfinite(w):
            w *= WIDTH_CORRECTION_FACTOR
        vis_widths_cm.append(w)

    vis_widths_cm = np.asarray(vis_widths_cm, float)
    valid = np.isfinite(vis_widths_cm)
    if valid.sum() >= 3:
        idx = np.arange(len(vis_widths_cm))
        vis_widths_cm[~valid] = np.interp(idx[~valid], idx[valid], vis_widths_cm[valid])
        vis_widths_cm = moving_average_1d(vis_widths_cm, VIS_WIDTH_SMOOTH_WIN)

    vis_widths_cm = np.maximum(vis_widths_cm, VIS_MIN_WIDTH_CM)
    vis_radii_raw = np.array([raw_radius_from_width_cm(w) for w in vis_widths_cm])

    if SHOW_MATPLOTLIB_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(length_cl[:, 0], length_cl[:, 1], length_cl[:, 2], linewidth=2)

        vm = np.isfinite(vis_radii_raw)
        if vm.any():
            sizes = (vis_radii_raw[vm] / np.nanmax(vis_radii_raw[vm])) * 80.0 + 10.0
            ax.scatter(
                vis_centers[vm, 0],
                vis_centers[vm, 1],
                vis_centers[vm, 2],
                s=sizes
            )

        set_axes_equal_3d(ax, length_cl[:, 0], length_cl[:, 1], length_cl[:, 2], pad=0.10)
        ax.set_title(f"{sample_name} - centerline + stable 1 cm rings")
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
        for i in range(N_VIS_RINGS):
            if not np.isfinite(vis_radii_raw[i]):
                continue
            ring_geoms.append(
                make_circle_polyline(
                    center=vis_centers[i],
                    normal=refined_axis,
                    radius=float(vis_radii_raw[i]),
                    color=palette[i % len(palette)],
                    n=VIS_CIRCLE_POINTS
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
        na_position="last"
    ).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "final_measurements.csv")
    if SAVE_CSV:
        final_df.to_csv(out_path, index=False)

    print("\n=== FINAL TABLE ===")
    print(final_df.to_string(index=False))
    if SAVE_CSV:
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
