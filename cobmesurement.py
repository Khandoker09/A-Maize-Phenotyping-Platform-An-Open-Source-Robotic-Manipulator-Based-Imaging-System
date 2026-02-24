import os
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

# =======================
# CONFIG
# =======================
IN_PATH = "presentation.ply"

# IMPORTANT: set correctly for your PLY units
# dm->cm = 10, m->cm = 100
UNIT_SCALE_TO_CM = 10.0

# Rings
N_RINGS = 55
SLICE_THICKNESS_FRAC = 0.012  # thickness for picking points around each ring (coarse-axis slab)

# Outlier removal
REMOVE_OUTLIERS = True
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.5

# Body detection (rough, along coarse PCA axis)
PROFILE_SAMPLES = 240
PROFILE_SLICE_FRAC = 0.010
PROFILE_SMOOTH_WIN = 11
BODY_THRESH_FRAC = 0.18
BODY_GAP_FILL = 5

# Skip end region for rings/metrics (tips are messy)
SKIP_END_FRAC = 0.03

# Outer radius method (prevents rings going inside)
ANGLE_BINS = 240
RADIAL_CAP_Q = 0.995        # remove extreme outliers before max
OUTER_SMOOTH_WIN = 5        # smooth ring radii along arc-length (odd 3/5/7). 0 disables.

# Ring thickness (visible)
RING_TUBE_FRAC = 0.018      # tube radius = frac * ring_radius

# Curved centerline extraction (robust)
CENTERLINE_SAMPLES = 140
CENTERLINE_SLICE_FRAC = 0.010
CENTERLINE_SMOOTH_WIN = 9

# Length stabilization
CENTERLINE_RESAMPLE_N = 240      # uniform arclength samples for stable length
CENTERLINE_LENGTH_SMOOTH_WIN = 7 # light smoothing after resample (odd)

# Diameter stats region (if you want "mid" only)
AVG_MODE = "all"  # "all" or "mid"
MID_FRAC_LOW = 0.10
MID_FRAC_HIGH = 0.90

# Visualization
POINT_SIZE = 2.5
BACKGROUND = np.array([0.05, 0.05, 0.05])
SHOW_MATPLOTLIB_3D = True
SHOW_OPEN3D = True

# Save outputs
SAVE_CSV = True
# =======================


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
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


# -------------------------
# Body interval detection (coarse, along PCA axis)
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
        return np.quantile(ts, 0.02), np.quantile(ts, 0.98)

    rid = radii.copy()
    # forward fill
    last = None
    for i in range(len(rid)):
        if np.isfinite(rid[i]):
            last = rid[i]
        elif last is not None:
            rid[i] = last
    # back fill
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
        return np.quantile(ts, 0.02), np.quantile(ts, 0.98)

    return float(ts[s]), float(ts[e - 1])


# -------------------------
# Robust curved centerline (trimmed centroid per slab)
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

        # Trim radial outliers (background, stray points)
        rcap = np.quantile(r, 0.98)
        keep = r <= rcap
        slab = slab[keep]
        if len(slab) < 180:
            continue

        cl.append(slab.mean(axis=0))

    cl = np.array(cl, float)
    if len(cl) < 10:
        # fallback straight
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
# Outer radius in local ring plane (prevents inside rings)
# -------------------------
def outer_radius_angle_binned(points_slice, plane_center, normal, bins=240, cap_q=0.995):
    u, v = plane_basis(normal)
    rel = points_slice - plane_center
    x = rel @ u
    y = rel @ v
    ang = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)
    if len(r) < 80:
        return np.nan

    rcap = np.quantile(r, cap_q)
    keep = r <= rcap
    ang = ang[keep]
    r = r[keep]

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    rmax_bins = []
    for i in range(bins):
        m = (ang >= edges[i]) & (ang < edges[i + 1])
        if np.any(m):
            rmax_bins.append(np.max(r[m]))

    if len(rmax_bins) < max(20, bins * 0.15):
        return float(np.quantile(r, 0.97))

    return float(np.median(np.array(rmax_bins)))


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
# Metrics (area, volume) along arc-length profile
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

    # Lateral surface area (approx): ∫ 2πr ds
    area = np.sum(2.0 * np.pi * r_avg * ds)

    # Volume (approx): ∫ π r^2 ds
    vol = np.sum(np.pi * (r_avg ** 2) * ds)
    return float(area), float(vol)


# -------------------------
# Matplotlib equal-aspect helper
# -------------------------
def set_axes_equal_3d(ax, X, Y, Z, pad=0.10):
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
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

    ax.set_xlim(x_mid - max_rng/2, x_mid + max_rng/2)
    ax.set_ylim(y_mid - max_rng/2, y_mid + max_rng/2)
    ax.set_zlim(z_mid - max_rng/2, z_mid + max_rng/2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


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

    # Coarse axis to order slices
    center, coarse_axis = pca_axis(pts)
    t_all = (pts - center) @ coarse_axis
    t_full_min = float(np.quantile(t_all, 0.001))
    t_full_max = float(np.quantile(t_all, 0.999))

    # Body interval (coarse)
    t_body_min, t_body_max = detect_body_interval(pts, center, coarse_axis, t_full_min, t_full_max)
    L_body = t_body_max - t_body_min

    # Robust centerline, then stabilize for length
    cl_raw = build_centerline(pts, center, coarse_axis, t_body_min, t_body_max)
    cl = smooth_centerline_for_length(cl_raw)  # stable length + stable tangents
    s_cl = centerline_arclength(cl)

    # Lengths
    L_curved_raw = float(s_cl[-1])
    L_curved_cm = L_curved_raw * UNIT_SCALE_TO_CM
    L_straight_raw = float(np.linalg.norm(cl[-1] - cl[0]))
    L_straight_cm = L_straight_raw * UNIT_SCALE_TO_CM

    # Ring positions along centerline arc length (skip ends)
    s0 = s_cl[0] + SKIP_END_FRAC * (s_cl[-1] - s_cl[0])
    s1 = s_cl[-1] - SKIP_END_FRAC * (s_cl[-1] - s_cl[0])
    ring_s = np.linspace(s0, s1, N_RINGS)

    ring_centers, idxs, _ = sample_centerline_by_arclength(cl, s_cl, ring_s)
    ring_normals = np.array([tangent_from_centerline(cl, int(i)) for i in idxs])

    # Select slice points around each ring using coarse-axis slabs (fast)
    half_thick = 0.5 * (SLICE_THICKNESS_FRAC * L_body)
    t_pts = (pts - center) @ coarse_axis
    t_ring = (ring_centers - center) @ coarse_axis

    ring_radii = np.full(N_RINGS, np.nan, float)
    slice_counts = np.zeros(N_RINGS, int)

    for i in range(N_RINGS):
        m = (t_pts >= (t_ring[i] - half_thick)) & (t_pts <= (t_ring[i] + half_thick))
        n_in = int(m.sum())
        slice_counts[i] = n_in
        if n_in < 250:
            continue

        ring_radii[i] = outer_radius_angle_binned(
            pts[m],
            plane_center=ring_centers[i],
            normal=ring_normals[i],
            bins=ANGLE_BINS,
            cap_q=RADIAL_CAP_Q
        )

    # Smooth radii to remove local dips/spikes
    valid = np.isfinite(ring_radii)
    if OUTER_SMOOTH_WIN > 1 and valid.sum() >= OUTER_SMOOTH_WIN:
        rr = ring_radii.copy()
        idx = np.arange(len(rr))
        rr[~valid] = np.interp(idx[~valid], idx[valid], rr[valid])
        rr = moving_average_1d(rr, OUTER_SMOOTH_WIN)
        ring_radii[valid] = rr[valid]

    # Width stats
    widths_cm = 2.0 * ring_radii * UNIT_SCALE_TO_CM
    valid_w = np.isfinite(widths_cm)

    if valid_w.any():
        if AVG_MODE.lower() == "mid":
            wv = widths_cm[valid_w]
            lo = int(np.floor(MID_FRAC_LOW * len(wv)))
            hi = int(np.ceil(MID_FRAC_HIGH * len(wv)))
            w_use = wv[lo:hi] if hi > lo else wv
        else:
            w_use = widths_cm[valid_w]
        avg_width_cm = float(np.mean(w_use))
        max_width_cm = float(np.max(widths_cm[valid_w]))
        min_width_cm = float(np.min(widths_cm[valid_w]))
    else:
        avg_width_cm = max_width_cm = min_width_cm = float("nan")

    # Area / volume (raw -> cm²/cm³)
    area_raw, vol_raw = compute_area_volume_from_profile(ring_s, ring_radii)
    area_cm2 = area_raw * (UNIT_SCALE_TO_CM ** 2) if np.isfinite(area_raw) else float("nan")
    vol_cm3 = vol_raw * (UNIT_SCALE_TO_CM ** 3) if np.isfinite(vol_raw) else float("nan")

    # Summary table
    summary = pd.DataFrame([{
        "straight_length_cm": L_straight_cm,
        "curved_length_cm": L_curved_cm,
        "surface_area_cm2": area_cm2,
        "volume_cm3": vol_cm3,
        "avg_width_cm": avg_width_cm,
        "max_width_cm": max_width_cm,
        "min_width_cm": min_width_cm,
        "rings_total": int(N_RINGS),
        "rings_valid": int(np.isfinite(ring_radii).sum()),
        "unit_scale_to_cm": UNIT_SCALE_TO_CM,
    }])

    print("\n=== SUMMARY TABLE ===")
    print(summary.to_string(index=False))

    # Save outputs
    if SAVE_CSV:
        base = os.path.splitext(IN_PATH)[0]
        rings_df = pd.DataFrame({
            "ring_id": np.arange(1, N_RINGS + 1),
            "s_along_centerline_raw": ring_s,
            "radius_raw": ring_radii,
            "radius_cm": ring_radii * UNIT_SCALE_TO_CM,
            "width_cm": widths_cm,
            "slice_points": slice_counts
        })
        rings_df.to_csv(base + "_rings_profile.csv", index=False)
        summary.to_csv(base + "_summary_metrics.csv", index=False)
        print(f"\nSaved: {base + '_rings_profile.csv'}")
        print(f"Saved: {base + '_summary_metrics.csv'}")

    # Matplotlib 3D (equal aspect)
    if SHOW_MATPLOTLIB_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2])

        vm = np.isfinite(ring_radii)
        if vm.any():
            sizes = (ring_radii[vm] / np.nanmax(ring_radii[vm])) * 80.0 + 10.0
            ax.scatter(ring_centers[vm, 0], ring_centers[vm, 1], ring_centers[vm, 2], s=sizes)

        set_axes_equal_3d(ax, cl[:, 0], cl[:, 1], cl[:, 2], pad=0.10)
        ax.set_title("Cob centerline + ring centers (equal aspect)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.show()

    # Open3D view
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
        vis.create_window("Cob rings (curved centerline)", 1280, 720)
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = float(POINT_SIZE)
        opt.background_color = BACKGROUND
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
