import os
import numpy as np
import pandas as pd
import open3d as o3d

# =======================
# CONFIG
# =======================
IN_PATH = "presentation.ply"
UNIT_SCALE_TO_CM = 10.0  # dm->cm

# Rings
N_RINGS = 55
SLICE_THICKNESS_FRAC = 0.012

# Outlier removal
REMOVE_OUTLIERS = True
OUTLIER_NB_NEIGHBORS = 30
OUTLIER_STD_RATIO = 2.5

# Body detection (rough, on coarse axis)
PROFILE_SAMPLES = 240
PROFILE_SLICE_FRAC = 0.010
PROFILE_SMOOTH_WIN = 11
BODY_THRESH_FRAC = 0.18
BODY_GAP_FILL = 5

# Ring placement margins (avoid messy tips)
SKIP_END_FRAC = 0.03

# Length touching (along curved centerline)
LENGTH_TOUCH_MARGIN_FRAC = 0.02

# Outer radius method (prevents rings going inside)
ANGLE_BINS = 240
RADIAL_CAP_Q = 0.995
OUTER_SMOOTH_WIN = 5  # smooth radii along the cob (odd: 3/5/7). set 0 to disable.

# Ring thickness
RING_TUBE_FRAC = 0.018  # tube radius = this * ring_radius

# Curved centerline
CENTERLINE_SAMPLES = 120          # number of slabs for centerline extraction
CENTERLINE_SLICE_FRAC = 0.010     # slab thickness as fraction of body length
CENTERLINE_SMOOTH_WIN = 9         # smooth centerline points (odd: 7/9/11)

# Average diameter
AVG_MODE = "mid"
MID_FRAC_LOW = 0.10
MID_FRAC_HIGH = 0.90

# Visualization
POINT_SIZE = 2.5
BACKGROUND = np.array([0.05, 0.05, 0.05])

SAVE_CSV = True
# =======================


# -------------------------
# Helpers
# -------------------------
def moving_average(x, win):
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
    if len(r) < 10:
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
        mask = slice_mask(pts, center, axis, t0, half_th)
        if mask.sum() < 100:
            continue
        radii[i] = radius_quantile(pts[mask], center, axis, q=0.90)

    valid = np.isfinite(radii)
    if valid.sum() < 10:
        return np.quantile(ts, 0.02), np.quantile(ts, 0.98)

    # fill nans
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
        return np.quantile(ts, 0.02), np.quantile(ts, 0.98)

    return float(ts[s]), float(ts[e - 1])


# -------------------------
# Curved centerline
# -------------------------
def build_centerline(pts, center, axis, t_body_min, t_body_max):
    L_body = t_body_max - t_body_min
    half_th = 0.5 * (CENTERLINE_SLICE_FRAC * L_body)

    ts = np.linspace(t_body_min, t_body_max, CENTERLINE_SAMPLES)
    cl = []
    valid_t = []

    for t0 in ts:
        m = slice_mask(pts, center, axis, t0, half_th)
        if m.sum() < 200:
            continue
        cl.append(pts[m].mean(axis=0))
        valid_t.append(t0)

    cl = np.array(cl, float)
    valid_t = np.array(valid_t, float)

    if len(cl) < 10:
        # fallback: straight line
        p0 = center + t_body_min * axis
        p1 = center + t_body_max * axis
        return np.vstack([p0, p1]), np.array([t_body_min, t_body_max])

    cl_s = moving_average(cl, CENTERLINE_SMOOTH_WIN)
    return cl_s, valid_t


def centerline_arclength(cl):
    d = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def sample_centerline_by_arclength(cl, s, s_query):
    # piecewise linear interpolation along arc length
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
# Outer radius using local tangent plane
# -------------------------
def outer_radius_angle_binned(points_slice, plane_center, normal, bins=240, cap_q=0.995):
    u, v = plane_basis(normal)

    rel = points_slice - plane_center
    x = rel @ u
    y = rel @ v
    ang = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y)

    if len(r) < 50:
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
# Draw torus ring
# -------------------------
def rotation_from_z_to_vec(v):
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(v, z):
        return np.eye(3)
    if np.allclose(v, -z):
        return np.array([[1, 0, 0],
                         [0,-1, 0],
                         [0, 0,-1]], dtype=float)
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
    points = np.asarray(points, float)
    lines = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, float), (len(lines), 1)))
    return ls


# -------------------------
# Main
# -------------------------
def main():
    pcd = o3d.io.read_point_cloud(IN_PATH)
    if len(pcd.points) == 0:
        raise RuntimeError("No points found in input PLY")

    if REMOVE_OUTLIERS:
        pcd, _ = pcd.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO)

    pts = np.asarray(pcd.points)
    center, coarse_axis = pca_axis(pts)
    t_all = (pts - center) @ coarse_axis

    t_full_min = float(np.quantile(t_all, 0.001))
    t_full_max = float(np.quantile(t_all, 0.999))

    # Body interval on coarse axis
    t_body_min, t_body_max = detect_body_interval(pts, center, coarse_axis, t_full_min, t_full_max)
    L_body = t_body_max - t_body_min

    # Build curved centerline (centroids of slabs)
    cl, cl_t = build_centerline(pts, center, coarse_axis, t_body_min, t_body_max)
    s = centerline_arclength(cl)

    # Length endpoints along centerline (touching)
    s_min = max(0.0, s[0] - 0.0)
    s_max = s[-1]
    s_touch = LENGTH_TOUCH_MARGIN_FRAC * (s_max - s_min)
    s_len_min = max(s_min, s_min - 0.0)  # already 0
    s_len_max = min(s_max, s_max + 0.0)  # already max
    # extend a bit by clamping to ends (visual touch by using endpoints anyway)
    # (If you want extra: you can reduce body threshold instead of extrapolating.)
    p_len0 = cl[0]
    p_len1 = cl[-1]

    # Ring placement along centerline (skip messy ends)
    s0 = s_min + SKIP_END_FRAC * (s_max - s_min)
    s1 = s_max - SKIP_END_FRAC * (s_max - s_min)
    ring_s = np.linspace(s0, s1, N_RINGS)

    ring_centers, idxs, _ = sample_centerline_by_arclength(cl, s, ring_s)
    ring_normals = np.array([tangent_from_centerline(cl, int(i)) for i in idxs])

    # For slicing: we still use coarse axis just to pick nearby points efficiently.
    # But for radius, we project onto the local ring plane.
    half_thick = 0.5 * (SLICE_THICKNESS_FRAC * (t_body_max - t_body_min))

    palette = [
        (0.0, 1.0, 0.0),
        (0.0, 0.7, 1.0),
        (1.0, 0.2, 1.0),
        (1.0, 0.8, 0.0),
        (1.0, 0.4, 0.2),
    ]

    ring_radii = np.full(N_RINGS, np.nan, float)
    rows = []

    # precompute coarse t for all points (speed)
    t_pts = (pts - center) @ coarse_axis
    # also coarse t for ring centers
    t_ring = (ring_centers - center) @ coarse_axis

    for i in range(N_RINGS):
        # pick slab by coarse t around this ring center
        m = (t_pts >= (t_ring[i] - half_thick)) & (t_pts <= (t_ring[i] + half_thick))
        n_in = int(m.sum())
        if n_in < 250:
            rows.append([i + 1, float(ring_s[i]), np.nan, np.nan, n_in])
            continue

        r_outer = outer_radius_angle_binned(
            pts[m],
            plane_center=ring_centers[i],
            normal=ring_normals[i],
            bins=ANGLE_BINS,
            cap_q=RADIAL_CAP_Q
        )

        ring_radii[i] = r_outer
        rows.append([i + 1, float(ring_s[i]), 2.0 * r_outer * UNIT_SCALE_TO_CM, r_outer * UNIT_SCALE_TO_CM, n_in])

    # Smooth radii along centerline
    valid = np.isfinite(ring_radii)
    if OUTER_SMOOTH_WIN > 1 and valid.sum() >= OUTER_SMOOTH_WIN:
        rr = ring_radii.copy()
        idx = np.arange(len(rr))
        rr[~valid] = np.interp(idx[~valid], idx[valid], rr[valid])
        rr = moving_average_1d(rr, OUTER_SMOOTH_WIN)
        ring_radii[valid] = rr[valid]

    # Build ring meshes
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

    # Curved length polyline
    length_line = make_polyline(cl, color=(1.0, 0.0, 0.0))

    # Compute length in cm (arc length)
    L_cm = float(s[-1] * UNIT_SCALE_TO_CM)

    # Average diameter in cm (mid portion)
    dcm = np.array([r[2] for r in rows], float)
    dcm = dcm[np.isfinite(dcm)]
    if len(dcm) > 0:
        if AVG_MODE == "mid":
            lo = int(np.floor(MID_FRAC_LOW * len(dcm)))
            hi = int(np.ceil(MID_FRAC_HIGH * len(dcm)))
            mid = dcm[lo:hi] if hi > lo else dcm
            avg_d_cm = float(np.mean(mid))
        else:
            avg_d_cm = float(np.mean(dcm))
    else:
        avg_d_cm = float("nan")

    print("\n=== SUMMARY ===")
    print(f"Curved length (cm): {L_cm:.2f}")
    print(f"Avg diameter (cm): {avg_d_cm:.2f}")

    if SAVE_CSV:
        df = pd.DataFrame(rows, columns=["ring_id", "s_along_centerline", "diameter_cm", "radius_cm", "slice_points"])
        base = os.path.splitext(IN_PATH)[0]
        out_csv = base + "_ring_values_cm.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    # Visualize
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
