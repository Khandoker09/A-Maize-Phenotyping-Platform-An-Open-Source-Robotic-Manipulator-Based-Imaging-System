"""
Maize cob: clean (remove blue/white noise without deleting sparse side) + OPTIONAL reconstruction to fill big holes
---------------------------------------------------------------------------------------------------------------

What this single script does (in order):

1) Load your PCD safely with Open3D (ASCII or binary)
2) STRICT but SAFE color filter:
   - remove very-white points
   - remove ONLY strong saturated blue/cyan points (won't eat cob)
3) Gentle denoise that DOES NOT delete sparse regions:
   - Statistical Outlier Removal (SOR) only (NO radius outlier removal)
4) OPTIONAL stand/fixture removal:
   - DBSCAN keep top K clusters (NOT just largest), so it won't drop cob pieces
5) Save cleaned point cloud (PLY)
6) OPTIONAL reconstruction to fill missing stripe:
   - Poisson surface reconstruction + low trimming percentile (keeps more surface)
   - optional hole filling (if your Open3D build supports it)
   - save mesh
   - optional resample mesh back into points (PLY)

Install:
    pip install open3d numpy

Run:
    python cob_all_in_one.py

Edit only the CONFIG section below.
"""

import os
import numpy as np
import open3d as o3d


# =========================
# CONFIG (EDIT THESE)
# =========================
IN_PCD = r"data\ada85s2 - Cloud.pcd"   # input PCD path (ASCII or binary)

# outputs (auto named if None)
OUT_CLEAN_PLY = None     # e.g. "cob_clean.ply"
OUT_MESH_PLY  = None     # e.g. "cob_filled_mesh.ply"
OUT_FILL_POINTS_PLY = None  # e.g. "cob_filled_points.ply"

# --- Step toggles ---
DO_COLOR_FILTER = True
DO_DENOISE = True

# If your stand/fixture is present, enable clustering removal.
# IMPORTANT: this keeps TOP_K clusters to avoid dropping parts of the cob.
DO_DBSCAN_KEEP_TOPK = False
DBSCAN_EPS = 0.010
DBSCAN_MIN_POINTS = 80
DBSCAN_KEEP_TOP_K = 2   # try 2-4

# Reconstruction to fill big missing stripe (hole filling / inference)
DO_RECONSTRUCT = True

# If you reconstruct, you can also resample points back from the mesh
RESAMPLE_POINTS_FROM_MESH = True
RESAMPLE_N_POINTS = 600_000

# --- SAFE denoise params (won't kill sparse side) ---
SOR_NB_NEIGHBORS = 25
SOR_STD_RATIO = 4.0     # bigger = less aggressive; 3.5-5.0

# --- Color filtering thresholds (safe defaults) ---
# white: very bright, very low saturation
WHITE_V_MIN = 0.97
WHITE_S_MAX = 0.08

# strong blue/cyan: ONLY highly saturated blue-ish points
BLUE_H_MIN = 0.52
BLUE_H_MAX = 0.72
BLUE_S_MIN = 0.55
BLUE_V_MIN = 0.25

# --- Reconstruction params ---
POISSON_DEPTH = 11      # 9-12. Higher = more detail and more artifacts.
NORMAL_RADIUS = 0.012   # depends on your scale. If too rough: increase a bit.
NORMAL_MAX_NN = 50
ORIENT_NORMALS_K = 80

# Poisson density trimming: LOWER keeps more surface (helps fill the stripe)
# Too low => extra blobs; too high => holes reappear.
TRIM_PERCENTILE = 4     # try 3,4,5,6

# Light mesh smoothing (too much will blur kernels)
TAUBIN_ITERS = 6

# Hole fill (may not exist in older Open3D)
TRY_FILL_HOLES = True
HOLE_SIZE = 20000       # larger fills larger holes (in triangles count scale)


# =========================
# Helpers
# =========================
def rgb_to_hsv_np(rgb01: np.ndarray):
    """rgb01 in [0,1], returns hsv in [0,1]. Vectorized."""
    mx = rgb01.max(axis=1)
    mn = rgb01.min(axis=1)
    diff = mx - mn

    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx.copy()

    nz = diff > 1e-12
    s[nz] = diff[nz] / np.maximum(mx[nz], 1e-12)

    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]

    idx = nz & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6

    idx = nz & (mx == g)
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2

    idx = nz & (mx == b)
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4

    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=1)


def keep_top_k_clusters(pcd: o3d.geometry.PointCloud, eps: float, min_points: int, k_keep: int):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    if labels.size == 0 or labels.max() < 0:
        print("[DBSCAN] No clusters found. Keeping everything.")
        return pcd

    valid = labels[labels >= 0]
    counts = np.bincount(valid)
    top = np.argsort(counts)[::-1][:k_keep]
    mask = np.isin(labels, top)

    kept = pcd.select_by_index(np.where(mask)[0])
    print(f"[DBSCAN] Kept top {k_keep} clusters -> {len(kept.points):,} points")
    return kept


def default_out_paths(in_path: str):
    base = os.path.splitext(os.path.basename(in_path))[0]
    clean = f"{base}_clean_safe.ply"
    mesh = f"{base}_filled_mesh.ply"
    pts  = f"{base}_filled_points.ply"
    return clean, mesh, pts


# =========================
# Main
# =========================
def main():
    clean_default, mesh_default, pts_default = default_out_paths(IN_PCD)
    out_clean = OUT_CLEAN_PLY or clean_default
    out_mesh = OUT_MESH_PLY or mesh_default
    out_fill_pts = OUT_FILL_POINTS_PLY or pts_default

    # 1) Load PCD safely with Open3D
    pcd_in = o3d.io.read_point_cloud(IN_PCD)
    if pcd_in.is_empty():
        raise ValueError(f"Failed to load point cloud or file is empty: {IN_PCD}")

    xyz = np.asarray(pcd_in.points, dtype=np.float64)
    print(f"[LOAD] Points: {xyz.shape[0]:,}")

    rgb01 = None
    if pcd_in.has_colors():
        rgb01 = np.asarray(pcd_in.colors, dtype=np.float64)
        if len(rgb01) == len(xyz):
            print("[LOAD] RGB colors detected.")
        else:
            print("[LOAD] Warning: colors length does not match points. Color filtering will be skipped.")
            rgb01 = None
    else:
        print("[LOAD] No RGB colors detected. Color filtering will be skipped.")

    # 2) Safe color filter
    if DO_COLOR_FILTER and rgb01 is not None:
        hsv = rgb_to_hsv_np(rgb01)

        white = (hsv[:, 2] > WHITE_V_MIN) & (hsv[:, 1] < WHITE_S_MAX)
        strong_blue = (
            (hsv[:, 0] > BLUE_H_MIN) & (hsv[:, 0] < BLUE_H_MAX) &
            (hsv[:, 1] > BLUE_S_MIN) & (hsv[:, 2] > BLUE_V_MIN)
        )

        keep = ~(white | strong_blue)

        print(f"[COLOR] Removed white: {int(white.sum()):,}")
        print(f"[COLOR] Removed strong blue/cyan: {int(strong_blue.sum()):,}")
        xyz = xyz[keep]
        rgb01 = rgb01[keep]
        print(f"[COLOR] After filter: {xyz.shape[0]:,}")
    else:
        print("[COLOR] OFF or no RGB colors available")

    # Create working Open3D cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb01 is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb01)

    # 3) Gentle denoise (keeps sparse side)
    if DO_DENOISE:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=SOR_NB_NEIGHBORS,
            std_ratio=SOR_STD_RATIO
        )
        print(f"[DENOISE] After SOR: {len(pcd.points):,}")
    else:
        print("[DENOISE] OFF")

    # 4) Optional: remove stand/fixture via clustering (top-K keep)
    if DO_DBSCAN_KEEP_TOPK:
        pcd = keep_top_k_clusters(
            pcd,
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            k_keep=DBSCAN_KEEP_TOP_K
        )

    # 5) Save cleaned point cloud
    ok = o3d.io.write_point_cloud(out_clean, pcd, write_ascii=True)
    if not ok:
        raise IOError(f"Failed to save cleaned point cloud: {out_clean}")
    print(f"[SAVE] Clean cloud: {out_clean}")

    # Visualize cleaned
    print("[VIEW] Close the viewer window to continue...")
    o3d.visualization.draw_geometries([pcd], window_name="CLEANED POINT CLOUD")

    # 6) Optional: reconstruct mesh to fill big holes/stripe
    if not DO_RECONSTRUCT:
        print("[RECON] OFF. Done.")
        return

    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty before reconstruction.")

    # Normals
    print("[RECON] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN
        )
    )

    print("[RECON] Orienting normals...")
    pcd.orient_normals_consistent_tangent_plane(ORIENT_NORMALS_K)

    # Poisson reconstruction
    print("[RECON] Running Poisson reconstruction...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )
    dens = np.asarray(dens)
    print(f"[RECON] Triangles (raw): {np.asarray(mesh.triangles).shape[0]:,}")

    # Trim LESS to preserve filled area
    thresh = np.percentile(dens, TRIM_PERCENTILE)
    mesh.remove_vertices_by_mask(dens < thresh)
    print(f"[RECON] Density trim percentile: {TRIM_PERCENTILE} -> threshold {thresh:.6f}")

    # Cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Light smoothing
    if TAUBIN_ITERS > 0:
        print(f"[RECON] Taubin smoothing: {TAUBIN_ITERS} iterations")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERS)

    mesh.compute_vertex_normals()

    # Optional hole fill
    if TRY_FILL_HOLES:
        try:
            mesh = mesh.fill_holes(hole_size=HOLE_SIZE)
            mesh.remove_non_manifold_edges()
            mesh.compute_vertex_normals()
            print("[RECON] Hole fill done.")
        except Exception as e:
            print("[RECON] fill_holes not available in your Open3D build. Skipping.")
            print("        Error:", e)

    # Save mesh
    ok = o3d.io.write_triangle_mesh(out_mesh, mesh, write_ascii=True)
    if not ok:
        raise IOError(f"Failed to save mesh: {out_mesh}")
    print(f"[SAVE] Mesh: {out_mesh}")

    print("[VIEW] Close the viewer window to continue...")
    o3d.visualization.draw_geometries([mesh], window_name="RECONSTRUCTED MESH (FILLED)")

    # 7) Optional: resample points from mesh
    if RESAMPLE_POINTS_FROM_MESH:
        print(f"[RESAMPLE] Sampling {RESAMPLE_N_POINTS:,} points from mesh...")
        pcd2 = mesh.sample_points_poisson_disk(number_of_points=RESAMPLE_N_POINTS)

        ok = o3d.io.write_point_cloud(out_fill_pts, pcd2, write_ascii=True)
        if not ok:
            raise IOError(f"Failed to save resampled point cloud: {out_fill_pts}")

        print(f"[SAVE] Resampled points: {out_fill_pts}")

        print("[VIEW] Close the viewer window to finish...")
        o3d.visualization.draw_geometries([pcd2], window_name="RESAMPLED POINT CLOUD (FROM FILLED MESH)")

    print("Done.")


if __name__ == "__main__":
    main()
