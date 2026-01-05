import numpy as np
import open3d as o3d

# =========================
# INPUT / OUTPUT
# =========================
IN_PCD  = r"colmap_out\onlyplant.pcd"
OUT_PCD = r"colmap_out\onlyplant_FINAL_clean_filled.pcd"

pcd0 = o3d.io.read_point_cloud(IN_PCD)
if pcd0.is_empty():
    raise RuntimeError(f"Empty input: {IN_PCD}")

print("Loaded:", len(pcd0.points))

# ---------- scale ----------
bbox0 = pcd0.get_axis_aligned_bounding_box()
diag0 = float(np.linalg.norm(bbox0.get_max_bound() - bbox0.get_min_bound()))
if diag0 <= 0:
    raise RuntimeError("Bad bbox diag")

# ============================================================
# 1) WHITE SPECKLE DENOISE (conservative, avoids leaf holes)
# ============================================================
pcd = pcd0

# Remove floating tiny noise first
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.6)

bbox = pcd.get_axis_aligned_bounding_box()
diag = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
R2 = diag / 360.0
pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=R2)

print("After denoise:", len(pcd.points))

# ============================================================
# 2) REMOVE BLUE OUTLINE (recolor blue-ish points, NO deletion)
# ============================================================
pts = np.asarray(pcd.points)
col = np.asarray(pcd.colors)

r, g, b = col[:, 0], col[:, 1], col[:, 2]
mx = np.max(col, axis=1)
mn = np.min(col, axis=1)
diff = mx - mn

# Hue (vectorized)
h = np.zeros_like(mx)
m = diff > 1e-8
idx = m & (mx == r); h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
idx = m & (mx == g); h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
idx = m & (mx == b); h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
h = h / 6.0

# Sat/Val
s = np.zeros_like(mx)
mv = mx > 1e-8
s[mv] = diff[mv] / mx[mv]
v = mx

# Blue-ish detection (tuned for blue-screen spill)
H_LO, H_HI = 0.53, 0.73
S_MIN, V_MIN = 0.18, 0.06
BR_GAP, BG_GAP = 0.06, 0.04

blueish = (
    (h >= H_LO) & (h <= H_HI) &
    (s >= S_MIN) & (v >= V_MIN) &
    ((b - r) >= BR_GAP) & ((b - g) >= BG_GAP)
)

blue_idx = np.where(blueish)[0]
print("Blue-ish points to recolor:", len(blue_idx))

kdt = o3d.geometry.KDTreeFlann(pcd)
new_col = col.copy()

KNN = 60
MIN_GOOD = 12

for i in blue_idx:
    _, idxs, _ = kdt.search_knn_vector_3d(pts[i], KNN)
    idxs = np.asarray(idxs, dtype=int)
    good = idxs[~blueish[idxs]]

    if good.size >= MIN_GOOD:
        new_col[i] = np.median(new_col[good], axis=0)
    else:
        c = new_col[i].copy()
        c[2] = min(c[2], max(c[0], c[1]) + 0.03)
        new_col[i] = c

pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(pts)
pcd_clean.colors = o3d.utility.Vector3dVector(new_col)

print("After de-blue (points unchanged):", len(pcd_clean.points))

# ============================================================
# 3) TARGETED HOLE FILL (Poisson internal, add points only in sparse areas)
# ============================================================
bbox = pcd_clean.get_axis_aligned_bounding_box()
diag = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
voxel = diag / 2600.0
if voxel <= 0:
    voxel = 0.001

# Uniform copy for Poisson
pcd_u = pcd_clean.voxel_down_sample(voxel)
print("Uniformized for Poisson:", len(pcd_u.points), "voxel:", voxel)

pcd_u.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 20, max_nn=120)
)
pcd_u.orient_normals_consistent_tangent_plane(120)

depth = 12
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_u, depth=depth)
dens = np.asarray(dens)

# Very gentle trim
keep_q = 0.002
thr = np.quantile(dens, keep_q)
mesh = mesh.select_by_index(np.where(dens >= thr)[0])

mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()

# Oversample candidates
N0 = len(pcd_clean.points)
Nsample = int(min(max(N0 * 1.5, 600000), 1800000))
pcd_s = mesh.sample_points_poisson_disk(number_of_points=Nsample)
s_pts = np.asarray(pcd_s.points)
print("Sampled candidates:", len(s_pts))

# Find sparse regions in ORIGINAL cleaned cloud
kdt_orig = o3d.geometry.KDTreeFlann(pcd_clean)
p_pts = np.asarray(pcd_clean.points)
p_col = np.asarray(pcd_clean.colors)

R = voxel * 4.0
MIN_NEI = 18

print("Finding sparse regions...")
sparse_mask = np.zeros(len(p_pts), dtype=bool)
for i in range(len(p_pts)):
    cnt, _, _ = kdt_orig.search_radius_vector_3d(p_pts[i], R)
    if cnt < MIN_NEI:
        sparse_mask[i] = True

sparse_idx = np.where(sparse_mask)[0]
print("Sparse points:", len(sparse_idx))

pcd_sparse = pcd_clean.select_by_index(sparse_idx.tolist())
kdt_sparse = o3d.geometry.KDTreeFlann(pcd_sparse)

# Select fill points near sparse regions
ADD_R = voxel * 8.0  # increase to *10 if holes remain
print("Selecting fill points...")
add_mask = np.zeros(len(s_pts), dtype=bool)
for i in range(len(s_pts)):
    cnt, _, _ = kdt_sparse.search_radius_vector_3d(s_pts[i], ADD_R)
    if cnt > 0:
        add_mask[i] = True

fill_pts = s_pts[add_mask]
print("Fill points selected:", len(fill_pts))

# Color transfer
fill_col = np.zeros((len(fill_pts), 3), dtype=np.float64)
for i in range(len(fill_pts)):
    _, idxs, _ = kdt_orig.search_knn_vector_3d(fill_pts[i], 1)
    fill_col[i] = p_col[int(idxs[0])]

# Merge (keep all original + fill)
all_pts = np.vstack([p_pts, fill_pts])
all_col = np.vstack([p_col, fill_col])

pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(all_pts)
pcd_out.colors = o3d.utility.Vector3dVector(all_col)

# Regularize density
pcd_out = pcd_out.voxel_down_sample(voxel)

# Gentle cleanup (avoid punching new holes)
pcd_out, _ = pcd_out.remove_statistical_outlier(nb_neighbors=25, std_ratio=3.0)

o3d.io.write_point_cloud(OUT_PCD, pcd_out)
print("Saved FINAL:", OUT_PCD)
print("DONE.")
