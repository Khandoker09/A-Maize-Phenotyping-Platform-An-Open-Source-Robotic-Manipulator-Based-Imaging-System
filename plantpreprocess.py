"""
Clean a plant point cloud by removing black/white outline noise and optionally filling small leaf holes.

Input requirement:
    Binary little-endian PLY with at least these vertex properties:
    x y z red green blue
    Optional normal properties nx ny nz are preserved/estimated as nearest-neighbour copies.

Install:
    pip install numpy scipy scikit-image

Basic run:
    python clean_plant_point_cloud_full.py sample1_clean.ply

Outputs:
    *_01_green_clean.ply                  -> clean plant-only point cloud
    *_02_reconstructed_conservative.ply   -> small-gap filled, safer for measurements
    *_03_reconstructed_visual.ply         -> stronger visual filling

Recommended:
    Use *_01_green_clean.ply or *_02_reconstructed_conservative.ply for analysis.
    Use *_03_reconstructed_visual.ply for poster/visual display only.
"""

from pathlib import Path
import argparse
import itertools
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import closing, opening, remove_small_holes, remove_small_objects, disk


# -----------------------------
# PLY read/write
# -----------------------------

def read_ply_binary_little_endian(path):
    path = Path(path)
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: end_header not found")
            header_lines.append(line.decode("latin1").rstrip("\n"))
            if line.strip() == b"end_header":
                break
        data_offset = f.tell()

    header = "\n".join(header_lines)
    if "format binary_little_endian" not in header:
        raise ValueError(
            "This script expects binary_little_endian PLY. "
            "If your file is ASCII PLY, export/save it as binary PLY from CloudCompare first."
        )

    vertex_count = None
    properties = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append((parts[1], parts[2]))

    if vertex_count is None:
        raise ValueError("Invalid PLY: vertex count not found")

    type_map = {
        "char": "i1", "int8": "i1",
        "uchar": "u1", "uint8": "u1",
        "short": "<i2", "int16": "<i2",
        "ushort": "<u2", "uint16": "<u2",
        "int": "<i4", "int32": "<i4",
        "uint": "<u4", "uint32": "<u4",
        "float": "<f4", "float32": "<f4",
        "double": "<f8", "float64": "<f8",
    }
    dtype = np.dtype([(name, type_map[typ]) for typ, name in properties])
    data = np.fromfile(path, dtype=dtype, offset=data_offset, count=vertex_count)
    return data


def write_ply_binary_little_endian(path, data):
    path = Path(path)
    type_to_ply = {
        "i1": "char", "u1": "uchar",
        "<i2": "short", "<u2": "ushort",
        "<i4": "int", "<u4": "uint",
        "<f4": "float", "<f8": "double",
        "|i1": "char", "|u1": "uchar",
    }
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Cleaned by clean_plant_point_cloud_full.py",
        f"element vertex {len(data)}",
    ]
    for name in data.dtype.names:
        dt = data.dtype.fields[name][0].str
        if dt not in type_to_ply:
            raise ValueError(f"Unsupported dtype {dt} for property {name}")
        lines.append(f"property {type_to_ply[dt]} {name}")
    lines.append("end_header")

    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        data.tofile(f)


# -----------------------------
# Utilities
# -----------------------------

def get_xyz_rgb_normals(data):
    required = ["x", "y", "z", "red", "green", "blue"]
    missing = [p for p in required if p not in data.dtype.names]
    if missing:
        raise ValueError(f"PLY is missing required properties: {missing}")

    points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    colors = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.float64)

    if all(n in data.dtype.names for n in ["nx", "ny", "nz"]):
        normals = np.column_stack([data["nx"], data["ny"], data["nz"]]).astype(np.float64)
    else:
        normals = np.zeros_like(points, dtype=np.float64)
    return points, colors, normals


def make_structured(points, colors, normals=None):
    if normals is None:
        normals = np.zeros_like(points, dtype=np.float64)

    out = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ],
    )
    out["x"] = points[:, 0].astype(np.float32)
    out["y"] = points[:, 1].astype(np.float32)
    out["z"] = points[:, 2].astype(np.float32)

    cc = np.clip(np.round(colors), 0, 255).astype(np.uint8)
    out["red"] = cc[:, 0]
    out["green"] = cc[:, 1]
    out["blue"] = cc[:, 2]

    out["nx"] = normals[:, 0].astype(np.float32)
    out["ny"] = normals[:, 1].astype(np.float32)
    out["nz"] = normals[:, 2].astype(np.float32)
    return out


def keep_largest_voxel_component(points, mask, voxel_size):
    """Remove disconnected floating islands using voxel connected components."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return mask

    p = points[idx]
    q = np.floor((p - p.min(axis=0)) / voxel_size).astype(np.int32)
    voxels, inverse = np.unique(q, axis=0, return_inverse=True)

    voxel_to_id = {tuple(v): i for i, v in enumerate(voxels)}
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]

    visited = np.zeros(len(voxels), dtype=bool)
    comp_id = np.full(len(voxels), -1, dtype=np.int32)
    comp_sizes = []
    cid = 0

    for i in range(len(voxels)):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        size = 0
        while stack:
            u = stack.pop()
            comp_id[u] = cid
            size += 1
            base = voxels[u]
            for off in offsets:
                key = (int(base[0] + off[0]), int(base[1] + off[1]), int(base[2] + off[2]))
                v = voxel_to_id.get(key)
                if v is not None and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comp_sizes.append(size)
        cid += 1

    largest = int(np.argmax(comp_sizes))
    keep_point = comp_id[inverse] == largest
    out = np.zeros_like(mask)
    out[idx[keep_point]] = True
    return out


# -----------------------------
# Step 1: remove black/white outline and non-green noise
# -----------------------------

def green_color_mask(colors, strength="soft"):
    """
    Remove black/white outline noise using color rules.

    Black/white points usually have low saturation or do not have green dominance.
    The main plant points have stronger green channel and positive ExG.
    """
    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    chroma = mx - mn
    saturation = chroma / np.where(mx == 0, 1, mx)
    exg = 2 * g - r - b  # Excess Green index

    if strength == "soft":
        # Keeps more plant material. Good default.
        mask = (
            (g >= r + 10) &
            (g >= b + 5) &
            (exg >= 18) &
            (saturation >= 0.12) &
            (g >= 35)
        )
    elif strength == "strict":
        # Removes more outline noise but can remove some real pale leaf points.
        mask = (
            (g >= r + 18) &
            (g >= b + 10) &
            (exg >= 35) &
            (saturation >= 0.18) &
            (g >= 45)
        )
    else:
        raise ValueError("strength must be 'soft' or 'strict'")
    return mask


def statistical_and_radius_cleanup(points, mask, k=16, std_ratio=1.5, radius=0.018, min_neighbors=6):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return mask

    p = points[idx]
    finite_local = np.isfinite(p).all(axis=1)
    idx = idx[finite_local]
    p = points[idx]

    tree = cKDTree(p)

    # Statistical outlier removal
    kk = min(k, len(p))
    dists, _ = tree.query(p, k=kk)
    if kk > 1:
        mean_dist = dists[:, 1:].mean(axis=1)
        threshold = mean_dist.mean() + std_ratio * mean_dist.std()
        stat_keep = mean_dist <= threshold
    else:
        stat_keep = np.ones(len(p), dtype=bool)

    # Radius neighbor removal
    counts = tree.query_ball_point(p, r=radius, return_length=True)
    radius_keep = counts >= min_neighbors

    keep_idx = idx[stat_keep & radius_keep]
    out = np.zeros(len(points), dtype=bool)
    out[keep_idx] = True
    return out


def clean_plant_points(points, colors, normals, color_strength="soft"):
    finite = np.isfinite(points).all(axis=1)

    mask = finite & green_color_mask(colors, strength=color_strength)

    # Remove isolated speckles while keeping dense plant leaf regions.
    mask = statistical_and_radius_cleanup(
        points,
        mask,
        k=16,
        std_ratio=1.5,
        radius=0.018,
        min_neighbors=6,
    )

    # Keep only the main connected plant body.
    mask = keep_largest_voxel_component(points, mask, voxel_size=0.020)

    return points[mask], colors[mask], normals[mask], mask


# -----------------------------
# Step 2: optional leaf hole reconstruction
# -----------------------------

def pca_frame(points):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    return center, vh


def aggregate_grid(uv, depth, colors, target_n=1100, pad_frac=0.02):
    mins = uv.min(axis=0)
    maxs = uv.max(axis=0)
    span = maxs - mins
    mins = mins - span * pad_frac
    maxs = maxs + span * pad_frac
    span = maxs - mins

    max_span = float(max(span[0], span[1]))
    px = max_span / target_n
    W = max(64, int(np.ceil(span[0] / px)) + 1)
    H = max(64, int(np.ceil(span[1] / px)) + 1)
    px = max(span[0] / (W - 1), span[1] / (H - 1))

    j = np.clip(np.round((uv[:, 0] - mins[0]) / px).astype(np.int32), 0, W - 1)
    i = np.clip(np.round((uv[:, 1] - mins[1]) / px).astype(np.int32), 0, H - 1)
    lin = i * W + j
    N = H * W

    counts = np.bincount(lin, minlength=N)
    depth_sum = np.bincount(lin, weights=depth, minlength=N)
    color_sums = [np.bincount(lin, weights=colors[:, c], minlength=N) for c in range(3)]

    occ = counts.reshape(H, W) > 0

    depth_img = np.zeros((H, W), dtype=np.float32)
    nz = counts > 0
    vals = np.zeros(N, dtype=np.float32)
    vals[nz] = depth_sum[nz] / counts[nz]
    depth_img[:] = vals.reshape(H, W)

    color_img = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        vals = np.zeros(N, dtype=np.float32)
        vals[nz] = color_sums[c][nz] / counts[nz]
        color_img[..., c] = vals.reshape(H, W)

    return {"mins": mins, "px": px, "H": H, "W": W, "occ": occ, "depth_img": depth_img, "color_img": color_img}


def fill_occupancy_mask(occ, mode="conservative"):
    occ2 = remove_small_objects(occ, 20)

    if mode == "conservative":
        # Small gap filling only.
        m = closing(occ2, disk(1))
        m = remove_small_holes(m, 120)
        m = opening(m, disk(1))
        m = remove_small_objects(m, 60)
    elif mode == "visual":
        # More visible filling, but more approximate.
        m = closing(occ2, disk(2))
        m = remove_small_holes(m, 900)
        m = closing(m, disk(1))
        m = opening(m, disk(1))
        m = remove_small_objects(m, 100)
    else:
        raise ValueError("mode must be conservative or visual")

    return m


def interpolate_new_pixels(grid, uv_orig, depth_orig, color_orig, mode="conservative"):
    occ = grid["occ"]
    filled = fill_occupancy_mask(occ, mode=mode)
    newmask = filled & (~occ)

    if not np.any(newmask):
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0, 3))

    ii, jj = np.nonzero(newmask)
    uv_new = np.column_stack([
        grid["mins"][0] + jj * grid["px"],
        grid["mins"][1] + ii * grid["px"],
    ])

    tree = cKDTree(uv_orig)
    k = 6 if mode == "conservative" else 8
    d, idx = tree.query(uv_new, k=min(k, len(uv_orig)))
    if idx.ndim == 1:
        idx = idx[:, None]
        d = d[:, None]

    weights = 1.0 / (d + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    depth_new = (depth_orig[idx] * weights).sum(axis=1)
    color_new = (color_orig[idx] * weights[..., None]).sum(axis=1)
    return uv_new, depth_new, color_new


def backproject(uv_new, depth_new, dims, depth_dim, center, basis):
    coords = np.zeros((len(uv_new), 3), dtype=np.float64)
    coords[:, dims[0]] = uv_new[:, 0]
    coords[:, dims[1]] = uv_new[:, 1]
    coords[:, depth_dim] = depth_new
    return coords @ basis + center


def reconstruct_leaf_holes(points, colors, normals, mode="conservative"):
    """
    Adds interpolated points into small holes using PCA projection grids.
    This is not ground truth. It only estimates missing surface points.
    """
    center, basis = pca_frame(points)
    coords = (points - center) @ basis.T

    if mode == "conservative":
        target_n = 1000
        min_dist = 0.0035
        max_dist = None
        voxel = 0.0030
        projections = [((0, 1), 2), ((0, 2), 1)]
    elif mode == "visual":
        target_n = 1250
        min_dist = 0.0025
        max_dist = 0.040
        voxel = 0.0025
        projections = [((0, 1), 2), ((0, 2), 1)]
    else:
        raise ValueError("mode must be conservative or visual")

    new_points_list = []
    new_colors_list = []

    for dims, depth_dim in projections:
        uv = coords[:, dims]
        depth = coords[:, depth_dim]
        grid = aggregate_grid(uv, depth, colors, target_n=target_n)
        uv_new, depth_new, color_new = interpolate_new_pixels(grid, uv, depth, colors, mode=mode)
        if len(uv_new):
            new_points_list.append(backproject(uv_new, depth_new, dims, depth_dim, center, basis))
            new_colors_list.append(color_new)

    if not new_points_list:
        return points, colors, normals, 0

    new_points = np.vstack(new_points_list)
    new_colors = np.vstack(new_colors_list)

    # Only keep points that fill gaps, not points that duplicate existing data.
    tree = cKDTree(points)
    d, nearest_idx = tree.query(new_points, k=1)
    keep = d > min_dist
    if max_dist is not None:
        keep &= d < max_dist

    new_points = new_points[keep]
    new_colors = new_colors[keep]
    nearest_idx = nearest_idx[keep]

    # Deduplicate reconstructed points.
    if len(new_points):
        q = np.floor(new_points / voxel).astype(np.int64)
        _, uniq_idx = np.unique(q, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        new_points = new_points[uniq_idx]
        new_colors = new_colors[uniq_idx]
        nearest_idx = nearest_idx[uniq_idx]

    new_normals = normals[nearest_idx] if len(new_points) else np.zeros((0, 3), dtype=np.float64)

    final_points = np.vstack([points, new_points])
    final_colors = np.vstack([colors, new_colors])
    final_normals = np.vstack([normals, new_normals])

    return final_points, final_colors, final_normals, len(new_points)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean plant point cloud and remove black/white outline noise.")
    parser.add_argument("input_ply", help="Input binary_little_endian PLY file")
    parser.add_argument(
        "--color_strength",
        choices=["soft", "strict"],
        default="soft",
        help="soft keeps more leaf points; strict removes more white/black outline noise",
    )
    parser.add_argument(
        "--no_reconstruction",
        action="store_true",
        help="Only create the cleaned green point cloud, without hole-filling outputs",
    )
    args = parser.parse_args()

    in_path = Path(args.input_ply)
    data = read_ply_binary_little_endian(in_path)
    points, colors, normals = get_xyz_rgb_normals(data)

    clean_points, clean_colors, clean_normals, clean_mask = clean_plant_points(
        points,
        colors,
        normals,
        color_strength=args.color_strength,
    )

    out_clean = in_path.with_name(in_path.stem + "_01_green_clean.ply")
    write_ply_binary_little_endian(out_clean, make_structured(clean_points, clean_colors, clean_normals))

    print("Input points:       ", f"{len(points):,}")
    print("Green clean points: ", f"{len(clean_points):,}", "->", out_clean)

    if not args.no_reconstruction:
        c_pts, c_col, c_nrm, c_added = reconstruct_leaf_holes(
            clean_points,
            clean_colors,
            clean_normals,
            mode="conservative",
        )
        out_conservative = in_path.with_name(in_path.stem + "_02_reconstructed_conservative.ply")
        write_ply_binary_little_endian(out_conservative, make_structured(c_pts, c_col, c_nrm))
        print("Conservative added:", f"{c_added:,}", "->", out_conservative)

        v_pts, v_col, v_nrm, v_added = reconstruct_leaf_holes(
            clean_points,
            clean_colors,
            clean_normals,
            mode="visual",
        )
        out_visual = in_path.with_name(in_path.stem + "_03_reconstructed_visual.ply")
        write_ply_binary_little_endian(out_visual, make_structured(v_pts, v_col, v_nrm))
        print("Visual added:      ", f"{v_added:,}", "->", out_visual)

    print("\nDone.")
    print("Use *_01_green_clean.ply or *_02_reconstructed_conservative.ply for analysis.")
    print("Use *_03_reconstructed_visual.ply only for display/poster visualization.")


if __name__ == "__main__":
    main()
