from pathlib import Path
import argparse
import itertools
import numpy as np
from scipy.spatial import cKDTree


# =========================================================
# PLY read / write
# =========================================================

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
            "This script expects binary_little_endian PLY.\n"
            "In CloudCompare: File > Save As > PLY binary_little_endian."
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
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }

    dtype = np.dtype([(name, type_map[typ]) for typ, name in properties])
    data = np.fromfile(path, dtype=dtype, offset=data_offset, count=vertex_count)

    return data


def write_ply_binary_little_endian(path, data):
    path = Path(path)

    type_to_ply = {
        "i1": "char",
        "|i1": "char",
        "u1": "uchar",
        "|u1": "uchar",
        "<i2": "short",
        "<u2": "ushort",
        "<i4": "int",
        "<u4": "uint",
        "<f4": "float",
        "<f8": "double",
    }

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Cleaned by plant_clean_black_outline_strong.py",
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


# =========================================================
# Basic helpers
# =========================================================

def get_xyz_rgb(data):
    required = ["x", "y", "z", "red", "green", "blue"]
    missing = [p for p in required if p not in data.dtype.names]

    if missing:
        raise ValueError(f"PLY is missing required properties: {missing}")

    points = np.column_stack(
        [data["x"], data["y"], data["z"]]
    ).astype(np.float64)

    colors = np.column_stack(
        [data["red"], data["green"], data["blue"]]
    ).astype(np.float64)

    # If color is saved as 0-1 float, convert to 0-255
    if np.nanmax(colors) <= 1.5:
        colors = colors * 255.0

    return points, colors


def rgb_to_hsv_np(colors_255):
    rgb = np.clip(colors_255 / 255.0, 0.0, 1.0)

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    delta = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc.copy()

    nonzero = maxc > 1e-12
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    mask = delta > 1e-12

    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g)
    bmask = mask & (maxc == b)

    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6.0
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4.0

    h = h * 60.0

    return h, s, v


# =========================================================
# Mode parameters
# =========================================================

def get_mode_params(mode):
    if mode == "rescue":
        return {
            "dark_min": 35,
            "green_seed_exg": 10,
            "green_candidate_exg": 0,
            "green_sat_min": 0.07,
            "brown_sat_min": 0.05,
            "gray_sat_max": 0.045,
        }

    if mode == "balanced":
        return {
            "dark_min": 45,
            "green_seed_exg": 14,
            "green_candidate_exg": 5,
            "green_sat_min": 0.09,
            "brown_sat_min": 0.08,
            "gray_sat_max": 0.055,
        }

    if mode == "strict":
        return {
            "dark_min": 60,
            "green_seed_exg": 20,
            "green_candidate_exg": 12,
            "green_sat_min": 0.12,
            "brown_sat_min": 0.13,
            "gray_sat_max": 0.070,
        }

    raise ValueError("mode must be rescue, balanced, or strict")


# =========================================================
# Strong black-outline plant mask
# =========================================================

def make_plant_masks(
    colors,
    mode="strict",
    dark_min=None,
    brown_sat_min=None,
):
    params = get_mode_params(mode)

    if dark_min is None:
        dark_min = params["dark_min"]

    if brown_sat_min is None:
        brown_sat_min = params["brown_sat_min"]

    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    h, s, v = rgb_to_hsv_np(colors)

    brightness = (r + g + b) / 3.0
    exg = 2.0 * g - r - b

    max_rgb = np.maximum.reduce([r, g, b])
    min_rgb = np.minimum.reduce([r, g, b])
    chroma = max_rgb - min_rgb

    # =====================================================
    # Strong hard-noise removal
    # =====================================================
    # This is the key part for your black outline.
    # It removes dark pixels even if they are slightly green/brown.
    black_outline = (
        (brightness < dark_min)
        | ((brightness < 75) & (max_rgb < 105))
        | ((brightness < 85) & (r < 95) & (g < 95) & (b < 95))
    )

    gray_black_noise = (
        (brightness < 100)
        & (s < 0.18)
        & (chroma < 32)
    )

    white_gray_noise = (
        (brightness > 220)
        & (s < 0.16)
    )

    hard_noise = black_outline | gray_black_noise | white_gray_noise

    # =====================================================
    # Plant candidates
    # =====================================================

    green_seed = (
        (h >= 35)
        & (h <= 140)
        & (s >= 0.14)
        & (v >= 0.10)
        & (brightness >= dark_min)
        & (exg >= params["green_seed_exg"])
        & (g >= b + 3)
        & (~hard_noise)
    )

    green_candidate = (
        (h >= 32)
        & (h <= 150)
        & (s >= params["green_sat_min"])
        & (v >= 0.10)
        & (brightness >= dark_min)
        & (exg >= params["green_candidate_exg"])
        & (g >= b - 2)
        & (~hard_noise)
    )

    yellow_green_candidate = (
        (h >= 28)
        & (h <= 95)
        & (s >= 0.10)
        & (v >= 0.10)
        & (brightness >= dark_min)
        & (r >= b + 4)
        & (g >= b + 3)
        & (~hard_noise)
    )

    # Brown/yellow stem rescue, but not very dark outline.
    brown_stem_candidate = (
        (h >= 15)
        & (h <= 85)
        & (s >= brown_sat_min)
        & (v >= 0.12)
        & (brightness >= max(dark_min, 50))
        & (brightness <= 205)
        & (r >= b + 4)
        & (g >= b - 4)
        & (~hard_noise)
    )

    plant_candidate = (
        green_candidate
        | yellow_green_candidate
        | brown_stem_candidate
    )

    green_seed = green_seed & plant_candidate

    return plant_candidate, green_seed


# =========================================================
# Connectivity cleanup
# =========================================================

def estimate_spacing(points, mask):
    idx = np.where(mask)[0]

    if len(idx) < 10:
        return 0.01

    p = points[idx]
    p = p[np.isfinite(p).all(axis=1)]

    if len(p) < 10:
        return 0.01

    tree = cKDTree(p)

    sample_n = min(len(p), 25000)

    if sample_n < len(p):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(p), sample_n, replace=False)
        sample = p[sample_idx]
    else:
        sample = p

    dists, _ = tree.query(sample, k=2)

    nn = dists[:, 1]
    nn = nn[np.isfinite(nn)]

    if len(nn) == 0:
        return 0.01

    return float(np.median(nn))


def keep_components_connected_to_green(
    points,
    candidate_mask,
    seed_mask,
    voxel_size=None,
    voxel_multiplier=8.0,
    min_component_points=6,
    min_seed_points=1,
):
    idx = np.where(candidate_mask)[0]

    if len(idx) == 0:
        return candidate_mask

    if seed_mask.sum() == 0:
        print("Warning: no green seed found. Keeping color candidates only.")
        return candidate_mask

    p = points[idx]

    finite_local = np.isfinite(p).all(axis=1)

    idx = idx[finite_local]
    p = points[idx]
    seed_local = seed_mask[idx]

    if len(idx) == 0:
        return np.zeros(len(points), dtype=bool)

    if voxel_size is None:
        median_nn = estimate_spacing(points, candidate_mask)
        voxel_size = median_nn * voxel_multiplier
        voxel_size = max(voxel_size, 1e-8)

        print(f"Median point spacing: {median_nn:.8f}")
        print(f"Auto voxel size:      {voxel_size:.8f}")

    q = np.floor((p - p.min(axis=0)) / voxel_size).astype(np.int32)

    voxels, inverse = np.unique(q, axis=0, return_inverse=True)

    voxel_has_seed = np.bincount(
        inverse,
        weights=seed_local.astype(np.float64),
        minlength=len(voxels),
    ) > 0

    voxel_point_count = np.bincount(
        inverse,
        minlength=len(voxels),
    )

    voxel_to_id = {tuple(v): i for i, v in enumerate(voxels)}

    offsets = [
        o for o in itertools.product((-1, 0, 1), repeat=3)
        if o != (0, 0, 0)
    ]

    visited = np.zeros(len(voxels), dtype=bool)
    comp_id = np.full(len(voxels), -1, dtype=np.int32)

    comp_seed_counts = []
    comp_point_counts = []

    cid = 0

    for start in range(len(voxels)):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True

        seed_count = 0
        point_count = 0

        while stack:
            u = stack.pop()
            comp_id[u] = cid

            if voxel_has_seed[u]:
                seed_count += 1

            point_count += int(voxel_point_count[u])

            base = voxels[u]

            for off in offsets:
                key = (
                    int(base[0] + off[0]),
                    int(base[1] + off[1]),
                    int(base[2] + off[2]),
                )

                v = voxel_to_id.get(key)

                if v is not None and not visited[v]:
                    visited[v] = True
                    stack.append(v)

        comp_seed_counts.append(seed_count)
        comp_point_counts.append(point_count)

        cid += 1

    comp_seed_counts = np.array(comp_seed_counts)
    comp_point_counts = np.array(comp_point_counts)

    keep_comp = (
        (comp_seed_counts >= min_seed_points)
        & (comp_point_counts >= min_component_points)
    )

    point_comp = comp_id[inverse]
    keep_local = keep_comp[point_comp]

    out = np.zeros(len(points), dtype=bool)
    out[idx[keep_local]] = True

    print(f"Candidate components: {len(comp_point_counts)}")
    print(f"Kept components:      {int(keep_comp.sum())}")

    return out


# =========================================================
# File picker
# =========================================================

def choose_input_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select input PLY point cloud",
        filetypes=[
            ("PLY files", "*.ply"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not file_path:
        raise SystemExit("No file selected.")

    return Path(file_path)


def choose_input_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(
        title="Select folder containing PLY files"
    )

    root.destroy()

    if not folder_path:
        raise SystemExit("No folder selected.")

    return Path(folder_path)


# =========================================================
# Output folder
# =========================================================

def get_output_folder(in_path, output_dir=None):
    in_path = Path(in_path)

    if output_dir is None:
        output_folder = in_path.parent / "cleaned_output"
    else:
        output_folder = Path(output_dir)

    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder


# =========================================================
# Processing
# =========================================================

def process_one_file(
    in_path,
    output_dir=None,
    mode="strict",
    dark_min=None,
    brown_sat_min=None,
    voxel_size=None,
    voxel_multiplier=8.0,
    min_component_points=6,
):
    in_path = Path(in_path)

    print("\n========================================")
    print("Processing:", in_path)
    print("========================================")

    data = read_ply_binary_little_endian(in_path)
    points, colors = get_xyz_rgb(data)

    finite = np.isfinite(points).all(axis=1)

    candidate_mask, green_seed_mask = make_plant_masks(
        colors,
        mode=mode,
        dark_min=dark_min,
        brown_sat_min=brown_sat_min,
    )

    candidate_mask = candidate_mask & finite
    green_seed_mask = green_seed_mask & finite

    final_mask = keep_components_connected_to_green(
        points,
        candidate_mask=candidate_mask,
        seed_mask=green_seed_mask,
        voxel_size=voxel_size,
        voxel_multiplier=voxel_multiplier,
        min_component_points=min_component_points,
        min_seed_points=1,
    )

    clean_data = data[final_mask]
    removed_data = data[~final_mask]

    output_folder = get_output_folder(in_path, output_dir)

    out_clean = output_folder / (in_path.stem + "_black_outline_removed.ply")
    out_removed = output_folder / (in_path.stem + "_removed_noise_check.ply")

    write_ply_binary_little_endian(out_clean, clean_data)
    write_ply_binary_little_endian(out_removed, removed_data)

    print("Input points:       ", f"{len(data):,}")
    print("Color candidates:   ", f"{candidate_mask.sum():,}")
    print("Green seed points:  ", f"{green_seed_mask.sum():,}")
    print("Final kept points:  ", f"{len(clean_data):,}")
    print("Removed points:     ", f"{len(removed_data):,}")
    print("Output folder:      ", output_folder)
    print("Clean output:       ", out_clean)
    print("Removed check file: ", out_removed)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strongly remove black outline from plant point cloud while keeping plant/stem."
    )

    parser.add_argument(
        "input_ply",
        nargs="?",
        help="Optional input PLY file. If empty, file picker opens.",
    )

    parser.add_argument(
        "--folder",
        action="store_true",
        help="Select folder and process all .ply files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: cleaned_output beside input PLY.",
    )

    parser.add_argument(
        "--mode",
        choices=["rescue", "balanced", "strict"],
        default="strict",
        help="strict removes more black outline; rescue keeps more dark stem.",
    )

    parser.add_argument(
        "--dark_min",
        type=float,
        default=None,
        help="Higher removes more black outline. Try 60, 65, or 70.",
    )

    parser.add_argument(
        "--brown_sat_min",
        type=float,
        default=None,
        help="Higher removes more brown/dark noise. Lower keeps more stem.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Manual voxel size for connectivity. Usually leave empty.",
    )

    parser.add_argument(
        "--voxel_multiplier",
        type=float,
        default=8.0,
        help="Higher connects plant parts across gaps. Default: 8.",
    )

    parser.add_argument(
        "--min_component_points",
        type=int,
        default=6,
        help="Very small components below this are removed. Default: 6.",
    )

    args = parser.parse_args()

    if args.folder:
        folder = choose_input_folder()
        ply_files = sorted(folder.glob("*.ply"))

        if not ply_files:
            raise SystemExit(f"No .ply files found in folder: {folder}")

        print(f"Found {len(ply_files)} PLY files in:")
        print(folder)

        for ply_path in ply_files:
            name = ply_path.name.lower()

            if (
                "_black_outline_removed" in name
                or "_removed_noise_check" in name
                or "_plant_keep_stem" in name
                or "_green_only" in name
                or "_removed_not_green" in name
                or "_outline_clean_safe" in name
            ):
                print("Skipping output file:", ply_path.name)
                continue

            process_one_file(
                ply_path,
                output_dir=args.output_dir,
                mode=args.mode,
                dark_min=args.dark_min,
                brown_sat_min=args.brown_sat_min,
                voxel_size=args.voxel_size,
                voxel_multiplier=args.voxel_multiplier,
                min_component_points=args.min_component_points,
            )

    else:
        if args.input_ply:
            in_path = Path(args.input_ply)
        else:
            in_path = choose_input_file()

        process_one_file(
            in_path,
            output_dir=args.output_dir,
            mode=args.mode,
            dark_min=args.dark_min,
            brown_sat_min=args.brown_sat_min,
            voxel_size=args.voxel_size,
            voxel_multiplier=args.voxel_multiplier,
            min_component_points=args.min_component_points,
        )

    print("\nDone.")
    print("Outputs saved in:")
    print("  cleaned_output")
    print("")
    print("Main clean file:")
    print("  *_black_outline_removed.ply")
    print("")
    print("Check removed points:")
    print("  *_removed_noise_check.ply")



if __name__ == "__main__":
    main()
