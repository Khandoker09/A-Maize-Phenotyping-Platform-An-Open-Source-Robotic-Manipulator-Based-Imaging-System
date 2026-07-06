import os
import re
import numpy as np
import open3d as o3d

# =====================================================
# USER SETTINGS
# =====================================================
INPUT_ROOT = r"C:\paper 3 umapp\labelled_parts"
OUTPUT_ROOT = r"C:\paper 3 umapp\labelled_training"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def read_ply_features(ply_path):
    """Read PLY and return x y z r g b nx ny nz."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"No points found in {ply_path}")

    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32) * 255.0
    else:
        colors = np.zeros_like(points, dtype=np.float32)

    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float32)
    else:
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals, dtype=np.float32)

    return np.hstack([points, colors, normals])


def get_labels_from_filename(filename):
    """
    Accepts:
    s1stem.ply, s1_stem.ply
    s1leaf1.ply, s1_leaf_01.ply, s1-leaf-02.ply

    semantic_label: 1=stem, 2=leaf
    instance_label: 1=stem, 2=leaf_01, 3=leaf_02, ...
    """
    name = filename.lower()

    if "stem" in name:
        return 1, 1

    leaf_match = re.search(r"leaf[\s_-]?(\d+)", name)
    if leaf_match:
        leaf_number = int(leaf_match.group(1))
        return 2, leaf_number + 1

    raise ValueError(
        f"Cannot understand label from filename: {filename}\n"
        f"Use names like s1stem.ply, s1leaf1.ply, s1_leaf_01.ply"
    )


def make_labelled_file_for_one_plant(plant_folder):
    plant_id = os.path.basename(plant_folder.rstrip("\\/"))
    all_rows = []

    ply_files = sorted([f for f in os.listdir(plant_folder) if f.lower().endswith(".ply")])
    if len(ply_files) == 0:
        print(f"No PLY files found in {plant_folder}")
        return

    for file in ply_files:
        ply_path = os.path.join(plant_folder, file)
        semantic_label, instance_label = get_labels_from_filename(file)
        features = read_ply_features(ply_path)
        n = features.shape[0]

        semantic_col = np.full((n, 1), semantic_label, dtype=np.float32)
        instance_col = np.full((n, 1), instance_label, dtype=np.float32)
        labelled_points = np.hstack([features, semantic_col, instance_col])
        all_rows.append(labelled_points)

        print(
            f"{plant_id}: {file} -> semantic={semantic_label}, "
            f"instance={instance_label}, points={n}"
        )

    final_data = np.vstack(all_rows)
    output_txt = os.path.join(OUTPUT_ROOT, f"{plant_id}_labelled.txt")
    output_npy = os.path.join(OUTPUT_ROOT, f"{plant_id}_labelled.npy")
    header = "x y z r g b nx ny nz semantic_label instance_label"

    np.savetxt(output_txt, final_data, fmt="%.6f", header=header, comments="")
    np.save(output_npy, final_data)

    print("\nSaved:")
    print(output_txt)
    print(output_npy)
    print(f"Total points: {final_data.shape[0]}")
    print("-" * 50)


def main():
    if not os.path.isdir(INPUT_ROOT):
        raise RuntimeError(f"INPUT_ROOT does not exist: {INPUT_ROOT}")

    for item in sorted(os.listdir(INPUT_ROOT)):
        plant_folder = os.path.join(INPUT_ROOT, item)
        if os.path.isdir(plant_folder):
            make_labelled_file_for_one_plant(plant_folder)


if __name__ == "__main__":
    main()
