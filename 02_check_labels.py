import os
import numpy as np
import open3d as o3d

# =====================================================
# USER SETTINGS
# =====================================================
INPUT_ROOT = r"C:\paper 3 umapp\labelled_training"
OUTPUT_ROOT = r"C:\paper 3 umapp\label_check_ply"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def make_colored_ply(npy_path, out_path):
    data = np.load(npy_path)
    xyz = data[:, 0:3]
    semantic = data[:, 9].astype(int)
    instance = data[:, 10].astype(int)

    colors = np.zeros((len(data), 3), dtype=np.float64)
    colors[semantic == 1] = [1.0, 0.0, 0.0]  # stem = red

    palette = {
        2: [0.0, 1.0, 0.0],
        3: [0.0, 0.8, 1.0],
        4: [1.0, 1.0, 0.0],
        5: [1.0, 0.5, 0.0],
        6: [0.7, 0.0, 1.0],
        7: [0.0, 1.0, 0.6],
        8: [1.0, 0.0, 0.6],
        9: [0.5, 1.0, 0.2],
    }
    for inst_id, col in palette.items():
        colors[instance == inst_id] = col

    colors[(semantic != 1) & (semantic != 2)] = [0.5, 0.5, 0.5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)


def main():
    if not os.path.isdir(INPUT_ROOT):
        raise RuntimeError(f"INPUT_ROOT does not exist: {INPUT_ROOT}")

    found = False
    for file in sorted(os.listdir(INPUT_ROOT)):
        if file.endswith("_labelled.npy"):
            found = True
            npy_path = os.path.join(INPUT_ROOT, file)
            plant_id = file.replace("_labelled.npy", "")
            out_path = os.path.join(OUTPUT_ROOT, f"{plant_id}_label_check.ply")
            make_colored_ply(npy_path, out_path)
            print("Saved:", out_path)

    if not found:
        print("No *_labelled.npy files found.")


if __name__ == "__main__":
    main()
