import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

# =====================================================
# USER SETTINGS
# =====================================================
MODEL_PATH = r"C:\paper 3 umapp\dl_output\best_stem_leaf_model.pth"
INPUT_NPY = r"C:\paper 3 umapp\labelled_training\s5_labelled.npy"
OUTPUT_DIR = r"C:\paper 3 umapp\measurement_output"
PLANT_ID = "s5"

# If point cloud is in meters and you want cm, use 100.0.
# If point cloud is in millimeters and you want cm, use 0.1.
# If point cloud is already in centimeters, use 1.0.
SCALE_TO_CM = 1.0

# Leaf splitting parameters
STEM_REMOVE_RATIO = 0.025
DBSCAN_EPS_MULT = 4.0
MIN_LEAF_CLUSTER_POINTS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SimplePointNetSeg(nn.Module):
    def __init__(self, input_dim=9, num_classes=2):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        local_feat = self.mlp1(x)
        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, local_feat.shape[2])
        combined = torch.cat([local_feat, global_feat], dim=1)
        logits = self.mlp2(combined)
        return logits.transpose(1, 2)


def normalize_xyz(xyz):
    xyz_norm = xyz - xyz.mean(axis=0)
    scale = np.max(np.linalg.norm(xyz_norm, axis=1))
    if scale < 1e-8:
        scale = 1.0
    return xyz_norm / scale


def predict_stem_leaf(data):
    xyz_original = data[:, 0:3].astype(np.float32)
    xyz_norm = normalize_xyz(xyz_original)
    rgb = data[:, 3:6].astype(np.float32) / 255.0
    normals = data[:, 6:9].astype(np.float32)
    features = np.concatenate([xyz_norm, rgb, normals], axis=1)

    model = SimplePointNetSeg(input_dim=9, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(features_tensor)
        pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return pred  # 0=stem, 1=leaf


def estimate_stem_vector(stem_xyz):
    if len(stem_xyz) < 5:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    centered = stem_xyz - stem_xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    stem_vec = vh[0]
    stem_vec = stem_vec / (np.linalg.norm(stem_vec) + 1e-8)
    if stem_vec[2] < 0:
        stem_vec = -stem_vec
    return stem_vec.astype(np.float32)


def nearest_distance_to_stem(leaf_xyz, stem_xyz):
    stem_pcd = o3d.geometry.PointCloud()
    stem_pcd.points = o3d.utility.Vector3dVector(stem_xyz)
    kdtree = o3d.geometry.KDTreeFlann(stem_pcd)

    distances = np.zeros(len(leaf_xyz), dtype=np.float32)
    for i, p in enumerate(leaf_xyz):
        _, _, dist2 = kdtree.search_knn_vector_3d(p, 1)
        distances[i] = math.sqrt(dist2[0])
    return distances


def auto_dbscan_eps(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(dists) == 0:
        return 0.01
    eps = np.percentile(dists, 80) * DBSCAN_EPS_MULT
    if eps <= 0:
        eps = np.mean(dists) * DBSCAN_EPS_MULT
    return float(max(eps, 0.01))


def cluster_leaf_instances(xyz, pred):
    stem_idx = np.where(pred == 0)[0]
    leaf_idx = np.where(pred == 1)[0]
    stem_xyz = xyz[stem_idx]
    leaf_xyz = xyz[leaf_idx]

    if len(stem_xyz) < 10:
        raise RuntimeError("Too few stem points predicted. Model prediction may be bad.")
    if len(leaf_xyz) < 10:
        raise RuntimeError("Too few leaf points predicted. Model prediction may be bad.")

    plant_diag = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
    stem_remove_radius = STEM_REMOVE_RATIO * plant_diag
    dist_to_stem = nearest_distance_to_stem(leaf_xyz, stem_xyz)

    body_mask = dist_to_stem > stem_remove_radius
    body_leaf_xyz = leaf_xyz[body_mask]
    body_leaf_global_idx = leaf_idx[body_mask]
    base_leaf_xyz = leaf_xyz[~body_mask]
    base_leaf_global_idx = leaf_idx[~body_mask]

    if len(body_leaf_xyz) < 10:
        print("Warning: too few body leaf points. Using all leaf points for clustering.")
        body_leaf_xyz = leaf_xyz
        body_leaf_global_idx = leaf_idx
        base_leaf_xyz = np.empty((0, 3), dtype=np.float32)
        base_leaf_global_idx = np.empty((0,), dtype=np.int64)

    eps = auto_dbscan_eps(body_leaf_xyz)
    print("Auto DBSCAN eps:", eps)
    print("Stem remove radius:", stem_remove_radius)

    body_pcd = o3d.geometry.PointCloud()
    body_pcd.points = o3d.utility.Vector3dVector(body_leaf_xyz)
    labels = np.array(body_pcd.cluster_dbscan(eps=eps, min_points=20, print_progress=False))
    raw_cluster_ids = sorted([c for c in np.unique(labels) if c != -1])

    kept_clusters = []
    for c in raw_cluster_ids:
        if np.sum(labels == c) >= MIN_LEAF_CLUSTER_POINTS:
            kept_clusters.append(c)

    if len(kept_clusters) == 0:
        raise RuntimeError("No leaf clusters found. Increase DBSCAN_EPS_MULT or lower MIN_LEAF_CLUSTER_POINTS.")

    instance_pred = np.zeros(len(xyz), dtype=np.int32)
    instance_pred[stem_idx] = 1

    for new_i, old_c in enumerate(kept_clusters, start=2):
        member_global_idx = body_leaf_global_idx[labels == old_c]
        instance_pred[member_global_idx] = new_i

    assigned_body_mask = instance_pred[body_leaf_global_idx] > 0
    assigned_points = body_leaf_xyz[assigned_body_mask]
    assigned_instances = instance_pred[body_leaf_global_idx[assigned_body_mask]]

    if len(assigned_points) > 0:
        assigned_pcd = o3d.geometry.PointCloud()
        assigned_pcd.points = o3d.utility.Vector3dVector(assigned_points)
        assigned_tree = o3d.geometry.KDTreeFlann(assigned_pcd)

        for gi, p in zip(body_leaf_global_idx, body_leaf_xyz):
            if instance_pred[gi] == 0:
                _, idx, _ = assigned_tree.search_knn_vector_3d(p, 1)
                instance_pred[gi] = assigned_instances[idx[0]]

        for gi, p in zip(base_leaf_global_idx, base_leaf_xyz):
            _, idx, _ = assigned_tree.search_knn_vector_3d(p, 1)
            instance_pred[gi] = assigned_instances[idx[0]]

    return instance_pred


def reorder_leaves_bottom_to_top(xyz, instance_pred):
    stem_xyz = xyz[instance_pred == 1]
    leaf_instance_ids = sorted([i for i in np.unique(instance_pred) if i >= 2])
    leaf_infos = []

    stem_pcd = o3d.geometry.PointCloud()
    stem_pcd.points = o3d.utility.Vector3dVector(stem_xyz)
    stem_tree = o3d.geometry.KDTreeFlann(stem_pcd)

    for inst_id in leaf_instance_ids:
        leaf_xyz = xyz[instance_pred == inst_id]
        best_dist = float("inf")
        best_leaf_point = None
        for p in leaf_xyz:
            _, _, dist2 = stem_tree.search_knn_vector_3d(p, 1)
            d = math.sqrt(dist2[0])
            if d < best_dist:
                best_dist = d
                best_leaf_point = p
        leaf_infos.append({"old_instance": inst_id, "attachment_z": best_leaf_point[2]})

    leaf_infos.sort(key=lambda x: x["attachment_z"])
    new_instance_pred = instance_pred.copy()
    new_instance_pred[instance_pred >= 2] = 0

    old_to_new = {}
    for leaf_number, info in enumerate(leaf_infos, start=1):
        old_id = info["old_instance"]
        new_id = leaf_number + 1
        old_to_new[old_id] = new_id
        new_instance_pred[instance_pred == old_id] = new_id

    return new_instance_pred, old_to_new


def measure_traits(xyz, instance_pred):
    stem_xyz = xyz[instance_pred == 1]
    stem_vec = estimate_stem_vector(stem_xyz)
    plant_height = (xyz[:, 2].max() - xyz[:, 2].min()) * SCALE_TO_CM
    results = []

    stem_pcd = o3d.geometry.PointCloud()
    stem_pcd.points = o3d.utility.Vector3dVector(stem_xyz)
    stem_tree = o3d.geometry.KDTreeFlann(stem_pcd)

    for inst_id in sorted([i for i in np.unique(instance_pred) if i >= 2]):
        leaf_number = inst_id - 1
        leaf_xyz = xyz[instance_pred == inst_id]
        if len(leaf_xyz) < MIN_LEAF_CLUSTER_POINTS:
            continue

        best_dist = float("inf")
        attachment = None
        for p in leaf_xyz:
            _, _, dist2 = stem_tree.search_knn_vector_3d(p, 1)
            d = math.sqrt(dist2[0])
            if d < best_dist:
                best_dist = d
                attachment = p

        distances = np.linalg.norm(leaf_xyz - attachment, axis=1)
        tip = leaf_xyz[np.argmax(distances)]
        straight_length = np.max(distances) * SCALE_TO_CM

        leaf_vec = tip - attachment
        leaf_vec = leaf_vec / (np.linalg.norm(leaf_vec) + 1e-8)
        dot_value = np.clip(np.dot(stem_vec, leaf_vec), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(dot_value))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        results.append({
            "plant_id": PLANT_ID,
            "leaf_id": leaf_number,
            "points": len(leaf_xyz),
            "plant_height_cm": plant_height,
            "leaf_straight_length_cm": straight_length,
            "leaf_angle_deg": angle_deg,
            "attachment_x": attachment[0],
            "attachment_y": attachment[1],
            "attachment_z": attachment[2],
            "tip_x": tip[0],
            "tip_y": tip[1],
            "tip_z": tip[2],
        })

    return results


def save_colored_instance_ply(xyz, instance_pred, out_path):
    colors = np.zeros((len(xyz), 3), dtype=np.float64)
    colors[instance_pred == 1] = [1.0, 0.0, 0.0]
    palette = [
        [0.0, 1.0, 0.0], [0.0, 0.8, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0], [0.7, 0.0, 1.0], [0.0, 1.0, 0.6],
        [1.0, 0.0, 0.6], [0.5, 1.0, 0.2],
    ]
    leaf_ids = sorted([i for i in np.unique(instance_pred) if i >= 2])
    for idx, inst_id in enumerate(leaf_ids):
        colors[instance_pred == inst_id] = palette[idx % len(palette)]
    colors[instance_pred == 0] = [0.5, 0.5, 0.5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)


def save_csv(results, out_csv):
    fieldnames = [
        "plant_id", "leaf_id", "points", "plant_height_cm",
        "leaf_straight_length_cm", "leaf_angle_deg",
        "attachment_x", "attachment_y", "attachment_z",
        "tip_x", "tip_y", "tip_z",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    print("Using device:", DEVICE)
    data = np.load(INPUT_NPY).astype(np.float32)
    xyz = data[:, 0:3]
    print("Loaded:", INPUT_NPY)
    print("Total points:", len(xyz))

    print("\nPredicting stem/leaf...")
    pred = predict_stem_leaf(data)
    print("Predicted stem points:", np.sum(pred == 0))
    print("Predicted leaf points:", np.sum(pred == 1))

    print("\nSplitting leaves...")
    instance_pred = cluster_leaf_instances(xyz, pred)

    print("\nOrdering leaves from bottom to top...")
    instance_pred, old_to_new = reorder_leaves_bottom_to_top(xyz, instance_pred)
    print("Leaf reordering old -> new:", old_to_new)

    print("\nMeasuring traits...")
    results = measure_traits(xyz, instance_pred)

    out_ply = os.path.join(OUTPUT_DIR, f"{PLANT_ID}_predicted_leaf_instances.ply")
    out_csv = os.path.join(OUTPUT_DIR, f"{PLANT_ID}_measurements.csv")
    out_npy = os.path.join(OUTPUT_DIR, f"{PLANT_ID}_predicted_labels.npy")

    save_colored_instance_ply(xyz, instance_pred, out_ply)
    save_csv(results, out_csv)

    predicted_semantic = np.zeros(len(pred), dtype=np.int32)
    predicted_semantic[pred == 0] = 1
    predicted_semantic[pred == 1] = 2
    combined = np.hstack([data[:, 0:9], predicted_semantic.reshape(-1, 1), instance_pred.reshape(-1, 1)])
    np.save(out_npy, combined)

    print("\nSaved:")
    print(out_ply)
    print(out_csv)
    print(out_npy)

    print("\nMeasurement result:")
    for row in results:
        print(
            f"Leaf {row['leaf_id']:02d} | points={row['points']} | "
            f"length={row['leaf_straight_length_cm']:.2f} cm | "
            f"angle={row['leaf_angle_deg']:.2f} deg | "
            f"height={row['plant_height_cm']:.2f} cm"
        )


if __name__ == "__main__":
    main()
