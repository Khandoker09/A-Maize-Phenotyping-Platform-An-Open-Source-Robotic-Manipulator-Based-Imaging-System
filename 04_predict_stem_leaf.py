import os
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

# =====================================================
# USER SETTINGS
# =====================================================
MODEL_PATH = r"C:\paper 3 umapp\dl_output\best_stem_leaf_model.pth"
INPUT_NPY = r"C:\paper 3 umapp\labelled_training\s5_labelled.npy"
OUTPUT_PLY = r"C:\paper 3 umapp\dl_output\s5_predicted_stem_leaf.ply"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


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


def predict():
    data = np.load(INPUT_NPY).astype(np.float32)
    xyz_original = data[:, 0:3]
    xyz = normalize_xyz(xyz_original)
    rgb = data[:, 3:6] / 255.0
    normals = data[:, 6:9]
    features = np.concatenate([xyz, rgb, normals], axis=1)

    model = SimplePointNetSeg(input_dim=9, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(features_tensor)
        pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # pred: 0=stem, 1=leaf
    colors = np.zeros((len(pred), 3), dtype=np.float64)
    colors[pred == 0] = [1.0, 0.0, 0.0]  # stem red
    colors[pred == 1] = [0.0, 1.0, 0.0]  # leaf green

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_original)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)
    o3d.io.write_point_cloud(OUTPUT_PLY, pcd)

    print("Saved prediction:", OUTPUT_PLY)
    print("Stem points:", np.sum(pred == 0))
    print("Leaf points:", np.sum(pred == 1))


if __name__ == "__main__":
    predict()
