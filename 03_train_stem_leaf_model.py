import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# USER SETTINGS
# =====================================================
DATA_ROOT = r"C:\paper 3 umapp\labelled_training"
OUTPUT_ROOT = r"C:\paper 3 umapp\dl_output"

NUM_POINTS = 4096
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 0.001
VAL_PLANT = "s5"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


class PlantSemanticDataset(Dataset):
    def __init__(self, files, num_points=4096, augment=False):
        self.files = files
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def normalize_xyz(self, xyz):
        xyz = xyz - xyz.mean(axis=0)
        scale = np.max(np.linalg.norm(xyz, axis=1))
        if scale < 1e-8:
            scale = 1.0
        return xyz / scale

    def __getitem__(self, idx):
        data = np.load(self.files[idx]).astype(np.float32)
        xyz = data[:, 0:3]
        rgb = data[:, 3:6] / 255.0
        normals = data[:, 6:9]
        semantic = data[:, 9].astype(np.int64)

        # original: 1=stem, 2=leaf
        # model:    0=stem, 1=leaf
        labels = np.zeros_like(semantic)
        labels[semantic == 1] = 0
        labels[semantic == 2] = 1

        xyz = self.normalize_xyz(xyz)
        features = np.concatenate([xyz, rgb, normals], axis=1)

        n = features.shape[0]
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)

        features = features[choice]
        labels = labels[choice]

        if self.augment:
            features = self.apply_augmentation(features)

        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()

    def apply_augmentation(self, features):
        xyz = features[:, 0:3]
        angle = np.random.uniform(-np.pi, np.pi)
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ rot.T
        xyz += np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
        features[:, 0:3] = xyz
        return features


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
        # x: B x N x C
        x = x.transpose(1, 2)
        local_feat = self.mlp1(x)
        global_feat = torch.max(local_feat, dim=2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, local_feat.shape[2])
        combined = torch.cat([local_feat, global_feat], dim=1)
        logits = self.mlp2(combined)
        return logits.transpose(1, 2)


def get_files():
    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT does not exist: {DATA_ROOT}")

    all_files = [os.path.join(DATA_ROOT, f) for f in sorted(os.listdir(DATA_ROOT)) if f.endswith("_labelled.npy")]
    train_files = []
    val_files = []

    for path in all_files:
        name = os.path.basename(path)
        if name.startswith(VAL_PLANT + "_"):
            val_files.append(path)
        else:
            train_files.append(path)

    print("Train files:")
    for f in train_files:
        print(" ", os.path.basename(f))
    print("Validation files:")
    for f in val_files:
        print(" ", os.path.basename(f))

    return train_files, val_files


def calculate_class_weights(train_files):
    counts = np.zeros(2, dtype=np.float64)
    for path in train_files:
        data = np.load(path)
        semantic = data[:, 9].astype(int)
        counts[0] += np.sum(semantic == 1)
        counts[1] += np.sum(semantic == 2)

    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    print("Class counts stem/leaf:", counts)
    print("Class weights stem/leaf:", weights)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def evaluate(model, loader):
    model.eval()
    total_correct = total_points = 0
    stem_correct = stem_total = 0
    leaf_correct = leaf_total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == labels).sum().item()
            total_points += labels.numel()

            stem_mask = labels == 0
            leaf_mask = labels == 1
            stem_correct += ((pred == labels) & stem_mask).sum().item()
            stem_total += stem_mask.sum().item()
            leaf_correct += ((pred == labels) & leaf_mask).sum().item()
            leaf_total += leaf_mask.sum().item()

    return (
        total_correct / max(total_points, 1),
        stem_correct / max(stem_total, 1),
        leaf_correct / max(leaf_total, 1),
    )


def train():
    train_files, val_files = get_files()
    if len(train_files) == 0:
        raise RuntimeError("No training files found.")
    if len(val_files) == 0:
        raise RuntimeError(f"No validation file found for {VAL_PLANT}. Change VAL_PLANT.")

    train_dataset = PlantSemanticDataset(train_files, NUM_POINTS, augment=True)
    val_dataset = PlantSemanticDataset(val_files, NUM_POINTS, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SimplePointNetSeg(input_dim=9, num_classes=2).to(DEVICE)
    class_weights = calculate_class_weights(train_files)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_ROOT, "best_stem_leaf_model.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.reshape(-1, 2), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, stem_acc, leaf_acc = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | Loss: {total_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Stem Acc: {stem_acc:.4f} | Leaf Acc: {leaf_acc:.4f}"
            )

    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)
    print("Saved model:", best_model_path)


if __name__ == "__main__":
    train()
