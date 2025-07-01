import numpy as np
import cv2
import glob
import os
import yaml

def calibrate_camera(image_dir, checkerboard_dims=(9, 6)):
    print(f"\nLooking for images in: {image_dir}")
    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)

    objpoints = []  # 3D real-world points
    imgpoints = []  # 2D image points

    images = sorted(glob.glob(os.path.join(image_dir, '*.png')) +
                glob.glob(os.path.join(image_dir, '*.jpg')) +
                glob.glob(os.path.join(image_dir, '*.jpeg')))

    print(f"Found {len(images)} images")

    if not images:
        print("⚠️ No images found!")
        return None, None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ Could not read: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

        if ret:
            print(f"✅ Checkerboard found in {fname}")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
        else:
            print(f"❌ Checkerboard NOT found in {fname}")

    if len(objpoints) < 3:
        print("⚠️ Not enough valid images with detected corners.")
        return None, None

    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"✅ Calibration done. Reprojection error: {ret}")
    return K, dist


def save_to_yml(filename, K, dist):
    data = {
        'camera_matrix': K.tolist(),
        'dist_coeff': dist.tolist()
    }
    with open(filename, 'w') as f:
        yaml.dump(data, f)
    print(f"📁 Saved calibration to: {filename}")

# ✅ Use your actual directory path
left_dir = '/home/umap3/calib/left'
right_dir = '/home/umap3/calib/right'

# 🔧 Calibrate
print("=== Calibrating LEFT camera ===")
K_left, dist_left = calibrate_camera(left_dir)

print("\n=== Calibrating RIGHT camera ===")
K_right, dist_right = calibrate_camera(right_dir)

# 💾 Save
if K_left is not None:
    save_to_yml('intrinsicl.yml', K_left, dist_left)

if K_right is not None:
    save_to_yml('intrinsicr.yml', K_right, dist_right)
