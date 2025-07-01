import cv2
import numpy as np
import glob
import yaml
import os

# Checkerboard configuration (adjust this to match your board)
CHECKERBOARD = (9, 6)  # (columns, rows) of inner corners
square_size = 1.0  # Set to real-world size of one square (e.g., in cm or mm)

# Calibration folders
left_dir = "/home/umap3/calib/left"
right_dir = "/home/umap3/calib/right"

# Load intrinsics
def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['camera_matrix']), np.array(data['dist_coeff'])

K1, D1 = load_yaml("intrinsicl.yml")
K2, D2 = load_yaml("intrinsicr.yml")

# Prepare object points like (0,0,0), (1,0,0), ..., scaled by square_size
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints_l = []  # 2D points in left images
imgpoints_r = []  # 2D points in right images

left_images = sorted(glob.glob(os.path.join(left_dir, '*.jpg')))
right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')))

if len(left_images) != len(right_images):
    print("⚠️ Mismatch in number of left and right images.")
    exit()

for left_path, right_path in zip(left_images, right_images):
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        print(f"✅ Checkerboard found in: {os.path.basename(left_path)} and {os.path.basename(right_path)}")
    else:
        print(f"❌ Skipped: {os.path.basename(left_path)} / {os.path.basename(right_path)}")

# Perform stereo calibration
print("\n🔧 Running stereo calibration...")
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K1, D1, K2, D2,
    gray_l.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

print(f"🎯 Stereo reprojection error: {ret:.6f}")

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, gray_l.shape[::-1], R, T, flags=cv2.CALIB_ZERO_DISPARITY
)

# Save results to YAML
calib_data = {
    'R': R.tolist(),
    'T': T.tolist(),
    'R1': R1.tolist(),
    'R2': R2.tolist(),
    'P1': P1.tolist(),
    'P2': P2.tolist(),
    'Q': Q.tolist()
}

with open("stereo_params.yml", 'w') as f:
    yaml.dump(calib_data, f)

print("✅ Stereo calibration saved to: stereo_params.yml")
