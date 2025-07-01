import cv2
import numpy as np
import yaml
import os
import glob

# Paths
left_dir = "/home/umap3/calib/left"
right_dir = "/home/umap3/calib/right"
output_dir = "/home/umap3/calib/rectified"
os.makedirs(output_dir, exist_ok=True)

# Checkerboard image size (you can update if needed)
image_size = cv2.imread(glob.glob(os.path.join(left_dir, "*.jpg"))[0]).shape[1::-1]

# Load calibration
def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['camera_matrix']), np.array(data['dist_coeff'])

K1, D1 = load_yaml("intrinsicl.yml")
K2, D2 = load_yaml("intrinsicr.yml")

with open("stereo_params.yml", 'r') as f:
    stereo_data = yaml.safe_load(f)

R1 = np.array(stereo_data['R1'])
R2 = np.array(stereo_data['R2'])
P1 = np.array(stereo_data['P1'])
P2 = np.array(stereo_data['P2'])

# Generate rectification maps
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

left_images = sorted(glob.glob(os.path.join(left_dir, "*.jpg")))
right_images = sorted(glob.glob(os.path.join(right_dir, "*.jpg")))

for i, (l_path, r_path) in enumerate(zip(left_images, right_images)):
    img_l = cv2.imread(l_path)
    img_r = cv2.imread(r_path)

    rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

    out_l_path = os.path.join(output_dir, f"rect_left_{i:03d}.jpg")
    out_r_path = os.path.join(output_dir, f"rect_right_{i:03d}.jpg")

    cv2.imwrite(out_l_path, rect_l)
    cv2.imwrite(out_r_path, rect_r)

    print(f"Saved rectified: {out_l_path}, {out_r_path}")

print("✅ All images rectified and saved to:", output_dir)
