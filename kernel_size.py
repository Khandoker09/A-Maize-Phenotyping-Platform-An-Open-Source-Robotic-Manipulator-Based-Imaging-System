import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cellpose import models
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# =========================================================
# INPUT / OUTPUT
# =========================================================
input_path          = "1.png"
output_numbered     = "cob_numbered_kernels.png"
output_sizemap      = "cob_kernel_sizemap.png"       # continuous colormap
output_outliers     = "cob_kernel_outliers.png"      # gray/black outlier style

# =========================================================
# MANUAL TOP CUTOFF
# =========================================================
KERNEL_TOP_Y = 395

# =========================================================
# HSV RANGE FOR ORANGE/YELLOW KERNEL BODY
# =========================================================
HSV_LOWER = np.array([6, 75, 75], dtype=np.uint8)
HSV_UPPER = np.array([28, 255, 255], dtype=np.uint8)

# =========================================================
# CELLPOSE 3.x PARAMETERS
# =========================================================
KERNEL_DIAMETER_PX  = 22
CELLPROB_THRESHOLD  = 0.0
FLOW_THRESHOLD      = 0.6
DOWNSCALE           = 1.0
USE_GPU             = False
MIN_AREA_PX         = 30

# =========================================================
# OUTLIER DETECTION (for outlier-style plot)
# =========================================================
# "outliers" = kernels whose area is outside [Q1 - K*IQR, Q3 + K*IQR].
# Increase K to flag fewer kernels, decrease to flag more.
IQR_K = 1.5

# =========================================================
# HELPERS
# =========================================================
def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out

# =========================================================
# LOAD IMAGE
# =========================================================
img_bgr = cv2.imread(input_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {input_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# =========================================================
# 1) MASK ONLY THE KERNEL BODY
# =========================================================
mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

k    = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
mask = keep_largest_component(mask)
mask[:KERNEL_TOP_Y, :] = 0

ys, xs = np.where(mask > 0)
if len(xs) == 0 or len(ys) == 0:
    raise RuntimeError("Mask is empty. Lower KERNEL_TOP_Y or relax HSV range.")

x1, x2 = xs.min(), xs.max()
y1, y2 = ys.min(), ys.max()

crop_bgr  = img_bgr[y1:y2+1, x1:x2+1]
crop_mask = mask[y1:y2+1, x1:x2+1]

# =========================================================
# 2) OPTIONAL DOWNSCALE FOR SPEED
# =========================================================
if DOWNSCALE > 1.0:
    h, w = crop_bgr.shape[:2]
    new_w = int(w / DOWNSCALE)
    new_h = int(h / DOWNSCALE)
    crop_small = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    eval_diam  = KERNEL_DIAMETER_PX / DOWNSCALE
else:
    crop_small = crop_bgr
    mask_small = crop_mask
    eval_diam  = KERNEL_DIAMETER_PX

# =========================================================
# 3) RUN CELLPOSE 3.x
# =========================================================
print("Loading Cellpose model (first run downloads cyto2 ~26MB)...")
model = models.Cellpose(gpu=USE_GPU, model_type="cyto2")

print("Running segmentation...")
masks_ws, flows, styles, diams = model.eval(
    crop_small,
    diameter=eval_diam,
    channels=[0, 0],
    flow_threshold=FLOW_THRESHOLD,
    cellprob_threshold=CELLPROB_THRESHOLD,
)

if DOWNSCALE > 1.0:
    masks_ws = cv2.resize(
        masks_ws.astype(np.int32),
        (crop_bgr.shape[1], crop_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
else:
    masks_ws = masks_ws.astype(np.int32)

masks_ws[crop_mask == 0] = 0

# =========================================================
# 4) ONE CENTROID + POLYGON + AREA PER KERNEL
# =========================================================
unique_ids = np.unique(masks_ws)
unique_ids = unique_ids[unique_ids > 0]

if len(unique_ids) == 0:
    raise RuntimeError(
        "Cellpose found no kernels. Try adjusting KERNEL_DIAMETER_PX "
        "or lowering CELLPROB_THRESHOLD."
    )

kernels = []  # list of dicts: {'pts', 'area', 'cx', 'cy'}
for lab_id in unique_ids:
    region = (masks_ws == lab_id).astype(np.uint8)
    area   = int(region.sum())
    if area < MIN_AREA_PX:
        continue

    contours, _ = cv2.findContours(region * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        continue
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        continue

    # shift from crop-space to full-image-space
    pts = cnt.squeeze(1).astype(np.float32)
    pts[:, 0] += x1
    pts[:, 1] += y1

    M = cv2.moments(region)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1

    kernels.append({'pts': pts, 'area': area, 'cx': cx, 'cy': cy})

# sort top-to-bottom, then left-to-right (for numbering)
kernels.sort(key=lambda d: (d['cy'], d['cx']))
areas = np.array([k['area'] for k in kernels], dtype=np.float32)

# =========================================================
# 5) ORIGINAL NUMBERED OUTPUT (unchanged)
# =========================================================
pil_img = Image.fromarray(img_rgb).convert("RGB")
draw    = ImageDraw.Draw(pil_img)
font_size = 12
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 255), width=2)
draw.line([(x1, KERNEL_TOP_Y), (x2, KERNEL_TOP_Y)], fill=(255, 0, 0), width=2)

for idx, kd in enumerate(kernels, start=1):
    gx, gy = kd['cx'], kd['cy']
    txt = str(idx)
    try:
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except:
        tw, th = draw.textsize(txt, font=font)
    tx = gx - tw // 2
    ty = gy - th // 2
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            draw.text((tx + dx, ty + dy), txt, font=font, fill=(255, 255, 255))
    draw.text((tx, ty), txt, font=font, fill=(0, 0, 255))

pil_img.save(output_numbered)

# =========================================================
# 6a) KERNEL SIZE MAP — CONTINUOUS COLORMAP
# =========================================================
fig, ax = plt.subplots(figsize=(8, 10), dpi=120)

polys = [Polygon(k['pts'], closed=True) for k in kernels]
norm  = mcolors.Normalize(vmin=areas.min(), vmax=areas.max())
pc = PatchCollection(polys, cmap=plt.cm.viridis, norm=norm,
                     edgecolor='black', linewidth=0.3)
pc.set_array(areas)
ax.add_collection(pc)

cbar = plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label('Kernel area (px²)')

pad = 20
ax.set_xlim(x1 - pad, x2 + pad)
ax.set_ylim(y2 + pad, y1 - pad)   # inverted y so it reads like the image
ax.set_aspect('equal')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title(f'Kernel size map   n={len(kernels)}   '
             f'min={int(areas.min())}  median={int(np.median(areas))}  max={int(areas.max())}')
plt.tight_layout()
plt.savefig(output_sizemap, dpi=150, bbox_inches='tight')
plt.close(fig)

# =========================================================
# 6b) KERNEL SIZE MAP — OUTLIER STYLE (matches reference image)
# =========================================================
q1, q3       = np.percentile(areas, [25, 75])
iqr          = q3 - q1
low_thresh   = q1 - IQR_K * iqr
high_thresh  = q3 + IQR_K * iqr

normals, outliers = [], []
for kd in kernels:
    poly = Polygon(kd['pts'], closed=True)
    if kd['area'] < low_thresh or kd['area'] > high_thresh:
        outliers.append(poly)
    else:
        normals.append(poly)

fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
ax.add_collection(PatchCollection(normals,
                                  facecolor=(0.55, 0.55, 0.55),
                                  edgecolor='black', linewidth=0.4))
ax.add_collection(PatchCollection(outliers,
                                  facecolor=(0.05, 0.05, 0.05),
                                  edgecolor='black', linewidth=0.4))

ax.set_xlim(x1 - pad, x2 + pad)
ax.set_ylim(y2 + pad, y1 - pad)
ax.set_aspect('equal')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_title(f'Kernel outliers (IQR, K={IQR_K})   '
             f'normal={len(normals)}  outlier={len(outliers)}')
plt.tight_layout()
plt.savefig(output_outliers, dpi=150, bbox_inches='tight')
plt.close(fig)

# =========================================================
# SUMMARY
# =========================================================
print("---------------------------------------------")
print("Manual top cutoff :", KERNEL_TOP_Y)
print("Kernel diameter px:", KERNEL_DIAMETER_PX)
print("Final kernel count:", len(kernels))
print("Area  min / median / mean / max : "
      f"{int(areas.min())} / {int(np.median(areas))} / "
      f"{int(areas.mean())} / {int(areas.max())}")
print("IQR thresholds    : low =", int(low_thresh), " high =", int(high_thresh))
print("Outliers flagged  :", len(outliers))
print("Saved:")
print("  ", output_numbered)
print("  ", output_sizemap)
print("  ", output_outliers)
