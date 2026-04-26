import os
import csv
import math
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

from cellpose import models


# =========================================================
# IMAGE PROCESSING HELPERS
# =========================================================

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def auto_detect_top_cutoff(mask, min_fraction=0.08, smooth_window=15, consecutive_rows=10, margin=12):
    """
    Automatically detect the top of the cob/kernel area.

    It scans from the top and finds the first sustained rows that contain
    enough kernel pixels. The margin moves the cutoff upward so the top
    kernels are not accidentally removed.
    """
    row_counts = np.sum(mask > 0, axis=1).astype(np.float32)

    if row_counts.max() == 0:
        return 0

    smooth_window = max(3, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1

    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smooth_counts = np.convolve(row_counts, kernel, mode='same')

    threshold = max(5, smooth_counts.max() * float(min_fraction))

    run = 0
    start_row = None
    for i, value in enumerate(smooth_counts):
        if value >= threshold:
            run += 1
            if run >= consecutive_rows:
                start_row = i - consecutive_rows + 1
                break
        else:
            run = 0

    if start_row is None:
        ys = np.where(mask > 0)[0]
        if len(ys) == 0:
            return 0
        start_row = int(ys.min())

    return max(0, int(start_row) - int(margin))


def classify_kernels_by_area(kernels, areas):
    """Classify kernels into small / medium / large using 33% and 66% quantiles."""
    small_thr, large_thr = np.percentile(areas, [33.33, 66.67])
    for kd in kernels:
        if kd['area'] <= small_thr:
            kd['size_class'] = 'small'
        elif kd['area'] <= large_thr:
            kd['size_class'] = 'medium'
        else:
            kd['size_class'] = 'large'
    return float(small_thr), float(large_thr)


def make_kernel_cutout(img_rgb, full_mask, kd, padding=8, tile_size=80):
    """Extract one masked kernel as a square RGB tile with white background."""
    pts = kd['pts'].astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    H, W = full_mask.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x3 = min(W, x + w + padding)
    y3 = min(H, y + h + padding)

    local_labels = full_mask[y0:y3, x0:x3]
    crop = img_rgb[y0:y3, x0:x3].copy()
    local_mask = (local_labels == kd['label_id']).astype(np.uint8) * 255

    white = np.full_like(crop, 255)
    crop = np.where(local_mask[..., None] > 0, crop, white)

    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    yy = (side - ch) // 2
    xx = (side - cw) // 2
    square[yy:yy+ch, xx:xx+cw] = crop
    tile = cv2.resize(square, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(tile)


def save_kernel_montage(kernels, img_rgb, masks_full, out_path, title, sort_key='area', tile_size=80, cols=10):
    ordered = sorted(kernels, key=lambda d: d[sort_key])
    if len(ordered) == 0:
        return

    rows = math.ceil(len(ordered) / cols)
    header_h = 42
    label_h = 22
    W = cols * tile_size
    H = header_h + rows * (tile_size + label_h)
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
        title_font = ImageFont.truetype('DejaVuSans.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)

    for i, kd in enumerate(ordered):
        row = i // cols
        col = i % cols
        x = col * tile_size
        y = header_h + row * (tile_size + label_h)
        tile = make_kernel_cutout(img_rgb, masks_full, kd, tile_size=tile_size)
        canvas.paste(tile, (x, y))
        label = f"#{kd['id']} {kd['area']} {kd['size_class'][0].upper()}"
        draw.text((x + 3, y + tile_size + 3), label, fill=(0, 0, 0), font=font)

    canvas.save(out_path)


def save_grouped_montage(kernels, img_rgb, masks_full, out_path, tile_size=80, cols=10):
    groups = [('small', 'SMALL'), ('medium', 'MEDIUM'), ('large', 'LARGE')]
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
        title_font = ImageFont.truetype('DejaVuSans.ttf', 18)
        section_font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except Exception:
        font = ImageFont.load_default()
        title_font = section_font = font

    label_h = 22
    section_h = 30
    title_h = 42
    W = cols * tile_size
    heights = [title_h]
    grouped = []
    for key, title in groups:
        items = sorted([k for k in kernels if k['size_class'] == key], key=lambda d: d['area'])
        grouped.append((key, title, items))
        rows = max(1, math.ceil(len(items) / cols))
        heights.append(section_h + rows * (tile_size + label_h))
    H = sum(heights)
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'Extracted kernels grouped by size class', fill=(0, 0, 0), font=title_font)

    y_cursor = title_h
    for key, title, items in grouped:
        draw.rectangle([0, y_cursor, W, y_cursor + section_h], fill=(235, 235, 235))
        draw.text((8, y_cursor + 6), f"{title}  n={len(items)}", fill=(0, 0, 0), font=section_font)
        y_cursor += section_h
        for i, kd in enumerate(items):
            row = i // cols
            col = i % cols
            x = col * tile_size
            y = y_cursor + row * (tile_size + label_h)
            tile = make_kernel_cutout(img_rgb, masks_full, kd, tile_size=tile_size)
            canvas.paste(tile, (x, y))
            label = f"#{kd['id']} {kd['area']}"
            draw.text((x + 3, y + tile_size + 3), label, fill=(0, 0, 0), font=font)
        rows = max(1, math.ceil(len(items) / cols))
        y_cursor += rows * (tile_size + label_h)

    canvas.save(out_path)


def process_cob_image(
    input_path,
    output_dir,
    kernel_top_y=395,
    auto_top_cutoff=True,
    disable_top_cutoff=False,
    top_margin=12,
    top_min_fraction=0.08,
    top_smooth_window=15,
    top_consecutive_rows=10,
    hsv_lower=(6, 75, 75),
    hsv_upper=(28, 255, 255),
    kernel_diameter_px=22,
    cellprob_threshold=0.0,
    flow_threshold=0.6,
    downscale=1.0,
    use_gpu=False,
    min_area_px=30,
    iqr_k=1.5,
):
    os.makedirs(output_dir, exist_ok=True)

    output_numbered = os.path.join(output_dir, 'cob_numbered_kernels.png')
    output_sizemap = os.path.join(output_dir, 'cob_kernel_sizemap.png')
    output_outliers = os.path.join(output_dir, 'cob_kernel_outliers.png')
    output_classified = os.path.join(output_dir, 'cob_kernel_size_classes.png')
    output_montage_all = os.path.join(output_dir, 'cob_kernels_sorted_by_size.png')
    output_montage_group = os.path.join(output_dir, 'cob_kernels_by_size_class.png')
    output_csv = os.path.join(output_dir, 'cob_kernel_measurements.csv')
    output_extract_dir = os.path.join(output_dir, 'extracted_kernels')

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Could not read image: {input_path}')

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = keep_largest_component(mask)

    # Automatic / manual / disabled top cutoff
    if disable_top_cutoff:
        used_top_cutoff = 0
    else:
        if auto_top_cutoff:
            used_top_cutoff = auto_detect_top_cutoff(
                mask,
                min_fraction=top_min_fraction,
                smooth_window=top_smooth_window,
                consecutive_rows=top_consecutive_rows,
                margin=top_margin,
            )
        else:
            used_top_cutoff = int(kernel_top_y)

        mask[:used_top_cutoff, :] = 0

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError('Mask is empty. Lower top cutoff or relax HSV range.')

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop_bgr = img_bgr[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1]

    if downscale > 1.0:
        h, w = crop_bgr.shape[:2]
        new_w = max(1, int(w / downscale))
        new_h = max(1, int(h / downscale))
        crop_small = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        eval_diam = kernel_diameter_px / downscale
    else:
        crop_small = crop_bgr
        eval_diam = kernel_diameter_px

    model = models.Cellpose(gpu=use_gpu, model_type='cyto2')
    masks_ws, flows, styles, diams = model.eval(
        crop_small,
        diameter=eval_diam,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    if downscale > 1.0:
        masks_ws = cv2.resize(
            masks_ws.astype(np.int32),
            (crop_bgr.shape[1], crop_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        masks_ws = masks_ws.astype(np.int32)

    masks_ws[crop_mask == 0] = 0

    unique_ids = np.unique(masks_ws)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        raise RuntimeError('Cellpose found no kernels. Try adjusting kernel diameter or thresholds.')

    kernels = []
    for lab_id in unique_ids:
        region = (masks_ws == lab_id).astype(np.uint8)
        area = int(region.sum())
        if area < min_area_px:
            continue

        contours, _ = cv2.findContours(region * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 3:
            continue

        pts = cnt.squeeze(1).astype(np.float32)
        pts[:, 0] += x1
        pts[:, 1] += y1

        M = cv2.moments(region)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00']) + x1
        cy = int(M['m01'] / M['m00']) + y1

        kernels.append({'label_id': int(lab_id), 'pts': pts, 'area': area, 'cx': cx, 'cy': cy})

    kernels.sort(key=lambda d: (d['cy'], d['cx']))
    if len(kernels) == 0:
        raise RuntimeError('No valid kernels remained after filtering.')

    areas = np.array([k['area'] for k in kernels], dtype=np.float32)
    for idx, kd in enumerate(kernels, start=1):
        kd['id'] = idx

    small_thr, large_thr = classify_kernels_by_area(kernels, areas)

    masks_full = np.zeros(mask.shape, dtype=np.int32)
    masks_full[y1:y2+1, x1:x2+1] = masks_ws

    # 1) Numbered image
    pil_img = Image.fromarray(img_rgb).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 255), width=2)
    if not disable_top_cutoff:
        draw.line([(x1, used_top_cutoff), (x2, used_top_cutoff)], fill=(255, 0, 0), width=2)

    for kd in kernels:
        gx, gy = kd['cx'], kd['cy']
        txt = str(kd['id'])
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
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

    # 2) Continuous size map
    fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
    polys = [Polygon(k['pts'], closed=True) for k in kernels]
    norm = mcolors.Normalize(vmin=areas.min(), vmax=areas.max())
    pc = PatchCollection(polys, cmap=plt.cm.viridis, norm=norm, edgecolor='black', linewidth=0.3)
    pc.set_array(areas)
    ax.add_collection(pc)
    cbar = plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Kernel area (px²)')
    pad = 20
    ax.set_xlim(x1 - pad, x2 + pad)
    ax.set_ylim(y2 + pad, y1 - pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(
        f'Kernel size map   n={len(kernels)}   min={int(areas.min())}  '
        f'median={int(np.median(areas))}  max={int(areas.max())}'
    )
    plt.tight_layout()
    plt.savefig(output_sizemap, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3) Outlier map
    q1, q3 = np.percentile(areas, [25, 75])
    iqr = q3 - q1
    low_thresh = q1 - iqr_k * iqr
    high_thresh = q3 + iqr_k * iqr

    normals, outliers = [], []
    for kd in kernels:
        poly = Polygon(kd['pts'], closed=True)
        if kd['area'] < low_thresh or kd['area'] > high_thresh:
            outliers.append(poly)
        else:
            normals.append(poly)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
    ax.add_collection(PatchCollection(normals, facecolor=(0.55, 0.55, 0.55), edgecolor='black', linewidth=0.4))
    ax.add_collection(PatchCollection(outliers, facecolor=(0.05, 0.05, 0.05), edgecolor='black', linewidth=0.4))
    ax.set_xlim(x1 - pad, x2 + pad)
    ax.set_ylim(y2 + pad, y1 - pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Kernel outliers (IQR, K={iqr_k})   normal={len(normals)}  outlier={len(outliers)}')
    plt.tight_layout()
    plt.savefig(output_outliers, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4) Extracted kernels, CSV, classified overlay, montages
    os.makedirs(output_extract_dir, exist_ok=True)
    for cls in ['small', 'medium', 'large']:
        os.makedirs(os.path.join(output_extract_dir, cls), exist_ok=True)

    for kd in kernels:
        tile = make_kernel_cutout(img_rgb, masks_full, kd, padding=8, tile_size=96)
        fname = f"kernel_{kd['id']:03d}_area_{kd['area']}_{kd['size_class']}.png"
        tile.save(os.path.join(output_extract_dir, kd['size_class'], fname))

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'area_px2', 'size_class', 'centroid_x', 'centroid_y'])
        for kd in kernels:
            writer.writerow([kd['id'], kd['area'], kd['size_class'], kd['cx'], kd['cy']])

    class_colors = {
        'small':  (0, 120, 255, 95),
        'medium': (0, 220, 0, 95),
        'large':  (255, 140, 0, 95),
    }
    class_outline = {
        'small':  (0, 80, 255),
        'medium': (0, 160, 0),
        'large':  (255, 90, 0),
    }
    classified = Image.fromarray(img_rgb).convert('RGBA')
    overlay = Image.new('RGBA', classified.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_cls = ImageDraw.Draw(classified)
    try:
        cls_font = ImageFont.truetype('DejaVuSans.ttf', 12)
    except Exception:
        cls_font = ImageFont.load_default()

    for kd in kernels:
        pts_list = [tuple(map(float, p)) for p in kd['pts']]
        draw_overlay.polygon(pts_list, fill=class_colors[kd['size_class']])
        draw_cls.line(pts_list + [pts_list[0]], fill=class_outline[kd['size_class']], width=2)
        draw_cls.text((kd['cx'] - 5, kd['cy'] - 6), kd['size_class'][0].upper(), fill=(0, 0, 0), font=cls_font)

    classified = Image.alpha_composite(classified, overlay).convert('RGB')
    draw_final = ImageDraw.Draw(classified)
    legend_x, legend_y = 10, 10
    for i, cls in enumerate(['small', 'medium', 'large']):
        yy = legend_y + i * 22
        draw_final.rectangle([legend_x, yy, legend_x + 16, yy + 16], fill=class_outline[cls])
        draw_final.text((legend_x + 22, yy), cls.upper(), fill=(0, 0, 0), font=cls_font)
    classified.save(output_classified)

    save_kernel_montage(
        kernels, img_rgb, masks_full, output_montage_all,
        title='All kernels sorted from smallest to largest',
        sort_key='area', tile_size=80, cols=10
    )
    save_grouped_montage(kernels, img_rgb, masks_full, output_montage_group, tile_size=80, cols=10)

    class_counts = {cls: sum(1 for k in kernels if k['size_class'] == cls) for cls in ['small', 'medium', 'large']}

    return {
        'outputs': {
            'Numbered kernels': output_numbered,
            'Continuous size map': output_sizemap,
            'Outlier map': output_outliers,
            'Size-class overlay': output_classified,
            'Kernels sorted by size': output_montage_all,
            'Kernels grouped by class': output_montage_group,
        },
        'csv_path': output_csv,
        'extract_dir': output_extract_dir,
        'stats': {
            'kernel_count': len(kernels),
            'area_min': int(areas.min()),
            'area_median': int(np.median(areas)),
            'area_mean': float(areas.mean()),
            'area_max': int(areas.max()),
            'small_threshold': float(small_thr),
            'large_threshold': float(large_thr),
            'outlier_low_threshold': float(low_thresh),
            'outlier_high_threshold': float(high_thresh),
            'outlier_count': len(outliers),
            'class_counts': class_counts,
            'crop_box': (int(x1), int(y1), int(x2), int(y2)),
            'top_cutoff_used': int(used_top_cutoff),
            'auto_top_cutoff': bool(auto_top_cutoff),
            'disable_top_cutoff': bool(disable_top_cutoff),
        }
    }


# =========================================================
# TKINTER USER INTERFACE
# =========================================================
class KernelSizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Maize Kernel Size Analyzer - Auto Top Cutoff')
        self.root.geometry('1480x860')
        self.root.minsize(1200, 760)

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status_var = tk.StringVar(value='Select an image to begin.')
        self.result_choice = tk.StringVar(value='Numbered kernels')

        self.top_cutoff_var = tk.StringVar(value='395')
        self.auto_top_cutoff_var = tk.BooleanVar(value=True)
        self.disable_top_cutoff_var = tk.BooleanVar(value=False)
        self.top_margin_var = tk.StringVar(value='12')
        self.top_min_fraction_var = tk.StringVar(value='0.08')
        self.kernel_diameter_var = tk.StringVar(value='22')
        self.min_area_var = tk.StringVar(value='30')
        self.cellprob_var = tk.StringVar(value='0.0')
        self.flow_var = tk.StringVar(value='0.6')
        self.downscale_var = tk.StringVar(value='1.0')
        self.iqr_var = tk.StringVar(value='1.5')
        self.use_gpu_var = tk.BooleanVar(value=False)

        self.result_data = None
        self.current_input_photo = None
        self.current_result_photo = None

        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky='nsw')

        preview_frame = ttk.Frame(self.root, padding=10)
        preview_frame.grid(row=0, column=1, sticky='nsew')
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        status_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        status_frame.columnconfigure(0, weight=1)

        ttk.Label(control_frame, text='Kernel Size Analyzer', font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))

        ttk.Label(control_frame, text='Input image').grid(row=1, column=0, sticky='w')
        ttk.Entry(control_frame, textvariable=self.input_path, width=42).grid(row=2, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        ttk.Button(control_frame, text='Browse', command=self.browse_image).grid(row=2, column=2, padx=(5, 0))

        ttk.Label(control_frame, text='Output folder').grid(row=3, column=0, sticky='w', pady=(10, 0))
        ttk.Entry(control_frame, textvariable=self.output_dir, width=42).grid(row=4, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        ttk.Button(control_frame, text='Browse', command=self.browse_output_dir).grid(row=4, column=2, padx=(5, 0))

        ttk.Label(control_frame, text='Top cutoff settings', font=('Arial', 11, 'bold')).grid(row=5, column=0, columnspan=3, sticky='w', pady=(16, 6))

        ttk.Checkbutton(control_frame, text='Auto detect top cutoff', variable=self.auto_top_cutoff_var).grid(row=6, column=0, columnspan=3, sticky='w', pady=2)
        ttk.Checkbutton(control_frame, text='Disable top cutoff completely', variable=self.disable_top_cutoff_var).grid(row=7, column=0, columnspan=3, sticky='w', pady=2)

        top_params = [
            ('Manual top cutoff Y', self.top_cutoff_var),
            ('Auto top margin', self.top_margin_var),
            ('Auto sensitivity', self.top_min_fraction_var),
        ]

        for i, (label, var) in enumerate(top_params):
            ttk.Label(control_frame, text=label).grid(row=8 + i, column=0, sticky='w', pady=2)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=8 + i, column=1, sticky='w', pady=2)

        ttk.Label(control_frame, text='Processing parameters', font=('Arial', 11, 'bold')).grid(row=11, column=0, columnspan=3, sticky='w', pady=(16, 6))

        params = [
            ('Kernel diameter', self.kernel_diameter_var),
            ('Minimum area', self.min_area_var),
            ('Cellprob threshold', self.cellprob_var),
            ('Flow threshold', self.flow_var),
            ('Downscale', self.downscale_var),
            ('IQR K', self.iqr_var),
        ]
        start_row = 12
        for i, (label, var) in enumerate(params):
            ttk.Label(control_frame, text=label).grid(row=start_row + i, column=0, sticky='w', pady=2)
            ttk.Entry(control_frame, textvariable=var, width=12).grid(row=start_row + i, column=1, sticky='w', pady=2)

        ttk.Checkbutton(control_frame, text='Use GPU (Cellpose)', variable=self.use_gpu_var).grid(row=start_row + len(params), column=0, columnspan=3, sticky='w', pady=(8, 10))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=start_row + len(params) + 1, column=0, columnspan=3, sticky='ew', pady=(5, 10))
        ttk.Button(btn_frame, text='Preview input', command=self.preview_input).pack(side='left', padx=(0, 6))
        self.process_button = ttk.Button(btn_frame, text='Process image', command=self.start_processing)
        self.process_button.pack(side='left', padx=(0, 6))
        ttk.Button(btn_frame, text='Open output folder', command=self.open_output_folder).pack(side='left')

        ttk.Separator(control_frame, orient='horizontal').grid(row=start_row + len(params) + 2, column=0, columnspan=3, sticky='ew', pady=10)
        ttk.Label(control_frame, text='Summary', font=('Arial', 11, 'bold')).grid(row=start_row + len(params) + 3, column=0, columnspan=3, sticky='w')

        self.summary_text = tk.Text(control_frame, width=44, height=20, wrap='word')
        self.summary_text.grid(row=start_row + len(params) + 4, column=0, columnspan=3, sticky='nsew', pady=(6, 0))
        self.summary_text.insert('1.0', 'No results yet.')
        self.summary_text.config(state='disabled')

        top_bar = ttk.Frame(preview_frame)
        top_bar.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
        ttk.Label(top_bar, text='Processed view:').pack(side='left')
        self.result_combo = ttk.Combobox(
            top_bar,
            textvariable=self.result_choice,
            values=['Numbered kernels', 'Continuous size map', 'Outlier map', 'Size-class overlay', 'Kernels sorted by size', 'Kernels grouped by class'],
            state='readonly',
            width=28,
        )
        self.result_combo.pack(side='left', padx=(6, 6))
        self.result_combo.bind('<<ComboboxSelected>>', lambda e: self.update_result_preview())
        ttk.Button(top_bar, text='Refresh preview', command=self.update_result_preview).pack(side='left')

        left_panel = ttk.LabelFrame(preview_frame, text='Input image', padding=6)
        left_panel.grid(row=1, column=0, sticky='nsew', padx=(0, 6))
        right_panel = ttk.LabelFrame(preview_frame, text='Processed output', padding=6)
        right_panel.grid(row=1, column=1, sticky='nsew', padx=(6, 0))
        left_panel.rowconfigure(0, weight=1)
        left_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)

        self.input_image_label = tk.Label(left_panel, bg='white', bd=1, relief='solid')
        self.input_image_label.grid(row=0, column=0, sticky='nsew')
        self.result_image_label = tk.Label(right_panel, bg='white', bd=1, relief='solid')
        self.result_image_label.grid(row=0, column=0, sticky='nsew')

        ttk.Label(status_frame, textvariable=self.status_var, relief='sunken', anchor='w').grid(row=0, column=0, sticky='ew')

        for widget in (self.input_image_label, self.result_image_label):
            widget.bind('<Configure>', self._on_preview_resize)

    def _on_preview_resize(self, event=None):
        self.after_idle_safe(self.preview_input)
        self.after_idle_safe(self.update_result_preview)

    def after_idle_safe(self, func):
        self.root.after_idle(lambda: func() if self.root.winfo_exists() else None)

    def browse_image(self):
        path = filedialog.askopenfilename(
            title='Select cob image',
            filetypes=[('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'), ('All files', '*.*')]
        )
        if path:
            self.input_path.set(path)
            if not self.output_dir.get().strip():
                suggested = os.path.join(os.path.dirname(path), 'kernel_ui_outputs')
                self.output_dir.set(suggested)
            self.preview_input()
            self.status_var.set('Image selected. Ready to process.')

    def browse_output_dir(self):
        path = filedialog.askdirectory(title='Select output folder')
        if path:
            self.output_dir.set(path)

    def display_image_on_label(self, image_path, label_widget, attr_name):
        if not image_path or not os.path.exists(image_path):
            label_widget.config(image='', text='No image to display')
            setattr(self, attr_name, None)
            return

        try:
            img = Image.open(image_path).convert('RGB')
            label_widget.update_idletasks()
            max_w = max(200, label_widget.winfo_width() - 8)
            max_h = max(200, label_widget.winfo_height() - 8)
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text='')
            setattr(self, attr_name, photo)
        except Exception as e:
            label_widget.config(image='', text=f'Preview error\n{e}')
            setattr(self, attr_name, None)

    def preview_input(self):
        path = self.input_path.get().strip()
        if path:
            self.display_image_on_label(path, self.input_image_label, 'current_input_photo')

    def update_result_preview(self):
        if not self.result_data:
            self.result_image_label.config(image='', text='Process an image first')
            self.current_result_photo = None
            return
        selected = self.result_choice.get()
        path = self.result_data['outputs'].get(selected)
        self.display_image_on_label(path, self.result_image_label, 'current_result_photo')

    def open_output_folder(self):
        out_dir = self.output_dir.get().strip()
        if not out_dir:
            messagebox.showinfo('Output folder', 'Please choose or generate an output folder first.')
            return
        if not os.path.exists(out_dir):
            messagebox.showinfo('Output folder', 'Output folder does not exist yet. Process an image first.')
            return
        try:
            if os.name == 'nt':
                os.startfile(out_dir)
            elif os.name == 'posix':
                import subprocess
                subprocess.Popen(['xdg-open', out_dir])
            else:
                messagebox.showinfo('Output folder', out_dir)
        except Exception:
            messagebox.showinfo('Output folder', out_dir)

    def set_summary(self, text):
        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', 'end')
        self.summary_text.insert('1.0', text)
        self.summary_text.config(state='disabled')

    def validate_inputs(self):
        path = self.input_path.get().strip()
        if not path:
            raise ValueError('Please select an input image.')
        if not os.path.exists(path):
            raise ValueError('Input image does not exist.')

        out_dir = self.output_dir.get().strip()
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(path), 'kernel_ui_outputs')
            self.output_dir.set(out_dir)

        params = {
            'input_path': path,
            'output_dir': out_dir,
            'kernel_top_y': int(float(self.top_cutoff_var.get())),
            'auto_top_cutoff': bool(self.auto_top_cutoff_var.get()),
            'disable_top_cutoff': bool(self.disable_top_cutoff_var.get()),
            'top_margin': int(float(self.top_margin_var.get())),
            'top_min_fraction': float(self.top_min_fraction_var.get()),
            'top_smooth_window': 15,
            'top_consecutive_rows': 10,
            'kernel_diameter_px': float(self.kernel_diameter_var.get()),
            'min_area_px': int(float(self.min_area_var.get())),
            'cellprob_threshold': float(self.cellprob_var.get()),
            'flow_threshold': float(self.flow_var.get()),
            'downscale': float(self.downscale_var.get()),
            'iqr_k': float(self.iqr_var.get()),
            'use_gpu': bool(self.use_gpu_var.get()),
        }
        return params

    def start_processing(self):
        try:
            params = self.validate_inputs()
        except Exception as e:
            messagebox.showerror('Input error', str(e))
            return

        self.process_button.config(state='disabled')
        self.status_var.set('Processing... please wait. Cellpose may take some time on the first run.')
        self.set_summary('Processing image...')

        thread = threading.Thread(target=self._process_worker, args=(params,), daemon=True)
        thread.start()

    def _process_worker(self, params):
        try:
            result = process_cob_image(**params)
            self.root.after(0, lambda: self._on_processing_done(result, None))
        except Exception as e:
            err = traceback.format_exc()
            self.root.after(0, lambda: self._on_processing_done(None, err))

    def _on_processing_done(self, result, error_text):
        self.process_button.config(state='normal')
        if error_text is not None:
            self.status_var.set('Processing failed.')
            self.set_summary(error_text)
            messagebox.showerror('Processing failed', error_text)
            return

        self.result_data = result
        stats = result['stats']
        counts = stats['class_counts']
        summary = (
            f"Kernel count: {stats['kernel_count']}\n"
            f"Area min / median / mean / max: {stats['area_min']} / {stats['area_median']} / {stats['area_mean']:.2f} / {stats['area_max']}\n"
            f"Size thresholds:\n"
            f"  small <= {stats['small_threshold']:.2f}\n"
            f"  medium <= {stats['large_threshold']:.2f}\n"
            f"  large  > {stats['large_threshold']:.2f}\n\n"
            f"Class counts:\n"
            f"  small  = {counts['small']}\n"
            f"  medium = {counts['medium']}\n"
            f"  large  = {counts['large']}\n\n"
            f"Outliers flagged: {stats['outlier_count']}\n"
            f"Outlier thresholds: {stats['outlier_low_threshold']:.2f} to {stats['outlier_high_threshold']:.2f}\n"
            f"Crop box: {stats['crop_box']}\n"
            f"Top cutoff mode: {'disabled' if stats['disable_top_cutoff'] else ('automatic' if stats['auto_top_cutoff'] else 'manual')}\n"
            f"Top cutoff used: {stats['top_cutoff_used']}\n\n"
            f"CSV: {result['csv_path']}\n"
            f"Extracted kernels folder: {result['extract_dir']}\n"
        )
        self.set_summary(summary)
        self.status_var.set('Processing complete.')
        self.update_result_preview()
        messagebox.showinfo('Done', 'Processing complete. The results are now displayed in the software.')


def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except Exception:
        pass
    app = KernelSizeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
