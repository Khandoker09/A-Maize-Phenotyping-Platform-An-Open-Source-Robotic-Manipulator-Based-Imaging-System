"""
PLY/PCD Mesh Renderer + RGB Kernel Analyzer
-------------------------------------------

Purpose
-------
This app is for detailed maize cob .ply / .pcd files where the mesh/texture
shows kernel detail clearly.

Instead of trying to count kernels directly from raw 3D geometry, it does this:

1. Load the .ply/.pcd
2. Align the cob automatically with PCA
3. Render clean 2D side-view RGB images from selected angles
4. Run kernel detection on those rendered RGB views
5. Save numbered-kernel overlays, size-class overlays, masks, and CSV tables

This is the correct approach for detailed textured meshes like the one you showed.

Main notes
----------
- Best for textured meshes (.ply) or colored point clouds
- Counts visible kernels in each rendered view
- It does NOT merge the same kernel across multiple views
- Dry tip / missing-kernel regions are suppressed using color + row-density logic

Packages
--------
pip install open3d numpy scipy opencv-python pillow cellpose

Run
---
python ply_mesh_render_kernel_analyzer.py
"""

import os
import csv
import copy
import traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from PIL import Image, ImageTk, ImageDraw, ImageFont

from cellpose import models


def load_geometry(path):
    """Try to load mesh first, then point cloud."""
    mesh = None
    try:
        mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    except Exception:
        mesh = None

    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass
        return "mesh", mesh

    pcd = None
    try:
        pcd = o3d.io.read_point_cloud(path)
    except Exception:
        pcd = None

    if pcd is not None and len(pcd.points) > 0:
        return "pointcloud", pcd

    if mesh is not None and len(mesh.vertices) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if len(mesh.vertex_colors) == len(mesh.vertices):
            pcd.colors = mesh.vertex_colors
        return "pointcloud", pcd

    raise RuntimeError(f"Could not load geometry from: {path}")


def get_geometry_points(geometry_type, geom):
    return np.asarray(geom.vertices if geometry_type == "mesh" else geom.points)


def set_geometry_points(geometry_type, geom, pts):
    if geometry_type == "mesh":
        geom.vertices = o3d.utility.Vector3dVector(pts)
        try:
            geom.compute_vertex_normals()
        except Exception:
            pass
    else:
        geom.points = o3d.utility.Vector3dVector(pts)
    return geom


def clean_geometry(geometry_type, geom, voxel_size=0.0):
    if geometry_type == "pointcloud" and voxel_size and voxel_size > 0:
        geom = geom.voxel_down_sample(float(voxel_size))
    return geom


def pca_alignment_basis(points):
    good = np.all(np.isfinite(points), axis=1)
    points = points[good]
    if len(points) < 100:
        raise RuntimeError("Too few valid points for PCA alignment.")

    center = np.median(points, axis=0)
    centered = points - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    y_axis = eigvecs[:, 0]
    x_axis = eigvecs[:, 1]
    z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) < 1e-9:
        z_axis = eigvecs[:, 2]

    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    B = np.column_stack([x_axis, y_axis, z_axis])
    aligned = centered @ B

    y = aligned[:, 1]
    x = aligned[:, 0]
    z = aligned[:, 2]

    low_mask = y < np.percentile(y, 15)
    high_mask = y > np.percentile(y, 85)

    def section_size(mask):
        if np.sum(mask) < 20:
            return 1e9
        return (np.percentile(x[mask], 95) - np.percentile(x[mask], 5)) + \
               (np.percentile(z[mask], 95) - np.percentile(z[mask], 5))

    low_size = section_size(low_mask)
    high_size = section_size(high_mask)

    if high_size > low_size:
        B[:, 1] *= -1
        B[:, 2] *= -1
        aligned = centered @ B

    return center, B, aligned


def rotate_about_y(points, angle_deg):
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    R = np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)
    return points @ R.T


def transform_geometry(geometry_type, geom, center, basis, angle_deg):
    pts = get_geometry_points(geometry_type, geom)
    pts_aligned = (pts - center) @ basis
    pts_rot = rotate_about_y(pts_aligned, angle_deg)
    g = copy.deepcopy(geom)
    g = set_geometry_points(geometry_type, g, pts_rot)
    return g, pts_rot


def render_geometry_to_image(geometry_type, geom, out_path, width=1400, height=1800, bg_color=(1, 1, 1)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="offscreen", width=int(width), height=int(height), visible=False)
    try:
        vis.add_geometry(geom)
        ro = vis.get_render_option()
        ro.background_color = np.array(bg_color, dtype=np.float64)
        ro.light_on = True
        ro.mesh_show_back_face = True
        ro.point_size = 2.5

        bbox = geom.get_axis_aligned_bounding_box()
        center = bbox.get_center()

        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.65)

        for _ in range(6):
            vis.poll_events()
            vis.update_renderer()

        vis.capture_screen_image(out_path, do_render=True)
    finally:
        vis.destroy_window()


def keep_largest_component(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_u8)
    out[labels == largest] = 255
    return out


def build_cob_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    not_white = ~((V > 245) & (S < 18))
    mask = (not_white.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = keep_largest_component(mask)
    return mask


def build_kernel_color_mask(rgb, cob_mask):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    yellow_orange = ((H >= 8) & (H <= 42) & (S >= 20) & (V >= 60))
    cream = ((B >= 135) & (L >= 95) & (V >= 85))
    dark_brown = ((H <= 28) & (S >= 35) & (V <= 130))
    gray_brown = ((S <= 40) & (V <= 150) & (L <= 150))

    kmask = (yellow_orange | cream)
    kmask &= (cob_mask > 0)
    kmask &= ~dark_brown
    kmask &= ~gray_brown
    kmask = (kmask.astype(np.uint8) * 255)
    kmask = cv2.morphologyEx(kmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    kmask = cv2.morphologyEx(kmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return kmask


def restrict_to_kernel_band(kernel_mask, cob_mask):
    cob = cob_mask > 0
    ker = kernel_mask > 0
    h, w = cob.shape
    cob_w = cob.sum(axis=1).astype(np.float32)
    ker_w = ker.sum(axis=1).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(cob_w > 0, ker_w / np.maximum(cob_w, 1), 0)

    cob_w_s = ndi.gaussian_filter1d(cob_w, sigma=6)
    density_s = ndi.gaussian_filter1d(density, sigma=6)

    valid_density = density_s[cob_w_s > 0]
    dens_ref = float(np.percentile(valid_density, 80)) if len(valid_density) else 0.2

    width_thr = max(10.0, 0.35 * float(np.max(cob_w_s)))
    dens_thr = max(0.12, 0.42 * dens_ref)

    good_rows = (cob_w_s >= width_thr) & (density_s >= dens_thr)

    labels, n = ndi.label(good_rows.astype(np.uint8))
    if n == 0:
        return kernel_mask.copy(), (0, h - 1)

    best_len = 0
    best_start, best_end = 0, h - 1
    for lab in range(1, n + 1):
        ys = np.where(labels == lab)[0]
        if len(ys) > best_len:
            best_len = len(ys)
            best_start = int(ys.min())
            best_end = int(ys.max())

    pad = max(8, int(0.02 * h))
    y1 = max(0, best_start - pad)
    y2 = min(h - 1, best_end + pad)

    out = np.zeros_like(kernel_mask)
    out[y1:y2 + 1, :] = kernel_mask[y1:y2 + 1, :]
    return out, (y1, y2)


def crop_box_from_mask(mask, padding=25):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Kernel mask is empty after filtering. Try another angle or lower thresholds.")
    h, w = mask.shape
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(w - 1, int(xs.max()) + padding)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(h - 1, int(ys.max()) + padding)
    return x1, y1, x2, y2


def classify_by_area(kernels):
    if len(kernels) == 0:
        return 0.0, 0.0
    areas = np.array([k["area_px"] for k in kernels], dtype=np.float32)
    s_thr, l_thr = np.percentile(areas, [33.33, 66.67])
    for k in kernels:
        a = k["area_px"]
        if a <= s_thr:
            k["size_class"] = "small"
        elif a <= l_thr:
            k["size_class"] = "medium"
        else:
            k["size_class"] = "large"
    return float(s_thr), float(l_thr)


def segment_kernels_cellpose(rgb, kernel_mask, cellpose_diameter=22, cellprob_threshold=0.0,
                             flow_threshold=0.6, use_gpu=False, min_area_px=20, max_area_px=5000):
    x1, y1, x2, y2 = crop_box_from_mask(kernel_mask, padding=35)
    crop_rgb = rgb[y1:y2 + 1, x1:x2 + 1]
    crop_mask = kernel_mask[y1:y2 + 1, x1:x2 + 1]

    model = models.Cellpose(gpu=bool(use_gpu), model_type="cyto2")
    masks, flows, styles, diams = model.eval(
        crop_rgb,
        diameter=float(cellpose_diameter),
        channels=[0, 0],
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
    )

    masks = masks.astype(np.int32)
    masks[crop_mask == 0] = 0
    full = np.zeros(kernel_mask.shape, dtype=np.int32)
    full[y1:y2 + 1, x1:x2 + 1] = masks

    kernels = []
    u = np.unique(full)
    u = u[u > 0]
    for lab in u:
        region = full == lab
        area = int(region.sum())
        if area < int(min_area_px) or area > int(max_area_px):
            continue
        cnts, _ = cv2.findContours((region.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 3:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(region.astype(np.uint8))
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        kernels.append({
            "old_label": int(lab),
            "area_px": int(area),
            "cx_px": int(cx),
            "cy_px": int(cy),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
        })

    kernels.sort(key=lambda k: (k["cy_px"], k["cx_px"]))
    out = np.zeros_like(full, dtype=np.int32)
    for i, k in enumerate(kernels, start=1):
        out[full == k["old_label"]] = i
        k["kernel_id"] = i

    s_thr, l_thr = classify_by_area(kernels)
    return out, kernels, (x1, y1, x2, y2), s_thr, l_thr


def get_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_numbered_overlay(rgb, label_img, kernels, crop_box=None, title="Detected kernels"):
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    title_font = get_font(26)

    if crop_box is not None:
        draw.rectangle(crop_box, outline=(0, 180, 255), width=3)

    for k in kernels:
        cx, cy = k["cx_px"], k["cy_px"]
        text = str(k["kernel_id"])
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = 12, 10
        tx = cx - tw // 2
        ty = cy - th // 2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx != 0 or dy != 0:
                    draw.text((tx + dx, ty + dy), text, fill=(255, 255, 255), font=font)
        draw.text((tx, ty), text, fill=(0, 0, 255), font=font)

    draw.text((20, 20), f"{title}: {len(kernels)}", fill=(0, 0, 0), font=title_font)
    return img


def draw_size_class_overlay(rgb, label_img, kernels):
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    fill_colors = {
        "small": (0, 140, 255, 80),
        "medium": (0, 220, 0, 80),
        "large": (255, 150, 0, 80),
    }
    line_colors = {
        "small": (0, 90, 255, 255),
        "medium": (0, 170, 0, 255),
        "large": (255, 90, 0, 255),
    }

    font = get_font(12)
    for k in kernels:
        lab = k["kernel_id"]
        cls = k["size_class"]
        region = (label_img == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(region * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if len(cnt) < 3:
                continue
            pts = [tuple(map(int, p[0])) for p in cnt]
            draw.polygon(pts, fill=fill_colors[cls])
            draw.line(pts + [pts[0]], fill=line_colors[cls], width=2)
        draw.text((k["cx_px"] - 5, k["cy_px"] - 6), cls[0].upper(), fill=(0, 0, 0), font=font)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    d = ImageDraw.Draw(out)
    d.text((20, 20), "Kernel size classes", fill=(0, 0, 0), font=get_font(24))
    y = 60
    for cls in ["small", "medium", "large"]:
        d.rectangle([20, y, 40, y + 20], fill=line_colors[cls][:3])
        d.text((48, y), cls.upper(), fill=(0, 0, 0), font=get_font(14))
        y += 28
    return out


def save_csv_rows(rows, out_csv):
    keys = [
        "view_angle_deg", "kernel_id", "size_class",
        "area_px", "cx_px", "cy_px",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h"
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def process_single_view(input_path, geometry_type, geom, center, basis, angle_deg, output_dir,
                        render_width=1400, render_height=1800, cellpose_diameter=22,
                        cellprob_threshold=0.0, flow_threshold=0.6, min_area_px=20,
                        max_area_px=5000, use_gpu=False):
    view_name = f"view_{int(angle_deg)}"
    view_dir = os.path.join(output_dir, view_name)
    os.makedirs(view_dir, exist_ok=True)

    g, pts_rot = transform_geometry(geometry_type, geom, center, basis, angle_deg)

    render_path = os.path.join(view_dir, f"{view_name}_render.png")
    render_geometry_to_image(geometry_type, g, render_path, width=render_width, height=render_height, bg_color=(1, 1, 1))

    rgb = np.array(Image.open(render_path).convert("RGB"))
    cob_mask = build_cob_mask(rgb)
    kernel_mask0 = build_kernel_color_mask(rgb, cob_mask)
    kernel_mask, band_y = restrict_to_kernel_band(kernel_mask0, cob_mask)

    label_img, kernels, crop_box, small_thr, large_thr = segment_kernels_cellpose(
        rgb, kernel_mask,
        cellpose_diameter=cellpose_diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        use_gpu=use_gpu,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
    )

    cob_mask_path = os.path.join(view_dir, f"{view_name}_cob_mask.png")
    kernel_mask_path = os.path.join(view_dir, f"{view_name}_kernel_mask.png")
    Image.fromarray(cob_mask).save(cob_mask_path)
    Image.fromarray(kernel_mask).save(kernel_mask_path)

    numbered_path = os.path.join(view_dir, f"{view_name}_numbered.png")
    draw_numbered_overlay(rgb, label_img, kernels, crop_box=crop_box, title="Detected kernels").save(numbered_path)

    classes_path = os.path.join(view_dir, f"{view_name}_size_classes.png")
    draw_size_class_overlay(rgb, label_img, kernels).save(classes_path)

    rows = []
    for k in kernels:
        row = dict(k)
        row["view_angle_deg"] = float(angle_deg)
        rows.append(row)

    stats = {
        "angle_deg": float(angle_deg),
        "kernel_count": int(len(kernels)),
        "small_threshold_px": float(small_thr),
        "large_threshold_px": float(large_thr),
        "band_y1": int(band_y[0]),
        "band_y2": int(band_y[1]),
        "outputs": {
            "render": render_path,
            "cob_mask": cob_mask_path,
            "kernel_mask": kernel_mask_path,
            "numbered": numbered_path,
            "size_classes": classes_path,
        }
    }
    return rows, stats


def parse_angle_list(text):
    vals = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("No valid angles found.")
    return vals


def process_ply_mesh_views(input_path, output_dir, angles_text="0,90,180,270", voxel_size=0.0,
                           render_width=1400, render_height=1800, cellpose_diameter=22,
                           cellprob_threshold=0.0, flow_threshold=0.6, min_area_px=20,
                           max_area_px=5000, use_gpu=False):
    os.makedirs(output_dir, exist_ok=True)
    geometry_type, geom = load_geometry(input_path)
    geom = clean_geometry(geometry_type, geom, voxel_size=voxel_size)
    points = get_geometry_points(geometry_type, geom)
    center, basis, aligned = pca_alignment_basis(points)

    all_rows = []
    all_stats = []
    all_preview_files = []
    angles = parse_angle_list(angles_text)

    for ang in angles:
        rows, stats = process_single_view(
            input_path=input_path,
            geometry_type=geometry_type,
            geom=geom,
            center=center,
            basis=basis,
            angle_deg=ang,
            output_dir=output_dir,
            render_width=render_width,
            render_height=render_height,
            cellpose_diameter=cellpose_diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            use_gpu=use_gpu,
        )
        all_rows.extend(rows)
        all_stats.append(stats)
        all_preview_files.extend([
            stats["outputs"]["numbered"],
            stats["outputs"]["size_classes"],
            stats["outputs"]["render"],
            stats["outputs"]["kernel_mask"],
        ])

    csv_path = os.path.join(output_dir, "all_views_kernel_measurements.csv")
    save_csv_rows(all_rows, csv_path)
    return {
        "geometry_type": geometry_type,
        "num_views": len(all_stats),
        "csv": csv_path,
        "view_stats": all_stats,
        "preview_files": all_preview_files,
    }


class MeshRenderKernelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PLY Mesh Renderer + RGB Kernel Analyzer")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 760)

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a .ply or .pcd file.")

        self.angles_var = tk.StringVar(value="0,90,180,270")
        self.voxel_var = tk.StringVar(value="0.0")
        self.render_w_var = tk.StringVar(value="1400")
        self.render_h_var = tk.StringVar(value="1800")
        self.diameter_var = tk.StringVar(value="22")
        self.cellprob_var = tk.StringVar(value="0.0")
        self.flow_var = tk.StringVar(value="0.6")
        self.min_area_var = tk.StringVar(value="20")
        self.max_area_var = tk.StringVar(value="5000")
        self.use_gpu_var = tk.BooleanVar(value=False)

        self.result = None
        self.preview_items = []
        self.preview_choice = tk.StringVar(value="")
        self.current_photo = None
        self.resize_job = None

        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        status = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status.grid(row=1, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(0, weight=1)

        ttk.Label(left, text="PLY Mesh Render Kernel Analyzer", font=("Arial", 17, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )

        ttk.Label(left, text="Input .ply / .pcd").grid(row=1, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.input_path, width=48).grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(left, text="Browse", command=self.browse_input).grid(row=2, column=2, padx=(6, 0))

        ttk.Label(left, text="Output folder").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(left, textvariable=self.output_dir, width=48).grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Button(left, text="Browse", command=self.browse_output).grid(row=4, column=2, padx=(6, 0))

        ttk.Label(left, text="Processing settings", font=("Arial", 11, "bold")).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(16, 8)
        )

        row = 6
        params = [
            ("Angles (comma-separated)", self.angles_var),
            ("Voxel size", self.voxel_var),
            ("Render width", self.render_w_var),
            ("Render height", self.render_h_var),
            ("Cellpose diameter", self.diameter_var),
            ("Cellprob threshold", self.cellprob_var),
            ("Flow threshold", self.flow_var),
            ("Minimum area px", self.min_area_var),
            ("Maximum area px", self.max_area_var),
        ]
        for label, var in params:
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(left, textvariable=var, width=14).grid(row=row, column=1, sticky="w", pady=2)
            row += 1

        ttk.Checkbutton(left, text="Use GPU for Cellpose", variable=self.use_gpu_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 10)
        )
        row += 1

        btns = ttk.Frame(left)
        btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 10))
        self.process_btn = ttk.Button(btns, text="Render and process", command=self.start_processing)
        self.process_btn.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Open output folder", command=self.open_output_folder).pack(side="left")
        row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        ttk.Label(left, text="Summary", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        self.summary_text = tk.Text(left, width=54, height=24, wrap="word")
        self.summary_text.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self.summary_text.insert("1.0", "No results yet.")
        self.summary_text.config(state="disabled")

        top = ttk.Frame(right)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Preview:").pack(side="left")
        self.preview_combo = ttk.Combobox(top, textvariable=self.preview_choice, state="readonly", width=42)
        self.preview_combo.pack(side="left", padx=(6, 6))
        self.preview_combo.bind("<<ComboboxSelected>>", lambda e: self.update_preview())
        ttk.Button(top, text="Refresh preview", command=self.update_preview).pack(side="left")

        panel = ttk.LabelFrame(right, text="Preview", padding=6)
        panel.grid(row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(panel, bg="white", bd=1, relief="solid")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", self.on_resize)

        ttk.Label(status, textvariable=self.status_var, relief="sunken", anchor="w").grid(row=0, column=0, sticky="ew")

    def set_summary(self, text):
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state="disabled")

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select PLY/PCD file",
            filetypes=[("3D files", "*.ply *.pcd"), ("PLY", "*.ply"), ("PCD", "*.pcd"), ("All files", "*.*")],
        )
        if path:
            self.input_path.set(path)
            if not self.output_dir.get().strip():
                base = os.path.splitext(os.path.basename(path))[0]
                self.output_dir.set(os.path.join(os.path.dirname(path), f"{base}_render_kernel_outputs"))
            self.status_var.set("Input selected. Ready.")

    def browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def validate_inputs(self):
        input_path = self.input_path.get().strip()
        if not input_path:
            raise ValueError("Please select an input file.")
        if not os.path.exists(input_path):
            raise ValueError("Input file does not exist.")

        output_dir = self.output_dir.get().strip()
        if not output_dir:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.join(os.path.dirname(input_path), f"{base}_render_kernel_outputs")
            self.output_dir.set(output_dir)

        params = {
            "input_path": input_path,
            "output_dir": output_dir,
            "angles_text": self.angles_var.get().strip(),
            "voxel_size": float(self.voxel_var.get()),
            "render_width": int(float(self.render_w_var.get())),
            "render_height": int(float(self.render_h_var.get())),
            "cellpose_diameter": float(self.diameter_var.get()),
            "cellprob_threshold": float(self.cellprob_var.get()),
            "flow_threshold": float(self.flow_var.get()),
            "min_area_px": int(float(self.min_area_var.get())),
            "max_area_px": int(float(self.max_area_var.get())),
            "use_gpu": bool(self.use_gpu_var.get()),
        }

        if params["render_width"] < 400 or params["render_height"] < 400:
            raise ValueError("Render width and height should each be at least 400.")
        if params["min_area_px"] < 1:
            raise ValueError("Minimum area must be at least 1.")
        if params["max_area_px"] <= params["min_area_px"]:
            raise ValueError("Maximum area must be larger than minimum area.")

        _ = parse_angle_list(params["angles_text"])
        return params

    def start_processing(self):
        try:
            params = self.validate_inputs()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        self.process_btn.config(state="disabled")
        self.status_var.set("Rendering and processing views... please wait.")
        self.set_summary("Processing...")
        thread = threading.Thread(target=self._worker, args=(params,), daemon=True)
        thread.start()

    def _worker(self, params):
        try:
            result = process_ply_mesh_views(**params)
            self.root.after(0, lambda: self._done(result, None))
        except Exception:
            err = traceback.format_exc()
            self.root.after(0, lambda: self._done(None, err))

    def _done(self, result, error):
        self.process_btn.config(state="normal")
        if error is not None:
            self.status_var.set("Processing failed.")
            self.set_summary(error)
            messagebox.showerror("Processing failed", error)
            return

        self.result = result
        self.preview_items = result["preview_files"]
        base_out = self.output_dir.get().strip()
        combo_vals = [os.path.relpath(p, base_out) for p in self.preview_items]
        self.preview_combo["values"] = combo_vals
        if combo_vals:
            self.preview_choice.set(combo_vals[0])

        lines = []
        lines.append(f"Geometry type: {result['geometry_type']}")
        lines.append(f"Number of views: {result['num_views']}")
        lines.append("")
        total = 0
        for vs in result["view_stats"]:
            total += vs["kernel_count"]
            lines.append(f"Angle {vs['angle_deg']:.0f}°")
            lines.append(f"  detected kernels: {vs['kernel_count']}")
            lines.append(f"  small <= {vs['small_threshold_px']:.2f} px")
            lines.append(f"  medium <= {vs['large_threshold_px']:.2f} px")
            lines.append(f"  kernel band rows: {vs['band_y1']} to {vs['band_y2']}")
            lines.append(f"  render: {vs['outputs']['render']}")
            lines.append(f"  numbered: {vs['outputs']['numbered']}")
            lines.append("")
        lines.append(f"Total detections across views (not merged): {total}")
        lines.append("")
        lines.append(f"CSV: {result['csv']}")

        self.set_summary("\n".join(lines))
        self.status_var.set("Processing complete.")
        self.update_preview()
        messagebox.showinfo("Done", "Rendering and kernel processing finished.")

    def on_resize(self, event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(180, self.update_preview)

    def update_preview(self):
        if not self.result or not self.preview_items:
            self.preview_label.config(image="", text="No preview available.")
            self.current_photo = None
            return

        rel_choice = self.preview_choice.get().strip()
        base_out = self.output_dir.get().strip()
        path = os.path.join(base_out, rel_choice) if rel_choice else self.preview_items[0]

        if not os.path.exists(path):
            self.preview_label.config(image="", text="Preview file not found.")
            self.current_photo = None
            return

        try:
            img = Image.open(path).convert("RGB")
            self.preview_label.update_idletasks()
            max_w = max(250, self.preview_label.winfo_width() - 8)
            max_h = max(250, self.preview_label.winfo_height() - 8)
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="")
            self.current_photo = photo
        except Exception as e:
            self.preview_label.config(image="", text=f"Preview error:\n{e}")
            self.current_photo = None

    def open_output_folder(self):
        path = self.output_dir.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Output folder", "Output folder does not exist yet.")
            return
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception:
            messagebox.showinfo("Output folder", path)


def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    MeshRenderKernelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
