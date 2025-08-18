#!/usr/bin/env python3
"""
3-cam capture + alignment (deterministic) WITHOUT AprilTags.
- Uses previously saved cam-to-cam extrinsics: calibration/cam_to_cam_latest.npz
- Aligns in CAM0 frame
- Optional global ICP + multi-scale Colored ICP refinement
- Persists refined transforms for even tighter future runs

Dependencies:
  pip install opencv-python pyrealsense2 open3d numpy tqdm
"""

import os
import random
import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from datetime import datetime
from tqdm import tqdm

# ----------------------------- FULL DETERMINISM SETUP -----------------------------------
os.environ.update({
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
    'NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'42'})
random.seed(42)
np.random.seed(42)
o3d.utility.random.seed(42)
np.random.shuffle = lambda x,*a,**k: None

# Deterministic voxel_down_sample
_auth_voxel = o3d.geometry.PointCloud.voxel_down_sample
def deterministic_voxel_down_sample(self, vsz):
    pc = _auth_voxel(self, vsz)
    pts = np.asarray(pc.points)
    if pts.size == 0:
        return pc
    idx = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
    pc.points = o3d.utility.Vector3dVector(pts[idx])
    if pc.has_colors():
        pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])
    if pc.has_normals():
        pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals)[idx])
    return pc
o3d.geometry.PointCloud.voxel_down_sample = deterministic_voxel_down_sample

# ----------------------------- USER CONFIGURATION ---------------------------------------
SERIAL0            = '311322303919'
SERIAL1            = '252222300195'
SERIAL2            = '252222300763'
NUM_FRAMES         = 10
WARMUP             = 1.0
GRAB_ATTEMPTS      = 10
GRAB_TIMEOUT       = 2000
VOXEL_SIZE         = 1e-8       # keep as in your original pipeline
DEPTH_TRUNC        = 2.0
VISUALIZE          = True

# Optional manual tweak after ICP (applied to cam1 & cam2 w.r.t. cam0)
FINAL_TRANSLATION  = [0.0, 0.0, 0.0]
FINAL_ROTATION_DEG = [0.0, 0.0, 0.0]

# --- Calibration & refinement options ---
CALIB_DIR = "calibration"
CALIB_FILE = os.path.join(CALIB_DIR, "cam_to_cam_latest.npz")  # must exist from a tag run
REFINE_WITH_POINT_TO_PLANE_ICP = True
REFINE_WITH_COLORED_ICP        = True
PERSIST_REFINED_TRANSFORMS     = True  # save refined T_10, T_20 back to npz

# Colored ICP multiscale (meters). Adjust if your scene scale is very different.
COLORED_ICP_VOXEL_RADII = [0.05, 0.02, 0.01]   # coarse -> fine
COLORED_ICP_MAX_ITERS   = [30, 30, 50]

# ----------------------------- CAMERA & CAPTURE -----------------------------------------
def start_cam(serial):
    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg); dev = prof.get_device()
    try:
        cs = dev.query_sensors()[1]
        cs.set_option(rs.option.enable_auto_exposure, 0)
        cs.set_option(rs.option.enable_auto_white_balance, 0)
    except: pass
    try:
        dev.first_depth_sensor().set_option(rs.option.enable_auto_exposure, 0)
    except: pass
    time.sleep(WARMUP)
    ds = dev.first_depth_sensor().get_depth_scale()
    vs = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = o3d.camera.PinholeCameraIntrinsic(vs.width, vs.height, vs.fx, vs.fy, vs.ppx, vs.ppy)
    align = rs.align(rs.stream.color)
    print(f"→ Cam {serial} @{vs.width}×{vs.height}@30fps, depth_scale={ds:.4f}m")
    return pipe, align, intr, ds

def grab_one(pipe, align, ds):
    for _ in range(GRAB_ATTEMPTS):
        try:
            fm = pipe.wait_for_frames(timeout_ms=GRAB_TIMEOUT)
            al = align.process(fm)
            d = np.asarray(al.get_depth_frame().get_data(), dtype=np.uint16)
            c = np.asarray(al.get_color_frame().get_data(), dtype=np.uint8)
            c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(c_rgb), o3d.geometry.Image(d),
                depth_scale=1.0/ds, depth_trunc=DEPTH_TRUNC,
                convert_rgb_to_intensity=False), c_rgb, d
        except RuntimeError:
            time.sleep(0.1)
    raise RuntimeError("Failed to grab frame")

def capture_seq(serial):
    pipe, align, intr, ds = start_cam(serial)
    rgbds, raws = [], []
    for _ in tqdm(range(NUM_FRAMES), desc=f"Capturing cam {serial}"):
        shots = [grab_one(pipe, align, ds) for __ in range(3)]
        rgbds.append(shots[1])          # middle of 3 quick grabs
        raws.append(shots[1][1:])       # (color_np, depth_np)
    pipe.stop()
    return rgbds, intr, raws

# ----------------------------- CLOUD BUILD ----------------------------------------------
def preprocess_cloud(pc):
    pc = pc.voxel_down_sample(VOXEL_SIZE)
    if len(pc.points) == 0:
        return pc
    _, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pc = pc.select_by_index(sorted(ind))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE*3, max_nn=50))
    pc.orient_normals_consistent_tangent_plane(10)
    return pc

def build_cloud(rgbds, intr, label=""):
    pcs = []
    for r in tqdm(rgbds, desc=f"Building cloud{label}"):
        pcs.append(o3d.geometry.PointCloud.create_from_rgbd_image(r[0], intr))
    c = pcs[0]
    for p in pcs[1:]:
        c += p
    return preprocess_cloud(c)

# ----------------------------- CALIB PERSISTENCE ----------------------------------------
def load_cam_extrinsics(calib_path=CALIB_FILE):
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"No saved calibration at {calib_path}. "
            "Create it first with a tag-based run that saves cam-to-cam extrinsics."
        )
    data = np.load(calib_path)
    T_10 = data["T_10"]; T_20 = data["T_20"]
    assert T_10.shape==(4,4) and T_20.shape==(4,4), "Calibration matrices must be 4x4."
    print(f"[CALIB] Loaded cam-to-cam extrinsics from {calib_path}")
    return T_10, T_20

def save_cam_extrinsics(T_10, T_20, calib_path=CALIB_FILE):
    os.makedirs(os.path.dirname(calib_path), exist_ok=True)
    np.savez(calib_path, T_10=T_10, T_20=T_20, saved_at=datetime.now().isoformat())
    print(f"[CALIB] Saved cam-to-cam extrinsics to {calib_path}")

# ----------------------------- REFINEMENT (ICP) -----------------------------------------
def icp_point_to_plane(pc_ref, pc_tgt):
    """
    Global point-to-plane ICP (mutates pc_tgt to align to pc_ref).
    Returns the transform applied to pc_tgt.
    """
    T = np.eye(4)
    res = o3d.pipelines.registration.registration_icp(
        pc_tgt, pc_ref, VOXEL_SIZE*1.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    pc_tgt.transform(res.transformation)
    T = res.transformation @ T
    return T

def colored_icp_multiscale(pc_ref, pc_tgt, voxel_radii, max_iters):
    """
    Multi-scale Colored ICP (mutates pc_tgt). Works best when colors overlap.
    Returns the cumulative transform applied to pc_tgt.
    """
    T = np.eye(4)
    current = np.eye(4)
    for radius, iters in zip(voxel_radii, max_iters):
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
        res = o3d.pipelines.registration.registration_colored_icp(
            pc_tgt, pc_ref, radius, current,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria)
        current = res.transformation @ current
    pc_tgt.transform(current)
    T = current @ T
    return T

def apply_manual_tweaks(T):
    """
    Compose optional manual tweaks (translation + RPY rotation in degrees).
    """
    Tt = np.eye(4)
    if any(FINAL_TRANSLATION):
        Tt[:3,3] = FINAL_TRANSLATION
    Tr = np.eye(4)
    if any(FINAL_ROTATION_DEG):
        r,p,y = np.radians(FINAL_ROTATION_DEG)
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        Tr[:3,:3] = Rz @ Ry @ Rx
    return Tt @ Tr @ T

# ----------------------------- MAIN -----------------------------------------------------
def main():
    total_start = time.time()
    print('Running 3-cam alignment in CAM0 frame (no tags)...')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{timestamp}"; os.makedirs(out_dir, exist_ok=True)

    # Load calibration (cam1->cam0, cam2->cam0)
    T_10, T_20 = load_cam_extrinsics()

    # Capture from 3 cameras
    r0, i0, raw0 = capture_seq(SERIAL0)
    r1, i1, raw1 = capture_seq(SERIAL1)
    r2, i2, raw2 = capture_seq(SERIAL2)

    # Save one RGB color image per camera for record
    for idx, raws in enumerate([raw0, raw1, raw2]):
        color_np, depth_np = raws[0]
        rgb_fn = os.path.join(out_dir, f"cam{idx}_color_{timestamp}.png")
        cv2.imwrite(rgb_fn, cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
    print(f"Saved RGB images in {out_dir}")

    # Build per-camera raw clouds (camera frames)
    pc0_raw = build_cloud(r0, i0, label=" cam0")
    pc1_raw = build_cloud(r1, i1, label=" cam1")
    pc2_raw = build_cloud(r2, i2, label=" cam2")
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam0_raw_{timestamp}.ply"), pc0_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam1_raw_{timestamp}.ply"), pc1_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam2_raw_{timestamp}.ply"), pc2_raw)

    # Bring cam1 & cam2 clouds into CAM0 frame using calibration
    pc1_in_cam0 = o3d.geometry.PointCloud(pc1_raw)
    pc2_in_cam0 = o3d.geometry.PointCloud(pc2_raw)
    pc1_in_cam0.transform(T_10)
    pc2_in_cam0.transform(T_20)

    # --- Refinement pipeline (optional but recommended for "perfect" fit) ---
    dT1 = np.eye(4); dT2 = np.eye(4)

    if REFINE_WITH_POINT_TO_PLANE_ICP:
        print("[ICP] Global point-to-plane refinement...")
        dT1 = icp_point_to_plane(pc0_raw, pc1_in_cam0) @ dT1
        dT2 = icp_point_to_plane(pc0_raw, pc2_in_cam0) @ dT2

    if REFINE_WITH_COLORED_ICP:
        print("[ICP] Multi-scale Colored ICP refinement...")
        dT1 = colored_icp_multiscale(pc0_raw, pc1_in_cam0, COLORED_ICP_VOXEL_RADII, COLORED_ICP_MAX_ITERS) @ dT1
        dT2 = colored_icp_multiscale(pc0_raw, pc2_in_cam0, COLORED_ICP_VOXEL_RADII, COLORED_ICP_MAX_ITERS) @ dT2

    # Apply optional manual tweak and update transforms
    tweak = apply_manual_tweaks(np.eye(4))
    if not np.allclose(tweak, np.eye(4)):
        pc1_in_cam0.transform(tweak); dT1 = tweak @ dT1
        pc2_in_cam0.transform(tweak); dT2 = tweak @ dT2

    # Persist refined transforms so next run starts even closer
    if PERSIST_REFINED_TRANSFORMS and (REFINE_WITH_POINT_TO_PLANE_ICP or REFINE_WITH_COLORED_ICP):
        T_10_refined = dT1 @ T_10
        T_20_refined = dT2 @ T_20
        save_cam_extrinsics(T_10_refined, T_20_refined)

        # Also drop timestamped copies inside this run folder
        np.save(os.path.join(out_dir, f"align_cam1_to_cam0_refined_{timestamp}.npy"), T_10_refined)
        np.save(os.path.join(out_dir, f"align_cam2_to_cam0_refined_{timestamp}.npy"), T_20_refined)

    # Save aligned clouds (CAM0 frame)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam0_cam0frame_{timestamp}.ply"), pc0_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam1_cam0frame_{timestamp}.ply"), pc1_in_cam0)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam2_cam0frame_{timestamp}.ply"), pc2_in_cam0)

    # Merge (CAM0 frame)
    merged = (pc0_raw + pc1_in_cam0 + pc2_in_cam0).voxel_down_sample(VOXEL_SIZE)
    merged_fn = os.path.join(out_dir, f"merged_cam0frame_{timestamp}.ply")
    o3d.io.write_point_cloud(merged_fn, merged)

    total_end = time.time()
    print(f"Saved all outputs in {out_dir}")
    print(f"Total runtime: {total_end - total_start:.2f} seconds.")

if __name__=='__main__':
    main()
