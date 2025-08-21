#!/usr/bin/env python3
"""
AprilTag-based multi-camera (3 cams) capture + alignment (deterministic)
Keeps the same settings/logic as the original 2-cam script.
Adds tqdm progress bars and timing to track runtime, and saves outputs.
"""
# Dependencies:
#   pip install opencv-python pyrealsense2 open3d numpy tqdm
#   pip install apriltag                 # primary AprilTag binding
#   pip install pupil-apriltags          # Windows alternative AprilTag binding

import os
import random
import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from datetime import datetime
from tqdm import tqdm

# AprilTag import with fallback
try:
    import apriltag
    opts = apriltag.DetectorOptions(
        families='tag36h11', border=1,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=True)
    _tag_detector = apriltag.Detector(opts)
except ImportError:
    try:
        from pupil_apriltags import Detector as PDetector
        _tag_detector = PDetector(families='tag36h11')
    except ImportError:
        raise ImportError(
            "AprilTag detector not found. Install 'apriltag' or 'pupil-apriltags'.")

# === FULL DETERMINISM SETUP ===
os.environ.update({
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
    'NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'42'})
random.seed(42)
np.random.seed(42)
o3d.utility.random.seed(42)
np.random.shuffle = lambda x,*a,**k:None

# Monkey-patch voxel_down_sample for deterministic ordering
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

# === USER CONFIGURATION ===
SERIAL0            = '311322303919'
SERIAL1            = '252222300195'
SERIAL2            = '252222300763'   # NEW third camera
TAG_ID             = 3        # AprilTag ID
TAG_SIZE           = 0.03026    # meters
NUM_FRAMES         = 10
WARMUP             = 1.0
GRAB_ATTEMPTS      = 10
GRAB_TIMEOUT       = 2000
VOXEL_SIZE         = 1e-8
DEPTH_TRUNC        = 2.0
VISUALIZE          = True
# Manual tweaks (applied to both cam1 and cam2 w.r.t. cam0)
FINAL_TRANSLATION  = [0.0, 0.0, 0.0]
FINAL_ROTATION_DEG = [0.0, 0.0, 0.0]

# === CAMERA & CAPTURE ===
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
    try: dev.first_depth_sensor().set_option(rs.option.enable_auto_exposure, 0)
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

# === CLOUD BUILD ===
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

# === APRILTAG POSE ===
def detect_tag_pose(color_img, intr, tid, tsize):
    img = np.asarray(color_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = _tag_detector.detect(gray)
    if VISUALIZE:
        vis = img.copy()
        if not dets: print("[AprilTag] none detected")
        for d in dets:
            pts = d.corners.astype(int).reshape(-1,1,2)
            cv2.polylines(vis, [pts], True, (0,255,0), 2)
            iid = d.tag_id if hasattr(d, 'tag_id') else d.id
            cv2.putText(vis, f"ID={iid}", tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow('AprilTag', vis); cv2.waitKey(500); cv2.destroyAllWindows()
    if not dets: raise RuntimeError('Tag not detected')
    chosen = next((d for d in dets if (d.tag_id if hasattr(d, 'tag_id') else d.id)==tid), dets[0])
    half = tsize / 2.0
    objp = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], dtype=np.float64)
    imgp = chosen.corners.astype(np.float64)
    K = intr.intrinsic_matrix
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, None)
    if not ok: raise RuntimeError('solvePnP failed')
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = tvec.ravel()
    return T

# === ICP (same logic as original, applied to any target vs reference) ===
def icp_to_reference(pc_ref, pc_tgt):
    T_full = np.eye(4)
    # Local planar ICP around tag plane
    half = TAG_SIZE / 2.0
    bbox = o3d.geometry.AxisAlignedBoundingBox([-half,-half,-0.01], [half,half,0.01])
    sub_ref, sub_tgt = pc_ref.crop(bbox), pc_tgt.crop(bbox)
    if len(sub_ref.points)>50 and len(sub_tgt.points)>50:
        res_loc = o3d.pipelines.registration.registration_icp(
            sub_tgt, sub_ref, VOXEL_SIZE*0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        pc_tgt.transform(res_loc.transformation); T_full = res_loc.transformation @ T_full
    # Global ICP
    res_glob = o3d.pipelines.registration.registration_icp(
        pc_tgt, pc_ref, VOXEL_SIZE*1.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    pc_tgt.transform(res_glob.transformation); T_full = res_glob.transformation @ T_full
    # Manual tweaks
    if any(FINAL_TRANSLATION):
        Tt = np.eye(4); Tt[:3,3] = FINAL_TRANSLATION
        pc_tgt.transform(Tt); T_full = Tt @ T_full
    if any(FINAL_ROTATION_DEG):
        r,p,y = np.radians(FINAL_ROTATION_DEG)
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        Rm = Rz @ Ry @ Rx; Tr = np.eye(4); Tr[:3,:3] = Rm
        pc_tgt.transform(Tr); T_full = Tr @ T_full
    return T_full

# === MAIN ===
def main():
    total_start = time.time()
    print('Running AprilTag alignment (3 cams)...')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{timestamp}"; os.makedirs(out_dir, exist_ok=True)

    # Capture from 3 cameras
    r0, i0, raw0 = capture_seq(SERIAL0)
    r1, i1, raw1 = capture_seq(SERIAL1)
    r2, i2, raw2 = capture_seq(SERIAL2)

    # Save one RGB color image per camera
    for idx, raws in enumerate([raw0, raw1, raw2]):
        color_np, depth_np = raws[0]
        rgb_fn = os.path.join(out_dir, f"cam{idx}_color_{timestamp}.png")
        cv2.imwrite(rgb_fn, cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
    print(f"Saved RGB images in {out_dir}")

    # Build per-camera raw clouds
    pc0_raw = build_cloud(r0, i0, label=" cam0")
    pc1_raw = build_cloud(r1, i1, label=" cam1")
    pc2_raw = build_cloud(r2, i2, label=" cam2")
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam0_raw_{timestamp}.ply"), pc0_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam1_raw_{timestamp}.ply"), pc1_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam2_raw_{timestamp}.ply"), pc2_raw)

    # Detect tags and align each to the tag frame
    # rawX lists contain tuples (color_np, depth_np), so pick element 0 for color image
    T0 = detect_tag_pose(raw0[0][0], i0, TAG_ID, TAG_SIZE)
    T1 = detect_tag_pose(raw1[0][0], i1, TAG_ID, TAG_SIZE)
    T2 = detect_tag_pose(raw2[0][0], i2, TAG_ID, TAG_SIZE)
    pc0_raw.transform(np.linalg.inv(T0))
    pc1_raw.transform(np.linalg.inv(T1))
    pc2_raw.transform(np.linalg.inv(T2))

    # Save aligned per-camera clouds (to the tag frame)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam0_aligned_{timestamp}.ply"), pc0_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam1_aligned_{timestamp}.ply"), pc1_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam2_aligned_{timestamp}.ply"), pc2_raw)

    # ICP: register cam1->cam0 and cam2->cam0 (same logic as original)
    T_full_1 = icp_to_reference(pc0_raw, pc1_raw)
    T_full_2 = icp_to_reference(pc0_raw, pc2_raw)

    # Final merge (3 cameras)
    merged = pc0_raw + pc1_raw + pc2_raw
    merged = merged.voxel_down_sample(VOXEL_SIZE)
    merged_fn = os.path.join(out_dir, f"merged_{timestamp}.ply")
    o3d.io.write_point_cloud(merged_fn, merged)

    # Save alignment matrices to map cam1->cam0 and cam2->cam0
    # (maintains original definition: inv(T0) @ T_full @ T1)
    T_cam1_to_cam0 = np.linalg.inv(T0) @ T_full_1 @ T1
    T_cam2_to_cam0 = np.linalg.inv(T0) @ T_full_2 @ T2
    np.save(os.path.join(out_dir, f"align_matrix_cam1_to_cam0_{timestamp}.npy"), T_cam1_to_cam0)
    np.save(os.path.join(out_dir, f"align_matrix_cam2_to_cam0_{timestamp}.npy"), T_cam2_to_cam0)

    total_end = time.time()
    print(f"Saved all outputs in {out_dir}")
    print(f"Total runtime: {total_end - total_start:.2f} seconds.")

if __name__=='__main__':
    main()