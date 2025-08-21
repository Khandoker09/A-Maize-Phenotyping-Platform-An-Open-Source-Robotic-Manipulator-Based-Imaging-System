#!/usr/bin/env python3
"""
AprilTag-based multi-camera (3 cams) capture + alignment (deterministic)

Alignment/ICP/pose math is IDENTICAL to your original.
Additions (do not change alignment):
  • Depth: 848x480@30, projector ON, laser max, High-Density preset, low confidence threshold.
  • Gentler per-camera preprocessing (keeps thin features).
  • Saves RGB + depth-debug PNGs for first frame of each cam.
  • Post-merge outputs:
      - merged_raw_*.ply      (raw merge, unchanged)
      - merged_thin_*.ply     (thin-preserve clean; best to see stems)
      - merged_*.ply          (strong clean: voxel + floor + crops)
  • TSDF fusion across all frames & cams:
      - tsdf_mesh_*.ply, tsdf_cloud_*.ply
"""

# Dependencies:
#   pip install opencv-python pyrealsense2 open3d numpy tqdm
#   pip install apriltag  (or pupil-apriltags)

import os
import random
import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from datetime import datetime
from tqdm import tqdm

# ---------- AprilTag import with fallback (unchanged) ----------
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
        raise ImportError("AprilTag detector not found. Install 'apriltag' or 'pupil-apriltags'.")

# ---------- Determinism (unchanged) ----------
os.environ.update({
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
    'NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'42'})
random.seed(42)
np.random.seed(42)
o3d.utility.random.seed(42)
np.random.shuffle = lambda x,*a,**k:None

_auth_voxel = o3d.geometry.PointCloud.voxel_down_sample
def deterministic_voxel_down_sample(self, vsz):
    pc = _auth_voxel(self, vsz)
    pts = np.asarray(pc.points)
    if pts.size == 0: return pc
    idx = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
    pc.points = o3d.utility.Vector3dVector(pts[idx])
    if pc.has_colors():  pc.colors  = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])
    if pc.has_normals(): pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals)[idx])
    return pc
o3d.geometry.PointCloud.voxel_down_sample = deterministic_voxel_down_sample

# ---------- USER CONFIG ----------
SERIAL0            = '311322303919'
SERIAL1            = '252222300195'
SERIAL2            = '252222300763'
TAG_ID             = 3              # AprilTag ID
TAG_SIZE           = 0.074          # meters
NUM_FRAMES         = 12            # ↑ more frames → better TSDF fill
WARMUP             = 1.0
GRAB_ATTEMPTS      = 30             # Increased from 10
GRAB_TIMEOUT       = 5000           # Increased from 2000
FRAME_WAIT         = 0.1            # Wait between frame attempts
VOXEL_SIZE         = 1e-8           # per-cam: effectively no downsample
DEPTH_TRUNC        = 1.0
VISUALIZE          = True
FINAL_TRANSLATION  = [0.0, 0.0, 0.0]
FINAL_ROTATION_DEG = [0.0, 0.0, 0.0]

# ---------- Post-merge quality (does NOT affect alignment) ----------
# Strong clean (your usual "merged_*.ply"):
POST_VOXEL_SIZE     = 0.002     # 2 mm
POST_OUTLIER_NN     = 30
POST_OUTLIER_STD    = 1.5
REMOVE_FLOOR        = True
FLOOR_DIST_THRESH   = 0.01      # m
FLOOR_RANSAC_N      = 3
FLOOR_NUM_ITERS     = 200
Z_CROP_ON           = True
Z_CROP_MIN, Z_CROP_MAX = -0.05, 0.75
XY_CROP_ON          = True
XY_HALF_EXTENT      = 0.60

# Thin-preserve clean (gentle – best for stems):
THIN_VOXEL          = None      # None = no downsample (set 0.0005 if you want tiny ds)
THIN_RADIUS         = 0.008     # 8 mm neighborhood
THIN_MIN_PTS        = 1         # keep any point with >=1 neighbor

# TSDF fusion
DO_TSDF             = True
TSDF_VOXEL          = 0.001     # 1 mm
TSDF_TRUNC_MULT     = 6.0       # more aggressive hole fill

# Recovery settings
RECOVERY_DELAY = 2.0  # seconds to wait between recovery attempts
MAX_RECOVERY_ATTEMPTS = 3  # number of times to try recovering the camera

# ---------- CAMERA & CAPTURE ----------
def start_cam(serial):
    print(f"Initializing camera {serial}...")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    
    # Try different profiles in order of preference
    profiles = [
        # Profile 1: High res
        [(rs.stream.depth, 848, 480, rs.format.z16, 30),
         (rs.stream.color, 640, 480, rs.format.bgr8, 30)],
        # Profile 2: Medium res
        [(rs.stream.depth, 640, 480, rs.format.z16, 30),
         (rs.stream.color, 640, 480, rs.format.bgr8, 30)],
        # Profile 3: Low res
        [(rs.stream.depth, 480, 270, rs.format.z16, 30),
         (rs.stream.color, 424, 240, rs.format.bgr8, 30)]
    ]
    
    for i, profile in enumerate(profiles, 1):
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            for stream in profile:
                cfg.enable_stream(*stream)
            prof = pipe.start(cfg)
            print(f"Successfully connected using profile {i}")
            break
        except Exception as e:
            print(f"Profile {i} failed: {str(e)}")
            if i == len(profiles):
                raise RuntimeError(f"All profiles failed for camera {serial}")
            continue
            
    dev = prof.get_device()
    print(f"Successfully connected to {serial}")

    # Color settings: AE/AWB off + increased contrast/saturation
    try:
        cs = dev.query_sensors()[1]  # color sensor
        cs.set_option(rs.option.enable_auto_exposure, 0)
        cs.set_option(rs.option.enable_auto_white_balance, 0)
        
        # Increase contrast and saturation
        try:
            cs.set_option(rs.option.contrast, 75)        # Default is 50, range 0-100
            cs.set_option(rs.option.saturation, 80)      # Default is 64, range 0-100
            cs.set_option(rs.option.sharpness, 70)       # Default is 50, range 0-100
            cs.set_option(rs.option.gamma, 300)          # Default is 300, range 100-500
            print(f"Set enhanced color settings for {serial}")
        except Exception as e:
            print(f"Note: Could not set some color options: {e}")
    except Exception as e:
        print(f"Warning: Could not configure color sensor: {e}")

    # Depth: AE off + projector ON + max laser + high-density + low confidence threshold
    try:
        dsens = dev.first_depth_sensor()
        dsens.set_option(rs.option.enable_auto_exposure, 0)
        try: dsens.set_option(rs.option.emitter_enabled, 1)
        except: pass
        try:
            rng = dsens.get_option_range(rs.option.laser_power)
            dsens.set_option(rs.option.laser_power, rng.max)
        except: pass
        try: dsens.set_option(rs.option.visual_preset, 4)  # 4 ≈ High Density
        except: pass
        try: dsens.set_option(rs.option.confidence_threshold, 1)  # accept weaker matches
        except: pass
    except: pass

    time.sleep(WARMUP)
    ds = dev.first_depth_sensor().get_depth_scale()
    vs = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = o3d.camera.PinholeCameraIntrinsic(vs.width, vs.height, vs.fx, vs.fy, vs.ppx, vs.ppy)
    align = rs.align(rs.stream.color)
    print(f"→ Cam {serial} @{vs.width}×{vs.height}@30fps, depth_scale={ds:.4f}m")
    return pipe, align, intr, ds

def capture_seq(serial):
    """Capture sequence with recovery attempts"""
    for recovery in range(MAX_RECOVERY_ATTEMPTS):
        try:
            pipe, align, intr, ds = start_cam(serial)
            rgbds, raws = [], []
            
            try:
                for _ in tqdm(range(NUM_FRAMES), desc=f"Capturing cam {serial}"):
                    # Flush any stale frames
                    for __ in range(5):
                        try:
                            pipe.wait_for_frames(timeout_ms=100)
                        except:
                            pass
                            
                    # Try to get frame set
                    success = False
                    for attempt in range(GRAB_ATTEMPTS):
                        try:
                            shots = [grab_one(pipe, align, ds) for __ in range(3)]
                            rgbds.append(shots[1])
                            raws.append(shots[1][1:])
                            success = True
                            break
                        except Exception as e:
                            print(f"Frame grab failed (attempt {attempt+1}/{GRAB_ATTEMPTS}): {str(e)}")
                            time.sleep(0.2)
                    
                    if not success:
                        raise RuntimeError("Failed to grab frames after all attempts")
                        
                pipe.stop()
                return rgbds, intr, raws, ds
                
            except Exception as e:
                print(f"Error during capture: {str(e)}")
                pipe.stop()
                if recovery < MAX_RECOVERY_ATTEMPTS - 1:
                    print(f"\nAttempting camera recovery ({recovery + 1}/{MAX_RECOVERY_ATTEMPTS})...")
                    print("Please wait...")
                    time.sleep(RECOVERY_DELAY)
                    continue
                raise
                
        except Exception as e:
            if recovery < MAX_RECOVERY_ATTEMPTS - 1:
                print(f"\nCamera start failed, retrying ({recovery + 1}/{MAX_RECOVERY_ATTEMPTS})...")
                print("Please wait...")
                time.sleep(RECOVERY_DELAY)
                continue
            raise
            
    raise RuntimeError(f"Failed to capture from camera {serial} after {MAX_RECOVERY_ATTEMPTS} attempts")

def grab_one(pipe, align, ds):
    """Grab one aligned RGBD frame"""
    fm = pipe.wait_for_frames(timeout_ms=GRAB_TIMEOUT)
    if not fm:
        raise RuntimeError("No frames received")
        
    # Align depth to color
    al = align.process(fm)
    if not al:
        raise RuntimeError("Frame alignment failed")
        
    # Get depth and color frames
    depth_frame = al.get_depth_frame()
    color_frame = al.get_color_frame()
    if not depth_frame or not color_frame:
        raise RuntimeError("Missing depth or color frame")
        
    # Convert to numpy arrays
    d = np.asarray(depth_frame.get_data(), dtype=np.uint16)
    c = np.asarray(color_frame.get_data(), dtype=np.uint8)
    
    if d.size == 0 or c.size == 0:
        raise RuntimeError("Empty frame data")
        
    # Convert color format
    c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(c_rgb),
        o3d.geometry.Image(d),
        depth_scale=1.0/ds,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False)
        
    return rgbd, c_rgb, d

# ---------- CLOUD BUILD ----------
def preprocess_cloud(pc):
    # Keep thin features: NO statistical outlier; use very gentle radius outlier
    pc = pc.voxel_down_sample(VOXEL_SIZE)  # VOXEL_SIZE=1e-8 => effectively no downsample
    if len(pc.points) == 0:
        return pc
    try:
        # keep points with >=1 neighbor within 6 mm (tune 0.004–0.008)
        _, ind = pc.remove_radius_outlier(nb_points=1, radius=0.006)
        pc = pc.select_by_index(sorted(ind))
    except Exception:
        pass

    # Normals (small radius so we don't require dense neighbors)
    if len(pc.points) > 0:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20))
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

# ---------- APRILTAG POSE (original) ----------
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

# ---------- ICP (original) ----------
def icp_to_reference(pc_ref, pc_tgt):
    T_full = np.eye(4)
    half = TAG_SIZE / 2.0
    bbox = o3d.geometry.AxisAlignedBoundingBox([-half,-half,-0.01], [half,half,0.01])
    sub_ref, sub_tgt = pc_ref.crop(bbox), pc_tgt.crop(bbox)
    if len(sub_ref.points)>50 and len(sub_tgt.points)>50:
        res_loc = o3d.pipelines.registration.registration_icp(
            sub_tgt, sub_ref, VOXEL_SIZE*0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        pc_tgt.transform(res_loc.transformation); T_full = res_loc.transformation @ T_full
    res_glob = o3d.pipelines.registration.registration_icp(
        pc_tgt, pc_ref, VOXEL_SIZE*1.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    pc_tgt.transform(res_glob.transformation); T_full = res_glob.transformation @ T_full
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

# ---------- Post-merge cleaners ----------
def clean_merged_cloud(pc):
    if len(pc.points) == 0: return pc
    pc = pc.voxel_down_sample(POST_VOXEL_SIZE)
    if len(pc.points) > 0:
        _, ind = pc.remove_statistical_outlier(nb_neighbors=POST_OUTLIER_NN, std_ratio=POST_OUTLIER_STD)
        pc = pc.select_by_index(sorted(ind))
    if REMOVE_FLOOR and len(pc.points) > 1000:
        try:
            plane_model, inliers = pc.segment_plane(distance_threshold=FLOOR_DIST_THRESH,
                                                    ransac_n=FLOOR_RANSAC_N,
                                                    num_iterations=FLOOR_NUM_ITERS)
            keep = np.setdiff1d(np.arange(len(pc.points)), np.asarray(inliers, dtype=int))
            if keep.size > 0:
                pc = pc.select_by_index(keep.tolist())
        except Exception:
            pass
    if Z_CROP_ON or XY_CROP_ON:
        xmin = -XY_HALF_EXTENT if XY_CROP_ON else -np.inf
        xmax =  XY_HALF_EXTENT if XY_CROP_ON else  np.inf
        ymin = -XY_HALF_EXTENT if XY_CROP_ON else -np.inf
        ymax =  XY_HALF_EXTENT if XY_CROP_ON else  np.inf
        zmin = Z_CROP_MIN if Z_CROP_ON else -np.inf
        zmax = Z_CROP_MAX if Z_CROP_ON else  np.inf
        aabb = o3d.geometry.AxisAlignedBoundingBox([xmin, ymin, zmin], [xmax, ymax, zmax])
        pc = pc.crop(aabb)
    return pc

def thin_preserve_cleanup(pc_raw, voxel=THIN_VOXEL, radius=THIN_RADIUS, min_pts=THIN_MIN_PTS):
    pc = pc_raw
    if voxel and voxel > 0:
        pc = pc.voxel_down_sample(voxel)  # tiny voxel, preserves detail
    if len(pc.points) > 0:
        _, ind = pc.remove_radius_outlier(nb_points=min_pts, radius=radius)
        pc = pc.select_by_index(sorted(ind))
    return pc

# ---------- TSDF fusion ----------
def fuse_tsdf_and_export(rgbd_seqs, intrinsics, extrinsics_tag_to_cam, out_dir, timestamp,
                         voxel=TSDF_VOXEL, trunc_mult=TSDF_TRUNC_MULT):
    try:
        integ = o3d.pipelines.integration
    except AttributeError:
        integ = o3d.integration
    vol = integ.ScalableTSDFVolume(
        voxel_length=float(voxel),
        sdf_trunc=float(trunc_mult * voxel),
        color_type=integ.TSDFVolumeColorType.RGB8
    )
    for seq, intr, T_tag_to_cam in zip(rgbd_seqs, intrinsics, extrinsics_tag_to_cam):
        extrinsic = T_tag_to_cam  # world(tag)->camera
        for (rgbd, _, _) in seq:
            vol.integrate(rgbd, intr, extrinsic)
    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    mesh_fn = os.path.join(out_dir, f"tsdf_mesh_{timestamp}.ply")
    o3d.io.write_triangle_mesh(mesh_fn, mesh)
    tsdf_pc = vol.extract_point_cloud()
    tsdf_pc_fn = os.path.join(out_dir, f"tsdf_cloud_{timestamp}.ply")
    o3d.io.write_point_cloud(tsdf_pc_fn, tsdf_pc)
    return mesh_fn, tsdf_pc_fn

# ---------- Debug depth saving ----------
def save_debug_depth_png(depth_u16, depth_scale_m, fn):
    d = depth_u16.astype(np.float32) * float(depth_scale_m)
    m = (d > 0) & (d < DEPTH_TRUNC)
    if not m.any():
        cv2.imwrite(fn, np.zeros_like(depth_u16))
        return
    dn = np.zeros_like(d)
    dn[m] = (d[m] - d[m].min()) / (d[m].max() - d[m].min() + 1e-9)
    cv2.imwrite(fn, (dn * 65535).astype(np.uint16))

# ---------- MAIN ----------
def main():
    total_start = time.time()
    print('Running AprilTag alignment (3 cams)...')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{timestamp}"; os.makedirs(out_dir, exist_ok=True)

    # Capture from 3 cameras
    r0, i0, raw0, ds0 = capture_seq(SERIAL0)
    r1, i1, raw1, ds1 = capture_seq(SERIAL1)
    r2, i2, raw2, ds2 = capture_seq(SERIAL2)

    # Save one RGB & debug depth per camera (first frame)
    for idx, (raws, dscale) in enumerate([(raw0, ds0), (raw1, ds1), (raw2, ds2)]):
        color_np, depth_np = raws[0]
        rgb_fn = os.path.join(out_dir, f"cam{idx}_color_{timestamp}.png")
        dep_fn = os.path.join(out_dir, f"cam{idx}_depthDebug_{timestamp}.png")
        cv2.imwrite(rgb_fn, cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
        save_debug_depth_png(depth_np, dscale, dep_fn)
    print(f"Saved RGB + depth debug images in {out_dir}")

    # Build per-camera raw clouds
    pc0_raw = build_cloud(r0, i0, label=" cam0")
    pc1_raw = build_cloud(r1, i1, label=" cam1")
    pc2_raw = build_cloud(r2, i2, label=" cam2")
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam0_raw_{timestamp}.ply"), pc0_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam1_raw_{timestamp}.ply"), pc1_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam2_raw_{timestamp}.ply"), pc2_raw)

    # Detect tags and align each to the tag frame (UNCHANGED)
    T0 = detect_tag_pose(raw0[0][0], i0, TAG_ID, TAG_SIZE)
    T1 = detect_tag_pose(raw1[0][0], i1, TAG_ID, TAG_SIZE)
    T2 = detect_tag_pose(raw2[0][0], i2, TAG_ID, TAG_SIZE)
    pc0_raw.transform(np.linalg.inv(T0))
    pc1_raw.transform(np.linalg.inv(T1))
    pc2_raw.transform(np.linalg.inv(T2))

    # Save aligned per-camera clouds
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam0_aligned_{timestamp}.ply"), pc0_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam1_aligned_{timestamp}.ply"), pc1_raw)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"cam2_aligned_{timestamp}.ply"), pc2_raw)

    # ICP (UNCHANGED)
    T_full_1 = icp_to_reference(pc0_raw, pc1_raw)
    T_full_2 = icp_to_reference(pc0_raw, pc2_raw)

    # Merge (UNCHANGED) and save RAW
    merged_raw = pc0_raw + pc1_raw + pc2_raw
    merged_raw_fn = os.path.join(out_dir, f"merged_raw_{timestamp}.ply")
    o3d.io.write_point_cloud(merged_raw_fn, merged_raw)
    print("  - Raw merged:   ", merged_raw_fn)

    # Thin-preserve cleanup (best for stems)
    merged_thin = thin_preserve_cleanup(merged_raw,
                                        voxel=THIN_VOXEL,
                                        radius=THIN_RADIUS,
                                        min_pts=THIN_MIN_PTS)
    merged_thin_fn = os.path.join(out_dir, f"merged_thin_{timestamp}.ply")
    o3d.io.write_point_cloud(merged_thin_fn, merged_thin)
    print("  - Thin-preserve:", merged_thin_fn)

    # Stronger cleaned version (floor removal + crops)
    merged_clean = clean_merged_cloud(merged_raw)
    merged_clean_fn = os.path.join(out_dir, f"merged_{timestamp}.ply")
    o3d.io.write_point_cloud(merged_clean_fn, merged_clean)
    print("  - Clean merged: ", merged_clean_fn)

    # TSDF fusion across all frames/cams
    if DO_TSDF:
        try:
            mesh_fn, tsdf_pc_fn = fuse_tsdf_and_export(
                rgbd_seqs=[r0, r1, r2],
                intrinsics=[i0, i1, i2],
                extrinsics_tag_to_cam=[T0, T1, T2],
                out_dir=out_dir, timestamp=timestamp,
                voxel=TSDF_VOXEL, trunc_mult=TSDF_TRUNC_MULT
            )
            print("  - TSDF mesh:   ", mesh_fn)
            print("  - TSDF cloud:  ", tsdf_pc_fn)
        except Exception as e:
            print("[TSDF] fusion skipped:", e)

    # Save alignment matrices (UNCHANGED definition)
    T_cam1_to_cam0 = np.linalg.inv(T0) @ T_full_1 @ T1
    T_cam2_to_cam0 = np.linalg.inv(T0) @ T_full_2 @ T2
    np.save(os.path.join(out_dir, f"align_matrix_cam1_to_cam0_{timestamp}.npy"), T_cam1_to_cam0)
    np.save(os.path.join(out_dir, f"align_matrix_cam2_to_cam0_{timestamp}.npy"), T_cam2_to_cam0)

    total_end = time.time()
    print(f"Saved all outputs in {out_dir}")
    print(f"Total runtime: {total_end - total_start:.2f} seconds.")

if __name__=='__main__':
    main()
