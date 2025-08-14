#!/usr/bin/env python3
# 3-cam AprilTag alignment using POSE-BASED init (works with ≥1 shared tag) + ICP refinement.

import os, random, time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from datetime import datetime
from tqdm import tqdm

# -------- AprilTag detector --------
try:
    import apriltag
    _tag_detector = apriltag.Detector(apriltag.DetectorOptions(
        families='tag36h11', border=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=True))
except ImportError:
    from pupil_apriltags import Detector as PDetector
    _tag_detector = PDetector(families='tag36h11')

# -------- Determinism --------
os.environ.update({
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'42'
})
random.seed(42); np.random.seed(42); o3d.utility.random.seed(42)
np.random.shuffle = lambda *a, **k: None
_auth_voxel = o3d.geometry.PointCloud.voxel_down_sample
def _det_voxel(self, vsz):
    pc = _auth_voxel(self, vsz)
    pts = np.asarray(pc.points)
    if pts.size:
        idx = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
        pc.points = o3d.utility.Vector3dVector(pts[idx])
        if pc.has_colors(): pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])
        if pc.has_normals(): pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals)[idx])
    return pc
o3d.geometry.PointCloud.voxel_down_sample = _det_voxel

# -------- Config --------
SERIAL0, SERIAL1, SERIAL2 = '311322303919', '252222300195', '252222300763'
TAG_SIZE = 0.074     # meters (black-square edge; size cancels in relative pose anyway)
NUM_FRAMES, WARMUP = 10, 1.0
GRAB_ATTEMPTS, GRAB_TIMEOUT = 10, 2000
DEPTH_TRUNC, VISUALIZE = 0.9, True

# Clouds / ICP
VOXEL_SIZE   = 0.0000001
CLOUD_VOXEL  = 0.0005
ICP_VOXEL    = 0.003
ICP_DIST_LOCAL  = 0.010
ICP_DIST_GLOBAL = 0.020
REFINE_VOXEL    = 0.0015
ICP_DIST_REFINE = 0.0035

# -------- Camera I/O --------
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
    except Exception: pass
    try: dev.first_depth_sensor().set_option(rs.option.enable_auto_exposure, 0)
    except Exception: pass
    time.sleep(WARMUP)
    ds = dev.first_depth_sensor().get_depth_scale()
    vs = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = o3d.camera.PinholeCameraIntrinsic(vs.width, vs.height, vs.fx, vs.fy, vs.ppx, vs.ppy)
    align = rs.align(rs.stream.color)
    print(f"→ Cam {serial} @{vs.width}×{vs.height}@30fps, depth_scale={ds:.6f} m/units")
    return pipe, align, intr, ds

def grab_one(pipe, align, ds):
    for _ in range(GRAB_ATTEMPTS):
        try:
            fm = pipe.wait_for_frames(timeout_ms=GRAB_TIMEOUT)
            al = align.process(fm)
            d = np.asarray(al.get_depth_frame().get_data(), dtype=np.uint16)
            c = np.asarray(al.get_color_frame().get_data(), dtype=np.uint8)
            c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(c_rgb), o3d.geometry.Image(d),
                depth_scale=1.0/ds, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
            return rgbd, c_rgb, d
        except RuntimeError:
            time.sleep(0.1)
    raise RuntimeError("Failed to grab frame")

def capture_seq(serial):
    pipe, align, intr, ds = start_cam(serial)
    rgbds, raws = [], []
    for _ in tqdm(range(NUM_FRAMES), desc=f"Capturing cam {serial}"):
        shots = [grab_one(pipe, align, ds) for __ in range(3)]
        rgbds.append(shots[1])
        raws.append((shots[1][1], shots[1][2], ds))  # (color_rgb, depth_u16, depth_scale)
    pipe.stop()
    return rgbds, intr, raws

# -------- Clouds --------
def preprocess_cloud(pc):
    pc = pc.voxel_down_sample(CLOUD_VOXEL)
    if len(pc.points)==0: return pc
    _, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pc = pc.select_by_index(sorted(ind))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=CLOUD_VOXEL*3, max_nn=50))
    pc.orient_normals_consistent_tangent_plane(10)
    return pc

def build_cloud(rgbds, intr, label=""):
    pcs = [o3d.geometry.PointCloud.create_from_rgbd_image(r[0], intr)
           for r in tqdm(rgbds, desc=f"Building cloud{label}")]
    c = pcs[0]
    for p in pcs[1:]: c += p
    return preprocess_cloud(c)

# -------- Tag pose utilities --------
def detect_tag_poses(color_rgb, intr):
    """
    Return dict: tag_id -> 4x4 pose T_cam_tag (maps tag frame -> camera frame)
    using solvePnP on the 4 tag corners. No depth needed; size cancels in relative pose.
    """
    img = np.asarray(color_rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dets = _tag_detector.detect(gray)

    if VISUALIZE:
        vis = img.copy()
        for d in dets:
            pts = d.corners.astype(int).reshape(-1,1,2)
            cv2.polylines(vis, [pts], True, (0,255,0), 2)
            tid = d.tag_id if hasattr(d,'tag_id') else d.id
            cv2.putText(vis, f"{tid}", tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        try: cv2.imshow('tags', vis); cv2.waitKey(120); cv2.destroyAllWindows()
        except cv2.error: pass

    K = intr.intrinsic_matrix
    poses = {}
    for d in dets:
        tid = d.tag_id if hasattr(d,'tag_id') else d.id
        imgp = d.corners.astype(np.float64)
        half = TAG_SIZE/2.0
        objp = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, None)
        if not ok: continue
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.ravel()
        poses[tid] = T  # T_cam_tag
    return poses

def invert(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

def quat_from_R(R):
    # w,x,y,z
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr+1.0)*2; q[0]=0.25*s
        q[1]=(R[2,1]-R[1,2])/s; q[2]=(R[0,2]-R[2,0])/s; q[3]=(R[1,0]-R[0,1])/s
    else:
        i = np.argmax([R[0,0],R[1,1],R[2,2]])
        if i==0:
            s=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; q[0]=(R[2,1]-R[1,2])/s
            q[1]=0.25*s; q[2]=(R[0,1]+R[1,0])/s; q[3]=(R[0,2]+R[2,0])/s
        elif i==1:
            s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; q[0]=(R[0,2]-R[2,0])/s
            q[1]=(R[0,1]+R[1,0])/s; q[2]=0.25*s; q[3]=(R[1,2]+R[2,1])/s
        else:
            s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; q[0]=(R[1,0]-R[0,1])/s
            q[1]=(R[0,2]+R[2,0])/s; q[2]=(R[1,2]+R[2,1])/s; q[3]=0.25*s
    return q/np.linalg.norm(q)

def R_from_quat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def average_SE3(transforms):
    """Average a list of 4x4 transforms (simple quat+mean method)."""
    if not transforms: return np.eye(4)
    qs = []; ts = []
    for T in transforms:
        qs.append(quat_from_R(T[:3,:3])); ts.append(T[:3,3])
    qs = np.array(qs)
    # Quaternion average via eigen on outer products
    M = (qs.T @ qs) / qs.shape[0]
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[0] < 0: q_avg = -q_avg
    R = R_from_quat(q_avg)
    t = np.mean(ts, axis=0)
    Tout = np.eye(4); Tout[:3,:3]=R; Tout[:3,3]=t
    return Tout

def init_from_tag_poses(poses_src, poses_tgt):
    """Build src->tgt init from any shared tag poses."""
    common = sorted(set(poses_src) & set(poses_tgt))
    if not common:
        return np.eye(4), []
    estimates = []
    for tid in common:
        Tsrc_tag = poses_src[tid]    # cam_src <- tag
        Ttgt_tag = poses_tgt[tid]    # cam_tgt <- tag
        Tsrc_to_tgt = Ttgt_tag @ invert(Tsrc_tag)
        estimates.append(Tsrc_to_tgt)
    return average_SE3(estimates), common

# -------- ICP --------
def _est_normals(pc, radius, max_nn=50):
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pc.orient_normals_consistent_tangent_plane(10); return pc

def run_icp_to_target(src_raw, tgt_raw, init=np.eye(4)):
    T = init.copy()

    # coarse
    src = src_raw.voxel_down_sample(ICP_VOXEL)
    tgt = tgt_raw.voxel_down_sample(ICP_VOXEL)
    if len(src.points)==0 or len(tgt.points)==0: return T
    _est_normals(src, ICP_VOXEL*3); _est_normals(tgt, ICP_VOXEL*3)

    # local near origin (keep small box; origin will be near tags after init)
    box_xy = 0.12/2.0
    bbox = o3d.geometry.AxisAlignedBoundingBox([-box_xy,-box_xy,-0.05],[box_xy,box_xy,0.05])
    sub_tgt, sub_src = tgt.crop(bbox), src.crop(bbox)
    if len(sub_tgt.points)>50 and len(sub_src.points)>50:
        res_loc = o3d.pipelines.registration.registration_icp(
            sub_src, sub_tgt, ICP_DIST_LOCAL, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        T = res_loc.transformation

    # global
    res_glob = o3d.pipelines.registration.registration_icp(
        src, tgt, ICP_DIST_GLOBAL, T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    T = res_glob.transformation

    # refine
    src_r = src_raw.voxel_down_sample(REFINE_VOXEL)
    tgt_r = tgt_raw.voxel_down_sample(REFINE_VOXEL)
    if len(src_r.points)==0 or len(tgt_r.points)==0: return T
    _est_normals(src_r, REFINE_VOXEL*3); _est_normals(tgt_r, REFINE_VOXEL*3)
    res_ref = o3d.pipelines.registration.registration_icp(
        src_r, tgt_r, ICP_DIST_REFINE, T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    return res_ref.transformation

# -------- Main --------
def main():
    t0 = time.time()
    print('Running 3-cam alignment (pose-based init from shared tags)...')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"output_{ts}"; os.makedirs(out_dir, exist_ok=True)

    # Capture
    r0,i0,raw0 = capture_seq(SERIAL0)
    r1,i1,raw1 = capture_seq(SERIAL1)
    r2,i2,raw2 = capture_seq(SERIAL2)

    # Save one RGB
    for idx, raws in enumerate([raw0, raw1, raw2]):
        color_np, depth_np, _ds = raws[0]
        cv2.imwrite(os.path.join(out_dir, f"cam{idx}_color_{ts}.png"),
                    cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))

    # Clouds (stay in camera frames)
    pc0 = build_cloud(r0, i0, " cam0"); o3d.io.write_point_cloud(os.path.join(out_dir,f"cam0_raw_{ts}.ply"), pc0)
    pc1 = build_cloud(r1, i1, " cam1"); o3d.io.write_point_cloud(os.path.join(out_dir,f"cam1_raw_{ts}.ply"), pc1)
    pc2 = build_cloud(r2, i2, " cam2"); o3d.io.write_point_cloud(os.path.join(out_dir,f"cam2_raw_{ts}.ply"), pc2)

    # Tag poses in camera frames
    poses0 = detect_tag_poses(raw0[0][0], i0)
    poses1 = detect_tag_poses(raw1[0][0], i1)
    poses2 = detect_tag_poses(raw2[0][0], i2)

    # Init cam1->cam0 from shared tag poses (works with ≥1 tag)
    init10, common10 = init_from_tag_poses(poses1, poses0)
    if not common10: print("[warn] cam1/cam0 have 0 shared tags — init=identity")
    T1_to_0 = run_icp_to_target(pc1, pc0, init=init10); pc1.transform(T1_to_0)

    # Build stronger target and init cam2->(cam0+cam1)
    merged_target = pc0 + pc1
    # Prefer shared tags with cam0; if none, it still uses whatever common existed
    init20, common20 = init_from_tag_poses(poses2, poses0)
    if not common20:
        # fall back to cam2 vs cam1 if that has tags in common
        init21, common21 = init_from_tag_poses(poses2, poses1)
        if common21: init20 = init21
        else: print("[warn] cam2 shares 0 tags with cam0/cam1 — init=identity")
    T2_to_01 = run_icp_to_target(pc2, merged_target, init=init20); pc2.transform(T2_to_01)

    # Merge & save (all now in cam0 frame)
    merged = (pc0 + pc1 + pc2).voxel_down_sample(VOXEL_SIZE)
    out_ply = os.path.join(out_dir, f"merged_{ts}.ply")
    o3d.io.write_point_cloud(out_ply, merged)
    print(f"Saved all outputs in {out_dir}")
    print(f"Total runtime: {time.time()-t0:.2f} s")

if __name__ == '__main__':
    main()




# #!/usr/bin/env python3
# # 3-cam AprilTag alignment (stable):
# #   - Init from shared AprilTag pose(s) (works with ≥1 common ID)
# #   - Pairwise ICP only: local (near tags) -> global -> very-fine
# #   - No tag-frame pretransforms, no pose-graph (prevents drift/explosions)

# import os, random, time
# import numpy as np
# import open3d as o3d
# import pyrealsense2 as rs
# import cv2
# from datetime import datetime
# from tqdm import tqdm

# # ---------------- AprilTag detector ----------------
# try:
#     import apriltag
#     _tag_detector = apriltag.Detector(
#         apriltag.DetectorOptions(
#             families='tag36h11', border=1,
#             quad_decimate=1.0, quad_sigma=0.0,
#             refine_edges=True
#         )
#     )
# except ImportError:
#     from pupil_apriltags import Detector as PDetector
#     _tag_detector = PDetector(families='tag36h11')

# # ---------------- Determinism ----------------------
# os.environ.update({
#     'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
#     'VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'42'
# })
# random.seed(42); np.random.seed(42); o3d.utility.random.seed(42)
# np.random.shuffle = lambda *a, **k: None

# _auth_voxel = o3d.geometry.PointCloud.voxel_down_sample
# def _det_voxel(self, vsz):
#     pc = _auth_voxel(self, vsz)
#     pts = np.asarray(pc.points)
#     if pts.size:
#         idx = np.lexsort((pts[:,2], pts[:,1], pts[:,0]))
#         pc.points = o3d.utility.Vector3dVector(pts[idx])
#         if pc.has_colors():
#             pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[idx])
#         if pc.has_normals():
#             pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals)[idx])
#     return pc
# o3d.geometry.PointCloud.voxel_down_sample = _det_voxel

# # ---------------- User config ----------------------
# SERIAL0, SERIAL1, SERIAL2 = '311322303919', '252222300195', '252222300763'
# TAG_SIZE = 0.074      # meters (black square edge; value cancels in relative pose)
# NUM_FRAMES, WARMUP = 10, 1.0
# GRAB_ATTEMPTS, GRAB_TIMEOUT = 10, 2000
# DEPTH_TRUNC, VISUALIZE = 1.0, True

# # Clouds / ICP tuning
# VOXEL_SIZE      = 0.000001   # final write-downsample (keep detail)
# CLOUD_VOXEL     = 0.0005     # 0.5 mm preprocess
# ICP_VOXEL       = 0.003      # coarse copies (3 mm)
# ICP_DIST_LOCAL  = 0.012      # 1.2 cm local pass (slightly looser to snap)
# ICP_DIST_GLOBAL = 0.020      # 2 cm global pass
# REFINE_VOXEL    = 0.0010     # very-fine copies (1.0 mm)
# ICP_DIST_REFINE = 0.0030     # 3.0 mm fine pass

# # ---------------- Camera I/O -----------------------
# def start_cam(serial):
#     pipe = rs.pipeline(); cfg = rs.config()
#     cfg.enable_device(serial)
#     cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#     cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     prof = pipe.start(cfg); dev = prof.get_device()
#     try:
#         cs = dev.query_sensors()[1]
#         cs.set_option(rs.option.enable_auto_exposure, 0)
#         cs.set_option(rs.option.enable_auto_white_balance, 0)
#     except Exception: pass
#     try: dev.first_depth_sensor().set_option(rs.option.enable_auto_exposure, 0)
#     except Exception: pass
#     time.sleep(WARMUP)
#     ds = dev.first_depth_sensor().get_depth_scale()
#     vs = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
#     intr = o3d.camera.PinholeCameraIntrinsic(vs.width, vs.height, vs.fx, vs.fy, vs.ppx, vs.ppy)
#     align = rs.align(rs.stream.color)
#     print(f"→ Cam {serial} @{vs.width}×{vs.height}@30fps, depth_scale={ds:.6f} m/units")
#     return pipe, align, intr, ds

# def grab_one(pipe, align, ds):
#     for _ in range(GRAB_ATTEMPTS):
#         try:
#             fm = pipe.wait_for_frames(timeout_ms=GRAB_TIMEOUT)
#             al = align.process(fm)
#             d = np.asarray(al.get_depth_frame().get_data(), dtype=np.uint16)
#             c = np.asarray(al.get_color_frame().get_data(), dtype=np.uint8)
#             c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
#             rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#                 o3d.geometry.Image(c_rgb), o3d.geometry.Image(d),
#                 depth_scale=1.0/ds, depth_trunc=DEPTH_TRUNC,
#                 convert_rgb_to_intensity=False)
#             return rgbd, c_rgb, d
#         except RuntimeError:
#             time.sleep(0.1)
#     raise RuntimeError("Failed to grab frame")

# def capture_seq(serial):
#     pipe, align, intr, ds = start_cam(serial)
#     rgbds, raws = [], []
#     for _ in tqdm(range(NUM_FRAMES), desc=f"Capturing cam {serial}"):
#         shots = [grab_one(pipe, align, ds) for __ in range(3)]
#         rgbds.append(shots[1])                      # middle shot
#         raws.append((shots[1][1], shots[1][2], ds)) # (color_rgb, depth_u16, depth_scale)
#     pipe.stop()
#     return rgbds, intr, raws

# # ---------------- Clouds --------------------------
# def preprocess_cloud(pc):
#     pc = pc.voxel_down_sample(CLOUD_VOXEL)
#     if len(pc.points)==0: return pc
#     _, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
#     pc = pc.select_by_index(sorted(ind))
#     pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=CLOUD_VOXEL*3, max_nn=50))
#     pc.orient_normals_consistent_tangent_plane(10)
#     return pc

# def build_cloud(rgbds, intr, label=""):
#     pcs = [o3d.geometry.PointCloud.create_from_rgbd_image(r[0], intr)
#            for r in tqdm(rgbds, desc=f"Building cloud{label}")]
#     c = pcs[0]
#     for p in pcs[1:]: c += p
#     return preprocess_cloud(c)

# # ---------------- Tag poses -----------------------
# def detect_tag_poses(color_rgb, intr):
#     """
#     Return dict: tag_id -> 4x4 pose T_cam_tag (maps tag -> camera).
#     """
#     img = np.asarray(color_rgb)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     dets = _tag_detector.detect(gray)

#     if VISUALIZE:
#         vis = img.copy()
#         for d in dets:
#             pts = d.corners.astype(int).reshape(-1,1,2)
#             cv2.polylines(vis, [pts], True, (0,255,0), 2)
#             tid = d.tag_id if hasattr(d,'tag_id') else d.id
#             cv2.putText(vis, f"{tid}", tuple(pts[0][0]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
#         try:
#             cv2.imshow('tags', vis); cv2.waitKey(120); cv2.destroyAllWindows()
#         except cv2.error: pass

#     K = intr.intrinsic_matrix
#     poses = {}
#     half = TAG_SIZE/2.0
#     objp = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]],
#                     dtype=np.float64)
#     for d in dets:
#         tid = d.tag_id if hasattr(d,'tag_id') else d.id
#         imgp = d.corners.astype(np.float64)
#         ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, None)
#         if not ok: continue
#         R, _ = cv2.Rodrigues(rvec)
#         T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.ravel()
#         poses[tid] = T  # T_cam_tag
#     return poses

# def invert(T):
#     R = T[:3,:3]; t = T[:3,3]
#     Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
#     return Ti

# def average_SE3(transforms):
#     """Average SE3 by quaternion mean + mean translation."""
#     if not transforms: return np.eye(4)
#     def quat_from_R(R):
#         q = np.empty(4); tr = np.trace(R)
#         if tr > 0:
#             s = np.sqrt(tr+1.0)*2; q[0]=0.25*s
#             q[1]=(R[2,1]-R[1,2])/s; q[2]=(R[0,2]-R[2,0])/s; q[3]=(R[1,0]-R[0,1])/s
#         else:
#             i = np.argmax([R[0,0],R[1,1],R[2,2]])
#             if i==0:
#                 s=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; q[0]=(R[2,1]-R[1,2])/s
#                 q[1]=0.25*s; q[2]=(R[0,1]+R[1,0])/s; q[3]=(R[0,2]+R[2,0])/s
#             elif i==1:
#                 s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; q[0]=(R[0,2]-R[2,0])/s
#                 q[1]=(R[0,1]+R[1,0])/s; q[2]=0.25*s; q[3]=(R[1,2]+R[2,1])/s
#             else:
#                 s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; q[0]=(R[1,0]-R[0,1])/s
#                 q[1]=(R[0,2]+R[2,0])/s; q[2]=(R[1,2]+R[2,1])/s; q[3]=0.25*s
#         return q/np.linalg.norm(q)
#     def R_from_quat(q):
#         w,x,y,z = q
#         return np.array([
#             [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
#             [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
#             [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
#         ])
#     qs, ts = [], []
#     for T in transforms:
#         qs.append(quat_from_R(T[:3,:3])); ts.append(T[:3,3])
#     qs = np.array(qs)
#     M = (qs.T @ qs) / qs.shape[0]
#     vals, vecs = np.linalg.eigh(M)
#     q_avg = vecs[:, np.argmax(vals)]
#     if q_avg[0] < 0: q_avg = -q_avg
#     R = R_from_quat(q_avg)
#     t = np.mean(ts, axis=0)
#     Tout = np.eye(4); Tout[:3,:3]=R; Tout[:3,3]=t
#     return Tout

# def init_from_tag_poses(poses_src, poses_tgt):
#     """
#     Build src->tgt init from any shared tag pose(s).
#     Works with ≥1 shared ID; averages multiple.
#     """
#     common = sorted(set(poses_src) & set(poses_tgt))
#     if not common:
#         return np.eye(4), []
#     estimates = []
#     for tid in common:
#         Tsrc_tag = poses_src[tid]    # cam_src <- tag
#         Ttgt_tag = poses_tgt[tid]    # cam_tgt <- tag
#         Tsrc_to_tgt = Ttgt_tag @ invert(Tsrc_tag)
#         estimates.append(Tsrc_to_tgt)
#     return average_SE3(estimates), common

# # ---------------- ICP utils -----------------------
# def _est_normals(pc, radius, max_nn=50):
#     pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
#     pc.orient_normals_consistent_tangent_plane(10); return pc

# def run_icp_to_target(src_raw, tgt_raw, init=np.eye(4)):
#     T = init.copy()

#     # Coarse (3 mm)
#     src = src_raw.voxel_down_sample(ICP_VOXEL)
#     tgt = tgt_raw.voxel_down_sample(ICP_VOXEL)
#     if len(src.points)==0 or len(tgt.points)==0: return T
#     _est_normals(src, ICP_VOXEL*3); _est_normals(tgt, ICP_VOXEL*3)

#     # Local near origin (tags should be near origin after init)
#     box_xy = 0.12/2.0
#     bbox = o3d.geometry.AxisAlignedBoundingBox([-box_xy,-box_xy,-0.05],
#                                                [ box_xy, box_xy, 0.05])
#     sub_tgt, sub_src = tgt.crop(bbox), src.crop(bbox)
#     if len(sub_tgt.points)>50 and len(sub_src.points)>50:
#         res_loc = o3d.pipelines.registration.registration_icp(
#             sub_src, sub_tgt, ICP_DIST_LOCAL, T,
#             o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
#         T = res_loc.transformation

#     # Global (2 cm)
#     res_glob = o3d.pipelines.registration.registration_icp(
#         src, tgt, ICP_DIST_GLOBAL, T,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
#     T = res_glob.transformation

#     # Very fine (1 mm / 3 mm)
#     src_r = src_raw.voxel_down_sample(REFINE_VOXEL)
#     tgt_r = tgt_raw.voxel_down_sample(REFINE_VOXEL)
#     if len(src_r.points)==0 or len(tgt_r.points)==0: return T
#     _est_normals(src_r, REFINE_VOXEL*3); _est_normals(tgt_r, REFINE_VOXEL*3)
#     res_ref = o3d.pipelines.registration.registration_icp(
#         src_r, tgt_r, ICP_DIST_REFINE, T,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
#     return res_ref.transformation

# # ---------------- Main ----------------------------
# def main():
#     t0 = time.time()
#     print('Running 3-cam alignment (pose-init + ICP, no pose-graph)...')
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_dir = f"output_{ts}"; os.makedirs(out_dir, exist_ok=True)

#     # Capture
#     r0,i0,raw0 = capture_seq(SERIAL0)
#     r1,i1,raw1 = capture_seq(SERIAL1)
#     r2,i2,raw2 = capture_seq(SERIAL2)

#     # Save one RGB
#     for idx, raws in enumerate([raw0, raw1, raw2]):
#         color_np, depth_np, _ds = raws[0]
#         cv2.imwrite(os.path.join(out_dir, f"cam{idx}_color_{ts}.png"),
#                     cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))

#     # Clouds (camera frames)
#     pc0 = build_cloud(r0, i0, " cam0"); o3d.io.write_point_cloud(os.path.join(out_dir,f"cam0_raw_{ts}.ply"), pc0)
#     pc1 = build_cloud(r1, i1, " cam1"); o3d.io.write_point_cloud(os.path.join(out_dir,f"cam1_raw_{ts}.ply"), pc1)
#     pc2 = build_cloud(r2, i2, " cam2"); o3d.io.write_point_cloud(os.path.join(out_dir,f"cam2_raw_{ts}.ply"), pc2)

#     # Tag poses in camera frames
#     poses0 = detect_tag_poses(raw0[0][0], i0)
#     poses1 = detect_tag_poses(raw1[0][0], i1)
#     poses2 = detect_tag_poses(raw2[0][0], i2)

#     # Init cam1->cam0 from any shared tag(s)
#     init10, common10 = init_from_tag_poses(poses1, poses0)
#     if not common10:
#         print("[warn] cam1/cam0 share 0 tags — using identity init")
#     T1_to_0 = run_icp_to_target(pc1, pc0, init=init10); pc1.transform(T1_to_0)

#     # Build stronger target (cam0+cam1) and init cam2
#     merged_target = pc0 + pc1
#     init20, common20 = init_from_tag_poses(poses2, poses0)
#     if not common20:
#         # try vs cam1 if no overlap with cam0
#         init21, common21 = init_from_tag_poses(poses2, poses1)
#         if common21: init20 = init21
#         else: print("[warn] cam2 shares 0 tags with cam0/cam1 — using identity init")
#     T2_to_01 = run_icp_to_target(pc2, merged_target, init=init20); pc2.transform(T2_to_01)

#     # Merge & save (all in cam0 frame)
#     merged = (pc0 + pc1 + pc2).voxel_down_sample(VOXEL_SIZE)
#     out_ply = os.path.join(out_dir, f"merged_{ts}.ply")
#     o3d.io.write_point_cloud(out_ply, merged)

#     print(f"Shared tags: cam1↔cam0={len(common10)}, cam2↔cam0={len(common20)}")
#     print(f"Saved all outputs in {out_dir}")
#     print(f"Total runtime: {time.time()-t0:.2f} s")

# if __name__ == '__main__':
#     main()
