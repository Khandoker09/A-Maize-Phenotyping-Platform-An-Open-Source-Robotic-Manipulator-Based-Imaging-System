#!/usr/bin/env python3
"""
live_tag_check_3cams.py
Live preview from 3 cameras showing AprilTag detections, pixel sizes, and per-camera status.

- Draws ALL tags per camera with ID + edge size in pixels.
- Shows detection count and smallest edge size per camera.
- Window border logic is configurable:
    * REQUIRE_SAME_ID = True  -> border is green only if TAG_ID is seen.
    * REQUIRE_SAME_ID = False -> border is green if any tag is seen.
- Tags smaller than MIN_EDGE_PX are outlined in ORANGE as a warning.

ESC to quit.
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time

# --- CONFIG ---
SERIALS = [
    '311322303919',  # cam0
    '252222300195',  # cam1
    '252222300763'   # cam2
]

TAG_ID = 3              # required tag ID (set to None to not require a specific ID)
REQUIRE_SAME_ID = True  # True: need TAG_ID for green border; False: any tag OK
MIN_EDGE_PX = 24        # warn if detected tag's edge is below this many pixels
WARMUP = 1.0

# AprilTag setup
try:
    import apriltag
    detector = apriltag.Detector(apriltag.DetectorOptions(
        families='tag36h11',
        border=1,
        quad_decimate=1.0,   # 1.0 = no decimation; increase (>1) makes it faster but less sensitive
        quad_sigma=0.0,
        refine_edges=True,
        nthreads=2           # bump if your CPU has more cores
    ))
except ImportError:
    from pupil_apriltags import Detector as PDetector
    detector = PDetector(families='tag36h11')

def start_cam(serial):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    time.sleep(WARMUP)
    return pipe

def detect_and_draw(color_bgr, serial):
    """
    Returns:
        vis_bgr  - annotated image
        statusOK - whether this camera meets the border rule
        count    - number of tags detected
        min_edge - smallest edge (pixels) among detections (np.inf if none)
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    count = 0
    saw_required = False
    min_edge = np.inf

    for r in results:
        tid = r.tag_id if hasattr(r, 'tag_id') else r.id
        corners = r.corners.astype(int)
        pts = corners.reshape(-1, 1, 2)

        # Edge size estimate (average of top and side)
        top_edge  = np.linalg.norm(r.corners[0] - r.corners[1])
        side_edge = np.linalg.norm(r.corners[1] - r.corners[2])
        edge_px = float((top_edge + side_edge) / 2.0)
        min_edge = min(min_edge, edge_px)

        # Color by size: orange if too small, green otherwise
        col = (0,165,255) if edge_px < MIN_EDGE_PX else (0,255,0)
        cv2.polylines(color_bgr, [pts], True, col, 2)
        label = f"ID={tid} {edge_px:.1f}px"
        org = tuple(pts[0][0])
        cv2.putText(color_bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        count += 1
        if TAG_ID is None or tid == TAG_ID:
            saw_required = True

    # Camera label + summary
    cv2.putText(color_bgr, f"Cam {serial}", (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    if count > 0:
        summary = f"tags:{count}  minEdge:{min_edge:.1f}px  (min ok≥{MIN_EDGE_PX})"
    else:
        summary = "tags:0"
    cv2.putText(color_bgr, summary, (12,58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

    # Border rule
    statusOK = saw_required if REQUIRE_SAME_ID else (count > 0)
    return color_bgr, statusOK, count, min_edge

def main():
    pipes = {s: start_cam(s) for s in SERIALS}

    try:
        while True:
            frames = {}
            for s in SERIALS:
                cf = pipes[s].wait_for_frames().get_color_frame()
                if not cf:
                    continue
                color = np.asanyarray(cf.get_data())

                vis, ok, cnt, min_edge = detect_and_draw(color, s)

                # Border color: green if OK, red otherwise
                border_color = (0, 255, 0) if ok else (0, 0, 255)
                vis = cv2.copyMakeBorder(vis, 6, 6, 6, 6,
                                         cv2.BORDER_CONSTANT, value=border_color)
                frames[s] = vis

            if len(frames) == 3:
                combined = np.hstack([frames[SERIALS[0]],
                                      frames[SERIALS[1]],
                                      frames[SERIALS[2]]])
                title = "3-Camera Tag Check"
                if REQUIRE_SAME_ID and TAG_ID is not None:
                    title += f"  [Require ID={TAG_ID}]"
                else:
                    title += "  [Any tag OK]"
                cv2.imshow(title, combined)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        for p in pipes.values():
            p.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
