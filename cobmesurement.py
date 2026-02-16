import os
import numpy as np
import pandas as pd
import open3d as o3d

# =======================
# CONFIG
# =======================
IN_PATH = "finalcob.pcd"

N_RINGS = 15
START_POS = 0.05
END_POS = 0.95
SLICE_THICKNESS_FRAC = 0.02
ROBUST_Q = 0.95
REMOVE_OUTLIERS = True

POINT_SIZE = 2.5
SAVE_CSV = True

# If your cloud units are decimeters (dm) and you want centimeters:
UNIT_SCALE_TO_CM = 10.0   # dm -> cm  (change to 100.0 if units are meters)

LABEL_SCALE = 0.10
LABEL_OFFSET = 1.15
LABEL_COLOR = (1.0, 1.0, 1.0)
WORLD_UP = np.array([0.0, 0.0, 1.0])

LETTER_SCALE_MULT = 1.0
LETTER_GAP_MULT = 0.45

# Better L label positioning:
SHOW_LENGTH_LABEL = True
L_LABEL_SCALE_FRAC_OF_LENGTH = 0.06   # text size ~ fraction of length
L_LABEL_ALONG_AXIS = 0.02             # move from tip along axis (in length fraction)
L_LABEL_OUTWARD = 0.22                # move outward from cob (in length fraction)
# =======================


# -------------------------
# Measurement helpers
# -------------------------
def pca_axis(points: np.ndarray):
    center = points.mean(axis=0)
    X = points - center
    C = (X.T @ X) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    return center, axis


def slice_mask(points, center, axis, t0, half_thickness):
    t = (points - center) @ axis
    return (t >= (t0 - half_thickness)) & (t <= (t0 + half_thickness))


def diameter_from_slice(points_slice, center, axis, robust_q):
    t = (points_slice - center) @ axis
    proj = np.outer(t, axis)
    radial = (points_slice - center) - proj
    r = np.linalg.norm(radial, axis=1)
    rad = float(np.quantile(r, robust_q))
    return 2.0 * rad, rad


# -------------------------
# Open3D helpers
# -------------------------
def make_lineset(points, lines, colors):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def merge_linesets(linesets):
    all_pts = []
    all_lines = []
    all_cols = []
    base = 0
    for ls in linesets:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        cols = np.asarray(ls.colors)
        all_pts.extend(pts.tolist())
        all_lines.extend((lines + base).tolist())
        all_cols.extend(cols.tolist())
        base += len(pts)
    return make_lineset(np.array(all_pts, float), np.array(all_lines, int), np.array(all_cols, float))


def make_axis_lineset(p0, p1, color=(1, 0, 0)):
    pts = np.vstack([p0, p1])
    lines = np.array([[0, 1]], dtype=np.int32)
    cols = np.array([color], dtype=np.float64)
    return make_lineset(pts, lines, cols)


def make_ring(center3d, axis_unit, radius, color=(0, 1, 0), n=180):
    axis_unit = axis_unit / np.linalg.norm(axis_unit)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis_unit)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_unit, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(axis_unit, u)

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring_pts = np.array([center3d + radius*(np.cos(a)*u + np.sin(a)*v) for a in angles])

    lines = np.array([[i, (i+1) % n] for i in range(n)], dtype=np.int32)
    cols = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
    ring = make_lineset(ring_pts, lines, cols)
    return ring, u, v


def build_label_basis(out_dir, world_up):
    n = out_dir / np.linalg.norm(out_dir)
    up = world_up / np.linalg.norm(world_up)
    v = up - np.dot(up, n) * n
    if np.linalg.norm(v) < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
        v = up - np.dot(up, n) * n
    v = v / np.linalg.norm(v)
    u = np.cross(v, n)
    u = u / np.linalg.norm(u)
    return u, v


def place_label_lineset(label_ls, origin3d, u, v):
    pts2 = np.asarray(label_ls.points)
    pts3 = origin3d + pts2[:, 0:1]*u.reshape(1, 3) + pts2[:, 1:2]*v.reshape(1, 3)
    label_ls.points = o3d.utility.Vector3dVector(pts3)
    return label_ls


# -------------------------
# 7-seg digits
# -------------------------
SEG = {
    "a": ((0.1, 2.0), (0.9, 2.0)),
    "b": ((0.9, 2.0), (0.9, 1.1)),
    "c": ((0.9, 0.9), (0.9, 0.0)),
    "d": ((0.1, 0.0), (0.9, 0.0)),
    "e": ((0.1, 0.0), (0.1, 0.9)),
    "f": ((0.1, 1.1), (0.1, 2.0)),
    "g": ((0.1, 1.0), (0.9, 1.0)),
}
DIGIT_TO_SEGS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abged",
    "3": "abgcd",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgcde",
    "7": "abc",
    "8": "abcdefg",
    "9": "abfgcd",
    "-": "g",
}

def make_7seg_char(ch, origin2d, s, color):
    ox, oy = origin2d
    pts, lines, cols = [], [], []

    def add_line(p, q):
        i0 = len(pts)
        pts.append([p[0], p[1], 0.0])
        pts.append([q[0], q[1], 0.0])
        lines.append([i0, i0 + 1])
        cols.append(color)

    if ch in DIGIT_TO_SEGS:
        for segname in DIGIT_TO_SEGS[ch]:
            p, q = SEG[segname]
            p2 = (ox + s * p[0], oy + s * p[1])
            q2 = (ox + s * q[0], oy + s * q[1])
            add_line(p2, q2)
    elif ch == ".":
        add_line((ox + s * 0.9, oy + s * 0.1), (ox + s * 0.9, oy + s * 0.25))

    return make_lineset(np.array(pts, float), np.array(lines, int), np.array(cols, float))


def make_7seg_string(text, s, color):
    char_w = 1.2 * s
    x = 0.0
    parts = []
    for ch in text:
        parts.append(make_7seg_char(ch, origin2d=(x, 0.0), s=s, color=color))
        x += char_w
    return merge_linesets(parts), x


# -------------------------
# Stroke letters: c, m, L, =
# -------------------------
def stroke_letter(letter, origin2d, s, color):
    ox, oy = origin2d
    segs = []
    def seg(p, q):
        segs.append(((ox + s * p[0], oy + s * p[1]), (ox + s * q[0], oy + s * q[1])))

    if letter == "c":
        seg((0.9, 1.7), (0.4, 2.0))
        seg((0.4, 2.0), (0.1, 1.5))
        seg((0.1, 1.5), (0.1, 0.5))
        seg((0.1, 0.5), (0.4, 0.0))
        seg((0.4, 0.0), (0.9, 0.3))
    elif letter == "m":
        seg((0.1, 0.0), (0.1, 2.0))
        seg((0.1, 2.0), (0.5, 1.2))
        seg((0.5, 1.2), (0.9, 2.0))
        seg((0.9, 2.0), (0.9, 0.0))
    elif letter == "L":
        seg((0.1, 2.0), (0.1, 0.0))
        seg((0.1, 0.0), (0.9, 0.0))
    elif letter == "=":
        seg((0.1, 1.35), (0.9, 1.35))
        seg((0.1, 0.65), (0.9, 0.65))

    pts = []
    lines = []
    cols = []
    for p, q in segs:
        i0 = len(pts)
        pts.append([p[0], p[1], 0.0])
        pts.append([q[0], q[1], 0.0])
        lines.append([i0, i0 + 1])
        cols.append(color)

    return make_lineset(np.array(pts, float), np.array(lines, int), np.array(cols, float))


def make_stroke_word(word, start_x, s, color):
    char_w = 1.2 * s
    x = start_x
    parts = []
    for ch in word:
        parts.append(stroke_letter(ch, origin2d=(x, 0.0), s=s, color=color))
        x += char_w
    return merge_linesets(parts), x


# -------------------------
# Main
# -------------------------
def main():
    pcd = o3d.io.read_point_cloud(IN_PATH)
    if REMOVE_OUTLIERS:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=4.0)

    pts = np.asarray(pcd.points)
    center, axis = pca_axis(pts)

    t_all = (pts - center) @ axis
    t_min, t_max = float(t_all.min()), float(t_all.max())
    L_raw = t_max - t_min

    # axis
    p0 = center + t_min * axis
    p1 = center + t_max * axis
    axis_ls = make_axis_lineset(p0, p1, color=(1, 0, 0))

    half_thick = 0.5 * (SLICE_THICKNESS_FRAC * L_raw)
    positions = np.linspace(START_POS, END_POS, N_RINGS)

    palette = [
        (0.0, 1.0, 0.0),
        (0.0, 0.7, 1.0),
        (1.0, 0.2, 1.0),
        (1.0, 0.8, 0.0),
        (1.0, 0.4, 0.2),
    ]

    rings = []
    labels = []
    rows = []

    L_cm = L_raw * UNIT_SCALE_TO_CM
    print("\n=== VALUES (cm) ===")
    print(f"L = {L_cm:.2f} cm\n")

    first_out_dir = None  # use for L label placement

    for i, pos01 in enumerate(positions, start=1):
        t0 = t_min + pos01 * (t_max - t_min)
        mask = slice_mask(pts, center, axis, t0, half_thick)
        n_in = int(mask.sum())
        col = palette[(i - 1) % len(palette)]

        if n_in < 80:
            rows.append([i, pos01, np.nan, n_in])
            continue

        d_raw, r_raw = diameter_from_slice(pts[mask], center, axis, ROBUST_Q)
        d_cm = d_raw * UNIT_SCALE_TO_CM
        rows.append([i, pos01, d_cm, n_in])

        c3 = center + t0 * axis
        ring, u_ring, v_ring = make_ring(c3, axis, r_raw, color=col, n=220)
        rings.append(ring)

        # remember an outward direction from the first valid ring
        if first_out_dir is None:
            first_out_dir = u_ring.copy()

        # label "<number> cm"
        digit_scale = LABEL_SCALE * r_raw
        num_text = f"{d_cm:.1f}"
        num_ls, end_x = make_7seg_string(num_text, s=digit_scale, color=LABEL_COLOR)

        gap = LETTER_GAP_MULT * digit_scale * 1.2
        cm_ls, _ = make_stroke_word("cm", start_x=end_x + gap,
                                    s=digit_scale * LETTER_SCALE_MULT, color=LABEL_COLOR)

        label_local = merge_linesets([num_ls, cm_ls])

        out_dir = u_ring
        u_text, v_text = build_label_basis(out_dir, WORLD_UP)
        label_origin = c3 + (r_raw * LABEL_OFFSET) * out_dir
        labels.append(place_label_lineset(label_local, label_origin, u_text, v_text))

        print(f"Ring {i:02d} @ {pos01*100:5.1f}%  D = {d_cm:6.2f} cm  (slice n={n_in})")

    # ---- Length label in 3D (better positioned using first ring outward direction) ----
    if SHOW_LENGTH_LABEL:
        if first_out_dir is None:
            first_out_dir = np.array([1.0, 0.0, 0.0])

        # Build local "L= <number> cm"
        l_scale = max(L_LABEL_SCALE_FRAC_OF_LENGTH * L_raw, 1e-6)

        prefix_ls, x_end = make_stroke_word("L=", start_x=0.0, s=l_scale, color=LABEL_COLOR)

        num_ls, x_num_end = make_7seg_string(f"{L_cm:.1f}", s=l_scale, color=LABEL_COLOR)
        num_pts = np.asarray(num_ls.points).copy()
        num_pts[:, 0] += x_end + (0.4 * l_scale)
        num_ls.points = o3d.utility.Vector3dVector(num_pts)

        cm_ls, _ = make_stroke_word("cm", start_x=(x_end + 0.4 * l_scale) + x_num_end + (0.4 * l_scale),
                                    s=l_scale, color=LABEL_COLOR)

        L_label_local = merge_linesets([prefix_ls, num_ls, cm_ls])

        # Place near top end (p1) slightly inside along axis and pushed outward
        out_dir = first_out_dir / np.linalg.norm(first_out_dir)
        uL, vL = build_label_basis(out_dir, WORLD_UP)

        origin = (p1 - (L_LABEL_ALONG_AXIS * L_raw) * axis) + (L_LABEL_OUTWARD * L_raw) * out_dir
        labels.append(place_label_lineset(L_label_local, origin, uL, vL))

    # save csv
    df = pd.DataFrame(rows, columns=["ring_id", "pos_0to1", "diameter_cm", "slice_points"])
    if SAVE_CSV:
        out_csv = os.path.splitext(IN_PATH)[0] + "_ring_values_cm.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved CSV: {out_csv}")

    geoms = [pcd, axis_ls] + rings + labels
    vis = o3d.visualization.Visualizer()
    vis.create_window("Cob diameters + LENGTH (cm) in 3D", 1280, 720)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(POINT_SIZE)
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
