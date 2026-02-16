#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RoArm-M2 simple kinematics simulator (FK) with:
- Tkinter UI to input angles (base/shoulder/elbow)
- 3D visualization (NO grid lines)
- Optional scan simulation (rings + sweep)
- Save animation as GIF (needs pillow)

Install:
  pip install numpy matplotlib pillow
Run:
  python kinematic_ui.py
"""

import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation, PillowWriter


# -----------------------------
# Helpers (same spirit as your app)
# -----------------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def clip_base_deg(angle, safe_limit):
    return clamp(angle, -safe_limit, safe_limit)

def plan_angles_centered(center_deg, half_span_deg, count, safe_limit_deg, start_b_deg=None):
    half_span_deg = min(float(half_span_deg), float(safe_limit_deg))
    center_deg = clip_base_deg(float(center_deg), safe_limit_deg)

    lo = clip_base_deg(center_deg - half_span_deg, safe_limit_deg)
    hi = clip_base_deg(center_deg + half_span_deg, safe_limit_deg)

    if count <= 1:
        return [clip_base_deg(center_deg, safe_limit_deg)]

    step = (hi - lo) / float(count - 1)
    arr = [lo + i * step for i in range(count)]

    if start_b_deg is None:
        return arr

    start_b_deg = clip_base_deg(float(start_b_deg), safe_limit_deg)
    best_i = min(range(len(arr)), key=lambda i: abs(arr[i] - start_b_deg))
    return arr[best_i:] + arr[:best_i]


# -----------------------------
# Forward kinematics (simple 2-link + base yaw)
# -----------------------------
def fk_points(cfg, b_deg, s_deg, e_deg):
    """
    Returns 3 points (shoulder, elbow, wrist) as (x,y,z) in cm.

    Model:
      - base yaw b rotates in XY
      - shoulder pitch s, elbow pitch e in vertical plane
      - link lengths L1, L2
      - offsets X0, Z0
    """
    L1 = float(cfg["L1_cm"])
    L2 = float(cfg["L2_cm"])
    X0 = float(cfg.get("X0_cm", 0.0))
    Z0 = float(cfg.get("Z0_cm", 0.0))

    # optional angle offsets
    b = math.radians(b_deg + cfg.get("b_offset_deg", 0.0))
    s = math.radians(s_deg + cfg.get("s_offset_deg", 0.0))
    e = math.radians(e_deg + cfg.get("e_offset_deg", 0.0))

    shoulder = np.array([X0 * math.cos(b), X0 * math.sin(b), Z0])

    # elbow
    r_el = X0 + L1 * math.cos(s)
    z_el = Z0 + L1 * math.sin(s)
    elbow = np.array([r_el * math.cos(b), r_el * math.sin(b), z_el])

    # wrist
    r_wr = X0 + L1 * math.cos(s) + L2 * math.cos(s + e)
    z_wr = Z0 + L1 * math.sin(s) + L2 * math.sin(s + e)
    wrist = np.array([r_wr * math.cos(b), r_wr * math.sin(b), z_wr])

    return shoulder, elbow, wrist


def build_scan_poses(cfg, total_images=120, start_base_deg=0.0):
    """
    Builds poses like your scan:
    - 3 rings (shoulder+elbow pairs)
    - base sweep across each ring
    """
    # Default rings from cfg
    rings = [
        (cfg["pose_low_sh_deg"],  cfg["pose_low_el_deg"]),
        ((cfg["pose_low_sh_deg"] + cfg["pose_high_sh_deg"]) / 2.0,
         (cfg["pose_low_el_deg"] + cfg["pose_high_el_deg"]) / 2.0),
        (cfg["pose_high_sh_deg"], cfg["pose_high_el_deg"]),
    ]

    # distribute frames among rings
    per = [total_images // 3] * 3
    for i in range(total_images % 3):
        per[i] += 1

    poses = []
    center = float(cfg["base_center_deg"])
    half = float(cfg["base_half_span_deg"])
    safe = float(cfg["safe_limit_deg"])

    for ring_i, ((s_deg, e_deg), ring_count) in enumerate(zip(rings, per), start=1):
        base_plan = plan_angles_centered(
            center_deg=center,
            half_span_deg=half,
            count=ring_count,
            safe_limit_deg=safe,
            start_b_deg=(start_base_deg if ring_i == 1 else None),
        )
        for b_deg in base_plan:
            poses.append((b_deg, s_deg, e_deg))

    return poses


# -----------------------------
# UI App
# -----------------------------
DEFAULT_CFG = {
    # --- kinematic parameters (MEASURE if possible) ---
    "L1_cm": 11.5,
    "L2_cm": 12.0,
    "Z0_cm": 6.0,
    "X0_cm": 0.0,

    # offsets (set to align "robot angles" to "math angles")
    "b_offset_deg": 0.0,
    "s_offset_deg": 0.0,
    "e_offset_deg": 0.0,

    # --- scan defaults (like your app) ---
    "safe_limit_deg": 175.0,
    "base_center_deg": 90.0,
    "base_half_span_deg": 155.0,
    "pose_low_sh_deg": -84.0,
    "pose_low_el_deg": 92.0,
    "pose_high_sh_deg": -60.0,
    "pose_high_el_deg": 70.0,
}


class KinematicUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RoArm-M2 Kinematics Simulator (FK)")
        self.geometry("1150x750")

        self.cfg = dict(DEFAULT_CFG)
        self.ani = None
        self.scan_poses = []
        self.trace = None

        self._build_ui()
        self._build_plot()
        self._apply_no_grid_style()
        self._draw_single_pose()

    def _build_ui(self):
        wrap = ttk.Frame(self)
        wrap.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(wrap)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right = ttk.Frame(wrap)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Inputs: angles ---
        boxA = ttk.LabelFrame(left, text="Angles (deg)")
        boxA.pack(fill=tk.X, pady=(0, 10))

        self.var_b = tk.DoubleVar(value=0.0)
        self.var_s = tk.DoubleVar(value=-82.0)
        self.var_e = tk.DoubleVar(value=90.0)

        self._row_entry(boxA, "Base b:", self.var_b)
        self._row_entry(boxA, "Shoulder s:", self.var_s)
        self._row_entry(boxA, "Elbow e:", self.var_e)

        ttk.Button(boxA, text="Update pose", command=self._draw_single_pose).pack(fill=tk.X, padx=8, pady=(6, 8))

        # --- Kinematic params ---
        boxK = ttk.LabelFrame(left, text="Kinematic params (cm)")
        boxK.pack(fill=tk.X, pady=(0, 10))

        self.var_L1 = tk.DoubleVar(value=self.cfg["L1_cm"])
        self.var_L2 = tk.DoubleVar(value=self.cfg["L2_cm"])
        self.var_Z0 = tk.DoubleVar(value=self.cfg["Z0_cm"])
        self.var_X0 = tk.DoubleVar(value=self.cfg["X0_cm"])

        self._row_entry(boxK, "L1 (shoulder→elbow):", self.var_L1)
        self._row_entry(boxK, "L2 (elbow→wrist):", self.var_L2)
        self._row_entry(boxK, "Z0 (shoulder height):", self.var_Z0)
        self._row_entry(boxK, "X0 (shoulder offset):", self.var_X0)

        ttk.Button(boxK, text="Apply params", command=self._apply_params).pack(fill=tk.X, padx=8, pady=(6, 8))

        # --- Scan simulation ---
        boxS = ttk.LabelFrame(left, text="Scan simulation (rings + sweep)")
        boxS.pack(fill=tk.X, pady=(0, 10))

        self.var_total = tk.IntVar(value=120)
        self.var_center = tk.DoubleVar(value=self.cfg["base_center_deg"])
        self.var_half = tk.DoubleVar(value=self.cfg["base_half_span_deg"])
        self.var_safe = tk.DoubleVar(value=self.cfg["safe_limit_deg"])

        self._row_entry(boxS, "Total frames:", self.var_total)
        self._row_entry(boxS, "Base center:", self.var_center)
        self._row_entry(boxS, "Half span:", self.var_half)
        self._row_entry(boxS, "Safe limit:", self.var_safe)

        self.var_show_trace = tk.BooleanVar(value=True)
        ttk.Checkbutton(boxS, text="Show wrist trace", variable=self.var_show_trace, command=self._draw_single_pose)\
            .pack(anchor="w", padx=8, pady=(0, 6))

        rowbtn = ttk.Frame(boxS)
        rowbtn.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(rowbtn, text="Play scan", command=self._play_scan).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(rowbtn, text="Stop", command=self._stop_animation).pack(side=tk.LEFT, padx=6)

        ttk.Button(boxS, text="Save scan as GIF…", command=self._save_gif).pack(fill=tk.X, padx=8, pady=(0, 8))

        # --- Status ---
        self.lbl = ttk.Label(left, text="Ready", foreground="#333")
        self.lbl.pack(fill=tk.X, pady=(6, 0))

        # Plot host
        self.plot_host = right

    def _row_entry(self, parent, label, var):
        r = ttk.Frame(parent)
        r.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(r, text=label, width=22).pack(side=tk.LEFT)
        e = ttk.Entry(r, textvariable=var, width=10)
        e.pack(side=tk.LEFT)
        return e

    def _build_plot(self):
        self.fig = plt.Figure(figsize=(7.2, 5.8), dpi=110)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_host)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # artists
        (self.arm_line,) = self.ax.plot([], [], [], linewidth=3)
        (self.wrist_pt,) = self.ax.plot([], [], [], marker="o")
        (self.trace_line,) = self.ax.plot([], [], [], linewidth=1)

        self.text2d = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes)

    def _apply_no_grid_style(self):
        # no grid lines
        self.ax.grid(False)

        # also hide the "pane grid" lines (matplotlib versions differ)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            try:
                axis._axinfo["grid"]["linewidth"] = 0
            except Exception:
                pass

        # optional: remove pane fill for clean look
        try:
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
        except Exception:
            pass

        self.ax.set_xlabel("X (cm)")
        self.ax.set_ylabel("Y (cm)")
        self.ax.set_zlabel("Z (cm)")
        self.ax.set_title("RoArm-M2 FK Visualization")

    def _apply_params(self):
        try:
            self.cfg["L1_cm"] = float(self.var_L1.get())
            self.cfg["L2_cm"] = float(self.var_L2.get())
            self.cfg["Z0_cm"] = float(self.var_Z0.get())
            self.cfg["X0_cm"] = float(self.var_X0.get())
            self.lbl.config(text="Params applied")
        except Exception as e:
            messagebox.showerror("Params", f"Invalid param: {e}")

    def _sync_scan_cfg(self):
        self.cfg["base_center_deg"] = float(self.var_center.get())
        self.cfg["base_half_span_deg"] = float(self.var_half.get())
        self.cfg["safe_limit_deg"] = float(self.var_safe.get())

    def _set_axes_limits(self, wrists_xyz):
        wrists_xyz = np.asarray(wrists_xyz, dtype=float)
        max_r = max(np.linalg.norm(wrists_xyz[:, :2], axis=1).max(), 1.0)
        zmin, zmax = wrists_xyz[:, 2].min(), wrists_xyz[:, 2].max()
        pad = 5.0
        self.ax.set_xlim(-max_r - pad, max_r + pad)
        self.ax.set_ylim(-max_r - pad, max_r + pad)
        self.ax.set_zlim(min(0.0, zmin - pad), zmax + pad)

    def _draw_single_pose(self):
        self._stop_animation()

        # update cfg from entries (safe now, no recursion)
        self._apply_params()

        b = float(self.var_b.get())
        s = float(self.var_s.get())
        e = float(self.var_e.get())

        shoulder, elbow, wrist = fk_points(self.cfg, b, s, e)

        xs = [shoulder[0], elbow[0], wrist[0]]
        ys = [shoulder[1], elbow[1], wrist[1]]
        zs = [shoulder[2], elbow[2], wrist[2]]

        self.arm_line.set_data(xs, ys)
        self.arm_line.set_3d_properties(zs)

        self.wrist_pt.set_data([wrist[0]], [wrist[1]])
        self.wrist_pt.set_3d_properties([wrist[2]])

        if self.var_show_trace.get():
            self.trace_line.set_data([wrist[0]], [wrist[1]])
            self.trace_line.set_3d_properties([wrist[2]])
        else:
            self.trace_line.set_data([], [])
            self.trace_line.set_3d_properties([])

        self._set_axes_limits([wrist])

        self.text2d.set_text(
            f"b={b:.1f}  s={s:.1f}  e={e:.1f}\n"
            f"Wrist: ({wrist[0]:.1f}, {wrist[1]:.1f}, {wrist[2]:.1f}) cm"
        )

        self.canvas.draw_idle()
        self.lbl.config(text="Pose updated")



    def _play_scan(self):
        self._stop_animation()
        self._apply_params()
        self._sync_scan_cfg()

        total = int(self.var_total.get())
        if total < 3:
            messagebox.showerror("Scan", "Total frames must be >= 3")
            return

        self.scan_poses = build_scan_poses(self.cfg, total_images=total, start_base_deg=float(self.var_b.get()))
        pts = [fk_points(self.cfg, b, s, e) for (b, s, e) in self.scan_poses]
        wrists = np.array([p[2] for p in pts])
        self._set_axes_limits(wrists)

        self.trace = np.zeros((len(pts), 3), dtype=float)

        def init():
            self.arm_line.set_data([], [])
            self.arm_line.set_3d_properties([])
            self.wrist_pt.set_data([], [])
            self.wrist_pt.set_3d_properties([])
            self.trace_line.set_data([], [])
            self.trace_line.set_3d_properties([])
            self.text2d.set_text("")
            return self.arm_line, self.wrist_pt, self.trace_line

        def update(i):
            shoulder, elbow, wrist = pts[i]
            xs = [shoulder[0], elbow[0], wrist[0]]
            ys = [shoulder[1], elbow[1], wrist[1]]
            zs = [shoulder[2], elbow[2], wrist[2]]

            self.arm_line.set_data(xs, ys)
            self.arm_line.set_3d_properties(zs)

            self.wrist_pt.set_data([wrist[0]], [wrist[1]])
            self.wrist_pt.set_3d_properties([wrist[2]])

            if self.var_show_trace.get():
                self.trace[i] = wrist
                tr = self.trace[: i + 1]
                self.trace_line.set_data(tr[:, 0], tr[:, 1])
                self.trace_line.set_3d_properties(tr[:, 2])
            else:
                self.trace_line.set_data([], [])
                self.trace_line.set_3d_properties([])

            b_deg, s_deg, e_deg = self.scan_poses[i]
            self.text2d.set_text(f"step {i+1}/{len(self.scan_poses)}  |  b={b_deg:.1f}  s={s_deg:.1f}  e={e_deg:.1f}")
            return self.arm_line, self.wrist_pt, self.trace_line

        self.ani = FuncAnimation(self.fig, update, frames=len(self.scan_poses), init_func=init, interval=40, blit=False)
        self.canvas.draw_idle()
        self.lbl.config(text=f"Playing scan ({len(self.scan_poses)} frames)")

    def _stop_animation(self):
        if self.ani is not None:
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
            self.ani = None

    def _save_gif(self):
        # Build scan if not already playing
        self._apply_params()
        self._sync_scan_cfg()

        total = int(self.var_total.get())
        if total < 3:
            messagebox.showerror("GIF", "Total frames must be >= 3")
            return

        # ask file path
        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif")],
            initialfile="scan_kinematics.gif",
        )
        if not path:
            return

        poses = build_scan_poses(self.cfg, total_images=total, start_base_deg=float(self.var_b.get()))
        pts = [fk_points(self.cfg, b, s, e) for (b, s, e) in poses]
        wrists = np.array([p[2] for p in pts])
        self._set_axes_limits(wrists)

        trace = np.zeros((len(pts), 3), dtype=float)

        def init():
            self.arm_line.set_data([], [])
            self.arm_line.set_3d_properties([])
            self.wrist_pt.set_data([], [])
            self.wrist_pt.set_3d_properties([])
            self.trace_line.set_data([], [])
            self.trace_line.set_3d_properties([])
            self.text2d.set_text("")
            return self.arm_line, self.wrist_pt, self.trace_line

        def update(i):
            shoulder, elbow, wrist = pts[i]
            xs = [shoulder[0], elbow[0], wrist[0]]
            ys = [shoulder[1], elbow[1], wrist[1]]
            zs = [shoulder[2], elbow[2], wrist[2]]

            self.arm_line.set_data(xs, ys)
            self.arm_line.set_3d_properties(zs)

            self.wrist_pt.set_data([wrist[0]], [wrist[1]])
            self.wrist_pt.set_3d_properties([wrist[2]])

            if self.var_show_trace.get():
                trace[i] = wrist
                tr = trace[: i + 1]
                self.trace_line.set_data(tr[:, 0], tr[:, 1])
                self.trace_line.set_3d_properties(tr[:, 2])
            else:
                self.trace_line.set_data([], [])
                self.trace_line.set_3d_properties([])

            b_deg, s_deg, e_deg = poses[i]
            self.text2d.set_text(f"step {i+1}/{len(poses)}  |  b={b_deg:.1f}  s={s_deg:.1f}  e={e_deg:.1f}")
            return self.arm_line, self.wrist_pt, self.trace_line

        # Create a temporary animation object and save
        ani = FuncAnimation(self.fig, update, frames=len(poses), init_func=init, interval=40, blit=False)

        try:
            fps = 25  # matches interval=40ms
            ani.save(path, writer=PillowWriter(fps=fps))
            self.lbl.config(text=f"Saved GIF: {path}")
            messagebox.showinfo("Saved", f"GIF saved:\n{path}")
        except Exception as e:
            messagebox.showerror("GIF save error", str(e))


def main():
    # nicer ttk on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = KinematicUI()
    app.mainloop()


if __name__ == "__main__":
    main()
