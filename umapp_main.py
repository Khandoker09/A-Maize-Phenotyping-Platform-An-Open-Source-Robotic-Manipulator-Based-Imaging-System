#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, sys, json, math, time, shlex, queue, signal
import threading, subprocess, webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image as PILImage, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_IS_WINDOWS = sys.platform.startswith("win")

# ══════════════════════════════════════════════════════════════
#  THEMES
# ══════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        "bg":        "#0f172a",
        "surface":   "#1e293b",
        "surface2":  "#243044",
        "border":    "#334155",
        "text":      "#e2e8f0",
        "text2":     "#94a3b8",
        "accent":    "#7c3aed",
        "accent2":   "#6d28d9",
        "success":   "#22c55e",
        "warning":   "#f59e0b",
        "error":     "#ef4444",
        "btn_fg":    "#ffffff",
        "entry_bg":  "#0f172a",
        "entry_fg":  "#e2e8f0",
        "log_bg":    "#0a0f1e",
        "log_fg":    "#94a3b8",
        "select_bg": "#7c3aed",
        "tab_bg":    "#1e293b",
        "tab_sel":   "#7c3aed",
    },
    "light": {
        "bg":        "#f1f5f9",
        "surface":   "#ffffff",
        "surface2":  "#f8fafc",
        "border":    "#e2e8f0",
        "text":      "#1e293b",
        "text2":     "#64748b",
        "accent":    "#3b82f6",
        "accent2":   "#2563eb",
        "success":   "#16a34a",
        "warning":   "#d97706",
        "error":     "#dc2626",
        "btn_fg":    "#ffffff",
        "entry_bg":  "#ffffff",
        "entry_fg":  "#1e293b",
        "log_bg":    "#f8fafc",
        "log_fg":    "#475569",
        "select_bg": "#3b82f6",
        "tab_bg":    "#ffffff",
        "tab_sel":   "#3b82f6",
    }
}

# ══════════════════════════════════════════════════════════════
#  DEFAULTS
# ══════════════════════════════════════════════════════════════
DEFAULT = {
    "app_title": "Umapp v1",
    "robot_ssid": "RoArm-M2",
    "robot_pass": "12345678",
    "robot_base": "http://192.168.4.1",
    "safe_limit_deg": 175.0,
    "pi_host": "192.168.4.20",
    "pi_port": 8765,
    "pi_auto_discover": True,
    "pi_subnet_prefix": "192.168.4.",
    "pi_scan_start": 2,
    "pi_scan_end": 80,
    "use_cameras": True,
    "rotate_180": True,
    "hold_last_pose": True,
    "lock_hand_deg": 62.0,
    "base_center_deg": 0.0,
    "base_half_span_deg": 175.0,
    "spd": 20,
    "acc": 20,
    "delay_s": 0.30,
    "mode": "Final",
    "final_total_images": 200,
    "test_total_images": 10,
    "active_profile": "",
    "height_low_cm": 10.0,
    "height_high_cm": 60.0,
    "pose_low_sh_deg": -84.0,
    "pose_low_el_deg": 92.0,
    "pose_high_sh_deg": -60.0,
    "pose_high_el_deg": 70.0,
    "ring_height_span_deg": 10.0,
    "ring_width_span_deg": 6.0,
    "elbow_push_gain": 0.8,
    "elbow_push_radius0_cm": 10.0,
    "sh_min": -120.0, "sh_max": 120.0,
    "el_min": 20.0,   "el_max": 110.0,
    "profiles": [{
        "name": "Plant_A",
        "height_cm": 20.0, "width_cm": 30.0, "clearance_cm": 5.0,
        "use_manual_rings": True,
        "manual_rings": [{"s":80.0,"e":75.0},{"s":75.0,"e":75.0},{"s":70.0,"e":70.0}],
    }],
}

# ══════════════════════════════════════════════════════════════
#  WIFI (Linux only)
# ══════════════════════════════════════════════════════════════
def _sh(cmd):
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def ensure_connected_to(ssid, password=None):
    if _IS_WINDOWS:
        return True
    try:
        rc, out, _ = _sh("nmcli -t -f ACTIVE,SSID dev wifi")
        for line in out.splitlines():
            if line.startswith("yes:") and line.split(":",1)[1] == ssid:
                return True
    except Exception:
        pass
    try:
        rc2, out2, _ = _sh("nmcli -t -f DEVICE,TYPE device status")
        dev = next((l.split(":")[0] for l in out2.splitlines()
                    if len(l.split(":"))>=2 and l.split(":")[1]=="wifi"), "")
        if not dev: raise RuntimeError("No Wi-Fi device")
        cmd = f'nmcli dev wifi connect "{ssid}"'
        if password: cmd += f' password "{password}"'
        cmd += f" ifname {dev}"
        rc3, out3, err3 = _sh(cmd)
        if rc3 != 0: raise RuntimeError(err3 or out3)
    except Exception as e:
        raise RuntimeError(str(e))
    return True

# ══════════════════════════════════════════════════════════════
#  ROBOT HTTP
# ══════════════════════════════════════════════════════════════
_SESSION = requests.Session()

def send_cmd(cfg, cmd, retries=3, log=None):
    url = cfg["robot_base"] + "/js?json=" + quote(json.dumps(cmd))
    for attempt in range(retries):
        try:
            return _SESSION.get(url, timeout=(3,6)).json()
        except Exception as e:
            if log: log(f"[WARN] Robot ({attempt+1}/{retries}): {e}")
            time.sleep(1)
    raise RuntimeError("Robot unreachable.")

def get_feedback(cfg, log=None):
    fb = send_cmd(cfg, {"T":105}, log=log)
    return json.loads(fb) if isinstance(fb, str) else fb

def torque(cfg, on=True, log=None):
    send_cmd(cfg, {"T":210,"cmd":(1 if on else 0)}, log=log)

def move_joints_deg(cfg, base=None, shoulder=None, elbow=None, hand=None,
                    spd=None, acc=None, log=None):
    fb = get_feedback(cfg, log=log)
    send_cmd(cfg, {"T":122,
        "b": base     if base     is not None else math.degrees(fb["b"]),
        "s": shoulder if shoulder is not None else math.degrees(fb["s"]),
        "e": elbow    if elbow    is not None else math.degrees(fb["e"]),
        "h": hand     if hand     is not None else math.degrees(fb["t"]),
        "spd": int(spd or cfg["spd"]), "acc": int(acc or cfg["acc"])}, log=log)
    time.sleep(0.45)
    return get_feedback(cfg, log=log)

# ══════════════════════════════════════════════════════════════
#  PI HTTP
# ══════════════════════════════════════════════════════════════
def _pi_base(cfg):
    return f"http://{cfg.get('pi_host','')}:{int(cfg.get('pi_port',8765))}"

def _pi_get(cfg, endpoint, params=None, timeout=10):
    return requests.get(_pi_base(cfg)+endpoint, params=params, timeout=timeout).json()

def _discover_http(prefix, port, start, end, timeout=0.5, max_workers=32):
    ips = [f"{prefix}{i}" for i in range(int(start), int(end)+1)]
    found = []
    def check(ip):
        try:
            r = requests.get(f"http://{ip}:{port}/status", timeout=timeout)
            if r.json().get("ok"): return ip
        except Exception: pass
        return None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ip in [f.result() for f in as_completed([ex.submit(check,ip) for ip in ips])]:
            if ip: found.append(ip)
    return sorted(found, key=lambda x:[int(p) for p in x.split(".")])

def ensure_pi_http(cfg, log=None):
    host = (cfg.get("pi_host") or "").strip()
    port = int(cfg.get("pi_port", 8765))
    if not host: return False, "Pi host not set"
    try:
        r = requests.get(f"http://{host}:{port}/status", timeout=5)
        if r.json().get("ok"): return True, f"HTTP OK  {host}:{port}"
    except Exception as e:
        if cfg.get("pi_auto_discover", True):
            prefix = (cfg.get("pi_subnet_prefix") or "192.168.4.").strip()
            start  = int(cfg.get("pi_scan_start", 2))
            end    = int(cfg.get("pi_scan_end",  80))
            if log: log(f"[Pi] Scanning {prefix}{start}–{end}…")
            found = _discover_http(prefix, port, start, end)
            if found:
                cfg["pi_host"] = found[0]
                if log: log(f"[Pi] Found at {found[0]}:{port} ✅")
                return True, f"HTTP OK  {found[0]}:{port}"
            return False, f"Not reachable ({e}) — nothing found"
        return False, str(e)
    return False, "Pi not responding"

def pi_get_storage(cfg, log=None):
    """Returns the active storage path on Pi (USB or SD card)."""
    try:
        data = _pi_get(cfg, "/storage", timeout=8)
        return data.get("path", "/home/umap/captures"), data.get("type", "SD")
    except Exception:
        return "/home/umap/captures", "SD"

def pi_mkdir(cfg, path, log=None):
    ok0, msg0 = ensure_pi_http(cfg, log=log)
    if not ok0: return False, msg0
    try:
        data = _pi_get(cfg, "/mkdir", params={"path": path}, timeout=15)
        return data.get("ok", False), data.get("error", "")
    except Exception as e:
        return False, str(e)

def trigger_capture_store_rotate(cfg, tag, target_dir, log=None):
    if not cfg["use_cameras"]: return True
    rotate = "1" if cfg.get("rotate_180") else "0"
    try:
        data = _pi_get(cfg, "/capture", params={"tag": tag}, timeout=30)
        if not data.get("ok"):
            if log: log("CAPTURE FAIL | {}".format(data.get("error","unknown")))
            return False
        captured_file = data.get("file","")
        if not captured_file:
            if log: log("CAPTURE FAIL | No file path"); return False
        data2 = _pi_get(cfg, "/move",
                        params={"src": captured_file, "dst": target_dir, "rotate": rotate},
                        timeout=30)
        if not data2.get("ok"):
            if log: log("STORE WARN | {}".format(data2.get("error","")))
        return True
    except Exception as e:
        if log: log(f"CAPTURE ERROR | {e}")
        return False

# ══════════════════════════════════════════════════════════════
#  PLANNING
# ══════════════════════════════════════════════════════════════
def clamp(x,a,b): return max(a,min(b,x))
def clip_base_deg(a,lim): return clamp(a,-lim,lim)

def plan_angles_centered(center, half, count, safe_lim, start_b=None):
    """Legacy centered sweep. Kept for compatibility.

    NOTE: This can be clipped if center+/-half exceeds the safe limit.
    For 360 object coverage, use plan_base_360_sweep() instead.
    """
    half = min(float(half), float(safe_lim))
    center = clip_base_deg(float(center), safe_lim)
    lo = clip_base_deg(center-half, safe_lim)
    hi = clip_base_deg(center+half, safe_lim)
    if count <= 1:
        return [center]
    arr = [lo + i*(hi-lo)/(count-1) for i in range(count)]
    if start_b is None:
        return arr
    start_b = clip_base_deg(float(start_b), safe_lim)
    # Do NOT rotate the list by slicing, because that creates a sudden jump
    # from +limit to -limit inside one ring. Instead choose the nearest end.
    if abs(start_b - hi) < abs(start_b - lo):
        arr.reverse()
    return arr

def choose_360_start_reverse(start_b, safe_lim):
    """Return True when it is shorter to start the 360 sweep from +safe_lim."""
    if start_b is None:
        return False
    safe_lim = abs(float(safe_lim))
    start_b = clip_base_deg(float(start_b), safe_lim)
    dist_to_lo = abs(start_b - (-safe_lim))
    dist_to_hi = abs(start_b - ( safe_lim))
    return dist_to_hi < dist_to_lo

def plan_base_360_sweep(count, safe_lim, reverse=False):
    """Plan the widest possible base sweep: -safe_limit to +safe_limit.

    With a +/-175 degree servo this gives 350 degrees of mechanical base travel.
    The remaining 10 degrees cannot be commanded by software, so the scan includes
    both end positions and alternates direction on the next ring to get the best
    possible overlap from the other side.
    """
    safe_lim = abs(float(safe_lim))
    lo, hi = -safe_lim, safe_lim
    count = int(count)
    if count <= 0:
        return []
    if count == 1:
        arr = [0.0]
    else:
        arr = [lo + i*(hi-lo)/(count-1) for i in range(count)]
    if reverse:
        arr.reverse()
    return arr


def compute_mid_pose(cfg, height_cm):
    h0,h1 = float(cfg["height_low_cm"]), float(cfg["height_high_cm"])
    t = clamp(0 if h1==h0 else (float(height_cm)-h0)/(h1-h0), 0,1)
    s = float(cfg["pose_low_sh_deg"]) + t*(float(cfg["pose_high_sh_deg"])-float(cfg["pose_low_sh_deg"]))
    e = float(cfg["pose_low_el_deg"]) + t*(float(cfg["pose_high_el_deg"])-float(cfg["pose_low_el_deg"]))
    return clamp(s,float(cfg["sh_min"]),float(cfg["sh_max"])), clamp(e,float(cfg["el_min"]),float(cfg["el_max"])), t

def compute_rings_auto(cfg, h,w,c):
    s_mid,e_mid,t = compute_mid_pose(cfg,h)
    ew = w + 2*c
    ws = clamp(20/max(1e-6,ew), 0.25, 1.0)
    sw = float(cfg["ring_width_span_deg"])*ws
    sh = float(cfg["ring_height_span_deg"])*clamp(0.7+0.6*t, 0.5, 1.3)
    r  = w/2+c
    ee = max(0,(r-float(cfg["elbow_push_radius0_cm"]))*float(cfg["elbow_push_gain"]))
    s2 = clamp(s_mid, float(cfg["sh_min"]), float(cfg["sh_max"]))
    e2 = clamp(e_mid+ee, float(cfg["el_min"]), float(cfg["el_max"]))
    def cp(s,e): return (clamp(s,float(cfg["sh_min"]),float(cfg["sh_max"])),
                         clamp(e,float(cfg["el_min"]),float(cfg["el_max"])))
    return ([cp(s2-(sw+sh),e2+(sw+sh)), cp(s2,e2), cp(s2+(sw+sh),e2-(sw+sh))],
            ws, ew, ee, r, (s2,e2))

def get_profile(cfg, name):
    return next((p for p in cfg.get("profiles",[]) if p.get("name")==name), None)

def compute_rings_from_profile(cfg, prof):
    h,w,c = float(prof.get("height_cm",20)), float(prof.get("width_cm",30)), float(prof.get("clearance_cm",5))
    if prof.get("use_manual_rings"):
        mr = [(float(r.get("s",-80)),float(r.get("e",90))) for r in prof.get("manual_rings",[])[:3]]
        if len(mr)==3:
            auto,ws,ew,ee,rad,mid = compute_rings_auto(cfg,h,w,c)
            return mr, ws, ew, ee, rad, mid, False
    auto,ws,ew,ee,rad,mid = compute_rings_auto(cfg,h,w,c)
    return auto, ws, ew, ee, rad, mid, True

# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root = root
        self.cfg  = deepcopy(DEFAULT)
        self.theme_name = "dark"
        self.T = THEMES[self.theme_name]

        self.root.title("Umapp v1")
        self.root.geometry("1320x880")
        self.root.minsize(1100, 700)

        self.log_q       = queue.Queue()
        self.abort_event = threading.Event()
        self.worker      = None
        self._live_running = False
        self._live_thread  = None
        self._live_photo   = None

        self._build_ui()
        self._apply_theme()
        self._poll_logs()
        signal.signal(signal.SIGINT, lambda *_: self.on_abort())
        self._ensure_active_profile()

    # ── logging ──────────────────────────────────────────────
    def log(self, msg):
        self.log_q.put(msg)

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.txt.configure(state="normal")
                ts = datetime.now().strftime("%H:%M:%S")
                # colour-code lines
                tag = "normal"
                if "✅" in msg or "OK" in msg or "DONE" in msg: tag = "ok"
                elif "FAIL" in msg or "ERROR" in msg or "❌" in msg: tag = "err"
                elif "WARN" in msg or "warning" in msg: tag = "warn"
                elif msg.startswith("---") or msg.startswith("Ring") or msg.startswith("Progress"): tag = "info"
                self.txt.insert("end", f"[{ts}] {msg}\n", tag)
                self.txt.see("end")
                self.txt.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)

    def set_progress(self, frac, label):
        frac = max(0.0, min(1.0, frac))
        self.progress["value"] = frac * 100
        self.lbl_prog.configure(text=label)

    # ── theme ─────────────────────────────────────────────────
    def toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.T = THEMES[self.theme_name]
        self._apply_theme()
        icon = "☀️" if self.theme_name == "dark" else "🌙"
        self.btn_theme.configure(text=icon)

    def _apply_theme(self):
        T = self.T
        self.root.configure(bg=T["bg"])
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".",
            background=T["bg"], foreground=T["text"],
            fieldbackground=T["entry_bg"], troughcolor=T["surface"],
            bordercolor=T["border"], darkcolor=T["surface"], lightcolor=T["surface"],
            selectbackground=T["select_bg"], selectforeground="#ffffff",
            font=("Segoe UI", 10))

        style.configure("TFrame",      background=T["bg"])
        style.configure("TLabel",      background=T["bg"],      foreground=T["text"],      font=("Segoe UI", 10))
        style.configure("TCheckbutton",background=T["bg"],      foreground=T["text"],      font=("Segoe UI", 10))
        style.configure("TEntry",      fieldbackground=T["entry_bg"], foreground=T["entry_fg"],
                        insertcolor=T["text"], bordercolor=T["border"], font=("Segoe UI", 10))
        style.configure("TCombobox",   fieldbackground=T["entry_bg"], foreground=T["entry_fg"],
                        selectbackground=T["select_bg"], font=("Segoe UI", 10))
        style.configure("TScale",      background=T["bg"], troughcolor=T["surface2"],
                        sliderrelief="flat")

        # LabelFrame
        style.configure("TLabelframe",       background=T["surface"], foreground=T["text2"],
                        bordercolor=T["border"], relief="flat", font=("Segoe UI", 9))
        style.configure("TLabelframe.Label",  background=T["surface"], foreground=T["text2"],
                        font=("Segoe UI", 9, "bold"))

        # Notebook
        style.configure("TNotebook",          background=T["bg"],      bordercolor=T["border"])
        style.configure("TNotebook.Tab",       background=T["surface"], foreground=T["text2"],
                        padding=[14,8], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
            background=[("selected", T["accent"])],
            foreground=[("selected", "#ffffff")])

        # Buttons — accent
        style.configure("Accent.TButton",
            background=T["accent"], foreground="#ffffff",
            relief="flat", borderwidth=0, padding=[14,8],
            font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton",
            background=[("active", T["accent2"]), ("pressed", T["accent2"])])

        # Buttons — secondary
        style.configure("TButton",
            background=T["surface2"], foreground=T["text"],
            relief="flat", borderwidth=0, padding=[12,6],
            font=("Segoe UI", 10))
        style.map("TButton",
            background=[("active", T["border"]), ("pressed", T["border"])])

        # Danger button
        style.configure("Danger.TButton",
            background=T["error"], foreground="#ffffff",
            relief="flat", borderwidth=0, padding=[14,8],
            font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton",
            background=[("active","#b91c1c"), ("pressed","#b91c1c")])

        # Progressbar
        style.configure("TProgressbar",
            troughcolor=T["surface2"], background=T["accent"],
            thickness=8, borderwidth=0)

        # Separator
        style.configure("TSeparator", background=T["border"])

        # Scrollbar
        style.configure("TScrollbar",
            background=T["surface2"], troughcolor=T["surface"],
            bordercolor=T["border"], arrowcolor=T["text2"],
            relief="flat")

        # log colours
        self.txt.configure(bg=T["log_bg"], fg=T["log_fg"],
                           insertbackground=T["text"], font=("Consolas",9),
                           selectbackground=T["select_bg"])
        self.txt.tag_configure("ok",   foreground=T["success"])
        self.txt.tag_configure("err",  foreground=T["error"])
        self.txt.tag_configure("warn", foreground=T["warning"])
        self.txt.tag_configure("info", foreground=T["accent"])
        self.txt.tag_configure("normal", foreground=T["log_fg"])

        # header / sidebar backgrounds
        for w in getattr(self, "_surface_frames", []):
            try: w.configure(bg=T["surface"])
            except Exception: pass
        for w in getattr(self, "_bg_frames", []):
            try: w.configure(bg=T["bg"])
            except Exception: pass

        # live view canvas
        if hasattr(self, "live_canvas"):
            self.live_canvas.configure(bg="#0a0f1e" if self.theme_name=="dark" else "#e2e8f0")

    # ── UI build ─────────────────────────────────────────────
    def _build_ui(self):
        self._surface_frames = []
        self._bg_frames      = []

        # ── Header bar ───────────────────────────────────────
        hdr = tk.Frame(self.root, height=56)
        hdr.pack(fill=tk.X, padx=0, pady=0)
        hdr.pack_propagate(False)
        self._surface_frames.append(hdr)

        tk.Label(hdr, text="Umapp v1", font=("Segoe UI", 18, "bold"),
                 fg="#7c3aed", bg=THEMES["dark"]["surface"]).pack(side=tk.LEFT, padx=20, pady=12)
        tk.Label(hdr, text="Photogeometry Platform", font=("Segoe UI", 10),
                 fg=THEMES["dark"]["text2"], bg=THEMES["dark"]["surface"]).pack(side=tk.LEFT)

        # Theme toggle
        self.btn_theme = tk.Button(hdr, text="☀️", font=("Segoe UI",14),
                                   command=self.toggle_theme, relief="flat",
                                   bg=THEMES["dark"]["surface"], fg=THEMES["dark"]["text"],
                                   cursor="hand2", bd=0, padx=12)
        self.btn_theme.pack(side=tk.RIGHT, pady=8, padx=8)

        # Browse images button
        ttk.Button(hdr, text="🌐 Browse Images", style="TButton",
                   command=self.open_browser).pack(side=tk.RIGHT, pady=10, padx=4)

        # ── Main body ────────────────────────────────────────
        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        left  = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left,  weight=3)
        body.add(right, weight=2)

        # ── Notebook ─────────────────────────────────────────
        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8,0))

        self.tab_conn     = ttk.Frame(nb)
        self.tab_profiles = ttk.Frame(nb)
        self.tab_modes    = ttk.Frame(nb)
        self.tab_sweep    = ttk.Frame(nb)
        self.tab_manual   = ttk.Frame(nb)
        self.tab_mapping  = ttk.Frame(nb)
        self.tab_liveview = ttk.Frame(nb)

        nb.add(self.tab_conn,     text=" ⚙  Connection ")
        nb.add(self.tab_profiles, text=" 🌿 Profiles ")
        nb.add(self.tab_modes,    text=" 🎯 Mode ")
        nb.add(self.tab_sweep,    text=" 🔄 Sweep ")
        nb.add(self.tab_manual,   text=" 🤖 Manual ")
        nb.add(self.tab_mapping,  text=" 📐 Mapping ")
        nb.add(self.tab_liveview, text=" 📷 Live View ")

        self._build_tab_conn()
        self._build_tab_profiles()
        self._build_tab_modes()
        self._build_tab_sweep()
        self._build_tab_manual()
        self._build_tab_mapping()
        self._build_tab_liveview()

        # ── Bottom bar (START / ABORT / progress) ────────────
        bbar = ttk.Frame(left)
        bbar.pack(fill=tk.X, padx=12, pady=10)

        self.btn_start = ttk.Button(bbar, text="▶  START SCAN", style="Accent.TButton",
                                    command=self.on_start)
        self.btn_abort = ttk.Button(bbar, text="■  ABORT",      style="Danger.TButton",
                                    command=self.on_abort, state="disabled")
        self.btn_start.pack(side=tk.LEFT)
        self.btn_abort.pack(side=tk.LEFT, padx=(10,0))

        self.lbl_prog = ttk.Label(bbar, text="Idle", foreground=THEMES["dark"]["text2"])
        self.lbl_prog.pack(side=tk.RIGHT, padx=(0,8))
        self.progress = ttk.Progressbar(bbar, orient=tk.HORIZONTAL,
                                        mode="determinate", length=260)
        self.progress.pack(side=tk.RIGHT, padx=(0,10))

        # ── Right panel: log ──────────────────────────────────
        rpad = ttk.Frame(right)
        rpad.pack(fill=tk.BOTH, expand=True, padx=(4,12), pady=8)

        log_hdr = ttk.Frame(rpad)
        log_hdr.pack(fill=tk.X, pady=(0,6))
        ttk.Label(log_hdr, text="Live Log", font=("Segoe UI",13,"bold")).pack(side=tk.LEFT)
        ttk.Button(log_hdr, text="Clear", command=lambda: (self.txt.configure(state="normal"),
                   self.txt.delete("1.0","end"), self.txt.configure(state="disabled"))).pack(side=tk.RIGHT)
        ttk.Button(log_hdr, text="Save…", command=self.save_log).pack(side=tk.RIGHT, padx=6)

        tf = ttk.Frame(rpad)
        tf.pack(fill=tk.BOTH, expand=True)
        self.txt = tk.Text(tf, wrap="word", state="disabled", relief="flat",
                           font=("Consolas",9), bd=0)
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sc = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self.txt.yview)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt.configure(yscrollcommand=sc.set)

        cfgb = ttk.Frame(rpad)
        cfgb.pack(fill=tk.X, pady=(8,0))
        ttk.Button(cfgb, text="Load config…", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(cfgb, text="Save config…", command=self.save_config).pack(side=tk.LEFT, padx=8)

    # ── helper to make a labelled group ──────────────────────
    def _group(self, parent, title, pady=(10,6)):
        lf = ttk.LabelFrame(parent, text=f"  {title}  ", padding=12)
        lf.pack(fill=tk.X, padx=12, pady=pady)
        return lf

    def _row(self, grid, r, label, var, c=0, width=None):
        ttk.Label(grid, text=label).grid(row=r, column=c, sticky="w", padx=(0,8), pady=5)
        kw = {"textvariable": var}
        if width: kw["width"] = width
        ttk.Entry(grid, **kw).grid(row=r, column=c+1, sticky="ew", pady=5, padx=(0,12))

    # ── Tab 1: Connection ─────────────────────────────────────
    def _build_tab_conn(self):
        # Robot group
        g = self._group(self.tab_conn, "Robot Wi-Fi + URL")
        g.columnconfigure(1, weight=1); g.columnconfigure(3, weight=1)

        self.var_ssid = tk.StringVar(value=self.cfg["robot_ssid"])
        self.var_pass = tk.StringVar(value=self.cfg["robot_pass"])
        self.var_base = tk.StringVar(value=self.cfg["robot_base"])

        self._row(g, 0, "SSID:",     self.var_ssid, 0)
        self._row(g, 0, "Robot URL:",self.var_base, 2)
        ttk.Label(g, text="Password:").grid(row=1, column=0, sticky="w", padx=(0,8), pady=5)
        ttk.Entry(g, textvariable=self.var_pass, show="*").grid(row=1, column=1, sticky="ew", pady=5, padx=(0,12))

        if _IS_WINDOWS:
            ttk.Label(g, text="ℹ  Windows: connect to robot Wi-Fi from taskbar",
                      foreground=THEMES["dark"]["text2"],
                      font=("Segoe UI",9)).grid(row=2, column=0, columnspan=4, sticky="w", pady=(0,4))

        bb = ttk.Frame(g); bb.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4,0))
        if not _IS_WINDOWS:
            ttk.Button(bb, text="Connect Wi-Fi", command=self.on_connect_wifi).pack(side=tk.LEFT)
        ttk.Button(bb, text="Test Robot",  command=self.on_test_robot).pack(side=tk.LEFT, padx=8)
        ttk.Button(bb, text="Read Angles", command=self.on_read_angles).pack(side=tk.LEFT)

        # Pi group
        g2 = self._group(self.tab_conn, "Raspberry Pi — HTTP (port 8765)")
        g2.columnconfigure(1, weight=1); g2.columnconfigure(3, weight=1)

        self.var_use_cam = tk.BooleanVar(value=self.cfg["use_cameras"])
        self.var_rot180  = tk.BooleanVar(value=self.cfg["rotate_180"])
        self.var_pi_host = tk.StringVar(value=self.cfg["pi_host"])
        self.var_pi_port = tk.StringVar(value=str(self.cfg.get("pi_port", 8765)))

        ttk.Checkbutton(g2, text="Trigger USB webcam during scan",
                        variable=self.var_use_cam).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(g2, text="Rotate images 180° (upside-down mount)",
                        variable=self.var_rot180).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        self._row(g2, 2, "Pi IP:",   self.var_pi_host, 0)
        self._row(g2, 2, "HTTP port:",self.var_pi_port, 2, width=8)

        bb2 = ttk.Frame(g2); bb2.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6,0))
        ttk.Button(bb2, text="Test Pi server",        command=self.on_test_pi).pack(side=tk.LEFT)
        ttk.Button(bb2, text="Test capture",          command=self.on_test_capture).pack(side=tk.LEFT, padx=8)
        ttk.Button(bb2, text="🌐 Browse images",      command=self.open_browser).pack(side=tk.LEFT)

        # Storage info
        g3 = self._group(self.tab_conn, "Image Storage")
        ttk.Label(g3, text="Images save to Pi SD card by default (/home/umap/captures/).\n"
                           "If USB drive is plugged in, USB is used instead.\n"
                           "Access images via browser: http://192.168.4.20:8765/browse",
                  font=("Segoe UI",9), foreground=THEMES["dark"]["text2"],
                  justify="left").pack(anchor="w")

    # ── Tab 2: Profiles ───────────────────────────────────────
    def _build_tab_profiles(self):
        top = ttk.Frame(self.tab_profiles); top.pack(fill=tk.X, padx=12, pady=(12,0))

        self.var_profile = tk.StringVar()
        self.cmb_profile = ttk.Combobox(top, textvariable=self.var_profile,
                                        state="readonly", width=22, font=("Segoe UI",10))
        self.cmb_profile.pack(side=tk.LEFT)
        ttk.Button(top, text="+ Add",    command=self.add_profile).pack(side=tk.LEFT, padx=(10,4))
        ttk.Button(top, text="✕ Delete", command=self.delete_profile).pack(side=tk.LEFT)

        # Active profile banner
        ab = ttk.Frame(self.tab_profiles); ab.pack(fill=tk.X, padx=12, pady=8)
        ttk.Label(ab, text="Active for scan:", font=("Segoe UI",10,"bold")).pack(side=tk.LEFT)
        self.var_active_profile = tk.StringVar(value=self.cfg.get("active_profile",""))
        ttk.Entry(ab, textvariable=self.var_active_profile, state="readonly", width=22,
                  font=("Segoe UI",10)).pack(side=tk.LEFT, padx=8)
        ttk.Button(ab, text="✅ Set selected as active",
                   command=self.set_selected_as_active_profile,
                   style="Accent.TButton").pack(side=tk.LEFT)

        pj = ttk.Frame(self.tab_profiles); pj.pack(fill=tk.X, padx=12, pady=(0,6))
        ttk.Button(pj, text="⬆ Export JSON", command=self.export_profiles_json).pack(side=tk.LEFT)
        ttk.Button(pj, text="⬇ Import JSON", command=self.import_profiles_json).pack(side=tk.LEFT, padx=8)

        ttk.Separator(self.tab_profiles).pack(fill=tk.X, padx=12, pady=4)

        # Editor
        g = self._group(self.tab_profiles, "Profile Editor")
        g.columnconfigure(1, weight=1); g.columnconfigure(3, weight=1)

        self.var_p_name   = tk.StringVar()
        self.var_p_h      = tk.DoubleVar(value=20.0)
        self.var_p_w      = tk.DoubleVar(value=30.0)
        self.var_p_c      = tk.DoubleVar(value=5.0)
        self.var_p_manual = tk.BooleanVar(value=True)

        self._row(g, 0, "Name:",         self.var_p_name, 0)
        self._row(g, 0, "Height (cm):",  self.var_p_h,    2)
        self._row(g, 1, "Width (cm):",   self.var_p_w,    0)
        self._row(g, 1, "Clearance (cm):",self.var_p_c,   2)
        ttk.Checkbutton(g, text="Use manual rings", variable=self.var_p_manual)\
            .grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        # Rings
        rg = self._group(self.tab_profiles, "Manual Rings (S = shoulder, E = elbow)")
        rg.columnconfigure((0,1,2,3,4,5), weight=1)

        self.var_r1s=tk.DoubleVar(value=80); self.var_r1e=tk.DoubleVar(value=75)
        self.var_r2s=tk.DoubleVar(value=75); self.var_r2e=tk.DoubleVar(value=75)
        self.var_r3s=tk.DoubleVar(value=70); self.var_r3e=tk.DoubleVar(value=70)

        for col, (title, vs, ve) in enumerate([
            ("Ring 1", self.var_r1s, self.var_r1e),
            ("Ring 2", self.var_r2s, self.var_r2e),
            ("Ring 3", self.var_r3s, self.var_r3e)]):
            ttk.Label(rg, text=title, font=("Segoe UI",9,"bold")).grid(row=0, column=col*2, columnspan=2, pady=(0,4))
            ttk.Label(rg, text="S:").grid(row=1, column=col*2, sticky="e", padx=(0,4))
            ttk.Entry(rg, textvariable=vs, width=7).grid(row=1, column=col*2+1, sticky="w", pady=3)
            ttk.Label(rg, text="E:").grid(row=2, column=col*2, sticky="e", padx=(0,4))
            ttk.Entry(rg, textvariable=ve, width=7).grid(row=2, column=col*2+1, sticky="w", pady=3)

        btns = ttk.Frame(self.tab_profiles); btns.pack(fill=tk.X, padx=12, pady=6)
        ttk.Button(btns, text="Load → editor", command=self.load_selected_profile_to_editor).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save → profile", command=self.save_editor_to_selected_profile).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Preview rings", command=self.preview_profile_rings).pack(side=tk.LEFT)

        self.cmb_profile.bind("<<ComboboxSelected>>", lambda *_: self.load_selected_profile_to_editor())
        self.refresh_profiles_dropdown()
        if self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
            self.load_selected_profile_to_editor()

    # ── Tab 3: Modes ─────────────────────────────────────────
    def _build_tab_modes(self):
        g = self._group(self.tab_modes, "Scan Mode")
        g.columnconfigure(1, weight=1)

        self.var_mode         = tk.StringVar(value=self.cfg["mode"])
        self.var_testname     = tk.StringVar(value="test1")
        self.var_final_images = tk.IntVar(value=self.cfg["final_total_images"])
        self.var_test_images  = tk.IntVar(value=self.cfg["test_total_images"])

        ttk.Label(g, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0,8), pady=6)
        ttk.Combobox(g, textvariable=self.var_mode, state="readonly",
                     values=["Calibration","Test","Final"],
                     font=("Segoe UI",10)).grid(row=0, column=1, sticky="ew", pady=6)
        self._row(g, 1, "Test name:",    self.var_testname,     0)
        self._row(g, 2, "Test images:",  self.var_test_images,  0)
        self._row(g, 3, "Final images:", self.var_final_images, 0)

        ap = ttk.Frame(g); ap.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8,0))
        ttk.Label(ap, text="Active profile:", font=("Segoe UI",10,"bold")).pack(side=tk.LEFT)
        ttk.Entry(ap, textvariable=self.var_active_profile, state="readonly",
                  width=22, font=("Segoe UI",10)).pack(side=tk.LEFT, padx=8)

    # ── Tab 4: Sweep ─────────────────────────────────────────
    def _build_tab_sweep(self):
        g = self._group(self.tab_sweep, "Sweep + Speed")
        g.columnconfigure(1, weight=1); g.columnconfigure(3, weight=1)

        self.var_spd        = tk.IntVar(value=self.cfg["spd"])
        self.var_acc        = tk.IntVar(value=self.cfg["acc"])
        self.var_delay      = tk.DoubleVar(value=self.cfg["delay_s"])
        self.var_hold       = tk.BooleanVar(value=self.cfg["hold_last_pose"])
        self.var_base_center= tk.DoubleVar(value=self.cfg["base_center_deg"])
        self.var_base_half  = tk.DoubleVar(value=self.cfg["base_half_span_deg"])
        self.var_safe_lim   = tk.DoubleVar(value=self.cfg["safe_limit_deg"])
        self.var_hand_lock  = tk.DoubleVar(value=self.cfg["lock_hand_deg"])

        self._row(g, 0, "Speed:",      self.var_spd,          0)
        self._row(g, 0, "Accel:",      self.var_acc,          2)
        self._row(g, 1, "Delay (s):",  self.var_delay,        0)
        self._row(g, 2, "Base center (°):", self.var_base_center, 0)
        self._row(g, 2, "Half span (°):",   self.var_base_half,   2)
        ttk.Label(g, text="360 mode uses -Safety limit to +Safety limit; center/half are kept only for compatibility.",
                  font=("Segoe UI",9), foreground=THEMES["dark"]["text2"]).grid(row=4, column=0, columnspan=4, sticky="w", pady=(2,4))
        self._row(g, 3, "Safety limit (°):",self.var_safe_lim,    0)
        self._row(g, 3, "Hand lock (°):",   self.var_hand_lock,   2)
        ttk.Checkbutton(g, text="Hold last pose (torque ON)",
                        variable=self.var_hold).grid(row=1, column=2, columnspan=2, sticky="w", pady=5)

        sp = ttk.Frame(self.tab_sweep); sp.pack(fill=tk.X, padx=12, pady=(0,8))
        ttk.Label(sp, text="Quick speed:", font=("Segoe UI",10)).pack(side=tk.LEFT)
        for lbl, s,a,d in [("🐢 Slow",10,10,0.60),("🚶 Normal",20,20,0.30),("🏃 Fast",30,30,0.15)]:
            ttk.Button(sp, text=lbl, command=lambda s=s,a=a,d=d: self.set_speed(s,a,d))\
                .pack(side=tk.LEFT, padx=6)

    def set_speed(self, spd, acc, delay):
        self.var_spd.set(int(spd)); self.var_acc.set(int(acc)); self.var_delay.set(float(delay))
        self.log(f"Speed: spd={spd} acc={acc} delay={delay}s")

    # ── Tab 5: Manual ─────────────────────────────────────────
    def _build_tab_manual(self):
        g = self._group(self.tab_manual, "Manual Joint Control")
        g.columnconfigure(1, weight=1)

        self.var_m_base = tk.DoubleVar(value=0.0)
        self.var_m_s    = tk.DoubleVar(value=-82.0)
        self.var_m_e    = tk.DoubleVar(value=90.0)
        self.var_m_h    = tk.DoubleVar(value=62.0)

        for row_i, (lbl, var, lo, hi) in enumerate([
            ("Base",     self.var_m_base, -175, 175),
            ("Shoulder", self.var_m_s,    -120, 120),
            ("Elbow",    self.var_m_e,      20, 110),
            ("Hand",     self.var_m_h,       0, 120)]):

            ttk.Label(g, text=lbl, font=("Segoe UI",10,"bold")).grid(row=row_i, column=0, sticky="w", padx=(0,10), pady=8)
            ttk.Scale(g, from_=lo, to=hi, variable=var, orient="horizontal")\
                .grid(row=row_i, column=1, sticky="ew", pady=8)
            val_lbl = ttk.Label(g, text=f"{var.get():.1f}°", width=7)
            val_lbl.grid(row=row_i, column=2, padx=8)
            var.trace_add("write", lambda *_, v=var, l=val_lbl: l.configure(text=f"{v.get():.1f}°"))
            bb = ttk.Frame(g); bb.grid(row=row_i, column=3)
            ttk.Button(bb, text="−", width=3, command=lambda v=var: v.set(v.get()-1)).pack(side=tk.LEFT)
            ttk.Button(bb, text="+", width=3, command=lambda v=var: v.set(v.get()+1)).pack(side=tk.LEFT, padx=2)

        btns = ttk.Frame(self.tab_manual); btns.pack(fill=tk.X, padx=12, pady=6)
        ttk.Button(btns, text="Read → sliders", command=self.on_read_angles_to_sliders).pack(side=tk.LEFT)
        ttk.Button(btns, text="▶ Move now", style="Accent.TButton",
                   command=self.on_manual_move).pack(side=tk.LEFT, padx=8)

    # ── Tab 6: Mapping ────────────────────────────────────────
    def _build_tab_mapping(self):
        g = self._group(self.tab_mapping, "Height → Pose Mapping (Auto Rings)")
        g.columnconfigure(1, weight=1); g.columnconfigure(3, weight=1)

        self.var_h_low   = tk.DoubleVar(value=self.cfg["height_low_cm"])
        self.var_h_high  = tk.DoubleVar(value=self.cfg["height_high_cm"])
        self.var_s_low   = tk.DoubleVar(value=self.cfg["pose_low_sh_deg"])
        self.var_e_low   = tk.DoubleVar(value=self.cfg["pose_low_el_deg"])
        self.var_s_high  = tk.DoubleVar(value=self.cfg["pose_high_sh_deg"])
        self.var_e_high  = tk.DoubleVar(value=self.cfg["pose_high_el_deg"])
        self.var_span_h  = tk.DoubleVar(value=self.cfg["ring_height_span_deg"])
        self.var_span_w  = tk.DoubleVar(value=self.cfg["ring_width_span_deg"])
        self.var_el_gain = tk.DoubleVar(value=self.cfg["elbow_push_gain"])
        self.var_el_r0   = tk.DoubleVar(value=self.cfg["elbow_push_radius0_cm"])

        pairs = [
            ("Height low (cm):",        self.var_h_low,  "Height high (cm):",       self.var_h_high),
            ("LOW Shoulder (°):",       self.var_s_low,  "LOW Elbow (°):",           self.var_e_low),
            ("HIGH Shoulder (°):",      self.var_s_high, "HIGH Elbow (°):",          self.var_e_high),
            ("Ring height span (°):",   self.var_span_h, "Ring width span (°):",     self.var_span_w),
            ("Elbow push gain:",        self.var_el_gain,"Elbow radius0 (cm):",      self.var_el_r0),
        ]
        for i, (l1,v1,l2,v2) in enumerate(pairs):
            self._row(g, i, l1, v1, 0)
            self._row(g, i, l2, v2, 2)

        ttk.Button(self.tab_mapping, text="Preview auto rings for active profile",
                   command=self.preview_auto_rings_for_active_profile)\
            .pack(anchor="w", padx=12, pady=6)

    # ── Tab 7: Live View ──────────────────────────────────────
    def _build_tab_liveview(self):
        if not _PIL_OK:
            f = ttk.Frame(self.tab_liveview); f.pack(expand=True)
            ttk.Label(f, text="⚠  Live View requires Pillow",
                      font=("Segoe UI",14,"bold")).pack(pady=(80,8))
            ttk.Label(f, text="pip install Pillow", font=("Segoe UI",11)).pack()
            return

        ctrl = ttk.Frame(self.tab_liveview); ctrl.pack(fill=tk.X, padx=12, pady=10)
        self.btn_lv = ttk.Button(ctrl, text="▶  Start Live View",
                                 style="Accent.TButton", command=self.toggle_live_view)
        self.btn_lv.pack(side=tk.LEFT)
        self.lbl_lv_status = ttk.Label(ctrl, text="Stopped",
                                       foreground=THEMES["dark"]["text2"],
                                       font=("Segoe UI",10))
        self.lbl_lv_status.pack(side=tk.LEFT, padx=15)
        self.lbl_lv_fps = ttk.Label(ctrl, text="", font=("Segoe UI",10,"bold"))
        self.lbl_lv_fps.pack(side=tk.RIGHT)
        ttk.Label(ctrl, text="Stops automatically when scan starts",
                  foreground=THEMES["dark"]["text2"],
                  font=("Segoe UI",9)).pack(side=tk.RIGHT, padx=10)

        self.live_canvas = tk.Canvas(self.tab_liveview, bg="#0a0f1e",
                                     relief="flat", bd=0)
        self.live_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,12))
        self.live_canvas.create_text(320, 240,
            text="Press  ▶ Start Live View  to see webcam feed",
            fill="#334155", font=("Segoe UI",13), tags="placeholder")

    def toggle_live_view(self):
        if self._live_running: self._stop_live_view()
        else:                  self._start_live_view()

    def _start_live_view(self):
        self._sync_cfg_from_ui()
        if not self.cfg.get("pi_host","").strip():
            messagebox.showerror("Live View","Set Pi IP in Connection tab first."); return
        self._live_running = True
        self.btn_lv.configure(text="■  Stop Live View")
        self.lbl_lv_status.configure(text="Connecting…", foreground=THEMES["dark"]["warning"])
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()

    def _stop_live_view(self):
        self._live_running = False
        if hasattr(self,"btn_lv"):
            self.btn_lv.configure(text="▶  Start Live View")
        if hasattr(self,"lbl_lv_status"):
            self.lbl_lv_status.configure(text="Stopped", foreground=self.T["text2"])
        if hasattr(self,"lbl_lv_fps"):
            self.lbl_lv_fps.configure(text="")

    def _live_loop(self):
        host = self.cfg.get("pi_host","")
        port = int(self.cfg.get("pi_port",8765))
        url  = f"http://{host}:{port}/preview"
        frames = 0; t0 = time.time()
        while self._live_running:
            try:
                r = requests.get(url, timeout=4)
                if r.status_code==200 and r.headers.get("Content-Type","").startswith("image"):
                    img = PILImage.open(io.BytesIO(r.content))
                    frames += 1
                    fps = frames / max(0.1, time.time()-t0)
                    self.root.after(0, lambda i=img, f=fps: self._update_live_frame(i,f))
                else:
                    err = ""
                    try: err = r.json().get("error","bad response")
                    except: pass
                    self.root.after(0, lambda e=err: self.lbl_lv_status.configure(
                        text=f"Error: {e[:55]}", foreground=self.T["error"]))
                    time.sleep(1); continue
            except Exception as e:
                self.root.after(0, lambda msg=str(e): self.lbl_lv_status.configure(
                    text=f"Error: {msg[:55]}", foreground=self.T["error"]))
                time.sleep(1); continue
            time.sleep(0.4)

    def _update_live_frame(self, img, fps):
        if not _PIL_OK or not hasattr(self,"live_canvas"): return
        cw = self.live_canvas.winfo_width()  or 640
        ch = self.live_canvas.winfo_height() or 480
        img.thumbnail((cw,ch), PILImage.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._live_photo = photo
        self.live_canvas.delete("all")
        self.live_canvas.create_image(cw//2, ch//2, anchor="center", image=photo)
        self.lbl_lv_status.configure(text="● Live", foreground=self.T["success"])
        self.lbl_lv_fps.configure(text=f"{fps:.1f} FPS")

    # ── profile helpers ───────────────────────────────────────
    def _ensure_active_profile(self):
        if not self.cfg.get("active_profile"):
            vals = [p.get("name","") for p in self.cfg.get("profiles",[]) if p.get("name")]
            if vals: self.cfg["active_profile"] = vals[0]
        self.var_active_profile.set(self.cfg.get("active_profile",""))
        self.refresh_profiles_dropdown()
        if self.cfg.get("active_profile") in (self.cmb_profile["values"] or []):
            self.var_profile.set(self.cfg["active_profile"])
            self.load_selected_profile_to_editor()

    def set_selected_as_active_profile(self):
        self._sync_cfg_from_ui()
        name = self.var_profile.get().strip()
        if not name or not get_profile(self.cfg, name):
            messagebox.showerror("Profile","Select a valid profile first."); return
        self.cfg["active_profile"] = name
        self.var_active_profile.set(name)
        self.log(f"Active profile ✅: {name}")

    def refresh_profiles_dropdown(self):
        names = [p.get("name","?") for p in self.cfg.get("profiles",[])]
        self.cmb_profile["values"] = names
        if names and self.var_profile.get() not in names:
            self.var_profile.set(names[0])

    def add_profile(self):
        self._sync_cfg_from_ui()
        nm = f"Plant_{len(self.cfg.get('profiles',[]))+1}"
        self.cfg.setdefault("profiles",[]).append({
            "name":nm,"height_cm":20.0,"width_cm":30.0,"clearance_cm":5.0,
            "use_manual_rings":True,
            "manual_rings":[{"s":80.0,"e":75.0},{"s":75.0,"e":75.0},{"s":70.0,"e":70.0}]})
        self.refresh_profiles_dropdown()
        self.var_profile.set(nm)
        self.load_selected_profile_to_editor()
        self.log(f"Added profile: {nm}")

    def delete_profile(self):
        self._sync_cfg_from_ui()
        name = self.var_profile.get()
        profs = [p for p in self.cfg.get("profiles",[]) if p.get("name")!=name]
        if len(profs)==len(self.cfg.get("profiles",[])): return
        self.cfg["profiles"] = profs
        self.refresh_profiles_dropdown()
        if self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
            self.load_selected_profile_to_editor()
        if self.cfg.get("active_profile")==name:
            self.cfg["active_profile"]=""; self._ensure_active_profile()
        self.log(f"Deleted profile: {name}")

    def load_selected_profile_to_editor(self):
        p = get_profile(self.cfg, self.var_profile.get())
        if not p: return
        self.var_p_name.set(p.get("name",""))
        self.var_p_h.set(float(p.get("height_cm",20)))
        self.var_p_w.set(float(p.get("width_cm",30)))
        self.var_p_c.set(float(p.get("clearance_cm",5)))
        self.var_p_manual.set(bool(p.get("use_manual_rings",True)))
        mr = p.get("manual_rings",[])
        if len(mr)>=3:
            self.var_r1s.set(float(mr[0].get("s",80))); self.var_r1e.set(float(mr[0].get("e",75)))
            self.var_r2s.set(float(mr[1].get("s",75))); self.var_r2e.set(float(mr[1].get("e",75)))
            self.var_r3s.set(float(mr[2].get("s",70))); self.var_r3e.set(float(mr[2].get("e",70)))

    def save_editor_to_selected_profile(self):
        self._sync_cfg_from_ui()
        name_sel = self.var_profile.get()
        p = get_profile(self.cfg, name_sel)
        if not p: return
        new_name = self.var_p_name.get().strip() or name_sel
        p.update({"name":new_name,"height_cm":float(self.var_p_h.get()),
                  "width_cm":float(self.var_p_w.get()),"clearance_cm":float(self.var_p_c.get()),
                  "use_manual_rings":bool(self.var_p_manual.get()),
                  "manual_rings":[{"s":float(self.var_r1s.get()),"e":float(self.var_r1e.get())},
                                  {"s":float(self.var_r2s.get()),"e":float(self.var_r2e.get())},
                                  {"s":float(self.var_r3s.get()),"e":float(self.var_r3e.get())}]})
        if new_name != name_sel:
            self.var_profile.set(new_name)
            if self.cfg.get("active_profile")==name_sel:
                self.cfg["active_profile"]=new_name
                self.var_active_profile.set(new_name)
        self.refresh_profiles_dropdown()
        self.log(f"Saved profile: {new_name}")

    def preview_profile_rings(self):
        self._sync_cfg_from_ui()
        p = get_profile(self.cfg, self.var_profile.get())
        if not p: return
        rings,_,_,_,_,_,auto = compute_rings_from_profile(self.cfg,p)
        self.log("Rings [{}]: {}".format("AUTO" if auto else "MANUAL",
            " | ".join(f"S={s:.1f} E={e:.1f}" for s,e in rings)))

    def preview_auto_rings_for_active_profile(self):
        self._sync_cfg_from_ui()
        p = get_profile(self.cfg, self.cfg.get("active_profile",""))
        if not p: self.log("No active profile."); return
        rings,_,_,_,_,mid = compute_rings_auto(self.cfg,
            float(p.get("height_cm",20)),float(p.get("width_cm",30)),float(p.get("clearance_cm",5)))
        self.log("AUTO rings: mid=S{:.1f} E{:.1f} | {}".format(
            mid[0],mid[1]," | ".join(f"S={s:.1f} E={e:.1f}" for s,e in rings)))

    def _sync_ui_from_cfg_profiles_only(self):
        self.refresh_profiles_dropdown()
        self.var_active_profile.set(self.cfg.get("active_profile",""))
        if self.cfg.get("active_profile") in (self.cmb_profile["values"] or []):
            self.var_profile.set(self.cfg["active_profile"])
        elif self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
        self.load_selected_profile_to_editor()

    def export_profiles_json(self):
        self._sync_cfg_from_ui()
        path = filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")])
        if not path: return
        with open(path,"w",encoding="utf-8") as f:
            json.dump({"active_profile":self.cfg.get("active_profile",""),
                       "profiles":self.cfg.get("profiles",[])},f,indent=2)
        self.log(f"Exported profiles: {path}")

    def import_profiles_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as f: payload = json.load(f)
            if isinstance(payload,dict) and "profiles" in payload:
                self.cfg["profiles"]=payload.get("profiles",[])
                if payload.get("active_profile"): self.cfg["active_profile"]=payload["active_profile"]
            elif isinstance(payload,list): self.cfg["profiles"]=payload
            else: raise ValueError("Invalid format")
            self.refresh_profiles_dropdown(); self._ensure_active_profile()
            self._sync_ui_from_cfg_profiles_only()
            self.log(f"Imported profiles: {path}")
        except Exception as e: messagebox.showerror("Import error",str(e))

    # ── sync ─────────────────────────────────────────────────
    def _sync_cfg_from_ui(self):
        self.cfg.update({
            "robot_ssid":  self.var_ssid.get().strip(),
            "robot_pass":  self.var_pass.get(),
            "robot_base":  self.var_base.get().strip(),
            "use_cameras": bool(self.var_use_cam.get()),
            "rotate_180":  bool(self.var_rot180.get()),
            "pi_host":     self.var_pi_host.get().strip(),
            "pi_port":     int(self.var_pi_port.get().strip() or "8765"),
            "mode":               self.var_mode.get(),
            "final_total_images": int(self.var_final_images.get()),
            "test_total_images":  int(self.var_test_images.get()),
            "spd":            int(self.var_spd.get()),
            "acc":            int(self.var_acc.get()),
            "delay_s":        float(self.var_delay.get()),
            "hold_last_pose": bool(self.var_hold.get()),
            "base_center_deg":    float(self.var_base_center.get()),
            "base_half_span_deg": float(self.var_base_half.get()),
            "safe_limit_deg":     float(self.var_safe_lim.get()),
            "lock_hand_deg":      float(self.var_hand_lock.get()),
            "height_low_cm":        float(self.var_h_low.get()),
            "height_high_cm":       float(self.var_h_high.get()),
            "pose_low_sh_deg":      float(self.var_s_low.get()),
            "pose_low_el_deg":      float(self.var_e_low.get()),
            "pose_high_sh_deg":     float(self.var_s_high.get()),
            "pose_high_el_deg":     float(self.var_e_high.get()),
            "ring_height_span_deg": float(self.var_span_h.get()),
            "ring_width_span_deg":  float(self.var_span_w.get()),
            "elbow_push_gain":      float(self.var_el_gain.get()),
            "elbow_push_radius0_cm":float(self.var_el_r0.get()),
            "active_profile": self.var_active_profile.get().strip(),
        })

    def _sync_ui_from_cfg(self):
        self.var_ssid.set(self.cfg.get("robot_ssid",""))
        self.var_pass.set(self.cfg.get("robot_pass",""))
        self.var_base.set(self.cfg.get("robot_base",""))
        self.var_use_cam.set(bool(self.cfg.get("use_cameras",True)))
        self.var_rot180.set(bool(self.cfg.get("rotate_180",True)))
        self.var_pi_host.set(self.cfg.get("pi_host",""))
        self.var_pi_port.set(str(self.cfg.get("pi_port",8765)))
        self.var_mode.set(self.cfg.get("mode","Final"))
        self.var_final_images.set(int(self.cfg.get("final_total_images",200)))
        self.var_test_images.set(int(self.cfg.get("test_total_images",10)))
        self.var_spd.set(int(self.cfg.get("spd",20)))
        self.var_acc.set(int(self.cfg.get("acc",20)))
        self.var_delay.set(float(self.cfg.get("delay_s",0.3)))
        self.var_hold.set(bool(self.cfg.get("hold_last_pose",True)))
        self.var_base_center.set(float(self.cfg.get("base_center_deg",0)))
        self.var_base_half.set(float(self.cfg.get("base_half_span_deg",175)))
        self.var_safe_lim.set(float(self.cfg.get("safe_limit_deg",175)))
        self.var_hand_lock.set(float(self.cfg.get("lock_hand_deg",62)))
        self.var_h_low.set(float(self.cfg.get("height_low_cm",10)))
        self.var_h_high.set(float(self.cfg.get("height_high_cm",60)))
        self.var_s_low.set(float(self.cfg.get("pose_low_sh_deg",-84)))
        self.var_e_low.set(float(self.cfg.get("pose_low_el_deg",92)))
        self.var_s_high.set(float(self.cfg.get("pose_high_sh_deg",-60)))
        self.var_e_high.set(float(self.cfg.get("pose_high_el_deg",70)))
        self.var_span_h.set(float(self.cfg.get("ring_height_span_deg",10)))
        self.var_span_w.set(float(self.cfg.get("ring_width_span_deg",6)))
        self.var_el_gain.set(float(self.cfg.get("elbow_push_gain",0.8)))
        self.var_el_r0.set(float(self.cfg.get("elbow_push_radius0_cm",10)))
        self.var_active_profile.set(self.cfg.get("active_profile",""))
        self.refresh_profiles_dropdown()
        if self.cfg.get("active_profile") in (self.cmb_profile["values"] or []):
            self.var_profile.set(self.cfg["active_profile"])
        elif self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
        self.load_selected_profile_to_editor()

    # ── config I/O ───────────────────────────────────────────
    def save_config(self):
        self._sync_cfg_from_ui()
        path = filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")])
        if not path: return
        with open(path,"w",encoding="utf-8") as f: json.dump(self.cfg,f,indent=2)
        messagebox.showinfo("Saved",f"Config saved:\n{path}")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as f: self.cfg=json.load(f)
            self._sync_ui_from_cfg(); self._ensure_active_profile()
            messagebox.showinfo("Loaded",f"Config loaded:\n{path}")
        except Exception as e: messagebox.showerror("Load error",str(e))

    def save_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",filetypes=[("Text","*.txt")])
        if not path: return
        self.txt.configure(state="normal")
        with open(path,"w",encoding="utf-8") as f: f.write(self.txt.get("1.0","end"))
        self.txt.configure(state="disabled")
        messagebox.showinfo("Saved",f"Log saved:\n{path}")

    # ── actions ───────────────────────────────────────────────
    def open_browser(self):
        self._sync_cfg_from_ui()
        host = self.cfg.get("pi_host","192.168.4.20")
        port = self.cfg.get("pi_port",8765)
        url  = f"http://{host}:{port}/browse"
        self.log(f"Opening browser: {url}")
        webbrowser.open(url)

    def on_connect_wifi(self):
        if _IS_WINDOWS: messagebox.showinfo("Wi-Fi","Connect from Windows taskbar."); return
        self._sync_cfg_from_ui()
        try:
            ensure_connected_to(self.cfg["robot_ssid"], self.cfg["robot_pass"])
            self.log("Connected to robot Wi-Fi ✅")
        except Exception as e:
            self.log(f"Wi-Fi error: {e}")
            messagebox.showerror("Wi-Fi",str(e))

    def on_test_robot(self):
        self._sync_cfg_from_ui()
        try:
            fb = get_feedback(self.cfg, log=self.log)
            self.log("Robot ✅  base={:.1f}°".format(math.degrees(fb.get("b",0))))
        except Exception as e:
            self.log(f"Robot test failed: {e}")
            messagebox.showerror("Robot",str(e))

    def on_read_angles(self):
        self._sync_cfg_from_ui()
        fb = get_feedback(self.cfg, log=self.log)
        self.log("Angles: b={:.1f} s={:.1f} e={:.1f} h={:.1f}".format(
            math.degrees(fb.get("b",0)), math.degrees(fb.get("s",0)),
            math.degrees(fb.get("e",0)), math.degrees(fb.get("t",0))))

    def on_read_angles_to_sliders(self):
        self._sync_cfg_from_ui()
        fb = get_feedback(self.cfg, log=self.log)
        self.var_m_base.set(math.degrees(fb.get("b",0)))
        self.var_m_s.set(   math.degrees(fb.get("s",0)))
        self.var_m_e.set(   math.degrees(fb.get("e",0)))
        self.var_m_h.set(   math.degrees(fb.get("t",0)))
        self.log("Angles loaded to sliders ✅")

    def on_manual_move(self):
        self._sync_cfg_from_ui()
        try:
            torque(self.cfg, on=True, log=self.log)
            move_joints_deg(self.cfg,
                base=float(self.var_m_base.get()), shoulder=float(self.var_m_s.get()),
                elbow=float(self.var_m_e.get()),   hand=float(self.var_m_h.get()),
                spd=self.cfg["spd"], acc=self.cfg["acc"], log=self.log)
            self.log("Manual move ✅")
            self.on_read_angles()
        except Exception as e:
            self.log(f"Move failed: {e}")
            messagebox.showerror("Manual move",str(e))

    def on_test_pi(self):
        self._sync_cfg_from_ui()
        ok, msg = ensure_pi_http(self.cfg, log=self.log)
        if not ok:
            messagebox.showerror("Pi Server",
                f"Cannot reach Pi:\n{msg}\n\nMake sure pi_server.py is running."); return
        self.var_pi_host.set(self.cfg.get("pi_host",""))
        self.log(f"Pi server ✅ — {msg}")
        storage, stype = pi_get_storage(self.cfg, log=self.log)
        self.log(f"Storage: {storage} ({stype})")

    def on_test_capture(self):
        self._sync_cfg_from_ui()
        ok, msg = ensure_pi_http(self.cfg, log=self.log)
        if not ok: messagebox.showerror("Pi",f"Cannot reach Pi:\n{msg}"); return
        self.var_pi_host.set(self.cfg.get("pi_host",""))
        storage, stype = pi_get_storage(self.cfg, log=self.log)
        dest = f"{storage}/captures/_test/{datetime.now().strftime('%Y%m%d')}"
        tag  = f"TEST_{datetime.now().strftime('%H%M%S')}"
        self.log(f"Test capture tag={tag}  dest={dest}")
        ok2 = trigger_capture_store_rotate(self.cfg, tag, dest, log=self.log)
        self.log(f"Test capture: {'OK ✅' if ok2 else 'FAIL ❌'}")

    def on_abort(self):
        self.abort_event.set()
        self.log("ABORT requested…")
        self.btn_abort.configure(state="disabled")

    # ── START SCAN ───────────────────────────────────────────
    def on_start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Running","Scan already running."); return

        self._sync_cfg_from_ui()
        test_name = self.var_testname.get().strip()
        if not test_name: messagebox.showerror("Test name","Enter a test name."); return

        active = self.cfg.get("active_profile","").strip()
        if not active: messagebox.showerror("Profile","Set an Active Profile first."); return
        prof = get_profile(self.cfg, active)
        if not prof: messagebox.showerror("Profile","Active profile not found."); return

        mode = self.var_mode.get()
        total = (int(self.cfg["test_total_images"])  if mode=="Test"  else
                 int(self.cfg["final_total_images"]) if mode=="Final" else 30)
        if total < 3: messagebox.showerror("Images","Total images must be >= 3."); return

        if self._live_running:
            self._stop_live_view()
            self.log("Live view stopped for scan.")
            time.sleep(0.6)

        self.abort_event.clear()
        self.btn_start.configure(state="disabled")
        self.btn_abort.configure(state="normal")
        self.set_progress(0.0,"Starting…")

        def worker():
            t0 = time.time()
            self.log(f"\n{'─'*50}")
            self.log(f"--- START  {mode} | {active} ---")
            self.log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                if self.cfg.get("use_cameras"):
                    ok_pi, msg_pi = ensure_pi_http(self.cfg, log=self.log)
                    if not ok_pi: raise RuntimeError(f"Pi not available: {msg_pi}")
                    self.root.after(0, lambda: self.var_pi_host.set(self.cfg.get("pi_host","")))
                    self.log("Pi ✅")

                torque(self.cfg, on=True, log=self.log)
                fb0 = get_feedback(self.cfg, log=self.log)
                start_b = math.degrees(fb0.get("b",0))
                self.log(f"Start base: {start_b:.1f}°")

                storage, stype = pi_get_storage(self.cfg, log=self.log) if self.cfg["use_cameras"] else ("/home/umap/captures","SD")
                date_folder = datetime.now().strftime("%Y%m%d")
                base_storage = f"{storage}/captures/{test_name}/{date_folder}"
                self.log(f"Storage ({stype}): {base_storage}/")

                if self.cfg["use_cameras"]:
                    ok_mk, msg_mk = pi_mkdir(self.cfg, base_storage)
                    if not ok_mk: raise RuntimeError(f"Cannot create storage: {msg_mk}")

                # Verification is shown directly in the Live Log only.
                # No CSV file is created. After every ring, the app reports:
                # planned base coverage, actual feedback coverage, missing degrees,
                # average base error, and failed captures.

                rings,_,_,_,_,_,auto = compute_rings_from_profile(self.cfg, prof)
                self.log("Rings [{}]: {}".format("AUTO" if auto else "MANUAL",
                    " | ".join(f"S={s:.1f} E={e:.1f}" for s,e in rings)))

                per = [total//3]*3
                for i in range(total%3): per[i]+=1
                self.log(f"Total={total} | per ring={per}")

                safe_lim = abs(float(self.cfg["safe_limit_deg"]))
                done     = 0

                # Full 360-style base planning:
                # The old code used center=90 and half=155, which was clipped to -65..175
                # and gave only 240 degrees. Here we always use the widest safe sweep:
                # -safe_limit .. +safe_limit. With +/-175 this is 350 degrees.
                base_lo, base_hi = -safe_lim, safe_lim
                base_coverage = base_hi - base_lo
                mechanical_gap = max(0.0, 360.0 - base_coverage)
                first_reverse = choose_360_start_reverse(start_b, safe_lim)
                self.log(f"Base sweep: {base_lo:.1f}° to {base_hi:.1f}° | coverage={base_coverage:.1f}°")
                if mechanical_gap <= 12.0:
                    self.log(f"Base gap accepted: {mechanical_gap:.1f}° missing is OK for this robot limit. Reverse sweep + edge shots will be used.")
                else:
                    self.log(f"[WARN] Mechanical base gap={mechanical_gap:.1f}°. Software cannot command this gap; using edge shots + reverse sweep from the other side.")

                for ring_i, ((s_deg,e_deg), ring_count) in enumerate(zip(rings,per), start=1):
                    if self.abort_event.is_set(): break
                    ring_reverse = first_reverse if (ring_i % 2 == 1) else (not first_reverse)
                    direction_txt = "+→-" if ring_reverse else "-→+"
                    self.log(f"Ring {ring_i}/3: S={s_deg:.1f} E={e_deg:.1f} ({ring_count} shots) | base {direction_txt}")
                    move_joints_deg(self.cfg, shoulder=s_deg, elbow=e_deg,
                                    hand=self.cfg["lock_hand_deg"],
                                    spd=self.cfg["spd"], acc=self.cfg["acc"], log=None)
                    plan = plan_base_360_sweep(ring_count, safe_lim, reverse=ring_reverse)

                    ring_actual = []
                    ring_planned = []
                    ring_failed = 0

                    for ring_shot, b_deg in enumerate(plan, start=1):
                        if self.abort_event.is_set(): break
                        safe_b = clip_base_deg(b_deg, safe_lim)
                        done  += 1
                        fb_move = move_joints_deg(self.cfg, base=safe_b, shoulder=s_deg, elbow=e_deg,
                                                  hand=self.cfg["lock_hand_deg"],
                                                  spd=self.cfg["spd"], acc=self.cfg["acc"], log=None)
                        actual_b = math.degrees(fb_move.get("b", 0))
                        err_b = actual_b - safe_b
                        ring_actual.append(actual_b)
                        ring_planned.append(safe_b)

                        tag = f"{test_name}_R{ring_i}_{str(done).zfill(3)}_b{int(round(safe_b))}_S{int(round(s_deg))}"
                        capture_ok = True
                        if self.cfg["use_cameras"]:
                            capture_ok = trigger_capture_store_rotate(self.cfg, tag, base_storage, log=self.log)
                            if not capture_ok:
                                ring_failed += 1
                                self.log(f"CAPTURE FAIL at {done}/{total} (tag={tag})")

                        eta_s    = int((time.time()-t0)/max(1,done)*(total-done))
                        pred_end = datetime.now()+timedelta(seconds=eta_s)
                        self.root.after(0,
                            lambda frac=float(done)/float(total), txt=f"{done}/{total}":
                            self.set_progress(frac, txt))
                        if done%10==0 or done==total:
                            self.log(f"Progress {done}/{total} | ETA {timedelta(seconds=eta_s)} | End {pred_end.strftime('%H:%M:%S')}")
                        time.sleep(float(self.cfg["delay_s"]))

                    if ring_actual:
                        planned_cov = max(ring_planned) - min(ring_planned)
                        actual_cov  = max(ring_actual)  - min(ring_actual)
                        planned_missing = max(0.0, 360.0 - planned_cov)
                        actual_missing  = max(0.0, 360.0 - actual_cov)
                        avg_abs_err = sum(abs(a-p) for a,p in zip(ring_actual, ring_planned)) / len(ring_actual)

                        if actual_missing <= 12.0 and ring_failed == 0:
                            status = "OK ✅"
                        elif actual_missing <= 12.0 and ring_failed > 0:
                            status = "ANGLE OK, but capture failed ⚠️"
                        else:
                            status = "LOW COVERAGE ⚠️"

                        self.log(
                            f"Ring {ring_i} 360 check: {status} | "
                            f"planned {min(ring_planned):.1f}°..{max(ring_planned):.1f}° "
                            f"= {planned_cov:.1f}° coverage, missing {planned_missing:.1f}° | "
                            f"actual {min(ring_actual):.1f}°..{max(ring_actual):.1f}° "
                            f"= {actual_cov:.1f}° coverage, missing {actual_missing:.1f}° | "
                            f"avg base error={avg_abs_err:.2f}°, failed captures={ring_failed}"
                        )
                        if actual_missing > 12.0:
                            self.log(f"[WARN] Ring {ring_i}: more than ~10° is missing. Check base limit, cable obstruction, speed/acceleration, or servo feedback.")

                self.log(f"Elapsed: {timedelta(seconds=int(time.time()-t0))}")
                if not self.cfg["hold_last_pose"]: torque(self.cfg, on=False, log=self.log)
                self.log("DONE ✅")
                self.log("360 verification was printed in the Live Log only; no CSV file was created.")
                self.log(f"Browse images → http://{self.cfg.get('pi_host','192.168.4.20')}:{self.cfg.get('pi_port',8765)}/browse")

            except Exception as e:
                self.log(f"ERROR: {e}")
                self.root.after(0, lambda: messagebox.showerror("Scan error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_start.configure(state="normal"))
                self.root.after(0, lambda: self.btn_abort.configure(state="disabled"))
                self.root.after(0, lambda: self.set_progress(0.0, "Idle"))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
