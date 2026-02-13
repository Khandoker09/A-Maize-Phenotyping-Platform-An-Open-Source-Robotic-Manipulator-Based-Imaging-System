#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import shlex
import queue
import signal
import threading
import subprocess
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ================== DEFAULT SETTINGS ==================
DEFAULT = {
    "app_title": "Umapp v1",

    "robot_ssid": "RoArm-M2",
    "robot_pass": "12345678",
    "robot_base": "http://192.168.4.1",
    "safe_limit_deg": 175.0,

    "pi_host": "192.168.4.20",
    "pi_ssh_port": 22,
    "pi_user": "umap",
    "pi_capture": "/home/umap/Desktop/capture_dual.py",

"pi_auto_discover": True,
"pi_subnet_prefix": "192.168.4.",
"pi_scan_start": 2,
"pi_scan_end": 80,
    "ssh_opts": [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "PreferredAuthentications=publickey",
        "-o", "PasswordAuthentication=no",
        "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
    ],

    "use_cameras": True,
    "rotate_180": True,

    # Hold last pose
    "hold_last_pose": True,

    # Hand
    "lock_hand_deg": 62.0,

    # Base sweep center: plant is placed 90° from base
    "base_center_deg": 90.0,
    "base_half_span_deg": 155.0,

    # Speed defaults
    "spd": 20,
    "acc": 20,
    "delay_s": 0.30,

    # Modes
    "mode": "Final",  # Calibration | Test | Final
    "final_total_images": 200,
    "test_total_images": 10,

    # Active profile used for scan
    "active_profile": "",

    # Height mapping (auto rings)
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

    # Angle clamps
    "sh_min": -120.0,
    "sh_max": -30.0,
    "el_min": 20.0,
    "el_max": 110.0,

    # Plant profiles
    "profiles": [
        {
            "name": "Plant_A",
            "height_cm": 20.0,
            "width_cm": 30.0,
            "clearance_cm": 5.0,
            "use_manual_rings": True,
            "manual_rings": [
                {"s": -84.0, "e": 92.0},
                {"s": -74.0, "e": 82.0},
                {"s": -64.0, "e": 72.0},
            ],
        }
    ],
}

# ================== LOCAL SYSTEM HELPERS ==================
def _run(cmd_list, timeout=None):
    return subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)

def _sh(cmd):
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def _wifi_device():
    rc, out, err = _sh("nmcli -t -f DEVICE,TYPE device status")
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) >= 2 and parts[1] == "wifi":
            return parts[0]
    raise RuntimeError("No Wi-Fi device found (nmcli).")

def ensure_connected_to(ssid, password=None):
    try:
        rc, out, err = _sh("nmcli -t -f ACTIVE,SSID dev wifi")
        for line in out.splitlines():
            if line.startswith("yes:"):
                cur = line.split(":", 1)[1]
                if cur == ssid:
                    return True
    except Exception:
        pass

    dev = _wifi_device()
    cmd = 'nmcli dev wifi connect "{}"'.format(ssid)
    if password:
        cmd += ' password "{}"'.format(password)
    cmd += " ifname {}".format(dev)
    rc, out, err = _sh(cmd)

    try:
        rc2, out2, err2 = _sh("nmcli -t -f ACTIVE,SSID dev wifi")
        for line in out2.splitlines():
            if line.startswith("yes:") and line.split(":", 1)[1] == ssid:
                return True
    except Exception:
        pass

    if rc != 0:
        raise RuntimeError(err or out or "Could not connect to SSID {}".format(ssid))
    return True

# ================== ROBOT (HTTP) ==================
_SESSION = requests.Session()

def send_cmd(cfg, cmd, retries=3, auto_wifi=True, log=None):
    url = "{}{}".format(cfg["robot_base"], "/js?json=" + quote(json.dumps(cmd)))
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=(3, 6))
            return r.json()
        except Exception as e:
            if log:
                log("[WARN] Robot cmd failed ({}/{}): {}".format(attempt + 1, retries, e))
            time.sleep(1)
            if auto_wifi:
                try:
                    ensure_connected_to(cfg["robot_ssid"], cfg["robot_pass"])
                except Exception as ee:
                    if log:
                        log("[WARN] Wi-Fi reconnect failed: {}".format(ee))
    raise RuntimeError("Robot unreachable after retries.")

def get_feedback(cfg, log=None):
    fb = send_cmd(cfg, {"T": 105}, log=log)
    if isinstance(fb, str):
        fb = json.loads(fb)
    return fb

def torque(cfg, on=True, log=None):
    send_cmd(cfg, {"T": 210, "cmd": (1 if on else 0)}, log=log)

def move_joints_deg(cfg, base=None, shoulder=None, elbow=None, hand=None, spd=None, acc=None, log=None):
    fb = get_feedback(cfg, log=log)
    payload = {
        "T": 122,
        "b": base if base is not None else math.degrees(fb["b"]),
        "s": shoulder if shoulder is not None else math.degrees(fb["s"]),
        "e": elbow if elbow is not None else math.degrees(fb["e"]),
        "h": hand if hand is not None else math.degrees(fb["t"]),
        "spd": int(spd if spd is not None else cfg["spd"]),
        "acc": int(acc if acc is not None else cfg["acc"]),
    }
    send_cmd(cfg, payload, log=log)
    time.sleep(0.45)
    return get_feedback(cfg, log=log)

# ================== PI (SSH) ==================
def _ping(host):
    return subprocess.run(["ping", "-c", "1", "-W", "2", host], stdout=subprocess.DEVNULL).returncode == 0


# Quick TCP check (distinguish 'refused' vs 'timeout')
def _tcp_check(host, port, timeout=0.8):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True, ""
    except Exception as e:
        return False, str(e)

def _is_ipv4(s):
    return bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", str(s).strip()))

def discover_ssh_hosts(prefix, port=22, start=2, end=80, timeout=0.25, max_workers=64):
    """
    Scan prefix like '192.168.4.' for hosts with TCP port open.
    Keeps scan small/fast by default (start..end).
    """
    start = int(start); end = int(end)
    start = max(1, min(254, start))
    end = max(1, min(254, end))
    if end < start:
        start, end = end, start

    ips = [f"{prefix}{i}" for i in range(start, end + 1)]
    found = []

    def check(ip):
        ok, _ = _tcp_check(ip, port, timeout=timeout)
        return ip if ok else None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(check, ip) for ip in ips]
        for fut in as_completed(futs):
            ip = fut.result()
            if ip:
                found.append(ip)

    found.sort(key=lambda x: [int(p) for p in x.split(".")])
    return found


def ensure_pi_ssh(cfg, log=None):
    """
    Robust Pi SSH availability check with **optional auto-discovery**.

    Why you see "Connection refused":
      - Pi IP changed (DHCP) OR
      - Another device took 192.168.4.2 OR
      - Pi not on the robot Wi‑Fi at that moment

    Strategy:
      1) Try configured host:port using a quick TCP connect.
      2) If it fails and auto-discovery is enabled, scan a small IP range for open SSH (port 22).
      3) If found, update cfg['pi_host'] to the discovered IP (and log it).
      4) Finally run a real SSH "true" command (source of truth).
    """
    host = (cfg.get("pi_host") or "").strip()
    port = int(cfg.get("pi_ssh_port", 22))
    user = (cfg.get("pi_user") or "").strip()
    opts = cfg.get("ssh_opts", [])

    if not user:
        return False, "Pi user not set"

    # 1) quick TCP check on configured host
    if host:
        ok_tcp, err_tcp = _tcp_check(host, port, timeout=0.6)
        if not ok_tcp:
            # 2) auto-discover
            if cfg.get("pi_auto_discover", True):
                prefix = (cfg.get("pi_subnet_prefix") or "192.168.4.").strip()
                start = int(cfg.get("pi_scan_start", 2))
                end = int(cfg.get("pi_scan_end", 80))
                if log:
                    log(f"[Pi] SSH not reachable at {host}:{port} ({err_tcp}). Scanning {prefix}{start}-{end}…")
                candidates = discover_ssh_hosts(prefix, port=port, start=start, end=end, timeout=0.20, max_workers=64)
                if candidates:
                    host = candidates[0]
                    cfg["pi_host"] = host  # update in-memory config
                    if log:
                        log(f"[Pi] Found SSH at {host}:{port}. Using this IP ✅")
                else:
                    return False, f"SSH not reachable at {host}:{port} ({err_tcp}); scan found no SSH hosts"
            else:
                return False, f"SSH not reachable at {host}:{port} ({err_tcp})"
    else:
        return False, "Pi host not set"

    # 3) real SSH test
    cmd = ["ssh", "-p", str(port), *opts, f"{user}@{host}", "true"]
    try:
        p = _run(cmd, timeout=8)
    except Exception as e:
        return False, str(e)

    if p.returncode == 0:
        return True, f"SSH OK {host}:{port}"

    msg = (p.stderr or "").strip() or (p.stdout or "").strip() or f"SSH failed {host}:{port}"
    return False, msg


def ssh_run(pi_user, pi_host, ssh_opts, remote_cmd, timeout=25, port=22):
    cmd = ["ssh", "-p", str(port), *ssh_opts, "{}@{}".format(pi_user, pi_host), remote_cmd]
    return _run(cmd, timeout=timeout)

def ssh_retry(cfg, remote_cmd_list, timeout=25, tries=3, log=None):
    ok, msg = ensure_pi_ssh(cfg, log=log)
    if not ok:
        # mimic ssh failure return
        return subprocess.CompletedProcess(args=[], returncode=255, stdout="", stderr=msg)
    remote = " ".join([shlex.quote(str(x)) for x in remote_cmd_list])
    last = None
    for i in range(tries):
        p = ssh_run(cfg["pi_user"], cfg["pi_host"], cfg["ssh_opts"], remote, timeout=timeout, port=int(cfg.get("pi_ssh_port", 22)))
        last = p
        if p.returncode == 0:
            return p
        if log:
            log("SSH WARN ({}/{}): {}".format(i + 1, tries, (p.stderr.strip() or p.stdout.strip())))
        time.sleep(0.6)
    return last

def pi_test_ssh(cfg):
    """
    Returns (ok, msg). If ok is False, msg contains the SSH error (e.g. "Connection refused").
    """
    p = ssh_run(cfg["pi_user"], cfg["pi_host"], cfg["ssh_opts"], "echo SSH_OK",
                timeout=8, port=int(cfg.get("pi_ssh_port", 22)))
    ok = (p.returncode == 0 and "SSH_OK" in (p.stdout or ""))
    msg = (p.stderr.strip() or p.stdout.strip())
    return ok, msg

def pi_detect_usb_mount(cfg, log=None):
    ok, msg = ensure_pi_ssh(cfg, log=log)
    if not ok:
        return ""
    script = r"""bash -lc '
for d in /media/*/*; do
  if [ -d "$d" ] && mountpoint -q "$d"; then
    echo "$d"
    exit 0
  fi
done
exit 0
'"""
    p = ssh_run(cfg["pi_user"], cfg["pi_host"], cfg["ssh_opts"], script, timeout=10, port=int(cfg.get("pi_ssh_port", 22)))
    return (p.stdout or "").strip()

def pi_mkdir(cfg, path, log=None):
    ok0, msg0 = ensure_pi_ssh(cfg, log=log)
    if not ok0:
        return False, msg0
    cmd = "bash -lc {}".format(shlex.quote('mkdir -p -- "{}"'.format(path)))
    p = ssh_run(cfg["pi_user"], cfg["pi_host"], cfg["ssh_opts"], cmd, timeout=15, port=int(cfg.get("pi_ssh_port", 22)))
    ok = (p.returncode == 0)
    msg = (p.stderr.strip() or p.stdout.strip())
    return ok, msg

_SAVED_RE = re.compile(r"Saved:\s+(\S+)\s+and\s+(\S+)", re.IGNORECASE)

def trigger_capture_store_rotate(cfg, tag, target_dir, log=None):
    if not cfg["use_cameras"]:
        return True

    # Do NOT rely on ping (it can be misleading). SSH is the truth.

    p = ssh_retry(cfg, ["python3", cfg["pi_capture"], str(tag)], timeout=60, tries=3, log=log)
    if p is None or p.returncode != 0:
        if log:
            log("CAPTURE FAIL | {}".format((p.stderr.strip() if p else "ssh failed")))
        return False

    out = p.stdout or ""
    m = _SAVED_RE.search(out)
    if not m:
        if log:
            log("CAPTURE WARN | Could not parse Saved line. Output:\n{}".format(out[:220]))
        return True

    p0, p1 = m.group(1), m.group(2)

    ok, msg = pi_mkdir(cfg, target_dir, log=log)
    if not ok:
        if log:
            log("CAPTURE FAIL | Could not create target folder: {}".format(msg))
        return False

    dest0 = os.path.join(target_dir, os.path.basename(p0))
    dest1 = os.path.join(target_dir, os.path.basename(p1))

    rotate_block = ""
    if cfg["rotate_180"]:
        rotate_block = r"""
python3 - <<'PY' "$f0" "$f1"
import sys
files = sys.argv[1:]
try:
    from PIL import Image
except Exception:
    print("NO_PIL")
    sys.exit(0)
for f in files:
    im = Image.open(f)
    im = im.rotate(180, expand=True)
    im.save(f, quality=95)
print("ROT_OK", len(files))
PY
"""
    else:
        rotate_block = "true\n"

    remote_script = "\n".join([
        "set -e",
        'mkdir -p -- "{}"'.format(target_dir),
        'mv -f "{}" "{}" "{}/"'.format(p0, p1, target_dir),
        'f0="{}"'.format(dest0),
        'f1="{}"'.format(dest1),
        rotate_block,
        'echo "OK_STORE"'
    ])

    p2 = ssh_run(cfg["pi_user"], cfg["pi_host"], cfg["ssh_opts"],
                 "bash -lc {}".format(shlex.quote(remote_script)),
                 timeout=60, port=int(cfg.get("pi_ssh_port", 22)))

    if p2.returncode != 0:
        if log:
            log("STORE/ROTATE WARN | {}".format(p2.stderr.strip() or p2.stdout.strip()))
        return True

    if cfg["rotate_180"] and log:
        txt = (p2.stdout or "")
        if "NO_PIL" in txt:
            log("ROTATE WARN | Pillow missing on Pi. Install: sudo apt install -y python3-pil")
        elif "ROT_OK" in txt:
            log("Rotate 180° OK ✅")

    return True

# ================== PLANNING ==================
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

def compute_mid_pose_from_height(cfg, height_cm):
    h0 = float(cfg["height_low_cm"])
    h1 = float(cfg["height_high_cm"])
    t = 0.0 if h1 == h0 else (float(height_cm) - h0) / (h1 - h0)
    t = clamp(t, 0.0, 1.0)

    s0 = float(cfg["pose_low_sh_deg"])
    e0 = float(cfg["pose_low_el_deg"])
    s1 = float(cfg["pose_high_sh_deg"])
    e1 = float(cfg["pose_high_el_deg"])

    s_mid = s0 + t * (s1 - s0)
    e_mid = e0 + t * (e1 - e0)

    s_mid = clamp(s_mid, float(cfg["sh_min"]), float(cfg["sh_max"]))
    e_mid = clamp(e_mid, float(cfg["el_min"]), float(cfg["el_max"]))
    return s_mid, e_mid, t

def compute_elbow_push(cfg, width_cm, clearance_cm):
    radius = float(width_cm) / 2.0 + float(clearance_cm)
    r0 = float(cfg["elbow_push_radius0_cm"])
    gain = float(cfg["elbow_push_gain"])
    extra = max(0.0, (radius - r0) * gain)
    return extra, radius

def compute_rings_auto(cfg, height_cm, width_cm, clearance_cm):
    s_mid, e_mid, t = compute_mid_pose_from_height(cfg, height_cm)

    effective_w = float(width_cm) + 2.0 * float(clearance_cm)
    width_scale = clamp(20.0 / max(1e-6, effective_w), 0.25, 1.0)

    span_w = float(cfg["ring_width_span_deg"]) * width_scale
    span_h = float(cfg["ring_height_span_deg"]) * clamp(0.7 + 0.6 * t, 0.5, 1.3)

    elbow_extra, radius = compute_elbow_push(cfg, width_cm, clearance_cm)
    e_mid2 = clamp(e_mid + elbow_extra, float(cfg["el_min"]), float(cfg["el_max"]))
    s_mid2 = clamp(s_mid, float(cfg["sh_min"]), float(cfg["sh_max"]))

    ring1 = (s_mid2 - (span_w + span_h), e_mid2 + (span_w + span_h))
    ring2 = (s_mid2,                   e_mid2)
    ring3 = (s_mid2 + (span_w + span_h), e_mid2 - (span_w + span_h))

    def clamp_pose(p):
        s, e = p
        return (
            clamp(s, float(cfg["sh_min"]), float(cfg["sh_max"])),
            clamp(e, float(cfg["el_min"]), float(cfg["el_max"]))
        )

    rings = [clamp_pose(ring1), clamp_pose(ring2), clamp_pose(ring3)]
    return rings, width_scale, effective_w, elbow_extra, radius, (s_mid2, e_mid2)

def get_profile(cfg, name):
    for p in cfg.get("profiles", []):
        if p.get("name") == name:
            return p
    return None

def compute_rings_from_profile(cfg, profile):
    h = float(profile.get("height_cm", 20.0))
    w = float(profile.get("width_cm", 30.0))
    c = float(profile.get("clearance_cm", 5.0))

    if bool(profile.get("use_manual_rings", False)):
        rings = []
        for r in profile.get("manual_rings", [])[:3]:
            rings.append((float(r.get("s", -80.0)), float(r.get("e", 90.0))))
        if len(rings) == 3:
            rings_auto, ws, ew, ee, rad, mid = compute_rings_auto(cfg, h, w, c)
            return rings, ws, ew, ee, rad, mid, False

    rings, ws, ew, ee, rad, mid = compute_rings_auto(cfg, h, w, c)
    return rings, ws, ew, ee, rad, mid, True

# ================== UI ==================
class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.cfg = deepcopy(DEFAULT)

        self.master.title(self.cfg["app_title"])
        self.master.geometry("1250x840")

        self.log_q = queue.Queue()
        self.abort_event = threading.Event()
        self.worker = None

        self._build_ui()
        self._poll_logs()
        signal.signal(signal.SIGINT, lambda *_: self.on_abort())

        # Ensure an active profile exists
        self._ensure_active_profile()

    # ---------- logging ----------
    def log(self, msg):
        self.log_q.put(msg)

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
        except queue.Empty:
            pass
        self.master.after(100, self._poll_logs)

    def set_progress(self, frac, label):
        frac = max(0.0, min(1.0, float(frac)))
        self.progress["value"] = frac * 100.0
        self.lbl_prog.configure(text=label)

    # ---------- UI build ----------
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        top = ttk.Frame(self.master)
        top.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Label(top, text="Umapp v1", font=("Arial", 20, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, text="(Photogeometry platform)",
                  foreground="#444").pack(side=tk.LEFT, padx=10)

        paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_conn = ttk.Frame(nb)
        self.tab_profiles = ttk.Frame(nb)
        self.tab_modes = ttk.Frame(nb)
        self.tab_sweep = ttk.Frame(nb)
        self.tab_manual = ttk.Frame(nb)
        self.tab_mapping = ttk.Frame(nb)

        nb.add(self.tab_conn, text="1) Connection")
        nb.add(self.tab_profiles, text="2) Plant Profiles")
        nb.add(self.tab_modes, text="3) Modes")
        nb.add(self.tab_sweep, text="4) Sweep + Speed")
        nb.add(self.tab_manual, text="5) Manual Move")
        nb.add(self.tab_mapping, text="6) Height Mapping")

        self._build_tab_conn()
        self._build_tab_profiles()
        self._build_tab_modes()
        self._build_tab_sweep()
        self._build_tab_manual()
        self._build_tab_mapping()

        # bottom bar
        bar = ttk.Frame(left)
        bar.pack(fill=tk.X, pady=(10, 0))

        self.btn_start = ttk.Button(bar, text="▶ START", command=self.on_start)
        self.btn_abort = ttk.Button(bar, text="■ ABORT", command=self.on_abort, state="disabled")
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_abort.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(bar, orient=tk.HORIZONTAL, mode="determinate", length=280)
        self.progress.pack(side=tk.RIGHT, padx=(8, 0))
        self.lbl_prog = ttk.Label(bar, text="Idle")
        self.lbl_prog.pack(side=tk.RIGHT)

        # logs
        ttk.Label(right, text="Live Log", font=("Arial", 14, "bold")).pack(anchor="w")
        frame = ttk.Frame(right)
        frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.txt = tk.Text(frame, wrap="word")
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.txt.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt.configure(yscrollcommand=scroll.set)

        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(bottom, text="Clear log", command=lambda: self.txt.delete("1.0", "end")).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Save log…", command=self.save_log).pack(side=tk.LEFT, padx=8)

        cfgb = ttk.Frame(right)
        cfgb.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(cfgb, text="Load full config JSON…", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(cfgb, text="Save full config JSON…", command=self.save_config).pack(side=tk.LEFT, padx=8)

    # ---------- config I/O ----------
    def save_config(self):
        self._sync_cfg_from_ui()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)
        messagebox.showinfo("Saved", "Config saved:\n{}".format(path))

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)
            self._sync_ui_from_cfg()
            self._ensure_active_profile()
            messagebox.showinfo("Loaded", "Config loaded:\n{}".format(path))
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def save_log(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.txt.get("1.0", "end"))
        messagebox.showinfo("Saved", "Log saved:\n{}".format(path))

    # ---------- PROFILES JSON (NEW) ----------
    def export_profiles_json(self):
        self._sync_cfg_from_ui()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        payload = {
            "active_profile": self.cfg.get("active_profile", ""),
            "profiles": self.cfg.get("profiles", []),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.log("Exported profiles JSON: {}".format(path))
        messagebox.showinfo("Profiles saved", "Profiles saved:\n{}".format(path))

    def import_profiles_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "profiles" in payload:
                self.cfg["profiles"] = payload.get("profiles", [])
                if payload.get("active_profile"):
                    self.cfg["active_profile"] = payload.get("active_profile", "")
            elif isinstance(payload, list):
                # allow simple list-of-profiles JSON
                self.cfg["profiles"] = payload
            else:
                raise ValueError("Invalid profiles JSON format.")
            self.refresh_profiles_dropdown()
            self._ensure_active_profile()
            self._sync_ui_from_cfg_profiles_only()
            self.log("Imported profiles JSON: {}".format(path))
            messagebox.showinfo("Profiles loaded", "Profiles loaded:\n{}".format(path))
        except Exception as e:
            messagebox.showerror("Import error", str(e))

    # ---------- tabs ----------
    def _build_tab_conn(self):
        box = ttk.LabelFrame(self.tab_conn, text="Robot Wi-Fi + URL")
        box.pack(fill=tk.X, padx=10, pady=10)

        self.var_ssid = tk.StringVar(value=self.cfg["robot_ssid"])
        self.var_pass = tk.StringVar(value=self.cfg["robot_pass"])
        self.var_base = tk.StringVar(value=self.cfg["robot_base"])

        g = ttk.Frame(box)
        g.pack(fill=tk.X, padx=10, pady=10)
        g.columnconfigure(1, weight=1)

        ttk.Label(g, text="Robot SSID:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(g, textvariable=self.var_ssid).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(g, text="Robot password:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(g, textvariable=self.var_pass, show="*").grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(g, text="Robot URL:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(g, textvariable=self.var_base).grid(row=2, column=1, sticky="ew", pady=4)

        b = ttk.Frame(box)
        b.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(b, text="Connect Wi-Fi", command=self.on_connect_wifi).pack(side=tk.LEFT)
        ttk.Button(b, text="Test Robot", command=self.on_test_robot).pack(side=tk.LEFT, padx=8)
        ttk.Button(b, text="Read Angles", command=self.on_read_angles).pack(side=tk.LEFT, padx=8)

        box2 = ttk.LabelFrame(self.tab_conn, text="Raspberry Pi (Cameras)")
        box2.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.var_use_cam = tk.BooleanVar(value=self.cfg["use_cameras"])
        self.var_rot180 = tk.BooleanVar(value=self.cfg["rotate_180"])
        self.var_pi_host = tk.StringVar(value=self.cfg["pi_host"])
        self.var_pi_port = tk.StringVar(value=str(self.cfg.get("pi_ssh_port", 22)))
        self.var_pi_user = tk.StringVar(value=self.cfg["pi_user"])
        self.var_pi_cap = tk.StringVar(value=self.cfg["pi_capture"])

        g2 = ttk.Frame(box2)
        g2.pack(fill=tk.X, padx=10, pady=10)
        g2.columnconfigure(1, weight=1)

        ttk.Checkbutton(g2, text="Trigger Pi cameras during scan", variable=self.var_use_cam)\
            .grid(row=0, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Checkbutton(g2, text="Rotate saved images 180° (robot upside down)", variable=self.var_rot180)\
            .grid(row=1, column=0, columnspan=2, sticky="w", pady=6)

        ttk.Label(g2, text="Pi IP:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(g2, textvariable=self.var_pi_host).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(g2, text="Pi SSH port:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(g2, textvariable=self.var_pi_port, width=8).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(g2, text="Pi user:").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(g2, textvariable=self.var_pi_user).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(g2, text="capture_dual.py path:").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(g2, textvariable=self.var_pi_cap).grid(row=5, column=1, sticky="ew", pady=4)

        bb = ttk.Frame(box2)
        bb.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(bb, text="Test Pi (ping + USB)", command=self.on_test_pi).pack(side=tk.LEFT)
        ttk.Button(bb, text="Test capture (save to USB)", command=self.on_test_capture).pack(side=tk.LEFT, padx=8)

    def _build_tab_profiles(self):
        box = ttk.LabelFrame(self.tab_profiles, text="Plant Profiles (create your own + save/load profiles JSON)")
        box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # selection row
        top = ttk.Frame(box)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))

        self.var_profile = tk.StringVar()
        self.cmb_profile = ttk.Combobox(top, textvariable=self.var_profile, state="readonly", width=24)
        self.cmb_profile.pack(side=tk.LEFT)

        ttk.Button(top, text="Refresh list", command=self.refresh_profiles_dropdown).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Add profile", command=self.add_profile).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Delete profile", command=self.delete_profile).pack(side=tk.LEFT, padx=8)

        ttk.Separator(box).pack(fill=tk.X, padx=10, pady=8)

        # Active profile row (NEW)
        ap = ttk.Frame(box)
        ap.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(ap, text="Active profile used for scan:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.var_active_profile = tk.StringVar(value=self.cfg.get("active_profile", ""))
        self.ent_active = ttk.Entry(ap, textvariable=self.var_active_profile, state="readonly", width=28)
        self.ent_active.pack(side=tk.LEFT, padx=8)
        ttk.Button(ap, text="Use selected profile for scan ✅", command=self.set_selected_as_active_profile)\
            .pack(side=tk.LEFT, padx=8)

        # Export/Import profiles JSON (NEW)
        pj = ttk.Frame(box)
        pj.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(pj, text="Export PROFILES to JSON…", command=self.export_profiles_json).pack(side=tk.LEFT)
        ttk.Button(pj, text="Import PROFILES from JSON…", command=self.import_profiles_json).pack(side=tk.LEFT, padx=8)

        # editor
        edit = ttk.Frame(box)
        edit.pack(fill=tk.X, padx=10, pady=10)
        edit.columnconfigure(1, weight=1)
        edit.columnconfigure(3, weight=1)

        self.var_p_name = tk.StringVar(value="")
        self.var_p_h = tk.DoubleVar(value=20.0)
        self.var_p_w = tk.DoubleVar(value=30.0)
        self.var_p_c = tk.DoubleVar(value=5.0)
        self.var_p_manual = tk.BooleanVar(value=True)

        def row(r, label, var, c):
            ttk.Label(edit, text=label).grid(row=r, column=c, sticky="w", padx=(0, 6), pady=6)
            ttk.Entry(edit, textvariable=var).grid(row=r, column=c + 1, sticky="ew", pady=6)

        row(0, "Profile name:", self.var_p_name, 0)
        row(0, "Height (cm):", self.var_p_h, 2)
        row(1, "Width (cm):", self.var_p_w, 0)
        row(1, "Clearance (cm):", self.var_p_c, 2)

        ttk.Checkbutton(edit, text="Use MANUAL rings (recommended for your setup)", variable=self.var_p_manual)\
            .grid(row=2, column=0, columnspan=2, sticky="w", pady=6)

        rings = ttk.LabelFrame(box, text="Manual rings (3 rings: shoulder + elbow)")
        rings.pack(fill=tk.X, padx=10, pady=(0, 10))
        rr = ttk.Frame(rings)
        rr.pack(fill=tk.X, padx=10, pady=10)

        self.var_r1s = tk.DoubleVar(value=-84.0); self.var_r1e = tk.DoubleVar(value=92.0)
        self.var_r2s = tk.DoubleVar(value=-74.0); self.var_r2e = tk.DoubleVar(value=82.0)
        self.var_r3s = tk.DoubleVar(value=-64.0); self.var_r3e = tk.DoubleVar(value=72.0)

        def rbox(col, title, vs, ve):
            f = ttk.Frame(rr)
            f.grid(row=0, column=col, sticky="ew", padx=10)
            ttk.Label(f, text=title, font=("Arial", 10, "bold")).pack(anchor="w")
            line = ttk.Frame(f); line.pack(fill=tk.X, pady=4)
            ttk.Label(line, text="S:").pack(side=tk.LEFT)
            ttk.Entry(line, textvariable=vs, width=8).pack(side=tk.LEFT, padx=4)
            ttk.Label(line, text="E:").pack(side=tk.LEFT, padx=(8, 0))
            ttk.Entry(line, textvariable=ve, width=8).pack(side=tk.LEFT, padx=4)

        rbox(0, "Ring1", self.var_r1s, self.var_r1e)
        rbox(1, "Ring2", self.var_r2s, self.var_r2e)
        rbox(2, "Ring3", self.var_r3s, self.var_r3e)

        btns = ttk.Frame(box)
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btns, text="Load selected → editor", command=self.load_selected_profile_to_editor).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save editor → selected", command=self.save_editor_to_selected_profile).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Preview rings (log)", command=self.preview_profile_rings).pack(side=tk.LEFT, padx=8)

        # Hook selection change: load profile into editor (and optionally set active)
        self.cmb_profile.bind("<<ComboboxSelected>>", self.on_profile_selected)

        self.refresh_profiles_dropdown()
        if self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
            self.load_selected_profile_to_editor()

    def _build_tab_modes(self):
        box = ttk.LabelFrame(self.tab_modes, text="Modes (select profile first, then Start)")
        box.pack(fill=tk.X, padx=10, pady=10)

        self.var_mode = tk.StringVar(value=self.cfg["mode"])
        self.var_testname = tk.StringVar(value="test1")
        self.var_final_images = tk.IntVar(value=self.cfg["final_total_images"])
        self.var_test_images = tk.IntVar(value=self.cfg["test_total_images"])

        g = ttk.Frame(box); g.pack(fill=tk.X, padx=10, pady=10)
        g.columnconfigure(1, weight=1)

        ttk.Label(g, text="Mode:").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Combobox(g, textvariable=self.var_mode, state="readonly",
                     values=["Calibration", "Test", "Final"]).grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(g, text="Test name:").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(g, textvariable=self.var_testname).grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(g, text="Test images (Test mode):").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(g, textvariable=self.var_test_images).grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(g, text="Final images (Final mode):").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(g, textvariable=self.var_final_images).grid(row=3, column=1, sticky="ew", pady=6)

        ap = ttk.Frame(box)
        ap.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(ap, text="Active profile for scan:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.ent_active2 = ttk.Entry(ap, textvariable=self.var_active_profile, state="readonly", width=28)
        self.ent_active2.pack(side=tk.LEFT, padx=8)
        ttk.Label(ap, text="(set it in Plant Profiles tab)", foreground="#444").pack(side=tk.LEFT)

    def _build_tab_sweep(self):
        box = ttk.LabelFrame(self.tab_sweep, text="Sweep + speed")
        box.pack(fill=tk.X, padx=10, pady=10)

        self.var_spd = tk.IntVar(value=self.cfg["spd"])
        self.var_acc = tk.IntVar(value=self.cfg["acc"])
        self.var_delay = tk.DoubleVar(value=self.cfg["delay_s"])
        self.var_hold = tk.BooleanVar(value=self.cfg["hold_last_pose"])

        self.var_base_center = tk.DoubleVar(value=self.cfg["base_center_deg"])
        self.var_base_half = tk.DoubleVar(value=self.cfg["base_half_span_deg"])
        self.var_safe_lim = tk.DoubleVar(value=self.cfg["safe_limit_deg"])
        self.var_hand_lock = tk.DoubleVar(value=self.cfg["lock_hand_deg"])

        g = ttk.Frame(box)
        g.pack(fill=tk.X, padx=10, pady=10)
        g.columnconfigure(1, weight=1)
        g.columnconfigure(3, weight=1)

        def row(r, label, var, c):
            ttk.Label(g, text=label).grid(row=r, column=c, sticky="w", padx=(0, 6), pady=6)
            ttk.Entry(g, textvariable=var).grid(row=r, column=c + 1, sticky="ew", pady=6)

        row(0, "Speed spd:", self.var_spd, 0)
        row(0, "Accel acc:", self.var_acc, 2)
        row(1, "Delay (sec):", self.var_delay, 0)
        ttk.Checkbutton(g, text="Hold last pose (keep torque ON)", variable=self.var_hold)\
            .grid(row=1, column=2, columnspan=2, sticky="w", pady=6)

        row(2, "Base center (°):", self.var_base_center, 0)
        row(2, "Half span (°):", self.var_base_half, 2)
        row(3, "Safety limit (°):", self.var_safe_lim, 0)
        row(3, "Hand lock (°):", self.var_hand_lock, 2)

        p = ttk.Frame(box)
        p.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(p, text="Quick speed:").pack(side=tk.LEFT)
        ttk.Button(p, text="Slow", command=lambda: self.set_speed(10, 10, 0.60)).pack(side=tk.LEFT, padx=6)
        ttk.Button(p, text="Normal", command=lambda: self.set_speed(20, 20, 0.30)).pack(side=tk.LEFT, padx=6)
        ttk.Button(p, text="Fast", command=lambda: self.set_speed(30, 30, 0.15)).pack(side=tk.LEFT, padx=6)

    def set_speed(self, spd, acc, delay):
        self.var_spd.set(int(spd))
        self.var_acc.set(int(acc))
        self.var_delay.set(float(delay))
        self.log("Speed preset: spd={} acc={} delay={}s".format(spd, acc, delay))

    def _build_tab_manual(self):
        box = ttk.LabelFrame(self.tab_manual, text="Manual move (for framing check)")
        box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.var_m_base = tk.DoubleVar(value=0.0)
        self.var_m_s = tk.DoubleVar(value=-82.0)
        self.var_m_e = tk.DoubleVar(value=90.0)
        self.var_m_h = tk.DoubleVar(value=62.0)

        g = ttk.Frame(box)
        g.pack(fill=tk.X, padx=10, pady=10)
        g.columnconfigure(1, weight=1)

        def slider_row(row, label, var, lo, hi, step=1.0):
            ttk.Label(g, text=label).grid(row=row, column=0, sticky="w", pady=6)
            s = ttk.Scale(g, from_=lo, to=hi, variable=var)
            s.grid(row=row, column=1, sticky="ew", pady=6)
            v = ttk.Label(g, text="{:.1f}°".format(var.get()))
            v.grid(row=row, column=2, sticky="e", padx=(8, 0))

            def update_label(*_):
                v.configure(text="{:.1f}°".format(var.get()))
            var.trace_add("write", update_label)

            bb = ttk.Frame(g)
            bb.grid(row=row, column=3, sticky="e", padx=(8, 0))
            ttk.Button(bb, text="-", width=3, command=lambda: var.set(var.get() - step)).pack(side=tk.LEFT)
            ttk.Button(bb, text="+", width=3, command=lambda: var.set(var.get() + step)).pack(side=tk.LEFT, padx=4)

        slider_row(0, "Base", self.var_m_base, -175, 175, step=1.0)
        slider_row(1, "Shoulder", self.var_m_s, -120, -30, step=1.0)
        slider_row(2, "Elbow", self.var_m_e, 20, 110, step=1.0)
        slider_row(3, "Hand", self.var_m_h, 0, 120, step=1.0)

        bb = ttk.Frame(box)
        bb.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(bb, text="Read angles → sliders", command=self.on_read_angles_to_sliders).pack(side=tk.LEFT)
        ttk.Button(bb, text="Move now", command=self.on_manual_move).pack(side=tk.LEFT, padx=8)

    def _build_tab_mapping(self):
        box = ttk.LabelFrame(self.tab_mapping, text="Height → Mid Pose mapping (Auto Rings)")
        box.pack(fill=tk.X, padx=10, pady=10)

        self.var_h_low = tk.DoubleVar(value=self.cfg["height_low_cm"])
        self.var_h_high = tk.DoubleVar(value=self.cfg["height_high_cm"])
        self.var_s_low = tk.DoubleVar(value=self.cfg["pose_low_sh_deg"])
        self.var_e_low = tk.DoubleVar(value=self.cfg["pose_low_el_deg"])
        self.var_s_high = tk.DoubleVar(value=self.cfg["pose_high_sh_deg"])
        self.var_e_high = tk.DoubleVar(value=self.cfg["pose_high_el_deg"])
        self.var_span_h = tk.DoubleVar(value=self.cfg["ring_height_span_deg"])
        self.var_span_w = tk.DoubleVar(value=self.cfg["ring_width_span_deg"])
        self.var_el_gain = tk.DoubleVar(value=self.cfg["elbow_push_gain"])
        self.var_el_r0 = tk.DoubleVar(value=self.cfg["elbow_push_radius0_cm"])

        g = ttk.Frame(box)
        g.pack(fill=tk.X, padx=10, pady=10)
        g.columnconfigure(1, weight=1)
        g.columnconfigure(3, weight=1)

        def row(r, label, var, c):
            ttk.Label(g, text=label).grid(row=r, column=c, sticky="w", padx=(0, 6), pady=6)
            ttk.Entry(g, textvariable=var).grid(row=r, column=c + 1, sticky="ew", pady=6)

        row(0, "Height low (cm):", self.var_h_low, 0)
        row(0, "Height high (cm):", self.var_h_high, 2)
        row(1, "Pose LOW Shoulder:", self.var_s_low, 0)
        row(1, "Pose LOW Elbow:", self.var_e_low, 2)
        row(2, "Pose HIGH Shoulder:", self.var_s_high, 0)
        row(2, "Pose HIGH Elbow:", self.var_e_high, 2)
        row(3, "Ring height span (deg):", self.var_span_h, 0)
        row(3, "Ring width span (deg):", self.var_span_w, 2)
        row(4, "Elbow push gain (deg/cm):", self.var_el_gain, 0)
        row(4, "Elbow push radius0 (cm):", self.var_el_r0, 2)

        ttk.Button(box, text="Preview AUTO rings for ACTIVE profile (log)", command=self.preview_auto_rings_for_active_profile)\
            .pack(anchor="w", padx=10, pady=(0, 10))

    # ---------- profile selection behavior ----------
    def on_profile_selected(self, *_):
        # Load into editor, but DO NOT auto-set active (user clicks button)
        self.load_selected_profile_to_editor()

    def _ensure_active_profile(self):
        # If active_profile empty, pick first profile name
        if not self.cfg.get("active_profile"):
            vals = [p.get("name", "") for p in self.cfg.get("profiles", []) if p.get("name")]
            if vals:
                self.cfg["active_profile"] = vals[0]
        self.var_active_profile.set(self.cfg.get("active_profile", ""))

        # ensure dropdown selection exists
        self.refresh_profiles_dropdown()
        if self.cfg.get("active_profile") in (self.cmb_profile["values"] or []):
            self.var_profile.set(self.cfg["active_profile"])
            self.load_selected_profile_to_editor()

    def set_selected_as_active_profile(self):
        self._sync_cfg_from_ui()
        name = self.var_profile.get().strip()
        if not name:
            messagebox.showerror("Profile", "Select a profile first.")
            return
        if not get_profile(self.cfg, name):
            messagebox.showerror("Profile", "Selected profile not found.")
            return
        self.cfg["active_profile"] = name
        self.var_active_profile.set(name)
        self.log("Active profile set ✅: {}".format(name))

    # ---------- profile helpers ----------
    def refresh_profiles_dropdown(self):
        names = [p.get("name", "Unnamed") for p in self.cfg.get("profiles", [])]
        self.cmb_profile["values"] = names
        if names and (self.var_profile.get() not in names):
            self.var_profile.set(names[0])

    def add_profile(self):
        self._sync_cfg_from_ui()
        new_name = "Plant_{}".format(len(self.cfg.get("profiles", [])) + 1)
        self.cfg.setdefault("profiles", []).append({
            "name": new_name,
            "height_cm": 20.0,
            "width_cm": 30.0,
            "clearance_cm": 5.0,
            "use_manual_rings": True,
            "manual_rings": [
                {"s": -84.0, "e": 92.0},
                {"s": -74.0, "e": 82.0},
                {"s": -64.0, "e": 72.0},
            ],
        })
        self.refresh_profiles_dropdown()
        self.var_profile.set(new_name)
        self.load_selected_profile_to_editor()
        self.log("Added profile: {}".format(new_name))

    def delete_profile(self):
        self._sync_cfg_from_ui()
        name = self.var_profile.get()
        if not name:
            return
        profs = self.cfg.get("profiles", [])
        profs2 = [p for p in profs if p.get("name") != name]
        if len(profs2) == len(profs):
            return
        self.cfg["profiles"] = profs2
        self.refresh_profiles_dropdown()
        if self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
            self.load_selected_profile_to_editor()
        # if deleted active profile, reselect
        if self.cfg.get("active_profile") == name:
            self.cfg["active_profile"] = ""
            self._ensure_active_profile()
        self.log("Deleted profile: {}".format(name))

    def load_selected_profile_to_editor(self):
        name = self.var_profile.get()
        p = get_profile(self.cfg, name)
        if not p:
            return
        self.var_p_name.set(p.get("name", ""))
        self.var_p_h.set(float(p.get("height_cm", 20.0)))
        self.var_p_w.set(float(p.get("width_cm", 30.0)))
        self.var_p_c.set(float(p.get("clearance_cm", 5.0)))
        self.var_p_manual.set(bool(p.get("use_manual_rings", True)))
        mr = p.get("manual_rings", [])
        if len(mr) >= 3:
            self.var_r1s.set(float(mr[0].get("s", -84.0))); self.var_r1e.set(float(mr[0].get("e", 92.0)))
            self.var_r2s.set(float(mr[1].get("s", -74.0))); self.var_r2e.set(float(mr[1].get("e", 82.0)))
            self.var_r3s.set(float(mr[2].get("s", -64.0))); self.var_r3e.set(float(mr[2].get("e", 72.0)))

    def save_editor_to_selected_profile(self):
        self._sync_cfg_from_ui()
        name_sel = self.var_profile.get()
        if not name_sel:
            return
        p = get_profile(self.cfg, name_sel)
        if not p:
            return
        new_name = self.var_p_name.get().strip() or name_sel
        p["name"] = new_name
        p["height_cm"] = float(self.var_p_h.get())
        p["width_cm"] = float(self.var_p_w.get())
        p["clearance_cm"] = float(self.var_p_c.get())
        p["use_manual_rings"] = bool(self.var_p_manual.get())
        p["manual_rings"] = [
            {"s": float(self.var_r1s.get()), "e": float(self.var_r1e.get())},
            {"s": float(self.var_r2s.get()), "e": float(self.var_r2e.get())},
            {"s": float(self.var_r3s.get()), "e": float(self.var_r3e.get())},
        ]
        if new_name != name_sel:
            self.var_profile.set(new_name)
            # also update active_profile if it was the old name
            if self.cfg.get("active_profile") == name_sel:
                self.cfg["active_profile"] = new_name
                self.var_active_profile.set(new_name)
        self.refresh_profiles_dropdown()
        self.log("Saved profile: {} (manual_rings={})".format(new_name, p["use_manual_rings"]))

    def preview_profile_rings(self):
        self._sync_cfg_from_ui()
        name = self.var_profile.get()
        p = get_profile(self.cfg, name)
        if not p:
            return
        rings, ws, ew, ee, rad, mid, used_auto = compute_rings_from_profile(self.cfg, p)
        self.log("--- PROFILE RINGS PREVIEW ---")
        self.log("Profile={} | H={} W={} clear={} | use_manual_rings={}".format(
            p.get("name"), p.get("height_cm"), p.get("width_cm"), p.get("clearance_cm"), p.get("use_manual_rings", False)
        ))
        self.log("Used {} rings".format("AUTO" if used_auto else "MANUAL"))
        self.log("Rings (S,E): " + " | ".join(["({:.1f},{:.1f})".format(s, e) for (s, e) in rings]))

    def preview_auto_rings_for_active_profile(self):
        self._sync_cfg_from_ui()
        ap = self.cfg.get("active_profile", "")
        p = get_profile(self.cfg, ap)
        if not p:
            self.log("No active profile selected.")
            return
        h = float(p.get("height_cm", 20.0))
        w = float(p.get("width_cm", 30.0))
        c = float(p.get("clearance_cm", 5.0))
        rings, ws, ew, ee, rad, mid = compute_rings_auto(self.cfg, h, w, c)
        self.log("--- AUTO RINGS PREVIEW (ACTIVE PROFILE) ---")
        self.log("Active profile={} | H={:.1f} W={:.1f} clear={:.1f}".format(ap, h, w, c))
        self.log("EffectiveW={:.1f} | width_scale={:.2f} | elbow_push=+{:.1f}°".format(ew, ws, ee))
        self.log("Mid pose: S={:.1f} E={:.1f}".format(mid[0], mid[1]))
        self.log("Rings (S,E): " + " | ".join(["({:.1f},{:.1f})".format(s, e) for (s, e) in rings]))

    def _sync_ui_from_cfg_profiles_only(self):
        self.refresh_profiles_dropdown()
        self.var_active_profile.set(self.cfg.get("active_profile", ""))
        if self.cfg.get("active_profile") in (self.cmb_profile["values"] or []):
            self.var_profile.set(self.cfg["active_profile"])
        elif self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
        self.load_selected_profile_to_editor()

    # ---------- sync UI <-> cfg ----------
    def _sync_cfg_from_ui(self):
        self.cfg["robot_ssid"] = self.var_ssid.get().strip()
        self.cfg["robot_pass"] = self.var_pass.get()
        self.cfg["robot_base"] = self.var_base.get().strip()

        self.cfg["use_cameras"] = bool(self.var_use_cam.get())
        self.cfg["rotate_180"] = bool(self.var_rot180.get())
        self.cfg["pi_host"] = self.var_pi_host.get().strip()
        try:
            self.cfg["pi_ssh_port"] = int(str(self.var_pi_port.get()).strip() or "22")
        except Exception:
            self.cfg["pi_ssh_port"] = 22
        self.cfg["pi_user"] = self.var_pi_user.get().strip()
        self.cfg["pi_capture"] = self.var_pi_cap.get().strip()

        self.cfg["mode"] = self.var_mode.get()
        self.cfg["final_total_images"] = int(self.var_final_images.get())
        self.cfg["test_total_images"] = int(self.var_test_images.get())

        self.cfg["spd"] = int(self.var_spd.get())
        self.cfg["acc"] = int(self.var_acc.get())
        self.cfg["delay_s"] = float(self.var_delay.get())
        self.cfg["hold_last_pose"] = bool(self.var_hold.get())

        self.cfg["base_center_deg"] = float(self.var_base_center.get())
        self.cfg["base_half_span_deg"] = float(self.var_base_half.get())
        self.cfg["safe_limit_deg"] = float(self.var_safe_lim.get())
        self.cfg["lock_hand_deg"] = float(self.var_hand_lock.get())

        self.cfg["height_low_cm"] = float(self.var_h_low.get())
        self.cfg["height_high_cm"] = float(self.var_h_high.get())
        self.cfg["pose_low_sh_deg"] = float(self.var_s_low.get())
        self.cfg["pose_low_el_deg"] = float(self.var_e_low.get())
        self.cfg["pose_high_sh_deg"] = float(self.var_s_high.get())
        self.cfg["pose_high_el_deg"] = float(self.var_e_high.get())
        self.cfg["ring_height_span_deg"] = float(self.var_span_h.get())
        self.cfg["ring_width_span_deg"] = float(self.var_span_w.get())
        self.cfg["elbow_push_gain"] = float(self.var_el_gain.get())
        self.cfg["elbow_push_radius0_cm"] = float(self.var_el_r0.get())

        # active profile (from UI mirror)
        self.cfg["active_profile"] = self.var_active_profile.get().strip()

    def _sync_ui_from_cfg(self):
        self.var_ssid.set(self.cfg.get("robot_ssid", ""))
        self.var_pass.set(self.cfg.get("robot_pass", ""))
        self.var_base.set(self.cfg.get("robot_base", ""))

        self.var_use_cam.set(bool(self.cfg.get("use_cameras", True)))
        self.var_rot180.set(bool(self.cfg.get("rotate_180", True)))
        self.var_pi_host.set(self.cfg.get("pi_host", ""))
        self.var_pi_port.set(str(self.cfg.get("pi_ssh_port", 22)))
        self.var_pi_user.set(self.cfg.get("pi_user", ""))
        self.var_pi_cap.set(self.cfg.get("pi_capture", ""))

        self.var_mode.set(self.cfg.get("mode", "Final"))
        self.var_final_images.set(int(self.cfg.get("final_total_images", 200)))
        self.var_test_images.set(int(self.cfg.get("test_total_images", 10)))

        self.var_spd.set(int(self.cfg.get("spd", 20)))
        self.var_acc.set(int(self.cfg.get("acc", 20)))
        self.var_delay.set(float(self.cfg.get("delay_s", 0.3)))
        self.var_hold.set(bool(self.cfg.get("hold_last_pose", True)))

        self.var_base_center.set(float(self.cfg.get("base_center_deg", 90.0)))
        self.var_base_half.set(float(self.cfg.get("base_half_span_deg", 155.0)))
        self.var_safe_lim.set(float(self.cfg.get("safe_limit_deg", 175.0)))
        self.var_hand_lock.set(float(self.cfg.get("lock_hand_deg", 62.0)))

        self.var_h_low.set(float(self.cfg.get("height_low_cm", 10.0)))
        self.var_h_high.set(float(self.cfg.get("height_high_cm", 60.0)))
        self.var_s_low.set(float(self.cfg.get("pose_low_sh_deg", -84.0)))
        self.var_e_low.set(float(self.cfg.get("pose_low_el_deg", 92.0)))
        self.var_s_high.set(float(self.cfg.get("pose_high_sh_deg", -60.0)))
        self.var_e_high.set(float(self.cfg.get("pose_high_el_deg", 70.0)))
        self.var_span_h.set(float(self.cfg.get("ring_height_span_deg", 10.0)))
        self.var_span_w.set(float(self.cfg.get("ring_width_span_deg", 6.0)))
        self.var_el_gain.set(float(self.cfg.get("elbow_push_gain", 0.8)))
        self.var_el_r0.set(float(self.cfg.get("elbow_push_radius0_cm", 10.0)))

        self.var_active_profile.set(self.cfg.get("active_profile", ""))
        self.refresh_profiles_dropdown()
        if self.cfg.get("active_profile") in (self.cmb_profile["values"] or []):
            self.var_profile.set(self.cfg["active_profile"])
        elif self.cmb_profile["values"]:
            self.var_profile.set(self.cmb_profile["values"][0])
        self.load_selected_profile_to_editor()

    # ---------- actions ----------
    def on_connect_wifi(self):
        self._sync_cfg_from_ui()
        try:
            ensure_connected_to(self.cfg["robot_ssid"], self.cfg["robot_pass"])
            self.log("Connected to robot Wi-Fi ✅")
        except Exception as e:
            self.log("Wi-Fi connect error: {}".format(e))
            messagebox.showerror("Wi-Fi", str(e))

    def on_test_robot(self):
        self._sync_cfg_from_ui()
        try:
            fb = get_feedback(self.cfg, log=self.log)
            b = math.degrees(fb.get("b", 0.0))
            self.log("Robot responds ✅ (base {:.1f}°)".format(b))
        except Exception as e:
            self.log("Robot test failed: {}".format(e))
            messagebox.showerror("Robot", str(e))

    def on_read_angles(self):
        self._sync_cfg_from_ui()
        fb = get_feedback(self.cfg, log=self.log)
        b = math.degrees(fb.get("b", 0.0))
        s = math.degrees(fb.get("s", 0.0))
        e = math.degrees(fb.get("e", 0.0))
        h = math.degrees(fb.get("t", 0.0))
        self.log("Angles: b={:.1f} s={:.1f} e={:.1f} h={:.1f}".format(b, s, e, h))

    def on_read_angles_to_sliders(self):
        self._sync_cfg_from_ui()
        fb = get_feedback(self.cfg, log=self.log)
        self.var_m_base.set(math.degrees(fb.get("b", 0.0)))
        self.var_m_s.set(math.degrees(fb.get("s", 0.0)))
        self.var_m_e.set(math.degrees(fb.get("e", 0.0)))
        self.var_m_h.set(math.degrees(fb.get("t", 0.0)))
        self.log("Loaded angles into sliders ✅")

    def on_manual_move(self):
        self._sync_cfg_from_ui()
        try:
            torque(self.cfg, on=True, log=self.log)
            move_joints_deg(
                self.cfg,
                base=float(self.var_m_base.get()),
                shoulder=float(self.var_m_s.get()),
                elbow=float(self.var_m_e.get()),
                hand=float(self.var_m_h.get()),
                spd=self.cfg["spd"],
                acc=self.cfg["acc"],
                log=self.log
            )
            self.log("Manual move done ✅")
            self.on_read_angles()
        except Exception as e:
            self.log("Manual move failed: {}".format(e))
            messagebox.showerror("Manual move", str(e))

    def on_test_pi(self):
        self._sync_cfg_from_ui()

        # 1) Robust SSH check (auto-discovery if enabled)
        ok, msg = ensure_pi_ssh(self.cfg, log=self.log)
        if not ok:
            messagebox.showerror(
                "Pi SSH",
                "SSH not available:\n" + str(msg) + "\n\n"
                "Permanent fix (recommended):\n"
                "1) Give the Pi a STATIC IP on the robot Wi‑Fi (e.g. 192.168.4.20)\n"
                "2) Or keep 'Pi auto-discover' enabled (this app can scan a small IP range)."
            )
            return


        # If ensure_pi_ssh updated pi_host, reflect in UI
        self.var_pi_host.set(self.cfg.get("pi_host", ""))
        self.log("SSH works ✅")

        # 3) USB mount check
        usb = pi_detect_usb_mount(self.cfg, log=self.log)
        if usb:
            self.log("USB detected on Pi ✅  mount={}".format(usb))
        else:
            self.log("USB not detected on Pi ❌ (no mounted /media/*/*)")

    def on_test_capture(self):
        self._sync_cfg_from_ui()

        # Ensure Pi SSH is reachable (auto-discovery if enabled)
        ok, msg = ensure_pi_ssh(self.cfg, log=self.log)
        if not ok:
            messagebox.showerror(
                "Pi SSH",
                "SSH not available on the Pi.\n\n"
                "Error: {}\n\n"
                "Permanent fix (recommended):\n"
                "1) Give the Pi a STATIC IP on the robot Wi‑Fi (e.g. 192.168.4.20)\n"
                "2) Or keep 'Pi auto-discover' enabled (this app can scan a small IP range)."
                .format(msg or "unknown error")
            )
            return


        # If ensure_pi_ssh updated pi_host, reflect in UI
        self.var_pi_host.set(self.cfg.get("pi_host", ""))

        # Now it's safe to query USB mounts over SSH
        usb = pi_detect_usb_mount(self.cfg, log=self.log)
        date_folder = datetime.now().strftime("%Y%m%d")
        if usb:
            dest = "{}/captures/_test/{}".format(usb, date_folder)
        else:
            dest = "/home/{}/captures/_test/{}".format(self.cfg["pi_user"], date_folder)

        tag = "UI_TEST_{}".format(datetime.now().strftime("%H%M%S"))
        self.log("Test capture tag={} dest={}".format(tag, dest))
        ok = trigger_capture_store_rotate(self.cfg, tag, dest, log=self.log)
        self.log("Test capture: {} tag={}".format("OK ✅" if ok else "FAIL ❌", tag))

    def on_abort(self):
        self.abort_event.set()
        self.log("ABORT requested… (stopping at next safe point)")
        self.btn_abort.configure(state="disabled")

    def on_start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Running", "Scan is already running.")
            return

        self._sync_cfg_from_ui()

        test_name = self.var_testname.get().strip()
        if not test_name:
            messagebox.showerror("Test name", "Please enter a test name.")
            return

        # MUST have active profile
        active = self.cfg.get("active_profile", "").strip()
        if not active:
            messagebox.showerror("Profile", "Please set an Active Profile in Plant Profiles tab.")
            return
        prof = get_profile(self.cfg, active)
        if not prof:
            messagebox.showerror("Profile", "Active profile not found. Re-set Active Profile.")
            return

        mode = self.var_mode.get()
        if mode == "Test":
            total_images = int(self.cfg["test_total_images"])
        elif mode == "Final":
            total_images = int(self.cfg["final_total_images"])
        else:
            total_images = 30

        if total_images < 3:
            messagebox.showerror("Images", "Total images must be >= 3.")
            return

        self.abort_event.clear()
        self.btn_start.configure(state="disabled")
        self.btn_abort.configure(state="normal")
        self.set_progress(0.0, "Starting…")

        def worker():
            start_wall = datetime.now()
            t0 = time.time()
            self.log("\n--- START ---")
            self.log("Mode: {} | Active Profile: {}".format(mode, active))
            self.log("Start time: {}".format(start_wall.strftime("%Y-%m-%d %H:%M:%S")))

            try:
                try:
                    ensure_connected_to(self.cfg["robot_ssid"], self.cfg["robot_pass"])
                    self.log("Robot Wi-Fi OK ✅")
                except Exception as e:
                    self.log("Wi-Fi warning: {} (continuing)".format(e))

                torque(self.cfg, on=True, log=self.log)

                # Ensure Pi SSH once at start (so we fail early with a clear message)
                if self.cfg.get("use_cameras"):
                    ok_pi, msg_pi = ensure_pi_ssh(self.cfg, log=self.log)
                    if not ok_pi:
                        raise RuntimeError(f"Pi SSH not available: {msg_pi}")
                    # reflect discovered IP in UI
                    self.master.after(0, lambda: self.var_pi_host.set(self.cfg.get("pi_host", "")))

                fb0 = get_feedback(self.cfg, log=self.log)
                start_b = math.degrees(fb0.get("b", 0.0))
                self.log("Start base: {:.1f}°".format(start_b))

                # profile inputs
                h_cm = float(prof.get("height_cm", 20.0))
                w_cm = float(prof.get("width_cm", 30.0))
                c_cm = float(prof.get("clearance_cm", 5.0))
                self.log("Test={} | H={:.1f}cm W={:.1f}cm clear={:.1f}cm".format(test_name, h_cm, w_cm, c_cm))

                # storage
                usb = pi_detect_usb_mount(self.cfg, log=self.log) if self.cfg["use_cameras"] else ""
                date_folder = datetime.now().strftime("%Y%m%d")
                if usb:
                    base_storage = "{}/captures/{}/{}".format(usb, test_name, date_folder)
                    self.log("USB: {}".format(usb))
                else:
                    base_storage = "/home/{}/captures/{}/{}".format(self.cfg["pi_user"], test_name, date_folder)
                    self.log("USB: not found (saving to Pi home)")

                if self.cfg["use_cameras"]:
                    ok, msg = pi_mkdir(self.cfg, base_storage)
                    if not ok:
                        raise RuntimeError("Could not create storage folder: {}".format(msg))
                self.log("Storage: {}/".format(base_storage))

                # rings based on ACTIVE profile
                rings, ws, ew, ee, rad, mid, used_auto = compute_rings_from_profile(self.cfg, prof)
                self.log("Rings source: {}".format("AUTO (Height Mapping)" if used_auto else "MANUAL (Profile)"))
                self.log("Rings (S,E): " + " | ".join(["({:.1f},{:.1f})".format(s, e) for (s, e) in rings]))

                per = [total_images // 3] * 3
                for i in range(total_images % 3):
                    per[i] += 1
                self.log("Total images={} | per ring={}".format(total_images, per))

                center = float(self.cfg["base_center_deg"])
                half = float(self.cfg["base_half_span_deg"])
                safe_lim = float(self.cfg["safe_limit_deg"])
                self.log("Sweep: center={:.1f}° half_span=±{:.1f}° | safe_limit=±{:.1f}°".format(center, half, safe_lim))
                self.log("Speed spd={} acc={} delay={:.2f}s".format(self.cfg["spd"], self.cfg["acc"], self.cfg["delay_s"]))
                self.log("Hold last pose: {}".format("YES" if self.cfg["hold_last_pose"] else "NO"))

                done = 0

                for ring_i, ((s_deg, e_deg), ring_count) in enumerate(zip(rings, per), start=1):
                    if self.abort_event.is_set():
                        break

                    self.log("Ring {}/3: S={:.1f} E={:.1f} images={}".format(ring_i, s_deg, e_deg, ring_count))

                    move_joints_deg(
                        self.cfg,
                        shoulder=s_deg, elbow=e_deg, hand=self.cfg["lock_hand_deg"],
                        spd=self.cfg["spd"], acc=self.cfg["acc"],
                        log=None
                    )

                    plan = plan_angles_centered(
                        center_deg=center,
                        half_span_deg=half,
                        count=ring_count,
                        safe_limit_deg=safe_lim,
                        start_b_deg=start_b if ring_i == 1 else None
                    )

                    for b_deg in plan:
                        if self.abort_event.is_set():
                            break

                        safe_b = clip_base_deg(b_deg, safe_lim)
                        step_idx = done + 1

                        move_joints_deg(
                            self.cfg,
                            base=safe_b,
                            shoulder=s_deg, elbow=e_deg, hand=self.cfg["lock_hand_deg"],
                            spd=self.cfg["spd"], acc=self.cfg["acc"],
                            log=None
                        )

                        tag = "{}_R{}_{}_b{}_S{}".format(
                            test_name, ring_i, str(step_idx).zfill(3),
                            int(round(safe_b)), int(round(s_deg))
                        )

                        if self.cfg["use_cameras"]:
                            okcap = trigger_capture_store_rotate(self.cfg, tag, base_storage, log=None)
                            if not okcap:
                                self.log("CAPTURE FAIL at {}/{} (tag={})".format(step_idx, total_images, tag))

                        done += 1

                        elapsed = time.time() - t0
                        rate = elapsed / max(1, done)
                        eta_s = int(rate * (total_images - done))
                        pred_end = datetime.now() + timedelta(seconds=eta_s)

                        self.master.after(
                            0,
                            lambda frac=float(done) / float(total_images), txt="{}/{}".format(done, total_images):
                            self.set_progress(frac, txt)
                        )

                        if done % 10 == 0 or done == total_images:
                            self.log("Progress {}/{} | ETA {} | PredEnd {}".format(
                                done, total_images, timedelta(seconds=eta_s), pred_end.strftime("%H:%M:%S")
                            ))

                        time.sleep(float(self.cfg["delay_s"]))

                end_wall = datetime.now()
                self.log("End time: {}".format(end_wall.strftime("%Y-%m-%d %H:%M:%S")))
                self.log("Elapsed: {}".format(timedelta(seconds=int(time.time() - t0))))

                if not self.cfg["hold_last_pose"]:
                    torque(self.cfg, on=False, log=self.log)
                    self.log("Torque OFF")
                    self.log("DONE ✅")
                else:
                    self.log("DONE ✅ (robot holds last pose)")

            except Exception as e:
                self.log("ERROR: {}".format(e))
                self.master.after(0, lambda: messagebox.showerror("Scan error", str(e)))
            finally:
                self.master.after(0, lambda: self.btn_start.configure(state="normal"))
                self.master.after(0, lambda: self.btn_abort.configure(state="disabled"))
                self.master.after(0, lambda: self.set_progress(0.0, "Idle"))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()

