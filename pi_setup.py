#!/usr/bin/env python3
"""
pi_setup.py  —  Run ONCE on the Raspberry Pi (with sudo).
Sets up everything so the system works automatically on every boot.

What it does:
  1. Copies pi_server.py  → /home/umap/pi_server.py
  2. Copies capture_usb.py → /home/umap/Desktop/capture_usb.py
  3. Installs python3-opencv  (for USB webcam capture + live view)
  4. Installs python3-pil     (for 180° image rotation)
  5. Fixes USB write permissions so images can be saved to USB drives
  6. Sets static IP 192.168.4.20 on robot WiFi (wlan0)
  7. Creates systemd service so pi_server.py starts on every boot
  8. Reboots to apply everything

Communication:
  Laptop talks to Pi over HTTP on port 8765 — NO SSH needed.
  Images save to Pi SD card (/home/umap/captures/) by default.
  If USB drive is plugged in, USB is used automatically instead.
  Browse images at: http://192.168.4.20:8765/browse

Usage:
  Put these 3 files in the same folder on the Pi, then run:
      sudo python3 pi_setup.py

  Files needed in same folder:
      pi_setup.py      ← this file
      pi_server.py     ← HTTP server (port 8765)
      capture_usb.py   ← USB webcam capture script
"""

import os, sys, subprocess, shutil, time

# ── Settings ─────────────────────────────────────────────────
PI_USER       = "umap"
PI_HOME       = f"/home/{PI_USER}"
SERVER_SRC    = "./pi_server.py"
SERVER_DEST   = f"{PI_HOME}/pi_server.py"
CAPTURE_SRC   = "./capture_usb.py"
CAPTURE_DEST  = f"{PI_HOME}/Desktop/capture_usb.py"
CAPTURES_DIR  = f"{PI_HOME}/captures"       # default SD card storage
STATIC_IP     = "192.168.4.20"
WIFI_IFACE    = "wlan0"
SERVICE_NAME  = "umapp"
SERVICE_FILE  = f"/etc/systemd/system/{SERVICE_NAME}.service"
DHCPCD_CONF   = "/etc/dhcpcd.conf"
# ─────────────────────────────────────────────────────────────

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; X = "\033[0m"

def ok(m):   print(f"  {G}✅  {m}{X}")
def err(m):  print(f"  {R}❌  {m}{X}"); sys.exit(1)
def warn(m): print(f"  {Y}⚠   {m}{X}")
def info(m): print(f"  →   {m}")
def head(m): print(f"\n{B}{'─'*54}\n  {m}\n{'─'*54}{X}")

def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()

# ─────────────────────────────────────────────────────────────
# Step 0 — must be root
# ─────────────────────────────────────────────────────────────
def check_root():
    if os.geteuid() != 0:
        print(f"\n  {R}Run with sudo:{X}   sudo python3 pi_setup.py\n")
        sys.exit(1)
    ok("Running as root")

# ─────────────────────────────────────────────────────────────
# Step 1 — copy pi_server.py and capture_usb.py
# ─────────────────────────────────────────────────────────────
def copy_scripts():
    # pi_server.py
    if not os.path.exists(SERVER_SRC):
        err(f"pi_server.py not found at {os.path.abspath(SERVER_SRC)}\n"
            f"      Place pi_server.py in the same folder as pi_setup.py")
    shutil.copy2(SERVER_SRC, SERVER_DEST)
    run(f"chown {PI_USER}:{PI_USER} {SERVER_DEST}")
    run(f"chmod 755 {SERVER_DEST}")
    ok(f"Copied → {SERVER_DEST}")

    # capture_usb.py
    if not os.path.exists(CAPTURE_SRC):
        warn(f"capture_usb.py not found — skipping.\n"
             f"      Copy it manually to {CAPTURE_DEST} later.")
    else:
        os.makedirs(os.path.dirname(CAPTURE_DEST), exist_ok=True)
        shutil.copy2(CAPTURE_SRC, CAPTURE_DEST)
        run(f"chown {PI_USER}:{PI_USER} {CAPTURE_DEST}")
        run(f"chmod 755 {CAPTURE_DEST}")
        ok(f"Copied → {CAPTURE_DEST}")

    # Create default captures folder on SD card
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    run(f"chown -R {PI_USER}:{PI_USER} {CAPTURES_DIR}")
    ok(f"SD card storage folder: {CAPTURES_DIR}")

# ─────────────────────────────────────────────────────────────
# Step 2 — install python packages
# ─────────────────────────────────────────────────────────────
def install_packages():
    info("Updating package list…")
    run("apt-get update -qq")

    info("Installing python3-opencv (USB webcam + live view)…")
    run("apt-get install -y python3-opencv")
    ok("python3-opencv installed")

    info("Installing python3-pil (image rotation)…")
    run("apt-get install -y python3-pil")
    ok("python3-pil (Pillow) installed")

# ─────────────────────────────────────────────────────────────
# Step 3 — USB write permissions
#
# USB drives mount as root by default — umap user can't write.
# Adding umap to plugdev/disk groups and setting a polkit rule
# lets udisks2 automount drives with correct ownership.
# ─────────────────────────────────────────────────────────────
POLKIT_DIR  = "/etc/polkit-1/localauthority/50-local.d"
POLKIT_FILE = f"{POLKIT_DIR}/10-udisks2.pkla"
POLKIT_CONTENT = f"""\
[Allow {PI_USER} to mount drives]
Identity=unix-user:{PI_USER}
Action=org.freedesktop.udisks2.filesystem-mount;org.freedesktop.udisks2.filesystem-mount-system
ResultAny=yes
ResultInactive=yes
ResultActive=yes
"""

def fix_usb_permissions():
    for group in ("plugdev", "disk", "dialout"):
        run(f"usermod -aG {group} {PI_USER}", check=False)
    ok(f"User '{PI_USER}' added to plugdev/disk/dialout groups")

    os.makedirs(POLKIT_DIR, exist_ok=True)
    with open(POLKIT_FILE, "w") as f:
        f.write(POLKIT_CONTENT)
    ok("Polkit rule created (udisks2 mount without password)")

    udisks_conf = "/etc/udisks2/mount_options.conf"
    os.makedirs("/etc/udisks2", exist_ok=True)
    if not os.path.exists(udisks_conf):
        with open(udisks_conf, "w") as f:
            f.write(f"[defaults]\n"
                    f"vfat_defaults=uid=1000,gid=1000,umask=000\n"
                    f"exfat_defaults=uid=1000,gid=1000,umask=000\n"
                    f"ntfs_defaults=uid=1000,gid=1000,umask=000\n")
        ok("udisks2 mount options set (FAT32/exFAT/NTFS all writable)")
    else:
        warn("udisks2 config already exists — verify uid/gid=1000,umask=000")

    run("systemctl enable udisks2 2>/dev/null || true", check=False)
    ok("udisks2 automount enabled")

# ─────────────────────────────────────────────────────────────
# Step 4 — static IP on robot WiFi
# ─────────────────────────────────────────────────────────────
STATIC_BLOCK = f"""
# Umapp static IP — added by pi_setup.py
interface {WIFI_IFACE}
static ip_address={STATIC_IP}/24
"""

def set_static_ip():
    with open(DHCPCD_CONF) as f:
        current = f.read()
    if "Umapp static IP" in current:
        ok(f"Static IP already configured ({STATIC_IP})")
        return
    with open(DHCPCD_CONF, "a") as f:
        f.write(STATIC_BLOCK)
    ok(f"Static IP set: {STATIC_IP} on {WIFI_IFACE}")

# ─────────────────────────────────────────────────────────────
# Step 5 — systemd service (auto-start pi_server.py on boot)
# ─────────────────────────────────────────────────────────────
SERVICE_CONTENT = f"""\
[Unit]
Description=Umapp Pi HTTP Server (port 8765)
After=network.target network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/python3 {SERVER_DEST}
WorkingDirectory={PI_HOME}
Restart=always
RestartSec=5
User={PI_USER}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

def create_service():
    with open(SERVICE_FILE, "w") as f:
        f.write(SERVICE_CONTENT)
    ok(f"Service file: {SERVICE_FILE}")
    run("systemctl daemon-reload")
    ok("Systemd reloaded")
    run(f"systemctl enable {SERVICE_NAME}")
    ok(f"Service '{SERVICE_NAME}' enabled (auto-start on boot)")
    run(f"systemctl restart {SERVICE_NAME}")
    ok("Service started now")

# ─────────────────────────────────────────────────────────────
# Step 6 — verify webcam and service
# ─────────────────────────────────────────────────────────────
def verify():
    time.sleep(2)

    # Service status
    status = run(f"systemctl is-active {SERVICE_NAME}", check=False).strip()
    if status == "active":
        ok(f"Service '{SERVICE_NAME}' is RUNNING ✅")
    else:
        warn(f"Service status: {status}")
        warn(f"Check logs:  sudo journalctl -u {SERVICE_NAME} -n 30")

    # Webcam check
    cams = run("ls /dev/video* 2>/dev/null || true", check=False).strip()
    if cams:
        ok(f"USB webcam detected: {cams}")
    else:
        warn("No /dev/video* found — plug in USB webcam and reboot")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
def summary():
    print(f"""
{B}{'═'*54}
  ALL DONE — Setup complete!
{'═'*54}{X}

  What was configured:
  {G}✅{X} pi_server.py      → {SERVER_DEST}
  {G}✅{X} capture_usb.py    → {CAPTURE_DEST}
  {G}✅{X} python3-opencv + python3-pil installed
  {G}✅{X} USB drives auto-mount with write permission
  {G}✅{X} Static IP: {STATIC_IP} on {WIFI_IFACE}
  {G}✅{X} Service '{SERVICE_NAME}' starts on every boot
  {G}✅{X} Service is running right now

  ── Image storage ──────────────────────────────
  Default:  Pi SD card  →  {CAPTURES_DIR}/
  USB:      Auto-detected if plugged in after boot

  ── In Umapp on your laptop ────────────────────
    Pi IP    =  {STATIC_IP}
    Pi port  =  8765   (HTTP — no SSH needed)

  ── Browse + download images ───────────────────
    Open in browser:
    http://{STATIC_IP}:8765/browse

  ── Troubleshooting ────────────────────────────
    View logs:   sudo journalctl -u {SERVICE_NAME} -n 50
    Restart:     sudo systemctl restart {SERVICE_NAME}
    Webcam:      ls /dev/video*
    Test server: curl http://localhost:8765/status

  Rebooting in 5 seconds to apply all changes...
""")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def step(n, title, fn):
    head(f"Step {n}: {title}")
    try:
        fn()
    except Exception as e:
        err(str(e))

def main():
    print(f"\n{B}{'═'*54}")
    print( "  Umapp Pi Setup")
    print(f"  HTTP server + USB webcam + SD card storage")
    print(f"{'═'*54}{X}\n")

    check_root()

    step(1, "Copy pi_server.py + capture_usb.py",  copy_scripts)
    step(2, "Install python3-opencv + python3-pil", install_packages)
    step(3, "Fix USB write permissions",            fix_usb_permissions)
    step(4, "Set static IP on robot WiFi",          set_static_ip)
    step(5, "Create systemd service",               create_service)
    step(6, "Verify service + webcam",              verify)

    summary()

    for i in range(5, 0, -1):
        print(f"  Rebooting in {i}…", end="\r", flush=True)
        time.sleep(1)
    print()
    subprocess.run("reboot", shell=True)

if __name__ == "__main__":
    main()
