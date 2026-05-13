#!/usr/bin/env python3
"""
pi_server.py  —  Run automatically via systemd (set up by pi_setup.py).

This version keeps your original server settings and behavior, including:
  • FALLBACK_DIR  = /home/umap/captures
  • PORT          = 8765
  • HOST          = 0.0.0.0
  • WEBCAM_INDEX  = 0
  • JPEG_QUALITY  = 95
  • PREVIEW_QUALITY = 75
  • shared _cam_lock for /capture and /preview
  • direct cv2 capture in /capture
  • USB detection and fallback behavior
  • /status /usb /usb_test /capture /preview /move /mkdir /file /delete

Added/fixed:
  • /browse     — browser UI for images and folders
  • /storage    — compatible with your desktop Umapp UI
  • safer /file — supports both absolute paths and browse-relative paths
"""

import os, json, shutil, time, re, threading, html, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote
from datetime import datetime
from pathlib import Path

# ── Settings ─────────────────────────────────────────────────
FALLBACK_DIR  = "/home/umap/captures"
PORT          = 8765
HOST          = "0.0.0.0"
WEBCAM_INDEX  = 0          # /dev/video0 — change if needed
JPEG_QUALITY  = 95         # quality for saved captures
PREVIEW_QUALITY = 75       # quality for live-view frames (smaller = faster)
# ─────────────────────────────────────────────────────────────

# One lock so /capture and /preview never race on the same webcam
_cam_lock = threading.Lock()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ── Webcam helper ─────────────────────────────────────────────
def _grab_frame(quality=JPEG_QUALITY):
    """
    Open webcam, grab one frame, release immediately, return JPEG bytes.
    Raises RuntimeError on any failure.
    Protected by _cam_lock — caller must hold it.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("cv2 not installed. Run: sudo apt install -y python3-opencv")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam /dev/video{WEBCAM_INDEX}")
    try:
        time.sleep(0.15)           # let auto-exposure settle
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Cannot read frame from webcam")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf)
    finally:
        cap.release()


# ── USB detection ─────────────────────────────────────────────
def _get_mounts():
    mounts = []
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mounts.append(parts[1])
    except Exception:
        pass
    return mounts

def _is_writable(path):
    test = os.path.join(path, ".umapp_write_test")
    try:
        os.makedirs(path, exist_ok=True)
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True
    except Exception:
        return False

def _detect_usb():
    mounts = set(_get_mounts())
    candidates = []
    for base in ("/media", "/mnt"):
        if not os.path.isdir(base):
            continue
        for sub in os.listdir(base):
            sub_path = os.path.join(base, sub)
            if sub_path in mounts:
                candidates.append(sub_path)
            elif os.path.isdir(sub_path):
                for drive in os.listdir(sub_path):
                    drive_path = os.path.join(sub_path, drive)
                    if drive_path in mounts:
                        candidates.append(drive_path)
    for path in candidates:
        if _is_writable(path):
            print(f"[USB] Found writable USB: {path}", flush=True)
            return path
        else:
            print(f"[USB] Mount found but NOT writable: {path}", flush=True)
    return ""

def _active_storage_root():
    """
    Returns the active root where the desktop app should create folders.
    USB is preferred when writable, otherwise FALLBACK_DIR.
    """
    usb = _detect_usb()
    if usb:
        return usb, "USB"
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    return FALLBACK_DIR, "SD"


# ── Safe mkdir with retries ───────────────────────────────────
def _safe_mkdir(path, retries=3, delay=1.0):
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.isdir(path) and _is_writable(path):
                return True, ""
            last_err = "Directory created but not writable"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            print(f"[mkdir] attempt {attempt} failed ({last_err}), retrying…", flush=True)
            time.sleep(delay)
    return False, last_err


# ── Safe file copy ────────────────────────────────────────────
def _safe_copy(src, dst_dir, retries=3):
    if not os.path.exists(src):
        return None, f"Source missing: {src}"
    if os.path.getsize(src) == 0:
        return None, f"Source is empty: {src}"
    dest = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dest):
        root, ext = os.path.splitext(dest)
        dest = f"{root}_{int(time.time())}{ext}"
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            shutil.copy2(src, dest)
            if not os.path.exists(dest):
                raise RuntimeError("Dest missing after copy")
            if os.path.getsize(dest) == 0:
                raise RuntimeError("Dest empty after copy")
            os.remove(src)
            return dest, ""
        except Exception as e:
            last_err = str(e)
            print(f"[copy] attempt {attempt} failed: {last_err}", flush=True)
            time.sleep(0.5)
    return None, last_err


# ── Browser helpers ───────────────────────────────────────────
def _safe_inside(root, requested):
    """
    Convert a browser path into a safe path inside root.
    Absolute paths are allowed only if they stay inside root.
    """
    root = Path(root).resolve()
    requested = unquote(str(requested or "")).strip()

    if requested.startswith("/"):
        target = Path(requested).resolve()
    else:
        target = (root / requested.strip("/")).resolve()

    if target == root:
        return target
    if root in target.parents:
        return target
    return root

def _browse_root():
    """
    Browse the active storage root.
    If it contains a captures folder, browse that first because your desktop UI
    creates storage/captures/test/date.
    """
    root, _stype = _active_storage_root()
    root = Path(root)

    captures = root / "captures"
    if captures.exists() and captures.is_dir():
        return str(captures)

    return str(root)

def _html_page(title, body):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
    :root {{
        --bg:#0f172a;
        --surface:#1e293b;
        --surface2:#243044;
        --border:#334155;
        --text:#e2e8f0;
        --muted:#94a3b8;
        --accent:#7c3aed;
        --accent2:#a78bfa;
        --danger:#ef4444;
    }}
    * {{ box-sizing:border-box; }}
    body {{
        margin:0;
        padding:24px;
        background:var(--bg);
        color:var(--text);
        font-family:Arial, Helvetica, sans-serif;
    }}
    h1 {{
        margin:0 0 8px 0;
        color:var(--accent2);
        font-size:28px;
    }}
    .top {{
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:18px;
        margin-bottom:22px;
    }}
    .meta {{
        color:var(--muted);
        margin-top:4px;
        word-break:break-word;
    }}
    .actions {{
        display:flex;
        gap:10px;
        flex-wrap:wrap;
    }}
    .btn {{
        display:inline-block;
        color:var(--text);
        text-decoration:none;
        background:var(--surface);
        border:1px solid var(--border);
        border-radius:10px;
        padding:10px 14px;
        font-weight:bold;
    }}
    .btn:hover {{ border-color:var(--accent); }}
    .grid {{
        display:grid;
        grid-template-columns:repeat(auto-fill, minmax(185px, 1fr));
        gap:16px;
    }}
    .card {{
        display:block;
        min-height:110px;
        padding:12px;
        background:var(--surface);
        border:1px solid var(--border);
        border-radius:14px;
        color:var(--text);
        text-decoration:none;
        overflow:hidden;
        transition:0.12s ease;
    }}
    .card:hover {{
        transform:translateY(-2px);
        border-color:var(--accent);
        background:var(--surface2);
    }}
    .card img {{
        width:100%;
        height:135px;
        object-fit:cover;
        display:block;
        border-radius:10px;
        background:#020617;
    }}
    .icon {{
        font-size:42px;
        margin-bottom:14px;
    }}
    .name {{
        margin-top:10px;
        color:#cbd5e1;
        font-size:13px;
        line-height:1.35;
        word-break:break-word;
    }}
    .folder .name {{
        color:var(--text);
        font-size:16px;
        font-weight:bold;
    }}
    .empty {{
        grid-column:1/-1;
        padding:30px;
        text-align:center;
        color:var(--muted);
        background:var(--surface);
        border:1px solid var(--border);
        border-radius:14px;
    }}
    @media (max-width:700px) {{
        body {{ padding:14px; }}
        .top {{ flex-direction:column; }}
        .grid {{ grid-template-columns:repeat(auto-fill, minmax(145px, 1fr)); }}
        .card img {{ height:110px; }}
    }}
</style>
</head>
<body>
{body}
</body>
</html>"""

def _make_browse_html(rel_path=""):
    root = Path(_browse_root()).resolve()
    current = _safe_inside(root, rel_path)

    if not current.exists():
        current = root
    if current.is_file():
        current = current.parent

    try:
        rel_current = "" if current == root else str(current.relative_to(root))
    except Exception:
        rel_current = ""

    rows = []

    if rel_current:
        parent = str(Path(rel_current).parent)
        if parent == ".":
            parent = ""
        rows.append(f"""
        <a class="card folder" href="/browse?path={quote(parent)}">
            <div class="icon">⬅️</div>
            <div class="name">Back</div>
        </a>
        """)

    try:
        items = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception:
        items = []

    for item in items:
        name = html.escape(item.name)
        item_rel = str(item.relative_to(root))
        item_q = quote(item_rel)

        if item.is_dir():
            rows.append(f"""
            <div class="card folder">
                <a class="open-link" href="/browse?path={item_q}">
                    <div class="icon">📁</div>
                    <div class="name">{name}</div>
                </a>
                <a class="delete-btn" href="/delete?path={item_q}"
                   onclick="return confirm('Delete this folder and everything inside it permanently?\\n\\n{name}');">
                   🗑 Delete folder
                </a>
            </div>
            """)
        elif item.suffix.lower() in IMAGE_EXTS:
            rows.append(f"""
            <div class="card image">
                <a class="open-link" href="/file?path={item_q}" target="_blank">
                    <img src="/file?path={item_q}" alt="{name}">
                    <div class="name">{name}</div>
                </a>
                <a class="delete-btn" href="/delete?path={item_q}"
                   onclick="return confirm('Delete this image permanently?\\n\\n{name}');">
                   🗑 Delete
                </a>
            </div>
            """)
        else:
            rows.append(f"""
            <div class="card file">
                <a class="open-link" href="/file?path={item_q}" target="_blank">
                    <div class="icon">📄</div>
                    <div class="name">{name}</div>
                </a>
                <a class="delete-btn" href="/delete?path={item_q}"
                   onclick="return confirm('Delete this file permanently?\\n\\n{name}');">
                   🗑 Delete
                </a>
            </div>
            """)

    if not rows:
        rows.append('<div class="empty">No files found in this folder.</div>')

    rel_display = "/" + html.escape(rel_current) if rel_current else "/"

    body = f"""
    <div class="top">
        <div>
            <h1>🌿 Umapp Image Browser</h1>
            <div class="meta">Browse root: {html.escape(str(root))}</div>
            <div class="meta">Current folder: {rel_display}</div>
        </div>
        <div class="actions">
            <a class="btn" href="/browse">Root</a>
            <a class="btn" href="/status">Status</a>
            <a class="btn" href="/storage">Storage</a>
        </div>
    </div>
    <div class="grid">
        {''.join(rows)}
    </div>
    """
    return _html_page("Umapp Image Browser", body)


# ── HTTP handler ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] " + fmt % args, flush=True)

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, code=200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data, content_type="application/octet-stream", code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _jpeg(self, data):
        self._bytes(data, "image/jpeg", 200)

    def _file(self, fp):
        """
        Supports:
          /file?path=/absolute/path/to/image.jpg
          /file?path=relative/path/from/browse/root.jpg
        """
        try:
            if not fp:
                self._json({"ok": False, "error": "no path"}, 400)
                return

            fp = unquote(fp)

            if fp.startswith("/"):
                path = Path(fp)
            else:
                path = _safe_inside(_browse_root(), fp)

            if not path.exists() or not path.is_file():
                self._json({"ok": False, "error": f"file not found: {path}"}, 404)
                return

            with open(path, "rb") as f:
                data = f.read()

            ctype, _ = mimetypes.guess_type(str(path))
            if not ctype:
                ctype = "application/octet-stream"

            self._bytes(data, ctype, 200)
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)
        path   = parsed.path

        def q(key, default=""):
            return qs.get(key, [default])[0]

        try:
            # ── / ────────────────────────────────────────────
            if path == "/":
                body = """
                <div class="top">
                    <div>
                        <h1>🌿 Umapp Pi Server</h1>
                        <div class="meta">Server is running.</div>
                    </div>
                    <div class="actions">
                        <a class="btn" href="/browse">Browse Images</a>
                        <a class="btn" href="/status">Status</a>
                        <a class="btn" href="/storage">Storage</a>
                    </div>
                </div>
                """
                self._html(_html_page("Umapp Pi Server", body))

            # ── /status ──────────────────────────────────────
            elif path == "/status":
                self._json({"ok": True, "webcam": WEBCAM_INDEX})

            # ── /storage ─────────────────────────────────────
            # This is needed by your desktop Umapp UI.
            elif path == "/storage":
                root, stype = _active_storage_root()
                self._json({"ok": True, "path": root, "type": stype})

            # ── /usb ─────────────────────────────────────────
            elif path == "/usb":
                self._json({"path": _detect_usb()})

            # ── /usb_test ────────────────────────────────────
            elif path == "/usb_test":
                usb = _detect_usb()
                if not usb:
                    self._json({"ok": False, "error": "No writable USB found",
                                "fallback": FALLBACK_DIR})
                else:
                    self._json({"ok": True, "path": usb})

            # ── /capture ─────────────────────────────────────
            # Grabs a frame directly via cv2, saves to /tmp, returns path.
            # Uses _cam_lock so it never conflicts with /preview.
            elif path == "/capture":
                tag = q("tag", f"cap_{int(time.time())}")
                print(f"[capture] tag={tag}", flush=True)

                ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{tag}_{ts}.jpg"
                filepath = f"/tmp/{filename}"

                acquired = _cam_lock.acquire(timeout=10)
                if not acquired:
                    self._json({"ok": False,
                                "error": "Webcam busy (preview running) — try again"}); return
                try:
                    jpeg_bytes = _grab_frame(quality=JPEG_QUALITY)
                finally:
                    _cam_lock.release()

                with open(filepath, "wb") as f:
                    f.write(jpeg_bytes)

                print(f"[capture] OK  file={filepath}", flush=True)
                self._json({"ok": True, "file": filepath})

            # ── /preview ─────────────────────────────────────
            # Returns one webcam frame as JPEG — used by Live View tab.
            # Nothing saved to disk. Uses _cam_lock to avoid conflict with /capture.
            elif path == "/preview":
                acquired = _cam_lock.acquire(timeout=3)
                if not acquired:
                    self._json({"ok": False,
                                "error": "Webcam busy (capture running)"}); return
                try:
                    jpeg_bytes = _grab_frame(quality=PREVIEW_QUALITY)
                finally:
                    _cam_lock.release()
                self._jpeg(jpeg_bytes)

            # ── /move ────────────────────────────────────────
            # Move a captured file to the final destination + optional 180° rotation.
            elif path == "/move":
                src     = q("src")
                dst_dir = q("dst")
                rotate  = q("rotate", "0") == "1"

                if not src or not dst_dir:
                    self._json({"ok": False, "error": "src and dst required"}); return

                ok_dir, dir_err = _safe_mkdir(dst_dir)
                if not ok_dir:
                    print(f"[move] dst not writable, falling back to {FALLBACK_DIR}", flush=True)
                    dst_dir = FALLBACK_DIR
                    ok_dir, dir_err = _safe_mkdir(dst_dir)
                    if not ok_dir:
                        self._json({"ok": False,
                                    "error": f"Cannot create dir: {dir_err}"}); return

                dest, copy_err = _safe_copy(src, dst_dir)
                if dest is None:
                    self._json({"ok": False, "error": copy_err}); return

                if rotate:
                    try:
                        from PIL import Image as PILImage
                        img = PILImage.open(dest)
                        img.rotate(180, expand=True).save(dest, quality=95)
                        print(f"[move] rotated {dest}", flush=True)
                    except ImportError:
                        print("[move] Pillow not installed — rotation skipped", flush=True)
                    except Exception as e:
                        print(f"[move] rotation warning: {e}", flush=True)

                if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                    self._json({"ok": False,
                                "error": f"File missing/empty after save: {dest}"}); return

                print(f"[move] saved → {dest}", flush=True)
                self._json({"ok": True, "dest": dest})

            # ── /mkdir ───────────────────────────────────────
            elif path == "/mkdir":
                fp = q("path")
                ok, err = _safe_mkdir(fp)
                self._json({"ok": ok, "error": err})

            # ── /browse ──────────────────────────────────────
            elif path == "/browse":
                rel = q("path", "")
                self._html(_make_browse_html(rel))

            # ── /file ────────────────────────────────────────
            elif path == "/file":
                fp = q("path")
                self._file(fp)

            # ── /delete ──────────────────────────────────────
            # Deletes a file or a folder from the browse root.
            # Browser links use relative paths; absolute paths are still accepted.
            elif path == "/delete":
                fp = q("path")
                try:
                    if not fp:
                        self._json({"ok": False, "error": "no path"}, 400)
                        return

                    fp = unquote(fp)

                    if fp.startswith("/"):
                        target = Path(fp)
                    else:
                        target = _safe_inside(_browse_root(), fp)

                    # Safety: never delete the browse root itself.
                    browse_root = Path(_browse_root()).resolve()
                    target_resolved = target.resolve()

                    if target_resolved == browse_root:
                        self._json({"ok": False, "error": "Refusing to delete browse root"}, 400)
                        return

                    if browse_root not in target_resolved.parents:
                        self._json({"ok": False, "error": "Refusing to delete outside browse root"}, 400)
                        return

                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        try:
                            os.remove(target)
                        except FileNotFoundError:
                            pass

                    # Browser-friendly redirect back to /browse.
                    self.send_response(303)
                    self.send_header("Location", "/browse")
                    self.end_headers()

                except Exception as e:
                    self._json({"ok": False, "error": str(e)}); return

            else:
                self._json({"ok": False, "error": f"Unknown endpoint: {path}"}, 404)

        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            self._json({"ok": False, "error": str(e)}, 500)


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'='*52}", flush=True)
    print(f"  Umapp Pi Server  —  port {PORT}", flush=True)
    print(f"  Webcam:    /dev/video{WEBCAM_INDEX}", flush=True)
    print(f"  Fallback:  {FALLBACK_DIR}", flush=True)
    print(f"  Endpoints: /status /storage /usb /capture /preview /move /mkdir /browse /file /delete /usb_test", flush=True)
    print(f"{'='*52}", flush=True)

    os.makedirs(FALLBACK_DIR, exist_ok=True)
    server = HTTPServer((HOST, PORT), Handler)
    print(f"  Listening on 0.0.0.0:{PORT} …\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.", flush=True)
