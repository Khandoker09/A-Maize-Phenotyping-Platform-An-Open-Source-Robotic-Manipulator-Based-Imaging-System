# cam_profiles.py
# pip install pyrealsense2 pandas
import pyrealsense2 as rs
import pandas as pd

def enum_name(e):
    try:
        return e.name
    except Exception:
        return str(e)

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    raise SystemExit("No RealSense devices found.")

supported_rows = []

# -------- 1) Enumerate ALL supported profiles (full resolutions list) --------
for dev in devices:
    serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "Unknown"
    dev_name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense"

    for sensor in dev.query_sensors():
        try:
            s_name = sensor.get_info(rs.camera_info.name)
        except Exception:
            s_name = "Sensor"

        # Every advertised stream profile on this sensor
        for p in sensor.get_stream_profiles():
            try:
                v = p.as_video_stream_profile()  # only take video profiles
            except Exception:
                continue  # skip non-video (e.g., motion)

            stream = enum_name(v.stream_type())
            fmt = enum_name(v.format())
            w, h = v.width(), v.height()
            fps = v.fps()

            supported_rows.append({
                "Serial Number": serial,
                "Device": dev_name,
                "Sensor": s_name,
                "Stream": stream,           # COLOR / DEPTH / INFRARED
                "Format": fmt,              # BGR8 / RGB8 / Z16 / Y8 ...
                "Width": w,
                "Height": h,
                "FPS": fps
            })

# Deduplicate identical rows (SDK can return duplicates across sensors)
supported_df = pd.DataFrame(supported_rows).drop_duplicates().sort_values(
    ["Serial Number", "Stream", "Width", "Height", "FPS", "Format"]
)
supported_df.to_csv("camera_supported_profiles.csv", index=False)

print("Saved full list of supported modes to camera_supported_profiles.csv")
print(supported_df.head(20))

# -------- 2) Record the ACTUAL active profiles negotiated right now --------
active_rows = []
pipes = []

try:
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        cfg = rs.config()
        cfg.enable_device(serial)  # let SDK pick valid modes
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        pipes.append((serial, pipe, profile))

    for serial, pipe, profile in pipes:
        # ensure running
        pipe.wait_for_frames()

        def get_active(vstream_type):
            try:
                vs = profile.get_stream(vstream_type).as_video_stream_profile()
                return f"{vs.width()}x{vs.height()} @ {vs.fps()}fps ({enum_name(vs.format())})"
            except Exception:
                return ""

        active_rows.append({
            "Serial Number": serial,
            "Active Color": get_active(rs.stream.color),
            "Active Depth": get_active(rs.stream.depth),
            "Active Infrared": get_active(rs.stream.infrared),
        })
finally:
    for _, pipe, _ in pipes:
        try:
            pipe.stop()
        except Exception:
            pass

active_df = pd.DataFrame(active_rows)
#active_df.to_csv("camera_active_profiles.csv", index=False)
print("\nSaved current active modes to camera_active_profiles.csv")
print(active_df)


# -------- Summary Table of Supported Outputs --------
summary_df = supported_df[["Stream", "Format", "Width", "Height", "FPS"]].drop_duplicates()
print("\nSupported Output Summary:")
print(summary_df.to_string(index=False))

# # check_usb_speed.py
# import pyrealsense2 as rs
# ctx = rs.context()
# devs = ctx.query_devices()
# print(f"Detected {len(devs)} device(s)")
# bad = False
# for d in devs:
#     name  = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "RealSense"
#     s     = d.get_info(rs.camera_info.serial_number) if d.supports(rs.camera_info.serial_number) else "?"
#     usb   = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else ""
#     print(f"- {name} | Serial: {s} | USB: {usb}")
#     if not usb or usb.startswith("2."):
#         bad = True
# if bad:
#     print("\n⚠️ One or more cameras are on USB2. Move them to USB 3.x ports/cables/hubs and try again.")
# else:
#     print("\n✅ All cameras are on USB 3.x. You’re good to stream multiple sensors.")
