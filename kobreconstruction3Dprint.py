# ============================================================
# Maize Cob Cleaner + Hole Filler + Printable STL/PLY Export
#
# Outputs:
#   1. clean point cloud PLY
#   2. clean colored mesh PLY
#   3. print-ready scaled STL
#
# Fixes:
#   - removes blue/noisy points
#   - keeps maize cob detail
#   - fills holes
#   - avoids Trimesh index-out-of-bounds error
#   - scales STL correctly for Cura
# ============================================================

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox


# ============================================================
# AUTO-INSTALL PACKAGES
# ============================================================

def install_if_missing(package, import_name=None):
    if import_name is None:
        import_name = package

    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_if_missing("open3d")
install_if_missing("numpy")
install_if_missing("trimesh")
install_if_missing("pymeshfix")

import open3d as o3d
import numpy as np
import trimesh
import pymeshfix


# ============================================================
# PRINT SIZE SETTINGS
# ============================================================

# Final cob length in millimeters.
# Cura reads STL units as millimeters.
TARGET_LENGTH_MM = 160

# Keep True so the STL is scaled correctly for Cura.
AUTO_SCALE_TO_MM = True


# ============================================================
# POINT CLOUD CLEANING SETTINGS
# ============================================================

AUTO_VOXEL = True

# Higher = more detail but slower.
# Good values:
#   450 = faster
#   600 = balanced/high detail
#   800 = more detail, slower
VOXEL_DIVISOR = 600

# Color filtering removes blue photogrammetry noise.
USE_COLOR_FILTER = True
BLUE_REJECTION_STRENGTH = 1.5

# Statistical noise removal.
# Larger STD_RATIO removes less.
NB_NEIGHBORS = 35
STD_RATIO = 3.0

# Radius filtering removes isolated floating points.
USE_RADIUS_FILTER = True
RADIUS_MULTIPLIER = 4.5
MIN_POINTS_IN_RADIUS = 4

# Export cleaned point cloud.
EXPORT_CLEAN_POINT_CLOUD = True


# ============================================================
# MESH SETTINGS
# ============================================================

# Poisson depth:
#   9 = faster
#   10 = better detail and hole filling
#   11 = very detailed but slow
POISSON_DEPTH = 10

# 0.0 keeps the full Poisson surface.
# This helps avoid holes caused by cutting weak-density parts.
DENSITY_QUANTILE = 0.0

# Light smoothing only.
# Too much smoothing destroys kernel detail.
SMOOTH_ITERATIONS = 2

# Larger crop avoids cutting off cob ends.
BBOX_SCALE = 1.15

# STL simplification.
# Use None to keep full mesh.
TARGET_TRIANGLES = 500000


# ============================================================
# FILE DIALOGS
# ============================================================

def choose_ply_file():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select original maize cob PLY",
        filetypes=[
            ("PLY files", "*.ply"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return path


def choose_output_folder():
    root = tk.Tk()
    root.withdraw()

    folder = filedialog.askdirectory(
        title="Select output folder"
    )

    root.destroy()
    return folder


# ============================================================
# POINT CLOUD HELPERS
# ============================================================

def remove_invalid_points(pcd):
    pts = np.asarray(pcd.points)
    valid = np.isfinite(pts).all(axis=1)

    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        valid = valid & np.isfinite(cols).all(axis=1)

    return pcd.select_by_index(np.where(valid)[0])


def get_adaptive_voxel_size(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    longest = float(np.max(extent))

    if longest <= 0:
        raise RuntimeError("Invalid point cloud bounding box.")

    voxel = longest / VOXEL_DIVISOR

    # Safety limits
    voxel = max(voxel, longest / 2000)
    voxel = min(voxel, longest / 250)

    return voxel


def color_filter_maize(pcd):
    if not pcd.has_colors():
        print("No colors found. Skipping color filter.")
        return pcd

    cols = np.asarray(pcd.colors)

    r = cols[:, 0]
    g = cols[:, 1]
    b = cols[:, 2]

    # Reject strong blue/purple noise.
    not_blue = b < BLUE_REJECTION_STRENGTH * np.maximum(r, g)

    # Keep yellow/orange/brown/white cob colors.
    maize_yellow = (r > 0.30) & (g > 0.18) & (r > b * 0.85)
    maize_brown = (r > 0.15) & (g > 0.08) & (b < 0.50)
    maize_white = (r > 0.50) & (g > 0.45) & (b > 0.35)

    keep = not_blue & (maize_yellow | maize_brown | maize_white)

    kept = int(np.sum(keep))
    total = len(cols)

    print(f"Color filter kept {kept} / {total} points")

    # Safety fallback
    if kept < total * 0.35:
        print("Color filter too aggressive. Keeping original point cloud.")
        return pcd

    return pcd.select_by_index(np.where(keep)[0])


def keep_largest_point_cluster(pcd, voxel_size):
    print("Keeping main cob cluster...")

    labels = np.array(
        pcd.cluster_dbscan(
            eps=voxel_size * 10,
            min_points=15,
            print_progress=True
        )
    )

    if labels.size == 0 or labels.max() < 0:
        print("No cluster found. Skipping cluster filter.")
        return pcd

    valid = labels >= 0
    counts = np.bincount(labels[valid])
    largest = counts.argmax()

    keep = np.where(labels == largest)[0]

    if len(keep) < len(pcd.points) * 0.35:
        print("Cluster filter too aggressive. Keeping previous cloud.")
        return pcd

    print(f"Cluster kept {len(keep)} / {len(pcd.points)} points")
    return pcd.select_by_index(keep)


# ============================================================
# MESH HELPERS
# ============================================================

def clean_open3d_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    try:
        mesh.remove_non_manifold_edges()
    except Exception as e:
        print(f"remove_non_manifold_edges warning: {e}")

    mesh.remove_unreferenced_vertices()

    return mesh


def remove_invalid_open3d_faces(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Mesh has no vertices or faces.")

    max_index = len(vertices) - 1

    valid = np.all((faces >= 0) & (faces <= max_index), axis=1)

    bad_count = int(np.sum(~valid))
    if bad_count > 0:
        print(f"Removing {bad_count} invalid Open3D faces.")

    faces = faces[valid]

    non_degenerate = (
        (faces[:, 0] != faces[:, 1]) &
        (faces[:, 1] != faces[:, 2]) &
        (faces[:, 0] != faces[:, 2])
    )

    degenerate_count = int(np.sum(~non_degenerate))
    if degenerate_count > 0:
        print(f"Removing {degenerate_count} degenerate Open3D faces.")

    faces = faces[non_degenerate]

    clean = o3d.geometry.TriangleMesh()
    clean.vertices = o3d.utility.Vector3dVector(vertices)
    clean.triangles = o3d.utility.Vector3iVector(faces)
    clean.compute_vertex_normals()
    clean.compute_triangle_normals()
    clean.remove_unreferenced_vertices()

    return clean


def keep_largest_mesh_component(mesh):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        return mesh

    largest = cluster_n_triangles.argmax()
    remove = triangle_clusters != largest

    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()

    return mesh


def transfer_colors_from_cloud_to_mesh(mesh, pcd):
    if not pcd.has_colors():
        return mesh

    print("Transferring colors to mesh...")

    tree = o3d.geometry.KDTreeFlann(pcd)
    cloud_colors = np.asarray(pcd.colors)
    vertices = np.asarray(mesh.vertices)

    mesh_colors = np.zeros((len(vertices), 3))

    for i, v in enumerate(vertices):
        _, idx, _ = tree.search_knn_vector_3d(v, 7)

        if len(idx) > 0:
            mesh_colors[i] = np.mean(cloud_colors[idx], axis=0)
        else:
            mesh_colors[i] = [1.0, 0.65, 0.05]

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    return mesh


def open3d_to_trimesh_safe(mesh):
    print("Converting Open3D mesh to Trimesh safely...")

    mesh = clean_open3d_mesh(mesh)
    mesh = remove_invalid_open3d_faces(mesh)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    print(f"Open3D vertices before Trimesh: {len(vertices)}")
    print(f"Open3D faces before Trimesh: {len(faces)}")

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Mesh has no vertices or faces before Trimesh conversion.")

    max_valid_index = len(vertices) - 1
    valid_faces_mask = np.all((faces >= 0) & (faces <= max_valid_index), axis=1)

    bad_faces = int(np.sum(~valid_faces_mask))
    if bad_faces > 0:
        print(f"Removed {bad_faces} invalid faces before Trimesh conversion.")

    faces = faces[valid_faces_mask]

    non_degenerate = (
        (faces[:, 0] != faces[:, 1]) &
        (faces[:, 1] != faces[:, 2]) &
        (faces[:, 0] != faces[:, 2])
    )

    degenerate_count = int(np.sum(~non_degenerate))
    if degenerate_count > 0:
        print(f"Removed {degenerate_count} degenerate faces before Trimesh conversion.")

    faces = faces[non_degenerate]

    if len(faces) == 0:
        raise RuntimeError("All faces were invalid or degenerate.")

    tm = trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces.copy(),
        process=False
    )

    try:
        tm.remove_duplicate_faces()
    except Exception:
        pass

    try:
        tm.remove_degenerate_faces()
    except Exception:
        pass

    try:
        tm.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        tm.process(validate=True)
    except Exception as e:
        print(f"Trimesh process warning: {e}")

    print(f"Trimesh vertices after conversion: {len(tm.vertices)}")
    print(f"Trimesh faces after conversion: {len(tm.faces)}")

    return tm


def extra_fill_holes(tm):
    print("Filling holes with Trimesh safely...")

    try:
        tm.remove_duplicate_faces()
    except Exception:
        pass

    try:
        tm.remove_degenerate_faces()
    except Exception:
        pass

    try:
        tm.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        trimesh.repair.fix_normals(tm)
    except Exception as e:
        print(f"fix_normals warning: {e}")

    try:
        trimesh.repair.fix_inversion(tm)
    except Exception as e:
        print(f"fix_inversion warning: {e}")

    try:
        trimesh.repair.fill_holes(tm)
    except Exception as e:
        print(f"fill_holes warning: {e}")

    try:
        tm.process(validate=True)
    except Exception as e:
        print(f"process warning: {e}")

    try:
        trimesh.repair.fill_holes(tm)
    except Exception as e:
        print(f"second fill_holes warning: {e}")

    try:
        trimesh.repair.fix_normals(tm)
    except Exception:
        pass

    return tm


def repair_with_meshfix(tm):
    print("Running strong hole repair with PyMeshFix...")

    vertices = np.asarray(tm.vertices)
    faces = np.asarray(tm.faces)

    meshfix = pymeshfix.MeshFix(vertices, faces)

    meshfix.repair(
        verbose=True,
        joincomp=True,
        remove_smallest_components=True
    )

    repaired = trimesh.Trimesh(
        vertices=meshfix.v,
        faces=meshfix.f,
        process=True
    )

    return repaired


def scale_trimesh_to_target_length_mm(tm, target_length_mm):
    ext = tm.extents
    longest = float(np.max(ext))

    if longest <= 0:
        raise RuntimeError("Cannot scale mesh because bounding box is invalid.")

    scale = target_length_mm / longest

    print(f"Current longest side before scaling: {longest}")
    print(f"Scale factor to millimeters: {scale}")
    print(f"Target printed length: {target_length_mm} mm")

    tm.apply_scale(scale)

    return tm


def save_colored_mesh_ply(mesh, output_path):
    o3d.io.write_triangle_mesh(
        output_path,
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
        write_triangle_uvs=False
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_maize_cob(input_ply, output_folder):
    base = os.path.splitext(os.path.basename(input_ply))[0]

    output_cloud = os.path.join(output_folder, base + "_clean_pointcloud.ply")
    output_mesh_ply = os.path.join(output_folder, base + "_clean_colored_mesh.ply")
    output_stl = os.path.join(
        output_folder,
        base + f"_PRINT_READY_{int(TARGET_LENGTH_MM)}mm.stl"
    )

    print("\nLoading original PLY...")
    pcd = o3d.io.read_point_cloud(input_ply)

    if pcd.is_empty():
        raise RuntimeError("Input PLY is empty or could not be read.")

    print(f"Original points: {len(pcd.points)}")
    print(f"Has colors: {pcd.has_colors()}")
    print(f"Has normals: {pcd.has_normals()}")

    pcd = remove_invalid_points(pcd)

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())

    print(f"Original bounding box extent: {extent}")

    if AUTO_VOXEL:
        voxel_size = get_adaptive_voxel_size(pcd)
    else:
        voxel_size = float(np.max(extent)) / VOXEL_DIVISOR

    print(f"Using voxel size: {voxel_size}")

    if USE_COLOR_FILTER:
        print("\nRemoving blue/color noise...")
        pcd = color_filter_maize(pcd)

    print("\nGentle downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"After downsampling: {len(pcd.points)}")

    print("\nStatistical outlier removal...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )
    print(f"After statistical filter: {len(pcd.points)}")

    if USE_RADIUS_FILTER:
        print("\nRadius outlier removal...")
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=MIN_POINTS_IN_RADIUS,
            radius=voxel_size * RADIUS_MULTIPLIER
        )
        print(f"After radius filter: {len(pcd.points)}")

    pcd = keep_largest_point_cluster(pcd, voxel_size)

    if len(pcd.points) < 2000:
        raise RuntimeError(
            "Too few points remain after cleaning.\n\n"
            "Try these settings:\n"
            "USE_RADIUS_FILTER = False\n"
            "STD_RATIO = 4.0\n"
            "VOXEL_DIVISOR = 800\n"
        )

    if EXPORT_CLEAN_POINT_CLOUD:
        print("\nSaving clean point cloud PLY...")
        o3d.io.write_point_cloud(
            output_cloud,
            pcd,
            write_ascii=False,
            compressed=False
        )

    print("\nEstimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 7,
            max_nn=100
        )
    )

    print("Orienting normals...")
    try:
        pcd.orient_normals_consistent_tangent_plane(150)
    except Exception as e:
        print(f"Normal orientation warning: {e}")

    print("\nPoisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=POISSON_DEPTH,
        width=0,
        scale=1.15,
        linear_fit=True
    )

    print(f"Raw mesh vertices: {len(mesh.vertices)}")
    print(f"Raw mesh triangles: {len(mesh.triangles)}")

    print("\nRemoving only very weak Poisson shell...")

    if DENSITY_QUANTILE > 0:
        densities = np.asarray(densities)
        threshold = np.quantile(densities, DENSITY_QUANTILE)
        remove = densities < threshold
        mesh.remove_vertices_by_mask(remove)
    else:
        print("DENSITY_QUANTILE = 0.0, keeping full Poisson surface.")

    print("\nCropping gently to cob bounding box...")
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(BBOX_SCALE, bbox.get_center())
    mesh = mesh.crop(bbox)

    print("\nCleaning mesh...")
    mesh = clean_open3d_mesh(mesh)
    mesh = remove_invalid_open3d_faces(mesh)

    print("Keeping largest mesh component...")
    mesh = keep_largest_mesh_component(mesh)

    print("Cleaning mesh again after component selection...")
    mesh = clean_open3d_mesh(mesh)
    mesh = remove_invalid_open3d_faces(mesh)

    if SMOOTH_ITERATIONS > 0:
        print(f"Light smoothing: {SMOOTH_ITERATIONS}")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=SMOOTH_ITERATIONS)
        mesh = clean_open3d_mesh(mesh)
        mesh = remove_invalid_open3d_faces(mesh)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print("\nAdding colors to PLY mesh...")
    colored_mesh = transfer_colors_from_cloud_to_mesh(mesh, pcd)

    print("Saving colored mesh PLY...")
    save_colored_mesh_ply(colored_mesh, output_mesh_ply)

    print("\nPreparing printable STL...")
    tm = open3d_to_trimesh_safe(mesh)

    print(f"Before repair watertight: {tm.is_watertight}")

    tm = extra_fill_holes(tm)

    if not tm.is_watertight:
        print("Mesh is still not watertight. Using PyMeshFix...")
        try:
            tm = repair_with_meshfix(tm)
            tm = extra_fill_holes(tm)
        except Exception as e:
            print(f"PyMeshFix failed. Continuing with Trimesh result.")
            print(f"PyMeshFix error: {e}")

    print("Keeping largest Trimesh body...")
    try:
        parts = tm.split(only_watertight=False)
        if len(parts) > 1:
            tm = max(
                parts,
                key=lambda x: x.volume if x.is_watertight else x.area
            )
    except Exception as e:
        print(f"Split warning: {e}")

    if TARGET_TRIANGLES is not None and len(tm.faces) > TARGET_TRIANGLES:
        print(f"Simplifying STL to about {TARGET_TRIANGLES} triangles...")
        try:
            tm = tm.simplify_quadric_decimation(TARGET_TRIANGLES)
        except Exception as e:
            print(f"Simplification skipped: {e}")

    try:
        tm.remove_duplicate_faces()
    except Exception:
        pass

    try:
        tm.remove_degenerate_faces()
    except Exception:
        pass

    try:
        tm.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        tm.process(validate=True)
    except Exception as e:
        print(f"Final Trimesh process warning: {e}")

    tm = extra_fill_holes(tm)

    if AUTO_SCALE_TO_MM:
        tm = scale_trimesh_to_target_length_mm(tm, TARGET_LENGTH_MM)

    print("\nFinal STL report:")
    print(f"Vertices: {len(tm.vertices)}")
    print(f"Faces: {len(tm.faces)}")
    print(f"Watertight: {tm.is_watertight}")
    print(f"Final size in mm: {tm.extents}")

    print("\nSaving print-ready STL...")
    tm.export(output_stl)

    print("\nDONE")
    print(f"Clean point cloud PLY:\n{output_cloud}")
    print(f"\nClean colored mesh PLY:\n{output_mesh_ply}")
    print(f"\nPrint-ready STL:\n{output_stl}")

    return output_cloud, output_mesh_ply, output_stl


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    try:
        input_file = choose_ply_file()

        if not input_file:
            print("No input file selected.")
            sys.exit()

        output_folder = choose_output_folder()

        if not output_folder:
            print("No output folder selected.")
            sys.exit()

        clean_cloud, clean_mesh, stl_file = process_maize_cob(
            input_file,
            output_folder
        )

        messagebox.showinfo(
            "Finished",
            "Finished successfully.\n\n"
            f"Clean point cloud:\n{clean_cloud}\n\n"
            f"Clean colored mesh:\n{clean_mesh}\n\n"
            f"Print-ready STL:\n{stl_file}"
        )

    except Exception as e:
        print("\nERROR:")
        print(e)
        messagebox.showerror("Error", str(e))
