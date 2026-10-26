"""
Check how well the raycasted (ideal) views line up with the real processed scans, before Voronoi.

Both clouds are in the CAD frame:
  - real scan : processed_data/{EXPERIMENT}/{WP}/viewpoint_simulated_noise_{i}.pcd   (from 4_3)
  - raycast   : viewpoints_candidate/testing_data/{EXPERIMENT}/{WP}/viewpoint_simulated_{i}.pcd  (re-raycast, +13.8 mm)
  - old raycast (optional): .../{EXPERIMENT}/old/{WP}/viewpoint_simulated_{i}.pcd     (original plan)

Usage (run from src/):
  python check_raycast_vs_scan.py TH1021AV 100         # one view: stats + windows
  python check_raycast_vs_scan.py TH1021AV all         # all views: summary table only
"""
import os
import sys
import numpy as np
import open3d as o3d

EXPERIMENT = "test_11_real_pocket"
WORKPIECE = sys.argv[1] if len(sys.argv) > 1 else "TH1021AV"
VIEW = sys.argv[2] if len(sys.argv) > 2 else "100"

SCAN_DIR = f"processed_data/{EXPERIMENT}/{WORKPIECE}"
RAY_DIR = f"viewpoints_candidate/testing_data/{EXPERIMENT}/{WORKPIECE}"
OLD_RAY_DIR = f"viewpoints_candidate/testing_data/{EXPERIMENT}/old/{WORKPIECE}"
RAW_DIR = f"pcd_data/testing_data/{EXPERIMENT}/{WORKPIECE}"
CAD_PATH = f"workpiece/{WORKPIECE}/workpiece.stl"


def load(path):
    return o3d.io.read_point_cloud(path) if os.path.exists(path) else None


def nn_stats(src, dst):
    """Mean / median distance from each point of src to its nearest point in dst (mm)."""
    d = np.asarray(src.compute_point_cloud_distance(dst))
    return d.mean(), np.median(d)


def compare(scan, ray):
    s2r_mean, s2r_med = nn_stats(scan, ray)
    r2s_mean, r2s_med = nn_stats(ray, scan)
    shift = np.asarray(scan.points).mean(0) - np.asarray(ray.points).mean(0)
    return dict(s2r_mean=s2r_mean, s2r_med=s2r_med, r2s_mean=r2s_mean, r2s_med=r2s_med, centroid_shift=shift)


def camera_frame(pose, size=20.0):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(pose)


def single_view(i):
    scan = load(f"{SCAN_DIR}/viewpoint_simulated_noise_{i}.pcd")
    ray = load(f"{RAY_DIR}/viewpoint_simulated_{i}.pcd")
    old = load(f"{OLD_RAY_DIR}/viewpoint_simulated_{i}.pcd")
    if scan is None or ray is None:
        sys.exit(f"Missing scan or raycast for view {i} (run 4_3 batch and the 1_3 re-raycast cell first).")

    print(f"{WORKPIECE} view {i}: scan {len(scan.points)} pts | raycast {len(ray.points)} pts"
          + (f" | old raycast {len(old.points)} pts" if old else ""))
    for name, r in [("new raycast", ray), ("old raycast", old)]:
        if r is None:
            continue
        c = compare(scan, r)
        print(f"  {name:12s}: scan->ray mean {c['s2r_mean']:.2f} / median {c['s2r_med']:.2f} mm | "
              f"ray->scan mean {c['r2s_mean']:.2f} / median {c['r2s_med']:.2f} mm | "
              f"centroid shift (scan - ray) {np.round(c['centroid_shift'], 2)} mm")

    # Camera poses in the CAD frame: raycast (new), planned (old), real (from T_average and the recorded TF)
    geoms = [scan.paint_uniform_color([0.0, 0.65, 0.93]), ray.paint_uniform_color([0.0, 0.8, 0.0])]
    cad = o3d.io.read_triangle_mesh(CAD_PATH)
    cad.compute_vertex_normals()
    cad_wire = o3d.geometry.LineSet.create_from_triangle_mesh(cad)
    cad_wire.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(cad_wire)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0))

    o3d.visualization.draw_geometries(geoms, window_name=f"{WORKPIECE} view {i}: scan (blue) vs new raycast (green), CAD (grey)")
    if old is not None:
        o3d.visualization.draw_geometries(
            [scan, old.paint_uniform_color([1.0, 0.0, 0.0]), cad_wire],
            window_name=f"{WORKPIECE} view {i}: scan (blue) vs OLD raycast (red), CAD (grey)")

    # Where the cameras are: new raycast pose, old planned pose and the real recorded pose
    poses = []
    for path, color in [(f"{RAY_DIR}/viewpoint_pose_{i}.npy", [0, 0.8, 0]), (f"{OLD_RAY_DIR}/viewpoint_pose_{i}.npy", [1, 0, 0])]:
        if os.path.exists(path):
            P = np.load(path)
            poses.append(camera_frame(P))
            s = o3d.geometry.TriangleMesh.create_sphere(radius=4.0).translate(P[:3, 3])
            s.paint_uniform_color(color)
            poses.append(s)
            print(f"  camera ({'new' if color[1] else 'old'} raycast) at {np.round(P[:3, 3], 1)}")
    if os.path.exists(f"{RAW_DIR}/T_average.npy") and os.path.exists(f"{RAW_DIR}/view{i}_tf.npy"):
        P_real = np.linalg.inv(np.load(f"{RAW_DIR}/T_average.npy")) @ np.load(f"{RAW_DIR}/view{i}_tf.npy")
        s = o3d.geometry.TriangleMesh.create_sphere(radius=4.0).translate(P_real[:3, 3])
        s.paint_uniform_color([0, 0.4, 1])
        poses += [camera_frame(P_real), s]
        print(f"  camera (real, via T_average) at {np.round(P_real[:3, 3], 1)}")
    o3d.visualization.draw_geometries(poses + [scan, cad_wire],
                                      window_name="Cameras: green = new raycast, red = old plan, blue = real (T_average)")


def all_views():
    rows = []
    files = sorted((int(f.split("_")[-1].split(".")[0]) for f in os.listdir(SCAN_DIR)
                    if f.startswith("viewpoint_simulated_noise_") and f.endswith(".pcd")))
    for i in files:
        scan = load(f"{SCAN_DIR}/viewpoint_simulated_noise_{i}.pcd")
        ray = load(f"{RAY_DIR}/viewpoint_simulated_{i}.pcd")
        old = load(f"{OLD_RAY_DIR}/viewpoint_simulated_{i}.pcd")
        if scan is None or ray is None or len(scan.points) == 0:
            continue
        new_c = compare(scan, ray)
        old_c = compare(scan, old) if old is not None else None
        rows.append((i, new_c, old_c))

    print(f"{WORKPIECE}: {len(rows)} views compared (distances in mm, mean over views)")
    for key, label in [("s2r_mean", "scan -> raycast (mean)"), ("r2s_mean", "raycast -> scan (mean)")]:
        new_v = np.array([r[1][key] for r in rows])
        line = f"  {label:24s}: new {new_v.mean():.2f}"
        if all(r[2] is not None for r in rows):
            old_v = np.array([r[2][key] for r in rows])
            line += f" | old {old_v.mean():.2f} | new better in {np.mean(new_v < old_v) * 100:.0f}% of views"
        print(line)
    shifts = np.array([r[1]["centroid_shift"] for r in rows])
    print(f"  centroid shift scan - new raycast (mean x, y, z): {np.round(shifts.mean(0), 2)} mm")
    worst = sorted(rows, key=lambda r: r[1]["r2s_mean"])[-5:]
    print("  worst views (raycast -> scan mean):", [(r[0], round(r[1]["r2s_mean"], 2)) for r in worst])


if __name__ == "__main__":
    if VIEW == "all":
        all_views()
    else:
        single_view(int(VIEW))
