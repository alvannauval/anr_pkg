"""Real-to-sim depth sensor model on the 2D camera grid. Shared code for the 12_x notebooks.

Part 1: camera model, real camera poses, CAD rendering of the per-pixel input features, and projection of a
3D point cloud onto the same pixel grid.
Part 2: CAD pose correction (T_average -> refined CAD pose) and the per-pixel labels y_drop and e_z.
Part 3: context inputs (edge distance, depth steps nearby) and the dataset over all views and workpieces.

Frames and units
  - Millimetres everywhere.
  - CAD frame: the workpiece.stl frame (origin = bottom centre). T_average.npy maps CAD -> robot base.
  - Camera frame: RealSense colour optical frame (x right, y down, z forward). view{i}_tf.npy maps
    camera -> robot base. It is the matrix 2_2_robot_multiview_ovp.py used to move the depth points into
    the base frame, so inv(view{i}_tf) puts a saved scan back exactly where the sensor measured it.
  - Camera pose in the CAD frame: T_cam_in_cad = inv(T_average) @ view{i}_tf.
  - Pixel centres sit at integer (u, v), as in 2_2 (x = (u - cx) * z / fx).
"""
import json
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
NATIVE_WIDTH, NATIVE_HEIGHT = 1920, 1080
# Factory calibration of the scanning camera (RealSense D435I, serial 335222073706), colour stream 1920x1080;
# the aligned depth uses the same grid. Source: `rs-enumerate-devices -c`, saved in CAMERA_CALIB_FILE.
# Distortion coefficients are all 0, so the pinhole model is exact. The fit from the saved depth images
# (fit_intrinsics, 12_1 section 2) agrees to within 0.5 px on every view, which confirms the scans used this stream.
CAMERA_CALIB_FILE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config",
                                                  "realsense_d435i_335222073706.txt"))
NATIVE_INTRINSICS = (1360.67224121094, 1360.55810546875, 975.655151367188, 560.14599609375)  # fx, fy, cx, cy

FEATURE_NAMES = ["xn", "yn", "z", "nx", "ny", "nz", "theta_inc"]


def read_rs_intrinsics(path=CAMERA_CALIB_FILE, stream="Color", width=NATIVE_WIDTH, height=NATIVE_HEIGHT):
    """One stream profile from an `rs-enumerate-devices -c` dump -> dict(fx, fy, cx, cy, distortion, coeffs)."""
    with open(path) as f:
        lines = f.read().splitlines()
    head = f'Intrinsic of "{stream}" / {width}x{height} /'
    start = next((i for i, line in enumerate(lines) if line.strip().startswith(head)), None)
    if start is None:
        raise ValueError(f"no '{head}' block in {path}")
    vals = {}
    for line in lines[start + 1:]:
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        vals[key.strip()] = value.strip()
    return dict(fx=float(vals["Fx"]), fy=float(vals["Fy"]), cx=float(vals["PPX"]), cy=float(vals["PPY"]),
                distortion=vals["Distortion"], coeffs=[float(c) for c in vals["Coeffs"].split()])


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def native(cls, intrinsics=NATIVE_INTRINSICS, width=NATIVE_WIDTH, height=NATIVE_HEIGHT):
        fx, fy, cx, cy = intrinsics
        return cls(float(fx), float(fy), float(cx), float(cy), int(width), int(height))

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def downscaled(self, s):
        """Coarser grid with the same optical centre and field of view. Each new pixel covers s x s native pixels,
        so its centre is the centre of that block: u_native = s * u + (s - 1) / 2.
        (The camera's own 960x540 profile lists cx / 2, 0.25 px off from this. That is a separate sensor mode; this
        grid is built from 1920x1080 data, and the CAD render and the real-scan projection both use it.)"""
        if s == 1:
            return self
        return Camera(self.fx / s, self.fy / s, (self.cx - (s - 1) / 2) / s, (self.cy - (s - 1) / 2) / s,
                      self.width // s, self.height // s)

    def normalized_grid(self):
        """(xn, yn) = ((u - cx) / fx, (v - cy) / fy) for every pixel, shape (H, W) each.
        xn, yn are the tangents of the ray angle, so they do not depend on the grid resolution."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return (u - self.cx) / self.fx, (v - self.cy) / self.fy

    def ray_dirs(self):
        """Ray direction per pixel in the camera frame, shape (H, W, 3), scaled so that the z component is 1.
        A hit at parameter t along this direction is at depth Z = t."""
        xn, yn = self.normalized_grid()
        return np.stack([xn, yn, np.ones_like(xn)], axis=-1)

    def project(self, P_cam):
        """3D points in the camera frame (N, 3) -> continuous pixel coordinates u, v and depth z."""
        z = P_cam[:, 2]
        return self.fx * P_cam[:, 0] / z + self.cx, self.fy * P_cam[:, 1] / z + self.cy, z

    def unproject(self, Z, mask=None):
        """Depth map (H, W) -> points (N, 3) in the camera frame, for the pixels in mask (default: finite Z)."""
        if mask is None:
            mask = np.isfinite(Z)
        xn, yn = self.normalized_grid()
        z = Z[mask]
        return np.stack([xn[mask] * z, yn[mask] * z, z], axis=-1)


# ---------------------------------------------------------------------------
# Real data: poses and scans
# ---------------------------------------------------------------------------
def real_dirs(experiment, workpiece):
    """Folders used for one real workpiece."""
    return dict(
        raw=f"pcd_data/testing_data/{experiment}/{workpiece}",
        processed=f"processed_data/{experiment}/{workpiece}",
        cad=f"workpiece/{workpiece}/workpiece.stl",
    )


def load_T_average(raw_dir):
    """CAD -> robot base ground truth, same convention as 4_3 (T_average.npy, no extra rotation)."""
    T = np.load(os.path.join(raw_dir, "T_average.npy"))
    T[3, :] = [0, 0, 0, 1]
    return T


def _view_file(raw_dir, view_idx, suffix):
    for name in (f"view{view_idx}{suffix}", f"view{view_idx:02d}{suffix}"):
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"view{view_idx}{suffix} not found in {raw_dir}")


def list_real_views(raw_dir):
    """Sorted view indices that have a scan and a camera TF."""
    views = []
    for f in os.listdir(raw_dir):
        m = re.fullmatch(r"view(\d+)\.pcd", f)
        if m and os.path.exists(os.path.join(raw_dir, f"view{m.group(1)}_tf.npy")):
            views.append(int(m.group(1)))
    return sorted(views)


def real_camera_pose(raw_dir, view_idx, T_cad_to_base=None):
    """Camera pose in the CAD frame for a real view: inv(T_average) @ view{i}_tf."""
    if T_cad_to_base is None:
        T_cad_to_base = load_T_average(raw_dir)
    T_base_cam = np.load(_view_file(raw_dir, view_idx, "_tf.npy"))
    return np.linalg.inv(T_cad_to_base) @ T_base_cam


def load_real_scan_base(raw_dir, view_idx):
    """Raw saved scan in the robot base frame, shape (N, 3)."""
    return np.asarray(o3d.io.read_point_cloud(_view_file(raw_dir, view_idx, ".pcd")).points)


def load_real_scan_cam(raw_dir, view_idx):
    """Raw saved scan (robot base frame) moved back into the camera frame of that view, shape (N, 3)."""
    T_base_cam = np.load(_view_file(raw_dir, view_idx, "_tf.npy"))
    return transform_points(load_real_scan_base(raw_dir, view_idx), np.linalg.inv(T_base_cam))


def transform_points(P, T):
    return P @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# CAD renderer
# ---------------------------------------------------------------------------
class CadRenderer:
    """Ray-casts the CAD mesh on a camera grid and returns the per-pixel input features."""

    TABLE_HALF_MM = 1000.0

    def __init__(self, mesh_path):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_triangle_normals()
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        self._table_scene = None

    @staticmethod
    def _rays(cam, T_cam_in_cad):
        R, t = T_cam_in_cad[:3, :3], T_cam_in_cad[:3, 3]
        d_cad = cam.ray_dirs() @ R.T
        return o3d.core.Tensor(np.concatenate([np.broadcast_to(t, d_cad.shape), d_cad], axis=-1).astype(np.float32))

    def scene_depth(self, cam, T_cam_in_cad):
        """Depth map (H, W) of the CAD standing on the table (the plane z = 0 of the CAD frame, which the refined
        pose of 12_2 puts exactly on the real table). NaN where a ray hits neither. Used for the context inputs only;
        the labels stay on the CAD pixels."""
        if self._table_scene is None:
            h = self.TABLE_HALF_MM
            table = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)),
                o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]])))
            self._table_scene = o3d.t.geometry.RaycastingScene()
            self._table_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
            self._table_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(table))
        return self._table_scene.cast_rays(self._rays(cam, T_cam_in_cad))["t_hit"].numpy().astype(np.float64)

    def render(self, cam, T_cam_in_cad):
        """Returns a dict of (H, W) maps; float maps are NaN where the ray misses the CAD.

        mask       ray hits the CAD
        z          depth along the optical axis (mm), the same quantity the RealSense reports
        range      distance from the optical centre along the ray (mm)
        xn, yn     normalised image coordinates (u - cx) / fx, (v - cy) / fy
        nx, ny, nz surface normal in the camera frame, oriented towards the camera
        theta_inc  angle of incidence (deg): angle between the surface normal and the reversed camera ray
        prim_id    hit triangle index in the mesh (-1 = miss)
        flipped    pixels whose mesh normal pointed away from the camera and was flipped
        """
        R = T_cam_in_cad[:3, :3]
        d_cam = cam.ray_dirs()
        ans = self.scene.cast_rays(self._rays(cam, T_cam_in_cad))

        z = ans["t_hit"].numpy().astype(np.float64)
        mask = np.isfinite(z)
        prim_id = ans["primitive_ids"].numpy().astype(np.int64)
        prim_id[~mask] = -1

        n_cam = ans["primitive_normals"].numpy().astype(np.float64) @ R      # R^T n for every pixel
        ray_unit = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)
        cos_in = -(n_cam * ray_unit).sum(-1)
        flipped = mask & (cos_in < 0)
        n_cam[flipped] *= -1.0
        theta = np.degrees(np.arccos(np.clip(np.abs(cos_in), 0.0, 1.0)))

        xn, yn = cam.normalized_grid()
        out = dict(mask=mask, prim_id=prim_id, flipped=flipped,
                   z=z, range=z * np.linalg.norm(d_cam, axis=-1),
                   xn=xn, yn=yn, nx=n_cam[..., 0], ny=n_cam[..., 1], nz=n_cam[..., 2], theta_inc=theta)
        for k in ("z", "range", "xn", "yn", "nx", "ny", "nz", "theta_inc"):
            out[k] = np.where(mask, out[k], np.nan)
        return out


def feature_stack(render):
    """(7, H, W) float32 array in FEATURE_NAMES order; NaN outside the CAD mask."""
    return np.stack([render[k] for k in FEATURE_NAMES]).astype(np.float32)


def cad_edges(render, crease_deg=20.0, jump_mm=3.0):
    """Pixels on a CAD edge: silhouette border, normal crease (> crease_deg between neighbours) or depth jump."""
    m = render["mask"]
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n = np.stack([render[k] for k in ("nx", "ny", "nz")], axis=-1)
    z = render["z"]
    cos_thr = np.cos(np.radians(crease_deg))
    H, W = m.shape
    with np.errstate(invalid="ignore"):
        for dv, du in ((0, 1), (1, 0)):
            a = (slice(0, H - dv), slice(0, W - du))
            b = (slice(dv, H), slice(du, W))
            crease = m[a] & m[b] & (((n[a] * n[b]).sum(-1) < cos_thr) | (np.abs(z[a] - z[b]) > jump_mm))
            edge[a] |= crease
            edge[b] |= crease
    return edge


def mask_bbox(mask, margin=0):
    """(v0, v1, u0, u1) slice bounds of the True pixels, grown by margin and clipped to the image."""
    vs, us = np.nonzero(mask)
    H, W = mask.shape
    return (max(vs.min() - margin, 0), min(vs.max() + margin + 1, H),
            max(us.min() - margin, 0), min(us.max() + margin + 1, W))


# ---------------------------------------------------------------------------
# Point cloud -> pixel grid
# ---------------------------------------------------------------------------
def project_to_grid(cam, P_cam, reduce="median"):
    """Bins camera-frame points into the pixel grid.

    Returns (Z, count): Z is (H, W) with the min / mean / median depth of the points in each pixel (NaN = empty),
    count is the number of points per pixel. The median takes the lower middle point of each pixel.
    """
    u, v, z = cam.project(P_cam)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    ok = (z > 0) & (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
    pix, z = vi[ok] * cam.width + ui[ok], z[ok]

    Z = np.full(cam.height * cam.width, np.nan)
    count = np.bincount(pix, minlength=cam.height * cam.width)
    if len(pix):
        order = np.lexsort((z, pix))                  # by pixel, then by depth
        pix, z = pix[order], z[order]
        starts = np.r_[0, np.flatnonzero(np.diff(pix)) + 1]
        n = np.diff(np.r_[starts, len(pix)])
        if reduce == "min":
            Z[pix[starts]] = z[starts]
        elif reduce == "mean":
            Z[pix[starts]] = np.add.reduceat(z, starts) / n
        elif reduce == "median":
            Z[pix[starts]] = z[starts + (n - 1) // 2]
        else:
            raise ValueError(reduce)
    return Z.reshape(cam.height, cam.width), count.reshape(cam.height, cam.width)


def near_cad_surface(cam, P_cam, render, gate_mm):
    """True for camera-frame points whose depth is within gate_mm of the CAD depth at their pixel.
    False where the pixel has no CAD (table, background) or the point is farther than the gate (e.g. table
    behind the workpiece edge)."""
    u, v, z = cam.project(P_cam)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inside = (z > 0) & (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
    z_cad = np.full(len(P_cam), np.nan)
    z_cad[inside] = render["z"][vi[inside], ui[inside]]
    with np.errstate(invalid="ignore"):
        return np.abs(z - z_cad) < gate_mm


# ---------------------------------------------------------------------------
# Intrinsics from the saved depth visualisation
# ---------------------------------------------------------------------------
_JET_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(256, 3)
_JET_TREE = cKDTree(_JET_LUT.astype(np.float32))


def read_viz_depth(viz_path, width=NATIVE_WIDTH):
    """Depth half of view*_viz.png -> normalised depth in [0, 1] (jet colormap inverted), NaN = no depth.
    2_2 saved (depth - d_min) / (d_max - d_min) with d_min, d_max over the whole frame, and black for no depth."""
    img = cv2.imread(viz_path)
    d = img[:, width:].reshape(-1, 3).astype(np.float32)
    dist, idx = _JET_TREE.query(d)
    valid = (d.sum(1) > 0) & (dist < 30)
    return np.where(valid, idx / 255.0, np.nan).reshape(img.shape[0], width)


def intrinsics_fit_error(intrinsics, norm_depth, P_cam):
    """Projects the scan with the given intrinsics and fits norm_depth = a * Z + b at the hit pixels.
    Returns (rmse of the fit, fraction of points that land on a pixel with depth)."""
    fx, fy, cx, cy = intrinsics
    H, W = norm_depth.shape
    u = np.round(fx * P_cam[:, 0] / P_cam[:, 2] + cx).astype(np.int64)
    v = np.round(fy * P_cam[:, 1] / P_cam[:, 2] + cy).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    n, z = norm_depth[v[ok], u[ok]], P_cam[ok, 2]
    has = np.isfinite(n)
    if has.sum() < 1000:
        return 1e9, 0.0
    A = np.c_[z[has], np.ones(has.sum())]
    coef, *_ = np.linalg.lstsq(A, n[has], rcond=None)
    return float(np.sqrt(np.mean((n[has] - A @ coef) ** 2))), float(has.mean())


def fit_intrinsics(raw_dir, view_idx, f_grid=np.arange(1300, 1481, 20), cx_grid=np.arange(920, 1031, 10),
                   cy_grid=np.arange(500, 621, 10)):
    """Finds (fx, fy, cx, cy) that make the saved scan line up with its depth image: coarse grid, then Nelder-Mead.
    Returns dict(intrinsics, rmse, frac_on_valid)."""
    norm = read_viz_depth(_view_file(raw_dir, view_idx, "_viz.png"))
    P = load_real_scan_cam(raw_dir, view_idx)
    best = min(((intrinsics_fit_error((f, f, cx, cy), norm, P)[0], (f, f, cx, cy))
                for f in f_grid for cx in cx_grid for cy in cy_grid), key=lambda r: r[0])
    res = minimize(lambda k: intrinsics_fit_error(k, norm, P)[0], np.array(best[1], float), method="Nelder-Mead",
                   options=dict(xatol=0.2, fatol=1e-6, maxiter=400))
    rmse, frac = intrinsics_fit_error(res.x, norm, P)
    return dict(intrinsics=tuple(float(x) for x in res.x), rmse=rmse, frac_on_valid=frac)


# ---------------------------------------------------------------------------
# Part 2: CAD pose correction
# ---------------------------------------------------------------------------
# T_average (ICP of the CAD to the merged scans, 4_1) is ~2 deg tilted and ~2 mm off (see 12_2). It is replaced by one rigid correction per
# workpiece, measured from many views pooled together, in two steps:
#   1. tilt + height from the table: the workpiece stands on the table, so the table must be the plane z = 0.
#   2. x, y and yaw from the top face seen from above: the outline of the top-face points (outer edge + pocket
#      opening) is turned and slid onto the CAD's, allowing for the sensor fattening the edges (topface_shift).
# Tilt and height come from the table, not the workpiece, so the sensor's depth errors on the workpiece (what the
# model has to learn) stay out of the pose.

def refined_pose_path(experiment, workpiece):
    return f"simulation/real2sim/{experiment}/{workpiece}/T_cad_refined.npy"


def save_refined_pose(experiment, workpiece, T_refined, report):
    path = refined_pose_path(experiment, workpiece)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, T_refined)
    with open(path.replace(".npy", ".json"), "w") as f:
        json.dump(report, f, indent=2)
    return path


def load_cad_pose(experiment, workpiece, refined=True):
    """CAD -> robot base: the refined pose saved by 12_2 (refined=True) or the original T_average.npy."""
    if refined:
        return np.load(refined_pose_path(experiment, workpiece))
    return load_T_average(real_dirs(experiment, workpiece)["raw"])


def cad_bounds(renderer):
    b = renderer.mesh.get_axis_aligned_bounding_box()
    return np.asarray(b.min_bound), np.asarray(b.max_bound)


def table_points(P_cad, lo, hi, margin=15.0, z_band=15.0, outer=150.0):
    """Points that can be table: within z_band of z = 0, at least `margin` outside the CAD footprint (skips the
    1-2 mm halo the sensor leaves next to the walls) and within `outer` of the CAD origin."""
    x, y, z = P_cad.T
    outside = (x < lo[0] - margin) | (x > hi[0] + margin) | (y < lo[1] - margin) | (y > hi[1] + margin)
    return (np.abs(z) < z_band) & outside & (np.abs(x) < outer) & (np.abs(y) < outer)


def fit_table_plane(P, cell=5.0, min_pts=20):
    """Robust plane n . p + d = 0 (n pointing up) through table points that are already close to z = 0.

    Single points on the table scatter by about +-2 mm, which is too thick for RANSAC to pin the tilt. Instead, the
    table is cut into cell x cell mm patches, each patch gives its median height, and a plane z = a x + b y + c is
    fitted to the patch medians by least squares; patches more than 3 robust sigmas off (objects, halo) are dropped
    and the plane refitted. Returns (n, d, info): info has n_patches, flatness_mm (robust sigma of the patch
    medians around the plane) and, for plots, patch_x / patch_y / patch_z (patch centres and median heights),
    patch_used and patch_residual (height above the fitted plane)."""
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    _, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.lexsort((P[:, 2], inv))
    inv_s, z_s = inv[order], P[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(inv_s)) + 1]
    n_in = np.diff(np.r_[starts, len(inv_s)])
    keep = n_in >= min_pts
    z_med = z_s[starts + (n_in - 1) // 2][keep]
    xy = (np.bincount(inv, weights=P[:, 0]) / counts, np.bincount(inv, weights=P[:, 1]) / counts)
    x, y = xy[0][inv_s[starts]][keep], xy[1][inv_s[starts]][keep]
    use = np.ones(len(z_med), bool)
    for _ in range(3):
        A = np.c_[x[use], y[use], np.ones(use.sum())]
        coef = np.linalg.lstsq(A, z_med[use], rcond=None)[0]
        res = z_med - np.c_[x, y, np.ones(len(x))] @ coef
        sigma = 1.4826 * np.median(np.abs(res[use]))
        use = np.abs(res) < 3 * max(sigma, 0.1)
    a, b, c = coef
    k = np.linalg.norm([a, b, 1.0])
    info = dict(n_patches=int(use.sum()), flatness_mm=float(sigma),
                patch_x=x, patch_y=y, patch_z=z_med, patch_used=use, patch_residual=res)
    return np.array([-a, -b, 1.0]) / k, float(-c / k), info


def table_correction(n, d):
    """Rigid transform (old CAD frame -> corrected CAD frame) that turns the plane n . p + d = 0 into z = 0:
    the smallest rotation taking n to +z, then a shift along z."""
    v, c = np.cross(n, [0.0, 0.0, 1.0]), float(n[2])
    s = np.linalg.norm(v)
    R = np.eye(3)
    if s > 1e-12:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = d
    return T


TOP_RES_MM = 0.25       # raster cell size for the top-face outline (seen from above)


def cad_top_mask(renderer, res=TOP_RES_MM, pad=12.0):
    """The CAD's top face seen from straight above, as an (H, W) bool raster of res mm cells.
    Returns (mask, x0, y0): cell (i, j) covers x in [x0 + j res, x0 + (j + 1) res), y likewise from y0."""
    lo, hi = cad_bounds(renderer)
    xs = np.arange(lo[0] - pad, hi[0] + pad, res) + res / 2
    ys = np.arange(lo[1] - pad, hi[1] + pad, res) + res / 2
    X, Y = np.meshgrid(xs, ys)
    z0 = hi[2] + 50.0
    rays = np.stack([X, Y, np.full_like(X, z0), np.zeros_like(X), np.zeros_like(X), -np.ones_like(X)], axis=-1)
    t = renderer.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    return np.abs(z0 - t - hi[2]) < 0.5, xs[0] - res / 2, ys[0] - res / 2


def rasterize_xy(P, x0, y0, shape, res=TOP_RES_MM, close_px=5):
    """Points (x, y in the first two columns) -> (H, W) bool raster of occupied cells, small gaps closed."""
    i = np.floor((P[:, 1] - y0) / res).astype(np.int64)
    j = np.floor((P[:, 0] - x0) / res).astype(np.int64)
    ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    M = np.zeros(shape, np.uint8)
    M[i[ok], j[ok]] = 1
    return cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8)).astype(bool)


def topface_shift(P_top, renderer, res=TOP_RES_MM, grow_mm=np.arange(0.0, 3.51, 0.5)):
    """x/y shift that slides the real top face (points at top height, CAD frame) onto the CAD's top face, both seen
    from above: the peak of the FFT cross-correlation of the two rasters, refined below one cell by a parabola.

    The real top face comes out wider than the CAD's: the sensor fattens edges, by ~2-3 mm when many views are
    pooled. A CAD face of the true size then fits anywhere inside that margin equally well, so the CAD face is also
    grown by each value in grow_mm and the size that overlaps best is used.
    Returns (tx, ty, info): adding (tx, ty) to the points aligns them. info: grow_mm (the best one), IoU before /
    after (against the grown CAD face) and the rasters."""
    cad, x0, y0 = cad_top_mask(renderer, res)
    real = rasterize_xy(P_top, x0, y0, cad.shape, res)
    F_real = np.fft.fft2(real.astype(float))
    best = None
    for g in grow_mm:
        r_px = int(round(g / res))
        grown = cad.astype(np.uint8)
        if r_px:
            grown = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1)))
        grown = grown.astype(bool)
        corr = np.real(np.fft.ifft2(F_real * np.conj(np.fft.fft2(grown.astype(float)))))
        k = np.unravel_index(corr.argmax(), corr.shape)      # corr[k] = sum_x real[x + k] * grown[x]
        iou_k = corr[k] / (real.sum() + grown.sum() - corr[k])
        if best is None or iou_k > best[0]:
            best = (iou_k, float(g), grown, corr, k)
    _, g, grown, corr, k = best
    H, W = corr.shape

    def subcell(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den < 0 else 0.0

    di = k[0] + subcell(corr[k[0] - 1, k[1]], corr[k], corr[(k[0] + 1) % H, k[1]])
    dj = k[1] + subcell(corr[k[0], k[1] - 1], corr[k], corr[k[0], (k[1] + 1) % W])
    di, dj = (di - H if di > H / 2 else di), (dj - W if dj > W / 2 else dj)
    tx, ty = -dj * res, -di * res
    moved = rasterize_xy(P_top[:, :2] + [tx, ty], x0, y0, cad.shape, res)

    def iou(A):
        return float((A & grown).sum() / max((A | grown).sum(), 1))

    return float(tx), float(ty), dict(grow_mm=g, iou_before=iou(real), iou_after=iou(moved), cad=cad,
                                      cad_grown=grown, real_before=real, real_after=moved,
                                      extent=(x0, x0 + W * res, y0, y0 + H * res))


def rotate_xy(P, yaw_deg):
    """x, y of the points turned by yaw_deg about the CAD z axis, shape (N, 2)."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return P[:, :2] @ np.array([[c, s], [-s, c]])


def topface_align(P_top, renderer, yaws=np.arange(-4.0, 4.001, 0.25)):
    """Yaw and x/y shift that put the real top face onto the CAD's, seen from above: for every yaw in `yaws` the top
    face is turned and then slid with topface_shift; the yaw with the best overlap wins (refined between grid steps
    by a parabola). Returns (T, info): T (4x4, a turn about z then a shift) moves the points onto the CAD; info is
    the final topface_shift info plus yaw_deg and yaw_scan = (yaws, IoU per yaw)."""
    ious = np.array([topface_shift(rotate_xy(P_top, yaw), renderer)[2]["iou_after"] for yaw in yaws])
    i = int(ious.argmax())
    yaw = float(yaws[i])
    if 0 < i < len(yaws) - 1:
        den = ious[i - 1] - 2 * ious[i] + ious[i + 1]
        if den < 0:
            yaw += 0.5 * (ious[i - 1] - ious[i + 1]) / den * (yaws[1] - yaws[0])
    tx, ty, info = topface_shift(rotate_xy(P_top, yaw), renderer)
    info["iou_before"] = topface_shift(P_top[:, :2], renderer)[2]["iou_before"]   # no turn, no shift
    info.update(yaw_deg=yaw, yaw_scan=(yaws, ious))
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = tx, ty
    return T, info


def refine_cad_pose(raw_dir, renderer, T_cad_to_base, views):
    """One rigid correction per workpiece from the scans of `views` pooled together (steps at the top of Part 2).
    Returns (T_refined, report, extra): T_refined is the corrected CAD -> base pose (same role as T_average),
    report the numbers (JSON-ready), extra the arrays for plots (top_points: the pooled top-face points after the
    table step, before the turn and shift)."""
    lo, hi = cad_bounds(renderer)
    T_inv = np.linalg.inv(T_cad_to_base)
    table_P, top_P = [], []
    for v in views:                                          # keep only what is needed, to save memory
        P = transform_points(load_real_scan_base(raw_dir, v), T_inv)
        x, y, z = P.T
        table_P.append(P[table_points(P, lo, hi)])
        top_P.append(P[(z > hi[2] - 8) & (z < hi[2] + 5) & (x > lo[0] - 10) & (x < hi[0] + 10)
                       & (y > lo[1] - 10) & (y < hi[1] + 10)])
    table_P = np.concatenate(table_P)
    n, d, table = fit_table_plane(table_P)
    T_table = table_correction(n, d)

    Q = transform_points(np.concatenate(top_P), T_table)
    top = Q[(Q[:, 2] > hi[2] - 2.5) & (Q[:, 2] < hi[2] + 1.5)]
    T_top, shift = topface_align(top, renderer)

    T_corr = T_top @ T_table                                 # old CAD frame -> corrected CAD frame
    T_refined = T_cad_to_base @ np.linalg.inv(T_corr)
    V = np.asarray(renderer.mesh.vertices)
    move = np.linalg.norm(transform_points(V, T_refined) - transform_points(V, T_cad_to_base), axis=1)
    report = dict(n_views=len(views), table_points=int(len(table_P)), top_points=int(len(top)),
                  table_tilt_deg=float(np.degrees(np.arccos(n[2]))), table_z_mm=float(-d / n[2]),
                  table_patches=table["n_patches"], table_flatness_mm=table["flatness_mm"],
                  yaw_deg=shift["yaw_deg"], shift_x_mm=float(T_top[0, 3]), shift_y_mm=float(T_top[1, 3]),
                  top_fattening_mm=shift["grow_mm"],
                  top_iou_before=shift["iou_before"], top_iou_after=shift["iou_after"],
                  cad_move_mean_mm=float(move.mean()), cad_move_max_mm=float(move.max()))
    return T_refined, report, dict(T_table=T_table, T_top=T_top, table=table, shift=shift, top_points=top)


def outline_iou(cam, render, P_cam, T_cam_in_cad, lo, hi, min_height=20.0):
    """Overlap (IoU) in the image between the CAD's raised region (more than min_height above the table) and the
    real one: pixels that got a real point more than min_height above the table. 1 = identical outlines. The sensor's
    own edge effects (dropouts, fattened edges) keep it below 1 even for a perfect pose, so it is for comparing
    poses, not an absolute score."""
    m = render["mask"]
    cad = np.zeros_like(m)
    cad[m] = transform_points(cam.unproject(render["z"], m), T_cam_in_cad)[:, 2] > min_height
    x, y, z = transform_points(P_cam, T_cam_in_cad).T
    raised = ((z > min_height) & (z < hi[2] + 10) & (x > lo[0] - 10) & (x < hi[0] + 10)
              & (y > lo[1] - 10) & (y < hi[1] + 10))
    _, cnt = project_to_grid(cam, P_cam[raised], reduce="min")
    real = cv2.morphologyEx((cnt > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return float((cad & real).sum() / max((cad | real).sum(), 1))


def view_pose_check(raw_dir, view_idx, renderer, cam, T_cad_to_base):
    """How well the CAD placed with T_cad_to_base fits one real view on its own (CAD frame, mm).
    table_tilt_deg / table_z_mm: this view's table plane (ideal 0 / 0)
    top_z_mm:                    median height of the top-face points (ideal = CAD top, 40)
    outline_iou:                 see outline_iou (higher = better)"""
    lo, hi = cad_bounds(renderer)
    T_cam = real_camera_pose(raw_dir, view_idx, T_cad_to_base)
    P_cam = load_real_scan_cam(raw_dir, view_idx)
    P_cad = transform_points(P_cam, T_cam)
    out = dict(view=view_idx)
    ring = table_points(P_cad, lo, hi)
    if ring.sum() > 2000:
        n, d, _ = fit_table_plane(P_cad[ring])
        out.update(table_tilt_deg=float(np.degrees(np.arccos(n[2]))), table_z_mm=float(-d / n[2]))
    x, y, z = P_cad.T
    top = ((np.abs(z - hi[2]) < 3) & (x > lo[0] + 3) & (x < hi[0] - 3) & (y > lo[1] + 3) & (y < hi[1] - 3))
    out["top_z_mm"] = float(np.median(z[top])) if top.sum() > 100 else np.nan
    out["outline_iou"] = outline_iou(cam, renderer.render(cam, T_cam), P_cam, T_cam, lo, hi)
    return out


# ---------------------------------------------------------------------------
# Part 2: per-pixel labels
# ---------------------------------------------------------------------------
def make_labels(cam, render, P_cam, gate_mm=10.0):
    """Training targets on the CAD pixels of one view (all maps (H, W)).

    Pairing rule: every real point goes to the pixel it projects into, and points farther than gate_mm from the
    CAD depth at that pixel are ignored (table, background, stray points). A CAD pixel is kept if a point remains.
      y_drop  1 = the sensor returned nothing there, 0 = kept; NaN outside the CAD
      e_z     Z_real - Z_CAD (mm) on kept pixels, Z_real = median depth of the pixel's points; NaN elsewhere.
              Positive = real surface farther from the camera than the CAD.
      z_real  Z_real on kept pixels, NaN elsewhere
      n_pts   real points per CAD pixel (0 outside the CAD)
    """
    m = render["mask"]
    near = near_cad_surface(cam, P_cam, render, gate_mm)
    Z, n_pts = project_to_grid(cam, P_cam[near], reduce="median")
    kept = m & np.isfinite(Z)
    return dict(y_drop=np.where(m, (~kept).astype(np.float32), np.nan),
                e_z=np.where(kept, Z - render["z"], np.nan),
                z_real=np.where(kept, Z, np.nan),
                n_pts=np.where(m, n_pts, 0))


# ---------------------------------------------------------------------------
# Part 3: context inputs and the dataset
# ---------------------------------------------------------------------------
# The 7 per-pixel features cannot tell a pixel next to an edge from one far away, nor the upper side of a depth
# step from the lower side, and the sensor treats these very differently (12_3 section 2):
#   edge_mm    distance to the nearest CAD edge (silhouette, crease, depth step), in the image, in mm at the
#              pixel's depth. Dropout falls from ~50% at an edge to ~1% beyond 10 mm.
#   deeper_mm  how much deeper the scene (CAD + table) gets within STEP_RADIUS_MM around the pixel
#   closer_mm  how much closer it gets
# Next to a drop-off (deeper_mm large: top face at the outer edge or the pocket rim) the sensor reads too far and
# drops a lot; next to a rise (closer_mm large: pocket floor along its walls) it reads too close. Both come from the
# sensor smoothing depth across the step. All three are computed from the CAD and the camera pose only, so the
# simulator can compute them for any new view.
INPUT_NAMES = FEATURE_NAMES + ["edge_mm", "deeper_mm", "closer_mm"]
STEP_RADIUS_MM = 5.0    # best of 2 / 3 / 5 / 8 / 12 mm on held-out workpieces (12_3 exploration)
STEP_CAP_MM = 50.0      # steps are capped here; nothing hit (beyond the table) counts as a 50 mm drop-off


def edge_distance(cam, render):
    """Distance from every pixel to the nearest CAD edge pixel (cad_edges), converted to mm with the pixel's own
    depth (mm per pixel = z / fx). Returns dict of (H, W) maps, NaN outside the CAD:
      edge_mm                 the distance
      edge_dx_mm, edge_dy_mm  offset to that nearest edge pixel (image right / down positive), kept for later tests
                              of the left/right asymmetry of stereo dropouts"""
    m = render["mask"]
    edge = cad_edges(render)
    dist, lab = cv2.distanceTransformWithLabels((~edge).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ev, eu = np.nonzero(edge)
    lab_u = np.zeros(lab.max() + 1)
    lab_v = np.zeros(lab.max() + 1)
    lab_u[lab[ev, eu]] = eu
    lab_v[lab[ev, eu]] = ev
    U, V = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    mm_per_px = render["z"] / cam.fx
    return dict(edge_mm=np.where(m, dist * mm_per_px, np.nan),
                edge_dx_mm=np.where(m, (lab_u[lab] - U) * mm_per_px, np.nan),
                edge_dy_mm=np.where(m, (lab_v[lab] - V) * mm_per_px, np.nan))


def step_context(cam, render, z_scene, radius_mm=STEP_RADIUS_MM, cap_mm=STEP_CAP_MM):
    """deeper_mm / closer_mm maps (NaN outside the CAD): max / min scene depth in a disc of radius_mm around each
    CAD pixel, minus the pixel's depth. The disc radius in pixels is set once per view from the median CAD depth."""
    m = render["mask"]
    z = render["z"]
    rad = max(1, int(round(radius_mm * cam.fx / np.nanmedian(z[m]))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    hit = np.isfinite(z_scene)
    far = cv2.dilate(np.where(hit, z_scene, np.nanmax(z[m]) + cap_mm).astype(np.float32), k)
    near = cv2.erode(np.where(hit, z_scene, 1e6).astype(np.float32), k)
    return dict(deeper_mm=np.where(m, np.clip(far - z, 0, cap_mm), np.nan),
                closer_mm=np.where(m, np.clip(z - near, 0, cap_mm), np.nan))


def input_maps(cam, renderer, T_cam_in_cad):
    """All model inputs for any camera pose, from the CAD alone (no real scan): the render dict (mask, z, normals,
    theta_inc, ...) extended with edge_mm, edge_dx_mm, edge_dy_mm, deeper_mm, closer_mm. Used by the simulator."""
    r = renderer.render(cam, T_cam_in_cad)
    r.update(edge_distance(cam, r))
    r.update(step_context(cam, r, renderer.scene_depth(cam, T_cam_in_cad)))
    return r


def view_maps(cam, renderer, raw_dir, view_idx, T_cad_to_base, gate_mm=10.0):
    """Everything for one real view on the grid: render, labels and the context maps. Returns (render, labels,
    context, T_cam_in_cad); context holds edge_mm, edge_dx_mm, edge_dy_mm, deeper_mm, closer_mm."""
    T_cam = real_camera_pose(raw_dir, view_idx, T_cad_to_base)
    r = renderer.render(cam, T_cam)
    labels = make_labels(cam, r, load_real_scan_cam(raw_dir, view_idx), gate_mm)
    ctx = edge_distance(cam, r)
    ctx.update(step_context(cam, r, renderer.scene_depth(cam, T_cam)))
    return r, labels, ctx, T_cam


def dataset_path(experiment, workpiece, cam):
    return f"simulation/real2sim/{experiment}/{workpiece}/dataset_{cam.width}x{cam.height}.npz"


def build_dataset(experiment, workpiece, cam, gate_mm=10.0, views=None, log_every=48):
    """One row per CAD pixel per view, for all real views of a workpiece, with the refined CAD pose of 12_2.
    Saves dataset_path(...) (compressed npz) and returns a per-view summary list of dicts."""
    d = real_dirs(experiment, workpiece)
    renderer = CadRenderer(d["cad"])
    T_cad = load_cad_pose(experiment, workpiece, refined=True)
    views = list_real_views(d["raw"]) if views is None else list(views)
    cols = {k: [] for k in ("view", "u", "v", "X", "edge_dx_mm", "edge_dy_mm", "y_drop", "e_z", "n_pts")}
    per_view, poses = [], []
    for i, v in enumerate(views):
        r, L, ctx, T_cam = view_maps(cam, renderer, d["raw"], v, T_cad, gate_mm)
        m = r["mask"]
        vv, uu = np.nonzero(m)
        maps = {**r, **ctx}
        cols["view"].append(np.full(len(vv), v, np.int16))
        cols["u"].append(uu.astype(np.int16))
        cols["v"].append(vv.astype(np.int16))
        cols["X"].append(np.stack([maps[k][m] for k in INPUT_NAMES], axis=1).astype(np.float32))
        cols["edge_dx_mm"].append(ctx["edge_dx_mm"][m].astype(np.float32))
        cols["edge_dy_mm"].append(ctx["edge_dy_mm"][m].astype(np.float32))
        cols["y_drop"].append(L["y_drop"][m].astype(np.uint8))
        cols["e_z"].append(L["e_z"][m].astype(np.float32))
        cols["n_pts"].append(np.minimum(L["n_pts"][m], 255).astype(np.uint8))
        poses.append(T_cam)
        per_view.append(dict(workpiece=workpiece, view=v, pixels=int(m.sum()),
                             drop_rate=float(L["y_drop"][m].mean()), ez_median=float(np.nanmedian(L["e_z"])),
                             touches_border=bool(m[0].any() | m[-1].any() | m[:, 0].any() | m[:, -1].any())))
        if log_every and (i + 1) % log_every == 0:
            print(f"  {workpiece}: {i + 1}/{len(views)} views", flush=True)
    path = dataset_path(experiment, workpiece, cam)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, **{k: np.concatenate(vals) for k, vals in cols.items()},
        views=np.array(views), T_cam_in_cad=np.array(poses), input_names=np.array(INPUT_NAMES),
        camera=np.array([cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height]),
        gate_mm=gate_mm, step_radius_mm=STEP_RADIUS_MM, step_cap_mm=STEP_CAP_MM,
        pose_file=refined_pose_path(experiment, workpiece))
    return per_view


def load_dataset(experiment, workpieces, cam):
    """All pixels of the given workpieces as one pandas DataFrame: workpiece, view, u, v, the INPUT_NAMES columns,
    edge_dx_mm, edge_dy_mm, y_drop (0/1), e_z (NaN where dropped), n_pts."""
    import pandas as pd
    frames = []
    for wp in workpieces:
        z = np.load(dataset_path(experiment, wp, cam))
        df = pd.DataFrame(z["X"], columns=[str(n) for n in z["input_names"]])
        df.insert(0, "v", z["v"])
        df.insert(0, "u", z["u"])
        df.insert(0, "view", z["view"])
        df.insert(0, "workpiece", wp)
        for k in ("edge_dx_mm", "edge_dy_mm", "y_drop", "e_z", "n_pts"):
            df[k] = z[k]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["workpiece"] = df["workpiece"].astype("category")
    return df
