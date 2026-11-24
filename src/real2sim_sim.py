"""Real-to-sim sensor model, Part 5: the simulator.

For a camera pose:
  1. render the CAD on the 960x540 grid and compute the model inputs   (real2sim.input_maps)
  2. the trained model gives p (dropout chance), mu and sigma per pixel  (real2sim_model)
  3. "roll the dice" with two blurred random fields, so that neighbouring pixels behave alike, like the real sensor:
       Cloud A (drops):  a pixel is dropped where Phi(A) < p     (Phi = normal CDF, so P(drop) = p exactly)
       Cloud B (depth):  simulated depth = CAD depth + mu + sigma * B, rounded to whole mm like the sensor
     Each cloud is unit-variance noise made of patches (white noise blurred by blur_px) plus a share fine_share of a
     finer grain (white noise blurred by fine_px; fine_px = 0 means every pixel on its own). The recipes are
     measured from the real residuals (12_5).
  4. unproject the kept pixels to 3D.

Part 6 (bottom of the file) turns a simulated scan into the same kind of file as a real one: full camera resolution,
about 15 frames merged with a small frame-to-frame flicker and 0.5 mm voxels (like 2_2), then the 4_3 steps (table
removed, CAD frame, crop).
"""
import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.special import ndtr

import real2sim as rs
import real2sim_model as rm


@dataclass(frozen=True)
class NoiseRecipe:
    blur_px: float            # size (Gaussian std, px) of the patches neighbouring pixels share
    fine_share: float = 0.0   # share of the variance in a finer grain (0 = patches only)
    fine_px: float = 0.0      # size of that grain (0 = every pixel on its own)

    def scaled(self, s):
        """The same recipe on a grid with s times more pixels per mm (sizes in px x s, so they stay the same in mm)."""
        return NoiseRecipe(self.blur_px * s, self.fine_share, self.fine_px * s)


def blur_unit(white, blur_px):
    """Blur unit white noise with a Gaussian of std blur_px and rescale it to unit variance (blur_px 0 = unchanged)."""
    if blur_px <= 0:
        return white
    k = cv2.getGaussianKernel(2 * int(np.ceil(4 * blur_px)) + 1, blur_px).ravel()
    out = cv2.sepFilter2D(white, cv2.CV_64F, k, k, borderType=cv2.BORDER_REFLECT)
    return out / np.sum(k ** 2)    # exact std of blurred unit white noise (separable kernel: sum(k^2) per axis)


def mix(patches, grain, fine_share):
    """Unit-variance mix of two unit-variance fields."""
    return np.sqrt(1.0 - fine_share) * patches + np.sqrt(fine_share) * grain


def noise_field(shape, recipe, rng):
    """Unit-variance random field (H, W) with the spatial correlation of the recipe:
    corr(d) = (1 - fine_share) exp(-d^2 / (4 blur_px^2)) + fine_share exp(-d^2 / (4 fine_px^2))."""
    patches = blur_unit(rng.standard_normal(shape), recipe.blur_px)
    if recipe.fine_share <= 0:
        return patches
    return mix(patches, blur_unit(rng.standard_normal(shape), recipe.fine_px), recipe.fine_share)


def field_correlation(maps, masks, max_lag=40):
    """Average spatial correlation vs pixel distance of value maps over their masks (same as used on the real
    residuals). maps, masks: lists of (H, W) arrays. Returns (distances 0..max_lag, correlation)."""
    num = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))
    cnt = np.zeros_like(num)
    for img, mk in zip(maps, masks):
        H, W = img.shape
        sh = (2 * H, 2 * W)
        f = np.where(mk, img, 0.0)
        F, M = np.fft.rfft2(f, sh), np.fft.rfft2(mk.astype(float), sh)
        n = np.fft.fftshift(np.fft.irfft2(F * np.conj(F), sh))
        c = np.fft.fftshift(np.fft.irfft2(M * np.conj(M), sh))
        win = (slice(H - max_lag, H + max_lag + 1), slice(W - max_lag, W + max_lag + 1))
        num += n[win]
        cnt += c[win]
    corr = num / np.maximum(cnt, 1)
    corr /= corr[max_lag, max_lag]
    yy, xx = np.mgrid[-max_lag:max_lag + 1, -max_lag:max_lag + 1]
    d = np.round(np.hypot(xx, yy)).astype(int).ravel()
    prof = np.bincount(d, corr.ravel()) / np.bincount(d)
    return np.arange(max_lag + 1), prof[:max_lag + 1]


class Simulator:
    """Simulates the RealSense scan of one CAD from any camera pose (CAD frame)."""

    def __init__(self, cad_path, model_path, cam, depth_noise, drop_noise, sigma_scale=1.0, round_mm=True,
                 device="cpu"):
        self.renderer = rs.CadRenderer(cad_path)
        self.model_path = model_path
        self.model, self.inputs, self.meta = rm.load_model(model_path, device)
        self.cam = cam
        self.depth_noise, self.drop_noise = depth_noise, drop_noise
        self.sigma_scale = sigma_scale
        self.round_mm = round_mm       # the sensor reports depth in whole millimetres

    def save_config(self, path, **extra):
        """Everything needed to rebuild this simulator for another CAD (see from_config)."""
        cfg = dict(model_path=self.model_path, depth_noise=asdict(self.depth_noise),
                   drop_noise=asdict(self.drop_noise), sigma_scale=self.sigma_scale, round_mm=self.round_mm, **extra)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_config(cls, path, cad_path, cam, device="cpu"):
        """Rebuilds a saved simulator for any CAD and any grid. The noise sizes are stored in pixels of the grid they
        were measured on (cfg["grid"]); on another grid they are scaled so the patches keep their size in mm.
        The model inputs are in mm and ray angles, so the model itself does not depend on the grid."""
        with open(path) as f:
            cfg = json.load(f)
        s = cam.width / cfg["grid"][0] if "grid" in cfg else 1.0
        return cls(cad_path, cfg["model_path"], cam, NoiseRecipe(**cfg["depth_noise"]).scaled(s),
                   NoiseRecipe(**cfg["drop_noise"]).scaled(s), sigma_scale=cfg["sigma_scale"],
                   round_mm=cfg["round_mm"], device=device)

    def predict_maps(self, T_cam_in_cad):
        """Inputs and model outputs as (H, W) maps (NaN outside the CAD): dict with the input_maps keys plus
        p, mu, sigma."""
        r = rs.input_maps(self.cam, self.renderer, T_cam_in_cad)
        m = r["mask"]
        X = np.stack([r[k][m] for k in self.inputs], axis=1).astype(np.float32)
        p, mu, sd = rm.predict(self.model, X)
        for k, vals in (("p", p), ("mu", mu), ("sigma", sd * self.sigma_scale)):
            r[k] = np.full(m.shape, np.nan)
            r[k][m] = vals
        return r

    def simulate(self, T_cam_in_cad, rng, maps=None):
        """One simulated scan. Returns dict:
          maps        predict_maps output (pass it in to reuse it for several draws of the same view)
          drop        (H, W) bool, CAD pixels the sensor drops
          z_sim       (H, W) simulated depth on kept pixels, NaN elsewhere
          z_static    the same before the rounding to whole mm (used by the Part 6 frame flicker)
          e_sim       z_sim - CAD depth
          points_cam  (N, 3) kept pixels unprojected, camera frame
          points_cad  (N, 3) the same points in the CAD frame"""
        r = self.predict_maps(T_cam_in_cad) if maps is None else maps
        m = r["mask"]
        A = noise_field(m.shape, self.drop_noise, rng)
        B = noise_field(m.shape, self.depth_noise, rng)
        drop = m & (ndtr(A) < r["p"])
        kept = m & ~drop
        z_static = r["z"] + r["mu"] + r["sigma"] * B
        z = np.round(z_static) if self.round_mm else z_static
        z_sim = np.where(kept, z, np.nan)
        P = self.cam.unproject(z_sim, kept)
        return dict(maps=r, drop=drop, z_sim=z_sim, z_static=np.where(kept, z_static, np.nan), e_sim=z_sim - r["z"],
                    points_cam=P, points_cad=rs.transform_points(P, T_cam_in_cad))


# ---------------------------------------------------------------------------
# Part 6: simulated scans in the format of the real ones
# ---------------------------------------------------------------------------
# The real files go through two steps, and the simulated ones copy both:
#   2_2_robot_multiview_ovp.py  full-resolution depth -> robot base frame -> voxel_down_sample(0.5 mm) -> view{i}.pcd
#   4_3_process_pcd_real        remove the table (points < 2 mm above it) -> CAD frame of T_average -> crop to the
#                               CAD bounds + 5 mm -> processed_data/.../viewpoint_simulated_noise_{i}.pcd
VOXEL_MM = 0.5
PLANE_OFFSET_MM = 2.0
CROP_MARGIN_MM = 5.0


@dataclass(frozen=True)
class Flicker:
    """Frame-to-frame shake of the depth. A saved real view merges about 15 frames (2_2 records for 1 s and keeps
    every 2nd frame). Each frame is rounded to whole mm on its own, so a pixel whose reading sits near a mm boundary
    gives two depths, i.e. two points one behind the other. The model of Parts 4-5 describes the merged scan (the
    median per pixel), so it does not contain this. Shake per frame (std, mm):
        a_mm * (z / 500 mm)^2 / cos(theta_inc)        (stereo noise grows with distance squared and with slant)
    a_mm is measured in 12_6 on the training workpieces."""
    a_mm: float = 0.0
    frames: int = 15

    def std(self, z, theta_deg):
        return self.a_mm * (z / 500.0) ** 2 / np.cos(np.radians(np.minimum(theta_deg, 80.0)))


def raw_view_cloud(simulator, T_cad_to_base, T_base_cam, rng, flicker=Flicker(), maps=None, voxel_mm=VOXEL_MM):
    """Simulated counterpart of a saved real view{i}.pcd: a simulated scan from camera pose T_base_cam with the
    workpiece at T_cad_to_base, merged over flicker.frames frames (see Flicker), in the robot base frame,
    voxel-downsampled like 2_2. maps: predict_maps output of this pose, to reuse it. Returns (points_base (N, 3),
    the simulate() dict)."""
    out = simulator.simulate(np.linalg.inv(T_cad_to_base) @ T_base_cam, rng, maps=maps)
    kept = np.isfinite(out["z_sim"])
    if flicker.a_mm > 0:
        r = out["maps"]
        zs = out["z_static"][kept]
        s = flicker.std(r["z"][kept], r["theta_inc"][kept])
        Z = zs[None, :] + s[None, :] * rng.standard_normal((flicker.frames, len(zs)))
        if simulator.round_mm:
            Z = np.round(Z)
        pairs = np.unique(np.c_[np.tile(np.arange(len(zs)), flicker.frames), Z.ravel()], axis=0)  # same depth = same point
        i, z = pairs[:, 0].astype(np.int64), pairs[:, 1]
    else:
        i, z = np.arange(kept.sum()), out["z_sim"][kept]
    xn, yn = simulator.cam.normalized_grid()
    P = rs.transform_points(np.c_[xn[kept][i] * z, yn[kept][i] * z, z], T_base_cam)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel_mm)
    return np.asarray(pcd.points), out


def process_like_4_3(points_base, T_cad_to_base, T_export, bounds, plane_offset=PLANE_OFFSET_MM,
                     margin=CROP_MARGIN_MM):
    """The 4_3 steps on a simulated view (base frame): remove the points within plane_offset of the table, move the
    rest into the CAD frame of T_export (CAD -> base; T_average, like the real files) and crop to the CAD bounds
    (lo, hi) + margin. The simulation has no table, so its plane is known exactly: z = 0 of the true CAD frame
    T_cad_to_base. (4_3 fits it with RANSAC; on a simulated view RANSAC would find the top face instead.)"""
    z_true = rs.transform_points(points_base, np.linalg.inv(T_cad_to_base))[:, 2]
    P = rs.transform_points(points_base[z_true > plane_offset], np.linalg.inv(T_export))
    lo, hi = np.asarray(bounds[0]) - margin, np.asarray(bounds[1]) + margin
    return P[np.all((P >= lo) & (P <= hi), axis=1)]


def save_pcd(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
