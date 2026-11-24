"""Reproducibility settings shared by all notebooks and scripts in src/.

Usage (in a notebook or script that runs from src/):
    from seed_utils import SEED, set_seed, segment_plane_repeatable
    set_seed(SEED)        # call before every random step (sampling, RANSAC, GRASP, ...)

Where randomness comes from in this project and how it is controlled:
  - Python `random`, NumPy                                -> set_seed()
  - Open3D CAD sampling (sample_points_uniformly,
    sample_points_poisson_disk)                            -> set_seed()  (o3d.utility.random.seed)
  - Open3D table-plane fit (segment_plane, RANSAC)        -> segment_plane_repeatable()
        Open3D splits the RANSAC iterations over several threads, so a seed alone does not
        make the result repeatable. The helper runs only this step on one thread (~20 ms per
        view); everything else, including ICP, keeps using all threads.
  - PyTorch (PointNet), if it is imported                 -> set_seed()
  - ICP (registration_icp) is deterministic and needs no seed.

A seed makes a run repeatable, not representative: a different seed draws a different CAD
sample and gives a slightly different ADD. Report results averaged over several seeds.
"""
import random
import sys

import numpy as np
import open3d as o3d
from threadpoolctl import threadpool_limits

SEED = 42


def set_seed(seed=SEED):
    """Seeds Python `random`, NumPy, Open3D and (if loaded) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def segment_plane_repeatable(pcd, distance_threshold, ransac_n=3, num_iterations=2000, seed=None):
    """Same as pcd.segment_plane(...), but repeatable: single-threaded RANSAC, optional seed.

    Returns (plane_model, inlier_indices) like Open3D's segment_plane.
    """
    if seed is not None:
        o3d.utility.random.seed(seed)
    with threadpool_limits(limits=1, user_api="openmp"):
        return pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n,
                                 num_iterations=num_iterations)
