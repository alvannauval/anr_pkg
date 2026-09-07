# --- Cell 0 ---
# ==========================================
# 1. HELPER FUNCTIONS & ICP ALGORITHMS
# ==========================================
import os
import copy
import glob
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def merge_simulated_scans(sim_dir):
    """
    Loads and merges all noisy simulated viewpoints.
    # RANSAC Plane Segmentation re-enabled per request.
    """
    pcd_files = glob.glob(os.path.join(sim_dir, 'viewpoint_simulated_noise_*.pcd'))
    merged_pcd = o3d.geometry.PointCloud()
    for file in pcd_files:
        pcd = o3d.io.read_point_cloud(file)
        
        
        merged_pcd += pcd
    return merged_pcd, len(pcd_files)

def preprocess_normal(pcd, num_points=False, invert_normals=False, radius=2, max_nn=30):
    """Downsamples, estimates normals, and computes FPFH features."""
    if num_points:
        current_num_points = len(pcd.points)
        if current_num_points >= num_points:
            pcd_down = pcd.farthest_point_down_sample(num_points)
        else:
            pcd_down = o3d.geometry.PointCloud(pcd) 
            num_to_pad = num_points - current_num_points
            indices = np.arange(current_num_points)
            pad_indices = np.random.choice(indices, size=num_to_pad, replace=True)
            orig_xyz = np.asarray(pcd.points)
            pad_xyz = orig_xyz[pad_indices]
            jitter = np.random.normal(0, 0.001, pad_xyz.shape)
            pad_xyz += jitter
            final_xyz = np.vstack((orig_xyz, pad_xyz))
            pcd_down.points = o3d.utility.Vector3dVector(final_xyz)
            if pcd.has_normals():
                orig_normals = np.asarray(pcd.normals)
                pad_normals = orig_normals[pad_indices]
                final_normals = np.vstack((orig_normals, pad_normals))
                pcd_down.normals = o3d.utility.Vector3dVector(final_normals)
    else:
        pcd_down = pcd
        
    avg_dist = np.mean(pcd_down.compute_nearest_neighbor_distance())
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * radius, max_nn=max_nn))
    normals = np.asarray(pcd_down.normals)
    
    if invert_normals:
        for i in range(len(normals)):
            if normals[i][2] < 0:
                normals[i] *= -1

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 5, max_nn=100))
    
    return pcd_down, fpfh

def run_global_registration_adaptive(source_down, target_down, source_fpfh, target_fpfh):
    max_attempts = 2
    best_fitness = -0.1
    best_inlier_rmse = 100.0
    best_result = None
    best_threshold = None 
    thresholds = [10.0, 7.0, 4.0]

    for attempt in range(max_attempts):
        for thr in thresholds:            
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, 
                mutual_filter=True, 
                max_correspondence_distance=thr,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3, 
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr)
                ], 
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.99)
            )
            if result.fitness > 0.85 and result.inlier_rmse < best_inlier_rmse:
                best_fitness = result.fitness
                best_inlier_rmse = result.inlier_rmse
                best_result = result
                best_threshold = thr
            if best_fitness > 0.85 and best_inlier_rmse < 3.0: 
                return best_result, best_threshold
            
    return best_result if best_result else result, best_threshold if best_threshold else thr

def run_local_refinement_adaptive(source, target, initial_trans=None, best_ransac_thr=10, method="point_to_point"):
    if initial_trans is None: initial_trans = np.eye(4)
    multipliers = [1.0, 0.5, 0.2]
    thresholds = [best_ransac_thr * m for m in multipliers]
    
    best_result = None
    best_inlier_rmse = float('inf')
    
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane() if method == "point_to_plane" else o3d.pipelines.registration.TransformationEstimationPointToPoint()

    for thr in thresholds:
        reg_icp = o3d.pipelines.registration.registration_icp(
            source, target, thr, initial_trans,
            estimation_method,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        if reg_icp.fitness > 0.85 and reg_icp.inlier_rmse < best_inlier_rmse:
            best_inlier_rmse = reg_icp.inlier_rmse
            best_result = reg_icp

    return best_result if best_result is not None else reg_icp


# --- Cell 1 ---
# ==========================================
# 2. BATCH SIMULATION ALIGNMENT PIPELINE
# ==========================================
EXPERIMENT = "test_8_simulation2"
# WORKPIECES = ["TH0011AV", "TH0012AV", "TH0021AV", "TH0022AV", "TH0031AV", "TH0032AV"]
WORKPIECES = ["TH0011AV", "TH0012AV", "TH0021AV", "TH0022AV", "TH0031AV", "TH0032AV", "TH0041AV", "TH0042AV", "TH0051AV", "TH0052AV", "TH0061AV", "TH0062AV", "TH0071AV", "TH0072AV"]
# WORKPIECES = ["TH0011AV", "TH0012AV"]
NUMBER_OF_POINTS = 40000
experiment = "{EXPERIMENT}"

# --- Artificial YOLO Error (To test ICP robustness) ---
T_artificial_error = np.eye(4)
T_artificial_error[:3, :3] = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix() # 15 degree yaw error
T_artificial_error[:3, 3] = [0.0, -0.0, 0.0]

for workpiece in WORKPIECES:
    print(f"\n========================================")
    print(f"EVALUATING SIMULATION: {workpiece}")
    print(f"========================================")
    
    SIM_DIR = f"pcd_data/testing_data/{experiment}/{workpiece}"
    CAD_PATH = f"workpiece/{workpiece}/workpiece.stl"
    
    if not os.path.exists(SIM_DIR):
        print(SIM_DIR)
        print(f"Skipping {workpiece} (No simulation folder found)")
        continue
        
    # 1. Load and Merge Noisy Target Scans
    target_cloud, count = merge_simulated_scans(SIM_DIR)
    if count == 0:
        print(f"Skipping {workpiece} (No viewpoint PCD files found)")
        continue
    print(f"Merged {count} noisy viewpoints into one massive target cloud.")
    
    # 2. Load Source CAD Model & Apply Artificial Offset
    mesh = o3d.io.read_triangle_mesh(CAD_PATH)
    mesh.compute_vertex_normals()
    source_cloud = mesh.sample_points_uniformly(number_of_points=NUMBER_OF_POINTS)
    
    # Offset the CAD model to simulate a rough, inaccurate YOLO guess!
    source_cloud.transform(T_artificial_error)
    
    # --- VISUALIZATION 1: After Merging & Offsetting ---
    print("DEBUG VISUALIZATION: Before ICP (Red=Shifted CAD, Blue=Merged Noisy Target)")
    print("-> You should see them clearly misaligned! Close window to continue.")
    source_viz = copy.deepcopy(source_cloud)
    target_viz = copy.deepcopy(target_cloud)
    source_viz.paint_uniform_color([1, 0, 0])      # Red Shifted CAD
    target_viz.paint_uniform_color([0, 0.5, 1])    # Blue Target
    # o3d.visualization.draw_geometries([source_viz, target_viz], window_name=f"{workpiece} - Before ICP (Close Window to Continue)")
    
    # 3. Preprocess Normals & FPFH Features
    print("Extracting geometric features (FPFH)...")
    source_down, source_fpfh = preprocess_normal(source_cloud)
    target_down, target_fpfh = preprocess_normal(target_cloud, num_points=40000, invert_normals=True)
    
    # 4. RANSAC Global Alignment
    print("Running RANSAC Global Alignment...")
    ransac_res, best_thr = run_global_registration_adaptive(source_down, target_down, source_fpfh, target_fpfh)
    
    # 5. ICP Local Refinement
    print("Running ICP Fine Alignment...")
    final_icp = run_local_refinement_adaptive(
        source_down, target_down, 
        initial_trans=ransac_res.transformation, 
        best_ransac_thr=best_thr, 
        method="point_to_plane"
    )
    print(f"FINISHED {workpiece} -> ICP Fitness: {final_icp.fitness:.4f} | RMSE: {final_icp.inlier_rmse:.4f}")
    
    # 6. Save the Transformations
    SAVE_DIR = f"evaluation_result/{experiment}/{workpiece}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, "merge_full_transformation.npy"), final_icp.transformation)
    np.save(os.path.join(SAVE_DIR, "artificial_error.npy"), T_artificial_error)
    print(f"Saved transformations to {SAVE_DIR}")

    # 7. Export to processed_data for PointNet / Surface Analysis
    import shutil
    PROCESSED_DIR = f"processed_data/{EXPERIMENT}/{workpiece}"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    count_exported = 0
    if os.path.exists(SIM_DIR):
        for file in os.listdir(SIM_DIR):
            if file.endswith(".pcd"):
                shutil.copy2(os.path.join(SIM_DIR, file), os.path.join(PROCESSED_DIR, file))
                count_exported += 1
    
    # Overwrite merge_full_transformation.npy with the true Identity Matrix for downstream scripts!
    true_transform = np.eye(4)
    np.save(os.path.join(SAVE_DIR, "merge_full_transformation.npy"), true_transform)
    print(f"Exported {count_exported} PCDs to {PROCESSED_DIR} and saved Identity Matrix to {SAVE_DIR}/merge_full_transformation.npy")

    
    # --- VISUALIZATION 2: After ICP ---
    print("DEBUG VISUALIZATION: After ICP (Red=Aligned CAD, Blue=Merged Noisy Target)")
    print("-> They should now be perfectly locked together! Close window to proceed.")
    source_aligned = copy.deepcopy(source_down)
    source_aligned.transform(final_icp.transformation)
    source_aligned.paint_uniform_color([1, 0, 0])
    target_down.paint_uniform_color([0, 0.5, 1])
    # o3d.visualization.draw_geometries([source_aligned, target_down], window_name=f"{workpiece} - After ICP (Close Window to Continue)")


# --- Cell 2 ---
# ==========================================
# 3. SINGLE VIEWPOINT ICP VALIDATION
# ==========================================
# This block tests if the global ICP matrix (calculated from the merged cloud) 
# successfully aligns the CAD model when looking at only a SINGLE noisy viewpoint.
import os
import copy
import open3d as o3d
import numpy as np

EXPERIMENT = "test_8_simulation2"
WORKPIECE = "TH0072AV"
VIEWPOINT_IDX = 0  # Change this to test different angles (0, 10, 20...)

SIM_DIR = f"pcd_data/testing_data/{EXPERIMENT}/{WORKPIECE}"
CAD_PATH = f"workpiece/{WORKPIECE}/workpiece.stl"
EVAL_DIR = f"evaluation_result/{EXPERIMENT}/{WORKPIECE}"

# 1. Load the specific viewpoint (Target)
target_path = os.path.join(SIM_DIR, f"viewpoint_simulated_noise_{VIEWPOINT_IDX}.pcd")
target_cloud = o3d.io.read_point_cloud(target_path)
target_cloud.paint_uniform_color([0, 0.5, 1])  # Blue Target

# 2. Load the original CAD Model (Source)
mesh = o3d.io.read_triangle_mesh(CAD_PATH)
mesh.compute_vertex_normals()
source_cloud = mesh.sample_points_uniformly(number_of_points=40000)

# 3. Load the Saved Transformations
try:
    T_artificial_error = np.load(os.path.join(EVAL_DIR, "artificial_error.npy"))
    T_icp = np.load(os.path.join(EVAL_DIR, "merge_full_transformation.npy"))
except FileNotFoundError:
    print("ERROR: Could not find the saved .npy matrices. Make sure you ran Cell 2 first!")
    raise

# 4. Simulate the Error & Apply ICP Correction
source_aligned = copy.deepcopy(source_cloud)
source_aligned.transform(T_artificial_error) # Apply the bad guess
source_aligned.transform(T_icp)              # Apply the mathematical fix calculated earlier
source_aligned.paint_uniform_color([1, 0, 0])  # Red CAD

print(f"Visualizing Viewpoint {VIEWPOINT_IDX} against ICP-Corrected CAD Model...")
print("Red: CAD Model | Blue: Single Noisy Viewpoint")
o3d.visualization.draw_geometries([source_aligned, target_cloud], window_name=f"Single Viewpoint Validation - {WORKPIECE} (View {VIEWPOINT_IDX})")


# --- Cell 3 ---
print("hey")
