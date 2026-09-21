# Robotic Scanning & Viewpoint Optimization Pipeline

Next-Best-View (NBV) planning and 6D pose estimation framework. Optimizes robotic camera viewpoints by balancing **Surface Coverability** (submodular gain) and **Sensor Quality** (Chamfer Distance noise).

---

## 1. Simulation Workflow (Synthetic)

1. **Viewpoint Generation & Raycasting** (`1_3_viewpoint_generation_manual.ipynb`)
   - Generates candidate camera poses around the CAD model on a hemisphere/dome.
   - Computes visible point clouds for each view using Hidden Point Removal (HPR).
   - Outputs `pcd_all.pcd`, `covered_indices.json`, and `viewpoint_pose_*.npy`.

2. **Sensor Noise Simulation** (`9_simulation_2.ipynb`)
   - Computes Angle of Incidence (AoI) and sensor distance ($Z$) for each raycasted point.
   - Applies physics-based Gaussian noise, depth variance, and specular dropout.
   - Outputs `viewpoint_simulated_noise_*.pcd`.

3. **Simulation Scan Alignment** (`4_2_process_pcd_simulation.ipynb`)
   - Merges noisy simulated scans and runs RANSAC global alignment + adaptive ICP against CAD.
   - Saves ground truth transformation matrix `merge_full_transformation.npy`.
   - Copies point clouds to `processed_data/`.

4. **Surface Feature Analysis** (`6_surface_analysis.ipynb`)
   - Uses Voronoi Nearest-Neighbor segmentation to partition scans into target features (`feature0`, `feature1`, etc.).
   - Computes exact Chamfer Distance error between noisy crops and ideal raycasted crops.
   - Exports feature point clouds and saves `metadata.csv`.

5. **PointNet Model Training & Inference** (`pointnet_pytorch_reflective/`)
   - Trains PointNet Mixture of Experts (MoE) on 7D features ($XYZ + \text{Normals} + \text{AoI}$) to predict Chamfer Distance noise.
   - Outputs `inference_results_moe.csv`.

6. **Viewpoint Optimization** (`7_optimization.ipynb`)
   - Runs Greedy or GRASP algorithm to select the best sequence of viewpoints based on Coverability ($\alpha$) and predicted Confidence ($\beta$).
   - Outputs optimal viewpoint sequence (e.g. 1, 2, 3, 4, 8, or 12 views).

7. **Multi-View Pose Estimation** (`10_pose_estimation.ipynb`)
   - Registers multi-view scans from the selected sequence to the CAD model.
   - Evaluates 6D pose accuracy using the Average Distance of Model Points (ADD) metric in mm.

---

## 2. Real Robot Data Workflow (Physical Scans)

1. **OptiTrack Ground Truth Calibration** (`11_optitrack_groundtruth.ipynb`)
   - Defines the Robot Base coordinate origin $(0,0,0)$ from physical marker clusters.
   - Computes the true ground truth workpiece pose matrix (`T_optitrack.npy` / `T_average.npy`).

2. **Physical Robot Multi-View Scanning** (`2_1_robot_multiview_hemisphere.py` / `2_2_robot_multiview_ovp.py`)
   - Commands the robot arm along candidate viewpoint trajectories.
   - Triggers RealSense depth camera captures and saves raw point clouds to `pcd_data/testing_data/{EXPERIMENT}/view*.pcd`.

3. **Real Scan Preprocessing & Alignment** (`4_3_process_pcd_real.ipynb`)
   - Removes table plane using RANSAC and crops workpiece using an Oriented Bounding Box (OBB).
   - Transforms scans directly into the CAD coordinate frame using inverted OptiTrack ground truth ($T_{inv\_gt}$).
   - Exports CAD-aligned point clouds to `processed_data/{EXPERIMENT}/{workpiece}/`.

4. **Real Surface Analysis & Error Calculation** (`6_surface_analysis.ipynb`)
   - Partitions real scanned features using Voronoi segmentation.
   - Calculates physical Chamfer Distance error against raycasted visible CAD surfaces.
   - Generates real ground truth `metadata.csv`.

5. **Real Data Viewpoint Optimization** (`7_optimization.ipynb`)
   - Runs Greedy or GRASP optimization using real measured errors (`USE_GROUND_TRUTH = True`) or PointNet predictions.
   - Automatically shifts CAD origin offset (e.g. $-11.2\text{ mm}$ for `workpiece31`) so Voronoi features match the scanned 40k cloud.

6. **Step-by-Step Real Pose Estimation** (`10_1_pose_estimation_real.ipynb`)
   - **Step 1 (Before ICP)**: Visualizes Initial Scanned Points (Blue) vs YOLO Guess (Orange) vs OptiTrack Ground Truth (Green).
   - **Step 2 (Coarse-to-Fine ICP)**: Executes 3-stage ICP registration ($10\text{ mm} \to 5\text{ mm} \to 2\text{ mm}$) to align YOLO guess to scans.
   - **Step 3 (After ICP)**: Visualizes final registered CAD (Red) vs Ground Truth (Green) and reports final ADD accuracy in mm.

---

## 3. File Breakdown & Block Classification

| File | Workflow | Purpose | Block Types Available |
|---|---|---|---|
| `1_3_viewpoint_generation_manual.ipynb` | Simulation | Camera pose generation & HPR raycasting | • **Debug**: Single viewpoint POV raycast<br>• **Batch**: All simulation workpieces (`TH0011AV`–`TH0072AV`) |
| `2_1_robot_multiview_hemisphere.py`<br>`2_2_robot_multiview_ovp.py` | Real | Physical scanning execution | • **Execution**: Robot motion + RealSense capture |
| `4_2_process_pcd_simulation.ipynb` | Simulation | Synthetic scan alignment & processing | • **Debug**: Single viewpoint ICP verification<br>• **Batch**: Full workpiece RANSAC+ICP baseline processing |
| `4_3_process_pcd_real.ipynb` | Real | Real scan cleaning & CAD frame alignment | • **Debug**: 2-window crop & OptiTrack alignment check<br>• **Batch**: Batch plane removal, crop, and CAD transform for all views |
| `6_surface_analysis.ipynb` | Sim & Real | Voronoi feature extraction & Chamfer Distance | • **Debug**: 2-window Voronoi & raycasted ideal vs actual scan<br>• **Batch**: Full dataset feature extraction & `metadata.csv` export |
| `7_optimization.ipynb` | Sim & Real | Next-Best-View (NBV) Viewpoint Optimizer | • **Debug**: Top/bottom 5 candidate preview & 3D camera pose visualizer<br>• **Single Workpiece**: Greedy/GRASP sequence run<br>• **Batch**: Multi-workpiece & multi-budget ($1, 2, 3, 4, 8, 12$ views)<br>• **Benchmark**: Manual / grid sequence evaluation |
| `7_1_alpha_beta_tuning.ipynb` | Sim & Real | $\alpha$/$\beta$ weight hyperparameter sweep | • **Batch**: Pareto frontier curve generation |
| `9_simulation_2.ipynb` | Simulation | Physics-based sensor noise simulator | • **Debug**: Single view AoI noise inspection<br>• **Batch**: Multi-workpiece noise generator & voxel averaging |
| `10_pose_estimation.ipynb` | Simulation | Multi-view 6D pose estimation & ADD | • **Batch**: Multi-workpiece ADD evaluation<br>• **Benchmark**: Exhaustive single-viewpoint sensitivity scan |
| `10_1_pose_estimation_real.ipynb` | Real | Step-by-step real robot scan pose estimation | • **Debug (Step 1)**: Visualizes Initial State (YOLO vs GT vs Scan)<br>• **Execution (Step 2)**: 3-stage coarse-to-fine ICP ($10 \to 5 \to 2\text{ mm}$)<br>• **Debug (Step 3)**: Final overlay visualizer + ADD readout |
| `11_optitrack_groundtruth.ipynb` | Real | Ground truth OptiTrack coordinate calibration | • **Execution**: Computes Robot Base origin & workpiece pose matrix |

---

## 4. Key Tips
- **CAD Origin Shift**: Real scans (`workpiece31`) have an OptiTrack-centered origin ($Z \in [-11.2, +28.8]\text{ mm}$). `7_optimization.ipynb` automatically shifts CAD features by $[0, 0, -11.2]\text{ mm}$ to match.
- **Chamfer Distance Reference**: In `6_surface_analysis.ipynb`, Chamfer Distance is measured against the **visible raycasted CAD surface**, not the full unraycasted 360° mesh.
- **Surface Names**: Scripts automatically support both `feature*.stl` and `surface*.stl`.
