# Autonomous Robotic Scanning & Viewpoint Optimization Pipeline (`anr_pkg`)

Next-Best-View (NBV) planning and 6D pose estimation framework.

---

## 1. Simulation Workflow (Synthetic)
1. **Viewpoint Generation & Raycasting** (`src/1_3_viewpoint_generation_manual.ipynb`)
2. **Sensor Noise Simulation** (`src/9_simulation_2.ipynb`)
3. **Simulation Scan Alignment** (`src/4_2_process_pcd_simulation.ipynb`)
4. **Surface Feature Analysis & Chamfer Error** (`src/6_surface_analysis.ipynb`)
5. **PointNet Model Training & Inference** (`pointnet_pytorch_reflective/`)
6. **Viewpoint Optimization** (`src/7_optimization.ipynb`)
7. **Multi-View 6D Pose Estimation & ADD** (`src/10_pose_estimation.ipynb`)

---

## 2. Real Robot Data Workflow (Physical Scans)
1. **OptiTrack Ground Truth Calibration** (`src/11_optitrack_groundtruth.ipynb`)
2. **Physical Robot Scanning** (`src/2_1_robot_multiview_hemisphere.py` / `src/2_2_robot_multiview_ovp.py`)
3. **Real Scan Preprocessing & CAD Frame Alignment** (`src/4_3_process_pcd_real.ipynb`)
4. **Real Surface Analysis & Physical Error** (`src/6_surface_analysis.ipynb`)
5. **Real Data Viewpoint Optimization** (`src/7_optimization.ipynb`)
6. **Step-by-Step Real Pose Estimation & ADD Validation** (`src/10_1_pose_estimation_real.ipynb`)

---

## 📖 Full Documentation
For the complete file breakdowns and block modes (Debug vs Batch vs Real vs Sim), see:
👉 **`src/README.md`**
