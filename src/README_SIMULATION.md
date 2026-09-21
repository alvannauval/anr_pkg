# Simulation & Viewpoint Planning Pipeline

> [!NOTE]
> For the complete documentation covering both **Simulation and Real Workflows** and file breakdowns, see **`README.md`**.

---

## 1. Concept

We optimize robotic scanning viewpoints by predicting scanning noise (Chamfer Distance) *before* physical execution:
1. **Simulation**: Virtual camera poses are raycasted (HPR), then physics noise (AoI + depth) is added (`9_simulation_2.ipynb`).
2. **Feature Extraction**: Features are cropped via Voronoi segmentation and Chamfer error is computed (`6_surface_analysis.ipynb`).
3. **PointNet Training**: Regression model learns to predict noise from perfect 7D CAD features ($XYZ + \text{Normals} + \text{AoI}$).
4. **Optimization**: Greedy / GRASP selects the best viewpoint sequence balancing Coverability and Quality (`7_optimization.ipynb`).
5. **Pose Estimation**: Multi-view scans are registered to the CAD model to evaluate ADD accuracy in mm (`10_pose_estimation.ipynb`).

---

## 2. Sequential Execution

1. `1_3_viewpoint_generation_manual.ipynb`: Viewpoint generation & raycasting.
2. `9_simulation_2.ipynb`: Physics-based noise simulation.
3. `4_2_process_pcd_simulation.ipynb`: RANSAC/ICP baseline alignment.
4. `6_surface_analysis.ipynb`: Feature cropping & `metadata.csv` generation.
5. `pointnet_pytorch_reflective/`: PointNet MoE model training & inference.
6. `7_optimization.ipynb`: Viewpoint sequence optimization.
7. `10_pose_estimation.ipynb`: Multi-view 6D pose estimation & ADD benchmarking.
