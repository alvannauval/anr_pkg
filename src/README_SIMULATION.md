# Simulation and Viewpoint Planning Pipeline

This document outlines the simulation, data generation, and training pipeline used to evaluate and predict point cloud scanning errors. 

## 1. How It Works

The core objective of this project is to optimize robotic scanning viewpoints by predicting the expected scanning noise (Chamfer Distance) *before* the physical scan is executed.

To achieve this, the pipeline relies on physics-based noise simulation and neural network regression:
1. **Simulation**: We take a perfect CAD model and generate a series of simulated viewpoints from a virtual camera. Physics-based noise (such as Gaussian dropout based on the Angle of Incidence) is applied to create a "Noisy Scan".
2. **Feature Extraction**: We isolate specific geometric features (e.g., `surface0.stl`) from both the Perfect CAD raycast and the Noisy Scan.
3. **Error Calculation**: We compute the Chamfer Distance between the Noisy feature and the Perfect CAD feature to act as our ground truth error metric.
4. **PointNet Training**: A PointNet regression model is trained. Crucially, the **Input** to the model is the *Noiseless, Perfect CAD Feature* (augmented with 7-dimensional data: `XYZ + Normals + Angle of Incidence`). The **Output** target is the calculated *Chamfer Distance*. The network learns to look at a perfect mathematical shape and predict how much noise a sensor will generate from a specific viewing angle.

---

## 2. How To (Execution Pipeline)

To generate the dataset from scratch and train the model, execute the following Jupyter Notebooks in sequential order.

### Phase 1: Viewpoint Generation
**Notebook**: `1_3_generate_viewpoint.ipynb` (or similar)
- **Purpose**: Generates the initial candidate viewpoints (camera poses) around the target CAD workpiece.
- **Action**: Defines the virtual camera sphere/dome around the workpiece and exports the transformation matrices representing where the camera will look from.

### Phase 2: Viewpoint Simulation
**Notebook**: `9_simulation.ipynb`
- **Purpose**: Simulates the physical robot scanning process over the generated viewpoints.
- **Action**: 
  - Takes the camera poses from Phase 1.
  - Emits virtual raycasts against the CAD model to generate perfect point clouds.
  - Applies physics-based dropouts and noise (based on Angle of Incidence) to generate the "Noisy Simulated Scans".

### Phase 3: Processing and ICP Alignment
**Notebook**: `4_2_process_pcd_simulation.ipynb`
- **Purpose**: Processes the raw simulated scans and aligns them perfectly to the CAD models.
- **Action**: 
  - Iterates through the simulated noisy viewpoints for the workpieces.
  - Merges the noisy viewpoints into a dense target cloud.
  - Performs RANSAC and ICP (Iterative Closest Point) fine alignment against the CAD geometry.
  - Generates the Ground-Truth transformation matrices (`merge_full_transformation.npy`) which will be used later for evaluating actual scans.
- **NOTE**: 216 Viewpoints * 6 Workpieces = 1296 viewpoints takes around 10 minutes.


### Phase 4: Surface Analysis and Feature Cropping
**Notebook**: `6_surface_analysis.ipynb`
- **Purpose**: Crops specific CAD features and generates the final `metadata.csv` required for PointNet.
- **Action**: 
  - Loads both the *Perfect CAD raycast* and the *Noisy Simulated Scan*.
  - Extracts the targeted surfaces (e.g., `surface0`, `surface1`) from the full point clouds.
  - Calculates the Chamfer Distance between the noisy crop and the perfect crop.
  - Exports the **Perfect CAD crops** (`viewpoint_simulated_X_surfaceY.pcd`) to the `processed_data/` directory.
  - Saves the mapping between the exported filenames and the Chamfer Distances into `metadata.csv` (using lowercase `filename`).
- **NOTE**: 216 Viewpoints * 6 Workpieces * 2 Surface = 2592 Samples takes around 3 minutes.
- **Visualization**: The final cell allows you to interactively compare the Noisy Simulated Intersect (Blue) against the Perfect Simulated Intersect (Green) to visually understand the physical noise.

### Phase 5: Model Training (Mixture of Experts)
**Notebook**: `1_training.ipynb` (located in `pointnet_pytorch_reflective/`)
- **Purpose**: Ingests the `metadata.csv` dataset and trains the PointNet Regression Model.
- **Action**:
  - **Cell 1 (Visualization)**: Allows you to interactively visualize the generated feature crops. Window 1 shows the exact cropped Perfect CAD feature colored by Angle of Incidence (what the network sees). Window 2 shows the full noisy scan context.
  - **Cell 2 (Training)**: The dataloader reads `metadata.csv`. To solve variance imbalance / Simpson's paradox, we utilize a **Mixture of Experts** approach by filtering the dataset using `TARGET_SURFACE_FILTER` to train isolated networks for specific feature geometries (e.g., flat surface vs pocket). It applies a `WeightedRandomSampler` to balance regression targets, extracts Angle of Incidence, normalizes spatially, and begins the AdamW optimization loop.

### Phase 6: Inference & Analytics
**Notebook**: `2_inference.ipynb` (located in `pointnet_pytorch_reflective/`)
- **Purpose**: Evaluates the trained PointNet expert models and visualizes predictive accuracy.
- **Action**:
  - Runs the trained `.pth` model against testing data to generate `inference_results.csv`.
  - **Histogram Analysis**: Plots the Chamfer Value distribution with clear ±1σ Standard Deviation markers.
  - **Parity Plots**: Plots Ground Truth Chamfer Distance vs Predicted Chamfer Distance, complete with best-fit linear regression lines. Supports filtering by surface or visualizing multiple workpieces simultaneously (`TARGET_WORKPIECES` list).


### Phase 7: Viewpoint Optimization (Next Best View)
**Notebook**: 7_optimization.ipynb
- **Purpose**: Uses predicted (or Ground Truth) Chamfer Distances and geometric coverage data to plan the mathematically optimal sequence of robotic camera viewpoints.
- **Action**:
  - **Data Ingestion**: Loads the predicted Chamfer Distances (`metadata.csv`) and the raycasted visibility data (`covered_indices.json`).
  - **Geometric Mapping (Voronoi)**: Utilizes a Voronoi Nearest-Neighbor mapping with a fixed `DISTANCE_THRESHOLD` (e.g., 2.0mm) to cleanly isolate target features. If a `background.stl` is provided, it intelligently absorbs the rest of the object to prevent boundary bleeding. Points assigned to the background are automatically discarded from optimization.
  - **Step 1: Baseline Confidence (The "Quality" Score)**: 
    - The algorithm calculates a **Point-Weighted Average** for every camera based on the Chamfer Distances of the features it sees (heavily weighting the score towards features that physically dominate the view).
    - It immediately calculates a `Filter Score = (Total Points Seen / Total Object Points) * Confidence` to filter out the worst 80% of candidates before optimization even begins, ensuring cameras must have both good accuracy *and* decent coverage to compete.
    - Surviving candidates have their Confidence scores normalized from `0.0` to `1.0`.
  - **Step 2: Submodular Coverability (The "Information Gain" Score)**:
    - At every step in the GRASP sequence, the algorithm analyzes the exact point indices a candidate camera covers.
    - It applies a **Submodular Decay Function** (`GAMMA = 0.5`) to encourage looking at new geometry:
      - If a point has never been seen: **1.0 points**
      - If a point was seen 1 time by previous cameras: **0.5 points**
      - If a point was seen 2 times by previous cameras: **0.25 points**
    - The sum of these values is divided by the total object points to create a dynamically updating `Coverability` score (0 to 1).
  - **Step 3: MOOP Utility Score**:
    - The algorithm selects the "Next Best View" by maximizing the combined Multi-Objective Utility function:
      `Utility Score = (ALPHA * Coverability) + (BETA * Confidence)`
  - **Manual Evaluation**: Contains a Reverse Engineering block to compare the physical error of manually selected camera grids against the GRASP-optimized sequence.
