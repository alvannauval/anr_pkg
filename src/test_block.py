import open3d as o3d
import numpy as np

feature_path = "surface/surface1.stl"
feature_mesh = o3d.io.read_triangle_mesh(feature_path)
feature_mesh.compute_vertex_normals()
feature_pcd = feature_mesh.sample_points_poisson_disk(number_of_points=1000)
feature_pcd.paint_uniform_color([1, 0, 0])

distance_threshold = 2
visualization = []

pcd_path = r"processed_data/test_3_dofconsistency_15/viewpoint_simulated_4.pcd"
actual_pcd = o3d.io.read_point_cloud(pcd_path)
actual_pcd.paint_uniform_color([0, 1, 0])

transform_path = r"evaluation_result/test_3_dofconsistency_15/merge_full_transformation.npy"
try:
    transformation_matrix = np.load(transform_path)
except Exception as e:
    print(f"Error loading {transform_path}: {e}")
    exit(1)

inverse_transformation = np.linalg.inv(transformation_matrix)
actual_pcd.transform(inverse_transformation)

dists = actual_pcd.compute_point_cloud_distance(feature_pcd)
dists = np.asarray(dists)

intersection_indices = np.where(dists < distance_threshold)[0]
intersection_pcd = actual_pcd.select_by_index(intersection_indices)

intersection_pcd.paint_uniform_color([0, 0, 1])

print(f"Found {len(intersection_indices)} intersecting points!")
print("Success!")
