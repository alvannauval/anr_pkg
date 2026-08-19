#!/usr/bin/env python3

import profile
import sys
import os
import math
import time
import cv2
import re
import numpy as np
import open3d as o3d
import rospy
import tf
import tf2_ros
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rospy
import tf
import tf2_ros
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
from std_msgs.msg import Float64MultiArray


def init_robot(robot_id="dsr01", model="m1013"):
    """Sets up the Doosan Robot API parameters."""
    DR_init.__dsr__id = robot_id
    DR_init.__dsr__model = model
    rospy.loginfo(f"Robot {robot_id} ({model}) initialized.")


class RealSenseROSManager:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None
        
        # Subscribe to Camera Info once to get intrinsics
        rospy.loginfo("Waiting for camera info...")
        self.camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        rospy.loginfo("Camera info received.")

        # Set up synchronized subscribers
        self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

    def sync_callback(self, color_msg, depth_msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # Usually aligned depth is 16UC1 in mm
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def wait_for_frames(self):
        """Wait until a new synchronized pair of frames is available."""
        self.latest_color = None
        self.latest_depth = None
        while not rospy.is_shutdown() and (self.latest_color is None or self.latest_depth is None):
            rospy.sleep(0.01)
        return self.latest_color, self.latest_depth


def get_robust_depth(depth_data, x, y, window_size=5):
    """
    Calculates the average depth in a window around (x, y) to avoid noise.
    depth_data is assumed to be in millimeters (16UC1).
    Returns depth in meters.
    """
    half_w = window_size // 2
    
    # Define the bounding box for the window (ROI)
    y_start, y_end = max(0, int(y)-half_w), min(depth_data.shape[0], int(y)+half_w+1)
    x_start, x_end = max(0, int(x)-half_w), min(depth_data.shape[1], int(x)+half_w+1)
    
    roi = depth_data[y_start:y_end, x_start:x_end]

    # Filter out zero values (invalid depth)
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) > 0:
        # Return mean depth converted to meters
        return np.mean(valid_depths) * 0.001
    else:
        return 0  # No valid depth found


def get_yolo_detection(rs_manager, model, target_class=None):
    global results
    """Detects object via YOLO OBB and returns camera-space coordinates."""
    print("Waiting for YOLO detection... Press 'q' to confirm.")
    
    # If target_class is provided as a string (e.g., "workpiece"), find its integer ID
    if isinstance(target_class, str):
        found_class = next((k for k, v in model.names.items() if v.lower() == target_class.lower()), None)
        if found_class is None:
            print(f"Error: Class '{target_class}' not found in the model. Available: {model.names}")
            return None, None
        target_class = found_class
    
    cx = rs_manager.camera_info.K[2]
    cy = rs_manager.camera_info.K[5]
    fx = rs_manager.camera_info.K[0]
    fy = rs_manager.camera_info.K[4]
    
    cv2.namedWindow("Detection (Press q)", cv2.WINDOW_NORMAL)
    
    while not rospy.is_shutdown():
        color_img, depth_img = rs_manager.wait_for_frames()
        
        # By passing classes=[target_class], YOLO strictly ignores all other objects
        if target_class is not None:
            results = model(color_img, conf=0.7, classes=[target_class])
        else:
            results = model(color_img, conf=0.7)
            
        if results[0].obb is not None and len(results[0].obb) > 0:
            # Since YOLO already filtered out everything else, index 0 is guaranteed
            # to be the highest confidence detection of your target_class
            box = results[0].obb[0]
                
            px, py, _, _, rotation = box.xywhr.cpu().numpy()[0]
            
            # --- Robust Depth Calculation ---
            # Define an offset for depth calculation (e.g., to avoid a hole in the center)
            offset_x = 0  # Adjust this value as needed
            offset_y = -30  # Adjust this value as needed
            depth_px = px + offset_x
            depth_py = py + offset_y
            
            # Temporal filtering is handled by the ROS node natively.
            dist = get_robust_depth(depth_img, depth_px, depth_py)
            
            if dist > 0:
                # Calculate 3D coordinates based on the original YOLO center (red dot), but use the depth from the offset point (green dot)
                X = (px - cx) * dist / fx
                Y = (py - cy) * dist / fy
                Z = dist
                cam_pts = [X, Y, Z]
                
                cv_frame = results[0].plot()
                # Red circle: Original YOLO bounding box center
                cv2.circle(cv_frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                # Green circle: Point where estimated depth is actually calculated
                cv2.circle(cv_frame, (int(depth_px), int(depth_py)), 5, (0, 255, 0), -1)
                
                cv2.imshow("Detection (Press q)", cv_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return [c * 1000 for c in cam_pts], np.degrees(box.xywhr.cpu().numpy()[0][4])
                elif key == 27: # ESC key
                    cv2.destroyAllWindows()
                    print("Process cancelled by user.")
                    sys.exit(0)
                
        else:
            print("Searching for the object")
                

def calculate_look_at_zyz(camera_pos, target_pos):
    """Calculates ZYZ Euler angles to orient camera toward the target."""
    z_axis = np.array(target_pos) - np.array(camera_pos)
    z_axis /= (np.linalg.norm(z_axis) + 1e-6)
    
    up = np.array([0, 1, 0]) if abs(z_axis[2]) > 0.9 else np.array([0, 0, 1])
    x_axis = np.cross(up, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    return R.from_matrix(rot_matrix).as_euler('zyz', degrees=True)

    
def capture_scan_view(rs_manager, T_base_camera, index, save_dir="pcd_data", duration=1.0, bbox_center=None, bbox_size=(150, 150, 200)):
    """
    Captures frames, merges PCD, and saves a side-by-side RGB+Depth visualization.
    Normalization ensures the depth image isn't just a solid blue block.
    If bbox_center is provided, it crops the point cloud to the 3D bounding box before merging.
    """    
    all_points = []
    last_color_image = None
    last_depth_data = None
    start_time = time.time()
    count = 0
    
    cx = rs_manager.camera_info.K[2]
    cy = rs_manager.camera_info.K[5]
    fx = rs_manager.camera_info.K[0]
    fy = rs_manager.camera_info.K[4]
    
    print(f"Scanning Viewpoint {index} for {duration}s...")

    while (time.time() - start_time) < duration:
        count += 1
        color_img, depth_img = rs_manager.wait_for_frames()
            
        # Store for visualization (most recent frame)
        last_depth_data = depth_img
        last_color_image = color_img

        # Logic to reduce point density: process every 2nd frame (prev 5)
        if count % 2 != 0:
            continue 

        # 1. Calculate Point Cloud (Vectorized)
        H, W = depth_img.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # valid depth is > 0
        valid = depth_img > 0
        z = depth_img[valid] * 0.001 # convert mm to meters for xyz calculation
        u = u[valid]
        v = v[valid]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        verts = np.stack((x, y, z), axis=-1) * 1000.0 # Convert back to mm for processing
        
        # 2. Transform to Base Frame
        # T_base_camera must be the 4x4 matrix from important_2 logic
        verts_base = (T_base_camera @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        
        # 3. Crop to Bounding Box
        if bbox_center is not None:
            bx, by, bz = bbox_center
            sx, sy, sz = bbox_size
            mask = (
                (verts_base[:, 0] >= bx - sx/2) & (verts_base[:, 0] <= bx + sx/2) &
                (verts_base[:, 1] >= by - sy/2) & (verts_base[:, 1] <= by + sy/2) &
                (verts_base[:, 2] >= bz - sz/2) & (verts_base[:, 2] <= bz + sz/2)
            )
            verts_base = verts_base[mask]

        if len(verts_base) > 0:
            all_points.append(verts_base)

    if len(all_points) == 0:
        print("Error: No data captured!")
        return

    # --- VISUALIZATION PROCESSING ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare RGB (BGR for OpenCV)
    # color_bgr = cv2.cvtColor(last_color_image, cv2.COLOR_RGB2BGR)

    # Prepare Normalized Depth (The "Blue Image" Fix)
    depth_mask = last_depth_data > 0
    if np.any(depth_mask):
        d_min = np.min(last_depth_data[depth_mask])
        d_max = np.max(last_depth_data[depth_mask])
        
        # Normalize to 0-255 range based on the distance of your object
        depth_norm = (last_depth_data - d_min) / (d_max - d_min + 1e-6)
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        # Make invalid/reflective holes pure black
        depth_viz[~depth_mask] = [0, 0, 0]
    else:
        depth_viz = np.zeros_like(last_color_image)

    # Create Side-by-Side image
    side_by_side = np.hstack((last_color_image, depth_viz))
    
    # Save Image
    viz_path = os.path.join(save_dir, f"view{index}_viz.png")
    cv2.imwrite(viz_path, side_by_side)

    # --- POINT CLOUD PROCESSING ---
    merged_verts = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_verts)
    
    # Voxel downsampling (2mm)
    pcd = pcd.voxel_down_sample(voxel_size=0.5) # at 350 mm working distance, the object resolution is 0.52 mm

    # Save PCD and TF
    pcd_path = os.path.join(save_dir, f"view{index}.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    np.save(os.path.join(save_dir, f"view{index}_tf.npy"), T_base_camera)

    print(f"Saved View {index}: {len(pcd.points)} points. Visualization: {viz_path}")


def pose_to_matrix(pose, zyz=False):
    """
    Converts [x, y, z, roll, pitch, yaw] in degrees to a 4x4 transformation matrix.
    If zyz=True, assumes the input is in ZYZ Euler Angles, otherwise XYZ Euler Angles.
    
    Args:
    - pose (list or array): [x, y, z, roll, pitch, yaw] in degrees.
    - zyz (bool): If True, interprets the last three values as ZYZ Euler angles. 
                  If False, uses XYZ Euler angles (default).
    
    Returns:
    - T (ndarray): The 4x4 homogeneous transformation matrix.
    """
    x, y, z = pose[:3]   # Translation vector
    roll, pitch, yaw = pose[3:]  # Rotation angles in degrees
    
    if zyz:
        # Convert ZYZ Euler Angles to a 3x3 rotation matrix
        rot_matrix = R.from_euler('zyz', [roll, pitch, yaw], degrees=True).as_matrix()
    else:
        # Convert XYZ Euler Angles to a 3x3 rotation matrix
        rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    # Build the 4x4 Homogeneous Transformation Matrix
    T = np.eye(4)  # Start with the identity matrix (4x4)
    T[:3, :3] = rot_matrix  # Set the upper-left 3x3 part to the rotation matrix
    T[:3, 3] = [x, y, z]    # Set the upper-right 3x1 part to the translation vector
    return T


def matrix_to_pose(matrix, zyz=False):
    """
    Converts a 4x4 homogeneous transform matrix into a 6D pose list.
    
    Args:
        matrix: 4x4 numpy array.
        zyz (bool): If True, returns [x, y, z, alpha, beta, gamma] using ZYZ order.
                   If False, returns [x, y, z, roll, pitch, yaw] using ZYX order.
    """
    # 1. Extract translation and rotation matrix
    x, y, z = matrix[:3, 3]
    rot_matrix = matrix[:3, :3]
    
    if isinstance(zyz, bool) and zyz:
        # Intrinsic ZYZ: R = Rz(alpha) * Ry(beta) * Rz(gamma)
        alpha, beta, gamma = R.from_matrix(rot_matrix).as_euler('ZYZ', degrees=True)
        return [x, y, z, alpha, beta, gamma]
    else:
        # Intrinsic ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        # SciPy returns [yaw, pitch, roll] for 'zyx'
        zyx = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
        # Reorder to [x, y, z, roll, pitch, yaw]
        return [x, y, z, zyx[2], zyx[1], zyx[0]]


def get_tf_matrix(tf_buffer, target, source):
    """
    Gets the transformation matrix between two frames, with translation in millimeters.
    Args:
    - tf_buffer: tf2_ros.Buffer object
    - target (str): The name of the target frame
    - source (str): The name of the source frame
    Returns:
    - tf_matrix (numpy.ndarray): The 4x4 transformation matrix
      Translation is in millimeters, rotation as usual.
    """
    try:
        # 1. Try to get the transform at the EXACT current time
        # This waits up to 2.0s for the TF buffer to receive the robot's new position
        now = rospy.Time.now()
        tf_buffer.can_transform(target, source, now, rospy.Duration(2.0))
        transform = tf_buffer.lookup_transform(target, source, now)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        # 2. If the exact time fails, fallback to the latest available (Time 0)
        transform = tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(1.0))
        
    translation = transform.transform.translation
    rotation = transform.transform.rotation

    # Convert translation to millimeters
    translation_vector = np.array([translation.x, translation.y, translation.z]) * 1000.0
    
    # Convert quaternion to 4x4 rotation matrix
    # Using np.eye(4) ensures a clean matrix structure
    tf_matrix = tf.transformations.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
    
    # Insert translation into the 4x4 matrix
    tf_matrix[:3, 3] = translation_vector
    
    return tf_matrix


def transform_to_cam(t_base2goal, t_link2cam):
    """
    Correctly transforms the target from Camera Goal to Link6 Base.
    Input: 4x4 Homogeneous Matrix (T_base2goal)
    Output: [x, y, z, A, B, C] in Doosan ZYZ
    """
    # 1. Calculate the Link6 position in Base frame
    # T_base2link = T_base2goal @ T_cam2link
    # (Note: np.linalg.inv(T_link2cam) is T_cam2link)
    t_base2link = t_base2goal @ np.linalg.inv(t_link2cam)

    # 2. Convert directly to ZYZ for the Doosan Robot
    target_link6_zyz = matrix_to_pose(t_base2link, zyz=True)

    return target_link6_zyz

    # Testing Functions (on main)
    # # start test
    # T_cam2ob = [obj_cam_pos[0], obj_cam_pos[1], obj_cam_pos[2], 0.0, 180.0, obb_angle]
    # T_base2cam = get_tf_matrix(tf_buffer, source='realsense_RGBframe', target='base_0')
    # T_base2ob = T_base2cam @ pose_to_matrix(T_cam2ob)
    # T_ob2cam_goal = [[-1, 0, 0, 0],
    #                  [0, 1, 0, 0],
    #                  [0, 0, -1, SCAN_HEIGHT],
    #                  [0, 0, 0, 1]]
    # T_base2goal = T_base2ob @ T_ob2cam_goal
    # goal_pose_cam = transform_to_cam(T_base2goal, T_link2cam)
    # # end of test


def zyz_to_rpy(zyz_angles, degrees=True):
    """
    Converts Euler ZYZ (Doosan style) to RPY (XYZ Euler).
    """
    # Create rotation object from ZYZ
    r = R.from_euler('zyz', zyz_angles, degrees=degrees)
    
    # Convert to RPY (extrinsic XYZ or intrinsic xyz depending on your transform_to_cam)
    # Most ROS-based 'RPY' uses 'xyz' (intrinsic) or 'XYZ' (extrinsic).
    rpy = r.as_euler('xyz', degrees=degrees)
    return rpy


def rpy_to_zyz(rpy_angles, degrees=True):
    """
    Converts RPY (XYZ Euler) to Euler ZYZ (Doosan style).
    RPY is usually interpreted as intrinsic xyz or fixed-axis XYZ.
    """
    # Create rotation object from RPY
    # Using 'xyz' (lowercase) denotes intrinsic rotations
    r = R.from_euler('xyz', rpy_angles, degrees=degrees)
    
    # Convert to ZYZ (intrinsic) for Doosan
    zyz = r.as_euler('zyz', degrees=degrees)
    return zyz


def home_robot():
    """Moves the robot to the predefined home joint position."""
    print("Moving to Home position...")
    movej([0, 0, 90, 0, 90, 0], v=30, a=60) 


def capture(index=0):
    T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
    time.sleep(1)
    capture_scan_view(rs_manager, T_current, index, save_dir=pcd_save_dir, duration=1.0, bbox_center=obj_base_pos)


def load_viewpoint_poses(folder_path):
    """
    Loads all .npy pose files from a directory into a list, 
    sorted numerically by the index in the filename.
    """
    # Regex to capture the index number from 'viewpoint_pose_X.npy'
    def extract_number(filename):
        match = re.search(r'viewpoint_pose_(\d+)\.npy', filename)
        return int(match.group(1)) if match else -1

    # Filter for .npy files and sort them (0, 1, 2... instead of 0, 1, 10...)
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    npy_files.sort(key=extract_number)

    # Load data into list
    viewpoint_poses = []
    for file_name in npy_files:
        full_path = os.path.join(folder_path, file_name)
        try:
            pose = np.load(full_path)
            viewpoint_poses.append(pose)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return viewpoint_poses

def get_current_posx_once():
    """Waits for one message from the rostopic and returns the list."""
    try:
        data = rospy.wait_for_message("/dsr01m1013/state/current_posx", Float64MultiArray, timeout=2.0)
        return list(data.data)
    except rospy.ROSException:
        rospy.logwarn("Posx topic timeout!")
        return None


def get_robot_pose_api():
    """Returns the [x, y, z, a, b, c] list directly from the API."""
    # get_current_posx() is part of the DSR_ROBOT module
    pos = get_current_posx() 
    if pos:
        # Doosan API sometimes returns a tuple (posx, sol_space)
        if isinstance(pos, tuple):
            return list(pos[0])
        return list(pos)
    return None

if __name__ == "__main__":
    rospy.init_node('unified_grasp_scan')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    

    # Robot Initialization
    sys.dont_write_bytecode = True
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../doosan-robot/common/imp")))
    import DR_init
    init_robot()
    from DSR_ROBOT import *

    # RealSense Initialization
    rs_manager = RealSenseROSManager()
    model = YOLO("model/workpiece1_OBB.pt")
    # model = YOLO("/home/smslab/catkin_ws/src/seongin/src/yolo11_log/yolo11x_S256_bs32_ep100/weights/best.pt")

    pcd_save_dir = r"pcd_data"
    path = r"viewpoints_candidate"
    
    # Configuration
    ENABLE_RECENTER = False
    TARGET_CLASS = 'Up'  # Set to None if no specific class filtering is needed

    # Detection & Localization
    obj_cam_pos, obb_angle = get_yolo_detection(rs_manager, model, target_class=TARGET_CLASS)
    T_init = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
    obj_base_pos = (T_init @ np.append(obj_cam_pos, 1))[:3]
    
    if ENABLE_RECENTER:
        print("Recentering robot to the workpiece...")
        
        # 1. Start with the current camera pose in base frame
        T_des_cam = np.copy(T_init)
        
        # 2. Update X and Y to match the object's base position
        T_des_cam[0, 3] = obj_base_pos[0]
        T_des_cam[1, 3] = obj_base_pos[1]
        
        # 3. Add yaw rotation based on YOLO obb_angle around the camera's local Z-axis
        rad = np.radians(obb_angle)
        Rz = np.array([
            [np.cos(rad), -np.sin(rad), 0, 0],
            [np.sin(rad),  np.cos(rad), 0, 0],
            [0,            0,           1, 0],
            [0,            0,           0, 1]
        ])
        T_des_cam = T_des_cam @ Rz
        
        # 4. Compute exactly where link6 needs to be to center the rotated camera
        T_link2cam_local = get_tf_matrix(tf_buffer, source='realsense_RGBframe', target='link6')
        T_des_link = T_des_cam @ np.linalg.inv(T_link2cam_local)
        
        # 5. Apply the safe movement keeping Z, Roll, Pitch locked
        current_pose = get_robot_pose_api()
        if current_pose:
            new_pose = list(current_pose)
            
            # Update X and Y to center the camera
            new_pose[0] = T_des_link[0, 3]
            new_pose[1] = T_des_link[1, 3]
            
            # Explicitly lock Z so it cannot move up or down
            new_pose[2] = current_pose[2]
            
            # Add the yaw rotation directly to the robot's C angle
            new_pose[5] += obb_angle
            
            movel(new_pose, v=100, a=200)
            time.sleep(1.0)
            
            print("Performing secondary scan after recentering...")
            obj_cam_pos, obb_angle = get_yolo_detection(rs_manager, model, target_class=TARGET_CLASS)
            T_init = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
            obj_base_pos = (T_init @ np.append(obj_cam_pos, 1))[:3]

    obj_base_pose = [obj_base_pos[0], obj_base_pos[1], obj_base_pos[2], 0.0, obb_angle, 0.0]
    np.save("pcd_data/initial_obj_pose.npy", obj_base_pose)
    
    print("obb_angle: ", obb_angle)

    viewpoint_poses = load_viewpoint_poses(path)

    T_link2cam = get_tf_matrix(tf_buffer, source='realsense_RGBframe', target='link6') 
    T_base2cam = get_tf_matrix(tf_buffer, source='realsense_RGBframe', target='base_0') # Changes overtime


    T_cam2ob = [obj_cam_pos[0], obj_cam_pos[1], obj_cam_pos[2], 0.0, 180.0, obb_angle]
    T_base2ob_yolo = T_base2cam @ pose_to_matrix(T_cam2ob)
    T_yolo2origin = np.array([[1, 0, 0,  0],
                              [0, 1, 0,  0],
                              [0, 0, 1,  -11.197], #-4.8 gatau ap # -8 #wp31 11.197
                              [0, 0, 0,  1]])
    
    np.save(os.path.join(pcd_save_dir, f"T_base2ob_yolo.npy"), T_base2ob_yolo)

    # Single Only
    # T_goal2ob_origin = viewpoint_poses[0]
    # T_base2goal = T_base2ob_yolo @ T_yolo2origin @ T_goal2ob_origin
    # goal_pose_cam = T_base2goal @ np.linalg.inv(T_link2cam)
    # goal_pose_cam = transform_to_cam(T_base2goal, T_link2cam)

    # Multi path based on custom waypoint
    goal_pose_cam = []
    for i in range(len(viewpoint_poses)):
        T_goal2ob_origin = viewpoint_poses[i]
        T_base2goal = T_base2ob_yolo @ T_yolo2origin @ T_goal2ob_origin
        goal_pose_cam.append(transform_to_cam(T_base2goal, T_link2cam))

    if False:
        for i in range(len(viewpoint_poses)):
            print(f"Moving to Viewpoint {i+1}...")
            movel(goal_pose_cam[i], v=100, a=200) # Doosan Move command
            time.sleep(1) 

            # Capture and merge from each viewpoint
            T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
            capture_scan_view(rs_manager, T_current, i+1, save_dir=pcd_save_dir, duration=1.0, bbox_center=obj_base_pos)
            time.sleep(0.5)

        # Return Home
        time.sleep(1)
        home_robot()    


def move_2():
    """
    Executes a specific sequence of movements to prevent cable tangling.
    It groups the viewpoints and returns home between groups.
    Skips scanning if a viewpoint has already been scanned in this sequence.
    """
    sequence_groups = [
        [7, 0, 1, 2, 3, 4, 5, 6],
        [15, 8, 9, 10, 11, 12, 13, 14],
        [23, 16, 17, 18, 19, 20, 21, 22]
    ]

    scanned_viewpoints = set()

    for group_idx, group in enumerate(sequence_groups):
        for vp_idx in group:
            # Skip if the index is out of range for the loaded poses
            if vp_idx >= len(goal_pose_cam):
                print(f"Warning: Viewpoint {vp_idx} is out of bounds. Skipping...")
                continue
                
            print(f"Moving to Viewpoint {vp_idx}...")
            movel(goal_pose_cam[vp_idx], v=75, a=150) # Doosan Move command
            time.sleep(2) 
            
            # Capture and merge from each viewpoint if not already scanned
            if vp_idx not in scanned_viewpoints:
                T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
                capture_scan_view(rs_manager, T_current, vp_idx, save_dir=pcd_save_dir, duration=1.0, bbox_center=obj_base_pos)
                scanned_viewpoints.add(vp_idx)
                time.sleep(0.5)
            else:
                print(f"Viewpoint {vp_idx} already scanned. Skipping capture.")
            
        # Return Home to prevent tangling (except after the last group)
        if group_idx < len(sequence_groups) - 1:
            print("Returning Home to prevent cable tangling...")
            time.sleep(1)
            home_robot()
            time.sleep(2)
            
    # Return Home at the end
    print("Sequence complete. Returning Home...")
    time.sleep(1)
    home_robot()


def move():
    # Original linear move
    for i in range(len(goal_pose_cam)):
        user_input = input(f"\\nReady for Viewpoint {i}. Press Enter to move (or 'q' to abort): ")
        if user_input.lower() == 'q':
            print("Aborting sequence...")
            break
            
        print(f"Moving to Viewpoint {i}...")
        movel(goal_pose_cam[i], v=75, a=150) # Doosan Move command
        time.sleep(2) 
        # Capture and merge from each viewpoint
        T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
        capture_scan_view(rs_manager, T_current, i, save_dir=pcd_save_dir, duration=1.0, bbox_center=obj_base_pos)
        time.sleep(0.5)
    # Return Home
    print("Returning Home...")
    time.sleep(1)
    home_robot()


def move_3():
    """
    Executes scanning only for viewpoints corresponding to 0, 30, 60, and 90 degrees azimuth.
    Since the maximum rotation is 90 degrees, no unwinding logic is needed.
    """
    viewpoints_per_angle = 12
    total_poses = len(goal_pose_cam)
    num_angles = total_poses // viewpoints_per_angle
    # Collect all indices that correspond to 0, 30, 60, and 90 degrees (offsets 0, 1, 2, 3)
    target_indices = []
    for i in range(num_angles):
        base_idx = i * viewpoints_per_angle
        target_indices.extend([base_idx, base_idx + 1, base_idx + 2, base_idx + 3])
    for vp_idx in target_indices:
        if vp_idx >= len(goal_pose_cam):
            print(f"Warning: Viewpoint {vp_idx} is out of bounds. Skipping...")
            continue
        print(f"Moving to Viewpoint {vp_idx} (0-90 deg sweep)...")
        movel(goal_pose_cam[vp_idx], v=75, a=150) # Doosan Move command
        time.sleep(2) 
        # Capture and merge
        T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
        capture_scan_view(rs_manager, T_current, vp_idx, save_dir=pcd_save_dir, duration=1.0, bbox_center=obj_base_pos)
        time.sleep(0.5)
    # Return Home at the end
    print("Sequence complete. Returning Home...")
    time.sleep(1)
    home_robot()


def test(posx=0,posy=0,posz=0):
    movel([558.9803466796875+posx, 37.472660064697266+posy, 411.4325256347656+posz, 7.729799270629883, -179.99227905273438, 7.732522964477539], v=100, a=200)
    time.sleep(1)
    


# ini starting point deket base
# movel([190.85684204101562, -114.87481689453125, 465.051025390625, 177.75711059570312, -150.01075744628906, 179.9525909423828], v=40, a=80)

# ini masih available (mundur 20 cm)
# movel([170.85684204101562, -114.87481689453125, 465.051025390625, 177.75711059570312, -150.01075744628906, 179.9525909423828], v=40, a=80)

# ini starting point mentok kiri
# movel([458.34783935546875, -471.1073303222656, 591.4693603515625, 87.81663513183594, 150.02391052246094, 90.02122497558594], v=40, a=80)

# ini kayaknya oke (kanan 20 cm)
# movel([458.34783935546875, -451.1073303222656, 591.4693603515625, 87.81663513183594, 150.02391052246094, 90.02122497558594], v=40, a=80)
