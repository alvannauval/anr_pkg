#!/usr/bin/env python3

import profile
import sys
import os
import math
import time
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO
import tf


def init_robot(robot_id="dsr01", model="m1013"):
    """Sets up the Doosan Robot API parameters."""
    DR_init.__dsr__id = robot_id
    DR_init.__dsr__model = model
    rospy.loginfo(f"Robot {robot_id} ({model}) initialized.")


def init_realsense():
    """Starts the RealSense pipeline and returns stream objects."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() # Usually 0.001 (1mm per unit)

    rospy.loginfo("RealSense Pipeline Started. Depth Scale is: {depth_scale}")
    return pipeline, align, intrinsics, depth_scale

def get_robust_depth(depth_frame, x, y, depth_scale, window_size=5):
    """
    Calculates the average depth in a window around (x, y) to avoid noise.
    """
    depth_data = np.asanyarray(depth_frame.get_data())
    half_w = window_size // 2
    
    # Define the bounding box for the window (ROI)
    y_start, y_end = max(0, int(y)-half_w), min(depth_data.shape[0], int(y)+half_w+1)
    x_start, x_end = max(0, int(x)-half_w), min(depth_data.shape[1], int(x)+half_w+1)
    
    roi = depth_data[y_start:y_end, x_start:x_end]

    # Filter out zero values (invalid depth)
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) > 0:
        # Return mean depth converted to meters
        return np.mean(valid_depths) * depth_scale
    else:
        return 0  # No valid depth found


def get_yolo_detection(pipeline, align, model, intrinsics, depth_scale):
    global results
    """Detects object via YOLO OBB and returns camera-space coordinates."""
    print("Waiting for YOLO detection... Press 'q' to confirm.")
    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_img = np.asanyarray(aligned.get_color_frame().get_data())

        if aligned:
            print("Frames aligned successfully.")
        else:
            print("Failed to align frames.")
        
        results = model(color_img, conf=0.92)
        if results[0].obb is not None:
            # results[0].obb is sorted by confidence; index 0 is the best
            box = results[0].obb[0]
            px, py, _, _, rotation = box.xywhr.cpu().numpy()[0]
            
            # --- Robust Depth Calculation ---
            depth_frame = aligned.get_depth_frame()
            dist = get_robust_depth(depth_frame, px, py, depth_scale)
            
            if dist > 0:
                cam_pts = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                cv_frame = results[0].plot()
                cv2.circle(cv_frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.imshow("Detection (Press q)", cv_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return [c * 1000 for c in cam_pts], np.degrees(results[0].obb[0].xywhr.cpu().numpy()[0][4])
                

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


def calculate_rotation_matrix(camera_pos, target_pos):
    """
    Creates a 3x3 rotation matrix for a 'Look-At' orientation.
    The camera's Z-axis will point directly at the target.
    """
    # 1. Z-axis: The direction the camera is looking
    z_axis = np.array(target_pos) - np.array(camera_pos)
    z_axis /= (np.linalg.norm(z_axis) + 1e-6)
    
    # 2. X-axis: Determine 'Right' 
    # We use a temporary UP vector. If looking straight down, use [0,1,0]
    temp_up = np.array([0, 0, 1])
    if abs(np.dot(z_axis, temp_up)) > 0.99:
        temp_up = np.array([0, 1, 0]) # Switch if looking parallel to Z
        
    x_axis = np.cross(temp_up, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-6)
    
    # 3. Y-axis: Determine 'Down' (or Up depending on camera convention)
    y_axis = np.cross(z_axis, x_axis)
    
    # Assemble the 3x3 matrix
    # Columns are x, y, z axes respectively
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    return rot_matrix



# def get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe'):
#     """Fetches the 4x4 Homogeneous Transformation matrix from TF2 (in mm)."""
#     try:
#         t = tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(2.0))
#         quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        
#         tf_matrix = np.eye(4)
#         tf_matrix[:3, :3] = R.from_quat(quat).as_matrix()
#         tf_matrix[:3, 3] = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]) * 1000.0
#         return tf_matrix
#     except Exception as e:
#         rospy.logerr(f"TF Lookup failed: {e}")
#         return None
    

def capture_scan_view(pipeline, align, T_base_camera, index, save_dir="PCD_Data", duration=1.0):
    """Captures multiple frames over a duration and merges them into one clean PCD."""    
    all_points = [] # List to store points from every frame
    start_time = time.time()
    count = 0
    
    print(f"Scanning Viewpoint {index} for {duration}s...")

    while (time.time() - start_time) < duration:
        count += 1
        frame = pipeline.wait_for_frames()
        aligned_frames = align.process(frame)

        depth_frame = aligned_frames.get_depth_frame()
        last_depth_data = np.asanyarray(depth_frame.get_data())
        if not depth_frame:
            continue
            
        if count % 5 != 0:
            continue 

        # Calculate Points
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3) * 1000.0
        
        verts_base = (T_base_camera @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        all_points.append(verts_base)

    if len(all_points) == 0:
        print("Error: No points captured!")
        return


    print("\nCurrent Transformation Base - Camera")
    print(T_base_camera)


    merged_verts = np.vstack(all_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_verts)
    pcd = pcd.voxel_down_sample(voxel_size=2.0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(last_depth_data, alpha=0.5), 
            cv2.COLORMAP_JET
        )
    img_file_path = os.path.join(save_dir, f"view{index:02d}_depth_color.png")
    cv2.imwrite(img_file_path, depth_colormap)

    pcd_file_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    tf_file_path = os.path.join(save_dir, f"view{index:02d}_tf.npy")
    np.save(tf_file_path, T_base_camera)
    o3d.io.write_point_cloud(pcd_file_path, pcd)
    print(f"Successfully saved merged {len(np.asarray(pcd.points))} points to {pcd_file_path}")


def home_robot():
    """Moves the robot to the predefined home joint position."""
    print("Moving to Home position...")
    movej([0, 0, 90, 0, 90, 0], v=15, a=30) 


def translation_matrix(x, y, z):
    """
    Creates a 4x4 homogenous transformation matrix 
    representing only a translation (no rotation).
    """
    T = np.eye(4) # Creates a 4x4 Identity Matrix
    T[0, 3] = x   # Set X translation
    T[1, 3] = y   # Set Y translation
    T[2, 3] = z   # Set Z translation
    return T

def pose_to_matrix(pose):
    """
    Converts [x, y, z, roll, pitch, yaw] in degrees to a 4x4 transformation matrix.
    Args:
    - pose (list or array): [x, y, z, roll, pitch, yaw] in degrees.
    
    Returns:
    - T (ndarray): The 4x4 homogeneous transformation matrix.
    """
    x, y, z = pose[:3]   # Translation vector
    roll, pitch, yaw = pose[3:]  # Rotation angles in degrees
    # 1. Create the 3x3 rotation matrix from roll, pitch, yaw (XYZ Euler Angles)
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    # 2. Build the 4x4 Homogeneous Transformation Matrix
    T = np.eye(4)  # Start with the identity matrix (4x4)
    T[:3, :3] = rot_matrix  # Set the upper-left 3x3 part to the rotation matrix
    T[:3, 3] = [x, y, z]    # Set the upper-right 3x1 part to the translation vector
    return T



def matrix_to_pose(matrix):
    """
    Converts a 4x4 homogeneous transform matrix into
    [x, y, z, roll, pitch, yaw] in degrees.
    Rotation is extracted using intrinsic ZYX order
    (R = Rz(yaw) * Ry(pitch) * Rx(roll)),
    which matches the standard ROS RPY convention.
    """
    # 1. Extract the translation (x, y, z)
    x, y, z = matrix[:3, 3]
    # 2. Extract the rotation matrix (top-left 3x3 block)
    rot_matrix = matrix[:3, :3]
    # 3. Convert rotation matrix to intrinsic ZYX Euler angles (degrees)
    # SciPy returns [yaw, pitch, roll] for 'zyx'
    zyx = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
    # Reorder to [roll, pitch, yaw]
    return [x, y, z, zyx[2], zyx[1], zyx[0]]


def matrix_to_pose_zyz(matrix):
    """
    Converts a 4x4 homogeneous transform matrix into
    [x, y, z, alpha, beta, gamma] in degrees.

    Rotation is extracted using intrinsic ZYZ Euler order:
        R = Rz(alpha) * Ry(beta) * Rz(gamma)
    """
    # 1. translation
    x, y, z = matrix[:3, 3]
    # 2. rotation matrix
    rot_matrix = matrix[:3, :3]
    # 3. ZYZ Euler extraction
    zyz = R.from_matrix(rot_matrix).as_euler('ZYZ', degrees=True)
    alpha, beta, gamma = zyz
    return [x, y, z, alpha, beta, gamma]




def get_tf_matrix_gpt(tf_buffer, target, source):
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


def capture():
    T_current = get_tf_matrix_gpt(tf_buffer, target='base_0', source='realsense_RGBframe')
    time.sleep(1)
    capture_scan_view(pipeline, align, T_current, 0, save_dir=pcd_save_dir, duration=1.0)


def transform_to_cam(pose):
    pose[0] -= 85.15
    pose[1] -= 32.5
    pose[2] += 122.85

    # force down
    pose[3] = 3.4242515563964844
    pose[4] = -179.9999542236328
    pose[5] = 3.4242515563964844

    return pose


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
    pipeline, align, intrinsics, depth_scale = init_realsense()
    model = YOLO("model/workpiece1_OBB.pt")

    # Detection & Localization
    obj_cam_pos, obb_angle = get_yolo_detection(pipeline, align, model, intrinsics, depth_scale)
    T_init = get_tf_matrix_gpt(tf_buffer, target='base_0', source='realsense_RGBframe')
    obj_base_pos = (T_init @ np.append(obj_cam_pos, 1))[:3]
    obj_base_pose = [obj_base_pos[0], obj_base_pos[1], obj_base_pos[2], 0.0, obb_angle, 0.0]
    np.save("PCD_Data/initial_obj_pose.npy", obj_base_pose)


    # Scanning Parameters
    SCAN_HEIGHT = 500         # m above the object
    VIEWPOINTS = 8              # Number of scans
    DESIRED_ANGLE_DEG = 75.0    # degrees
    pcd_save_dir = "PCD_Data"   
    
    SCAN_RADIUS = SCAN_HEIGHT * math.tan(math.radians(90.0 - DESIRED_ANGLE_DEG)) 

    # Multi View Scanning
    print(f"Starting Multi-view Scan...")
    print(f"Scanning from viewpoint no 1")
    capture_scan_view(pipeline, align, T_init, 0, save_dir=pcd_save_dir, duration=1.0)

    link6_path = []      # target based on link6
    camera_path = []    # target based on camera

    T_link2cam = get_tf_matrix_gpt(tf_buffer, source='link6', target='realsense_RGBframe')
        

    # Planning Phase
    for i in range(VIEWPOINTS):
        # Calculate circular position
        angle = math.radians((360.0 / VIEWPOINTS) * i)
        tx = obj_base_pose[0] + SCAN_RADIUS * math.cos(angle)
        ty = obj_base_pose[1] + SCAN_RADIUS * math.sin(angle)
        tz = obj_base_pose[2] + SCAN_HEIGHT


        
        # This matrix describes the camera's orientation in the Base frame
        # zyz = calculate_look_at_zyz([tx, ty, tz], obj_base_pose[:3])
        # target_pose = [tx, ty, tz, zyz[0], zyz[1], zyz[2]]


        # T_base_obj = translation_matrix(tx, ty, tz)
        # T_base_link6_new = T_base_obj @ T_cam_to_link6
        # final_xyz = list(T_base_link6_new[:3, 3])
        # wrist_pose = [final_xyz[0], final_xyz[1], final_xyz[2], target_pose[3], target_pose[4], target_pose[5]]

        link6_pose = [tx, ty, tz, 0, 0, 0]

        camera_pose = transform_to_cam(link6_pose)

        link6_path.append(link6_pose)
        camera_path.append(camera_pose)

    # Moving Phase
def move_path():
    for i in range(VIEWPOINTS):
        print(f"Moving to Viewpoint {i}...")
        movel(camera_path[i], v=100, a=200) # Doosan Move command
        time.sleep(1) 
        # Capture and merge from each viewpoint
        T_current = get_tf_matrix_gpt(tf_buffer, target='base_0', source='realsense_RGBframe')
        capture_scan_view(pipeline, align, T_current, i+1, save_dir=pcd_save_dir, duration=1.0)
        time.sleep(0.5)

    # # Return Home
    # home_robot()


# def move_to_camera_above_object_transformation():
#     global center_pose, center_pose_transformed, T_init
    # known yolo position in cam frame
    # obj_cam_pos 
    # obb_angle


T_link2cam = get_tf_matrix_gpt(tf_buffer, source='realsense_RGBframe', target='link6')
# T_cam2ob = [obj_cam_pos[0], obj_cam_pos[1], obj_cam_pos[2], 0.0, 180.0, 90.0] #T_cam2ob def look inside
T_cam2ob = [obj_cam_pos[0], obj_cam_pos[1], obj_cam_pos[2], 0.0, 180.0, -90.0] #T_cam2ob def look outside
# transform to base frame
T_base2cam = get_tf_matrix_gpt(tf_buffer, source='realsense_RGBframe', target='base_0') #visually, it's directing from
T_base2ob = T_base2cam @ pose_to_matrix(T_cam2ob)

T_ob2cam_goal = [[0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, -1, SCAN_HEIGHT],
                [0, 0, 0, 1]]


T_base2link = T_base2ob @ T_ob2cam_goal @ np.linalg.inv(T_link2cam)
# target_link6 = matrix_to_pose(T_base2link)
target_link6 = matrix_to_pose_zyz(T_base2link)

# movel(target_link6, v=25, a=150)



    # method 1
    # move link6 to position where camera is above object
    # Formula derivation
    # T_base_link6 @ T_link6_cam = T_base_wp @ T_wp_cam 
    # T_base_link6 @ T_link6_cam = T_base_goal
    # T_base_link6 = T_base_goal @ inv(T_link6_cam)

    # T_base_goal = pose_to_matrix(goal_base_pose) # makes sense
    # T_link6_cam = get_tf_matrix_gpt(tf_buffer, source='link6', target='realsense_RGBframe') # same with rosrun tf
    # T_base_link6 = T_base_goal @ np.linalg.inv(T_link6_cam)


    # target_link6[3] = 3.4242515563964844
    # target_link6[4] = -179.9999542236328
    # target_link6[5] = 3.4242515563964844



    # method 2
    # T_base_goal = obj_base_pose
    # T_base_goal[2,3] += SCAN_HEIGHT

    # T_cam_link6 = get_tf_matrix_gpt(tf_buffer, source='realsense_RGBframe', target='link6')

    # T_base_link6 = T_base_goal @ T_cam_link6



    # damnn 3
    # r_goal = goal_base_pose[:3]
    # T_cam_link6 = get_tf_matrix_gpt(tf_buffer, source='realsense_RGBframe', target='link6')
    # r_l_g = T_cam_link6[:3, 3]

    # link6_position_for_camera = r_goal + r_l_g

    # target = list(link6_position_for_camera) + [3.4242515563964844, -179.9999542236328, 3.4242515563964844]
    # target[2] = 350

    # target_r_goal = list(r_goal) + [3.4242515563964844, -179.9999542236328, 3.4242515563964844]




    # ke tengah link6
target = list(goal_base_pose[:3]) + [3.4242515563964844, -179.9999542236328, 3.4242515563964844]

    # posisi link6 top of object
    # [611.1490142611697, -10.020748209501974, 502.9430853352909, 3.4242515563964844, -179.9999542236328, 3.4242515563964844]

    # posisi cam top of object

target_transformed = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -179.9999542236328, 3.4242515563964844]

test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 10, -179.9999542236328, 15]

# perfect down
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 0.01, -179.9999542236328, 0.01]
# yaw 5
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 5, -179.9999542236328, 0.01]
# yaw 0
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 5, -179.9999542236328, 5]
# yaw 10
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 5, -179.9999542236328, -5]



# pitch 15
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 0.01, -165.0, 0.01]
# pitch 15 and roll 15
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 15, -165, 15]
# pitch 15 and yaw 15
test = [goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 15, -165, 0.01]



def test():
    path = []
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -179.9999542236328, 3.4242515563964844])
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -175, 3.4242515563964844])
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -170, 3.4242515563964844])
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -165, 3.4242515563964844])
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -160, 3.4242515563964844])
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -155, 3.4242515563964844])
    path.append([goal_base_pose[0]-85.15, goal_base_pose[1]-32.5, goal_base_pose[2]+122.85, 3.4242515563964844, -150, 3.4242515563964844])

    for i in range(len(path)):
        movel(path[i], v=15, a=30)
        time.sleep(1)


    # transformasi
    # T_wp_base = get_tf_matrix_gpt(tf_buffer, source='realsense_RGBframe', target='base_0')
    # T_cam_link6 = get_tf_matrix_gpt(tf_buffer, source='link6', target='realsense_RGBframe')



    # goal_base_pose_2 = target
    # goal_base_pose_2[0] -= 32
    # goal_base_pose_2[1] -= 85.1
    # goal_base_pose_2[3] = 3.4242515563964844
    # goal_base_pose_2[4] = -179.9999542236328
    # goal_base_pose_2[5] = 3.4242515563964844


    # get_tf_matrix_gpt(tf_buffer, source='link6', target='base_0')

    # obj_base_pose
    # b2wp= obj_base_pose
    # b2wp[4] = 0.0

    # T_b2wp = pose_to_matrix(b2wp)

    # T_wp2cam_goal = np.array([
    #                 [1, 0,  0, 0],
    #                 [0, -1,  0, 0],
    #                 [0, 0, -1, SCAN_HEIGHT],
    #                 [0, 0,  0, 1]])
    
    # T_link2cam = get_tf_matrix(tf_buffer, source='link6', target='realsense_RGBframe')


    # T_b2link = T_b2wp @ T_wp2cam_goal @ np.linalg.inv(T_link2cam)
    
    # b2link = matrix_to_pose(T_b2link)

    # movel(b2link, v=50, a=150)


    # print(matrix_to_pose(get_tf_matrix(tf_buffer, source='base_0', target='link6')))

    # print(get_tf_matrix(tf_buffer, source='base_0', target='link6'))


    # # manual manipulation, any z, look downward
    # center_pose = [obj_base_pose[0], obj_base_pose[1], SCAN_HEIGHT, obj_base_pose[3], obj_base_pose[4], obj_base_pose[5]]
    # # move link6 to center above object
    # # movel(center_pose, v=25, a=150) 

    # # transform center_pose to matrix form
    # # T_base_center_pose = translation_matrix(center_pose[0], center_pose[1], SCAN_HEIGHT)
    # T_base_center_pose = pose_to_matrix(center_pose)
    
    # # transformation from link6 to camera
    # T_link6_cam = get_tf_matrix(tf_buffer, source='link6', target='realsense_RGBframe')
    # T_cam_link6 = np.linalg.inv(T_link6_cam)

    # # do the transformation
    # T_base_link6_new = T_base_center_pose @ T_cam_link6
    # center_pose_transformed = matrix_to_pose(T_base_link6_new)



    # # final_xyz = list(T_base_link6_new[:3, 3])
    # # # manual manipuation again
    # # center_pose_transformed = [final_xyz[0], final_xyz[1], SCAN_HEIGHT, 3.4242515563964844, -179.9999542236328, 3.4242515563964844]
    # # time.sleep(1)
    # movel(center_pose_transformed, v=50, a=150)

    # # then do this
    # time.sleep(2)
    # T_init = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
    # capture_scan_view(pipeline, align, T_init, 0, save_dir=pcd_save_dir, duration=1.0)

    # print("Object Pose: ", obj_base_pose)
    # print("Camera Above Object Pose: \n", T_init)




# # TEST FUNCTIONS
# def move_to_camera_above_object_manual():
#     global center_pose_manual, center_pose_transformed_manual, T_init_manual
#     # known yolo position
#     obj_base_pose
    
#     center_pose_manual = [obj_base_pose[0], obj_base_pose[1], SCAN_HEIGHT, 3.4242515563964844, -179.9999542236328, 3.4242515563964844]

#     # if do
#     # T_cam_to_link6 = get_tf_matrix(tf_buffer, target='realsense_RGBframe', source='link6')
#     # output is like this
#     # array([[          0,          -1,          -0,        32.5],
#     #     [          1,           0,           0,       85.15],
#     #     [          0,          -0,           1,     -122.85],
#     #     [          0,           0,           0,           1]])
    
#     # move link6 to center above object
#     # movel(center_pose, v=50, a=150)

#     # trial and error so camera move to above object
#     center_pose_transformed_manual = [center_pose_manual[0]-85.15, center_pose_manual[1]-32.5, center_pose_manual[2], center_pose_manual[3], center_pose_manual[4], center_pose_manual[5]]
    
#     time.sleep(1)
#     movel(center_pose_transformed_manual, v=50, a=150)

#     # then do this
#     time.sleep(2)
#     T_init_manual = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
#     capture_scan_view(pipeline, align, T_init_manual, 0, save_dir=pcd_save_dir, duration=1.0)

#     print("Object Pose: ", obj_base_pose)
#     print("Camera Above Object Pose: \n", T_init_manual)








# T1 = get_tf_matrix_gpt(tf_buffer, target='base_0', source='realsense_RGBframe')
# T2 = get_tf_matrix_gpt(tf_buffer, target='base_0', source='link6')
# T3 = get_tf_matrix_gpt(tf_buffer, target='link6', source='realsense_RGBframe')



