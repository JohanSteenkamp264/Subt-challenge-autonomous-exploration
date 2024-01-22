#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError

import matplotlib.pyplot as plt
import matplotlib.image as img

import rospy
import ros_numpy as rnp
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu, CameraInfo, MagneticField, PointCloud2
from tf2_msgs.msg import TFMessage

import sys
import os
import shutil

from path_recorder import get_path_recorder_from_file

from sklearn.cluster import DBSCAN
from pid import PIDArduino

import open3d as o3d

"""Helper Functions"""


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]

    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def quaternion_multiply(Q0, Q1):
    """
    Multiplies two quaternions.

    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from Q0
    w0 = Q0[0]
    x0 = Q0[1]
    y0 = Q0[2]
    z0 = Q0[3]

    # Extract the values from Q1
    w1 = Q1[0]
    x1 = Q1[1]
    y1 = Q1[2]
    z1 = Q1[3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


class dataset_collector:
    def __init__(self):
        if len(sys.argv) < 2:
            raise Exception("No dataset save path provided")
        else:
            self.Save_Path = sys.argv[1]
            # create recording paths
            if os.path.isdir(self.Save_Path):
                shutil.rmtree(self.Save_Path)

            os.makedirs(self.Save_Path)
            os.mkdir(self.Save_Path + "/rgb")
            os.mkdir(self.Save_Path + "/depth")
            os.mkdir(self.Save_Path + "/lidar")

            os.mkdir(self.Save_Path + "/depth_up")
            os.mkdir(self.Save_Path + "/depth_down")

            os.mkdir(self.Save_Path + "/rgb_up")
            os.mkdir(self.Save_Path + "/rgb_down")

        # possible_path_exploration
        self.flag_exit_program = False
        self.num_positions_not_exploured_in_previous_runs = 0
        self.THR_DISTACE_FOR_EXPLOURED = 5.0

        # Obsticle Avoicance
        self.bounding_box_size = [1.0, 1.0, 1.0]
        self.Point_Cloud_Occupancy = []
        self.OCCUPANCY_DISTANCE = 7.5
        self.NUM_POSITIONS_CHECKED_FOR_COLLISIONS = 20
        self.PATH_DIVISIONS = 9
        self.max_collision_avoidance_iterations = 10

        # Dataset generation
        self.PATH_WRITTEN = False
        self.limit_Dataset_genetation = True
        self.limit_index = 6000
        self.Path_Traversed_to_origin = False

        self.depth_horizontal_range = 69.0 * (np.pi / 180.0)
        self.depth_vertical_range = 42.0 * (np.pi / 180.0)
        self.depth_shape = (240, 320)
        self.horizontal_map = np.zeros(self.depth_shape, dtype=np.float32)
        self.vertical_map = np.zeros(self.depth_shape, dtype=np.float32)

        h_map = np.linspace(self.depth_horizontal_range / 2.0, -self.depth_horizontal_range / 2.0, self.depth_shape[1])

        for height in range(self.depth_shape[0]):
            self.horizontal_map[height] = h_map

        v_map = np.linspace(self.depth_vertical_range / 2.0, -self.depth_vertical_range / 2.0, self.depth_shape[0])
        for width in range(self.depth_shape[1]):
            self.vertical_map[:, width] = v_map

        self.bridge = CvBridge()
        self.move_cmd = Twist()

        self.dist_up = 0.0
        self.ave_dist_up = 10.0
        self.dist_down = 0.0
        self.ave_dist_down = 10.0
        self.dist_front = 0.0
        self.dist_len_ave = 200.0

        # State Machine
        self.random_exploration = True
        self.last_followed_path = None
        self.Recent_Exploration_added_nodes = []
        self.NUM_POSITIONS_EXPLORE_TO_TRAVERSE = 240
        self.num_positions_explored = 0

        self.search_path = False
        self.time_search_started = 0.0

        self.transverse = False
        self.Transversal_Path = None
        self.THRESHOLD_REMOVE_FROM_PATH = 4.0
        self.num_unexplored_positions = 0
        self.THR_UNEXPLORED = 80

        self.linspace_colours = [[255, 0, 0], [169, 114, 28], [84, 228, 56], [255, 85, 85], [169, 199, 113],
                                 [84, 57, 142], [255, 170, 170], [169, 28, 199], [84, 142, 227], [255, 255, 255]]
        self.sample_time = 1.0 / 20.0
        self.map_saved_file = "saved_map.txt"
        self.path_record_planner = get_path_recorder_from_file(self.map_saved_file)

        # Velocity Control
        self.MAX_X_VELOCITY = 1.0
        self.MAX_VELOSITY_DISTANCE = 10.0
        self.MAX_DIST_1_LIMIT = 5.0
        self.VEL_L1 = 0.5
        self.MAX_DIST_2_LIMIT = 1.5
        self.VEL_L2 = 0.2

        self.pid_velocity_x = PIDArduino(sampletime=self.sample_time, kp=-0.4, ki=-0.1, kd=-0.001,
                                         out_min=0.0,
                                         out_max=self.MAX_X_VELOCITY, time=rospy.get_time)
        self.pid_velocity_z = PIDArduino(sampletime=self.sample_time, kp=-0.25, ki=-0.1, kd=-0.0010,
                                         out_min=-self.MAX_X_VELOCITY * (1.5 / 4.0),
                                         out_max=self.MAX_X_VELOCITY * (1.5 / 4.0), time=rospy.get_time)
        self.pid_rotation_z = PIDArduino(sampletime=self.sample_time, kp=-0.5, ki=-0.1, kd=-0.0010,
                                         out_min=-self.MAX_X_VELOCITY * (0.3 / 4.0),
                                         out_max=self.MAX_X_VELOCITY * (0.3 / 4.0), time=rospy.get_time)

        self.move_cmd = Twist()
        self.cmd_vel_Publisher = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)
        rospy.init_node('dataset_collector')

        self.f_rgb = open(self.Save_Path + "/rgb.txt", 'w')
        self.f_rgb.write("# Time[s], filename\n")
        self.i_rgb = 0
        self.image_front = None
        self.image_front_sunscriber = rospy.Subscriber('/X1/front_rgbd/image_raw', Image, self.callback_image_front)

        self.f_rgb_up = open(self.Save_Path + "/rgb_up.txt", 'w')
        self.f_rgb_up.write("# Time[s], filename\n")
        self.i_rgb_up = 0
        self.image_up = None
        self.image_up_sunscriber = rospy.Subscriber('/X1/up_rgbd/image_raw', Image, self.callback_image_up)

        self.f_rgb_down = open(self.Save_Path + "/rgb_down.txt", 'w')
        self.f_rgb_down.write("# Time[s], filename\n")
        self.i_rgb_down = 0
        self.image_down = None
        self.image_down_sunscriber = rospy.Subscriber('/X1/down_rgbd/image_raw', Image, self.callback_image_down)

        self.depth_horizontal_range = 87.0 * (np.pi / 180.0)
        self.depth_vertical_range = 58.0 * (np.pi / 180.0)

        self.f_depth = open(self.Save_Path + "/depth.txt", 'w')
        self.f_depth.write("# Time[s], filename\n")
        self.i_depth = 0
        self.depth_front = None
        self.depth_front_subcriber = rospy.Subscriber('/X1/front_rgbd/depth/image_raw', Image,
                                                      self.callback_depth_front)

        self.f_depth_up = open(self.Save_Path + "/depth_up.txt", 'w')
        self.f_depth_up.write("# Time[s], filename\n")
        self.i_depth_up = 0
        self.depth_up = None
        self.depth_up_subcriber = rospy.Subscriber('/X1/up_rgbd/depth/image_raw', Image, self.callback_depth_up)

        self.f_depth_down = open(self.Save_Path + "/depth_down.txt", 'w')
        self.f_depth_down.write("# Time[s], filename\n")
        self.i_depth_down = 0
        self.depth_down = None
        self.depth_down_subcriber = rospy.Subscriber('/X1/down_rgbd/depth/image_raw', Image, self.callback_depth_down)

        self.f_lidar = open(self.Save_Path + "/lidar.txt", 'w')
        self.f_lidar.write("# Time[s], filename\n")
        self.i_lidar = 0
        self.lidar_scan = None
        self.Lidar_depth_image = None
        self.lidar_subscriber = rospy.Subscriber('/X1/points', PointCloud2, self.callback_lidar)

        self.f_gt = open(self.Save_Path + "/groundtruth.txt", 'w')
        self.f_gt.write("# Time[s], px[m], py[m], pz[m], qx, qy, qz, qw\n")
        self.pose_subscriber = rospy.Subscriber('/X1/pose_static', TFMessage, self.callback_pose)

        self.f_cmd_vel = open(self.Save_Path + "/velocity_control.txt", 'w')
        self.f_cmd_vel.write(
            "# Time[s], d_px[m/s^2], d_py[m/s^2], d_pz[m/s^2], d_wx[rad/s], d_wy[rad/s], d_wz[rad/s]\n")

        self.f_imu = open(self.Save_Path + "/imu.txt", 'w')
        self.f_imu.write(
            "# Time[s], ax[m/s^2], ay[m/s^2], az[m/s^2], wx[rad/s], wy[rad/s], wz[rad/s]\n")
        self.imu_subscriber = rospy.Subscriber('/X1/imu/data', Imu, self.callback_imu)

        self.f_mag = open(self.Save_Path + "/magnetometer.txt", 'w')
        self.f_mag.write("# Time[s], mx[Tesla], my[Tesla], mz[Tesla]\n")
        self.magnetometer_subscriber = rospy.Subscriber('/X1/magnetic_field', MagneticField, self.callback_magnetometer)

        # Pose
        self.last_translational_pose = np.zeros(3)
        self.last_rotational_pose = np.zeros(3)
        self.last_rotational_pose_quaternion = np.zeros(4)

        self.lift_off()
        rospy.Timer(rospy.Duration(self.sample_time), self.timer_vel_publisher_callback)

        while not rospy.is_shutdown() and not self.flag_exit_program:
            time.sleep(10.0)

    def lift_off(self):
        rospy.sleep(0.5)

        self.move_cmd.linear.z = 1.0
        self.cmd_vel_Publisher.publish(self.move_cmd)

        rospy.sleep(0.5)

        self.move_cmd.linear.z = 0.0
        self.cmd_vel_Publisher.publish(self.move_cmd)


    def callback_lidar(self, lidar):
        xyz_array = rnp.point_cloud2.pointcloud2_to_array(lidar)

        self.f_lidar.write("{},{}".format(rospy.get_time(), "/lidar/{:010d}.npy\n".format(self.i_lidar)))
        np.save(self.Save_Path + "/lidar/{:010d}.npy".format(self.i_lidar), xyz_array)
        self.i_lidar += 1

        self.lidar_scan = xyz_array

    def callback_depth_up(self, depth):
        dpth_float = self.bridge.imgmsg_to_cv2(depth)
        self.f_depth_up.write("{},{}".format(rospy.get_time(), "/depth_up/{:010d}.npy\n".format(self.i_depth_up)))
        np.save(self.Save_Path + "/depth_up/{:010d}.npy".format(self.i_depth_up), dpth_float)
        self.i_depth_up += 1

        self.dist_up = np.min(
            dpth_float[dpth_float.shape[0] // 2 - 50: dpth_float.shape[0] // 2 + 50,
            dpth_float.shape[1] // 2 - 50: dpth_float.shape[1] // 2 + 50])
        if self.dist_up > 10.0:
            self.dist_up = 10.0
        if self.dist_up < 0.0:
            self.dist_up = 0.0

        self.ave_dist_up = ((self.dist_len_ave - 1) * self.ave_dist_up + self.dist_up) / self.dist_len_ave

        self.depth_up = dpth_float

    def callback_depth_down(self, depth):
        dpth_float = self.bridge.imgmsg_to_cv2(depth)

        self.f_depth_down.write(
            "{},{}".format(rospy.get_time(), "/depth_down/{:010d}.npy\n".format(self.i_depth_down)))
        np.save(self.Save_Path + "/depth_down/{:010d}.npy".format(self.i_depth_down), dpth_float)
        self.i_depth_down += 1

        self.dist_down = np.min(
            dpth_float[dpth_float.shape[0] // 2 - 50: dpth_float.shape[0] // 2 + 50,
            dpth_float.shape[1] // 2 - 50: dpth_float.shape[1] // 2 + 50])
        if self.dist_down > 10.0:
            self.dist_down = 10.0

        if self.dist_down < 0.0:
            self.dist_down = 0.0

        self.ave_dist_down = ((self.dist_len_ave - 1) * self.ave_dist_down + self.dist_down) / self.dist_len_ave

        self.depth_down = dpth_float

    def callback_depth_front(self, depth):
        dpth_float = self.bridge.imgmsg_to_cv2(depth)

        self.f_depth.write("{},{}".format(rospy.get_time(), "/depth/{:010d}.npy\n".format(self.i_depth)))
        np.save(self.Save_Path + "/depth/{:010d}.npy".format(self.i_depth), dpth_float)
        self.i_depth += 1

        self.dist_front = np.min(
            dpth_float[dpth_float.shape[0] // 2 - 50: dpth_float.shape[0] // 2 + 50,
            dpth_float.shape[1] // 2 - 50: dpth_float.shape[1] // 2 + 50])
        if self.dist_front > 10.0:
            self.dist_front = 10.0

        self.depth_front = dpth_float

    def callback_image_front(self, image):
        rgb = self.bridge.imgmsg_to_cv2(image)

        self.f_rgb.write("{},{}".format(rospy.get_time(), "/rgb/{:010d}.png\n".format(self.i_rgb)))
        cv2.imwrite(self.Save_Path + "/rgb/{:010d}.png".format(self.i_rgb), rgb)
        self.i_rgb += 1

        self.image_front = rgb
        return

    def callback_image_up(self, image):
        rgb = self.bridge.imgmsg_to_cv2(image)

        self.f_rgb_up.write("{},{}".format(rospy.get_time(), "/rgb_up/{:010d}.png\n".format(self.i_rgb_up)))
        cv2.imwrite(self.Save_Path + "/rgb_up/{:010d}.png".format(self.i_rgb_up), rgb)
        self.i_rgb_up += 1

        self.image_up = rgb
        return

    def callback_image_down(self, image):
        rgb = self.bridge.imgmsg_to_cv2(image)

        self.f_rgb_down.write("{},{}".format(rospy.get_time(), "/rgb_down/{:010d}.png\n".format(self.i_rgb_down)))
        cv2.imwrite(self.Save_Path + "/rgb_down/{:010d}.png".format(self.i_rgb_down), rgb)
        self.i_rgb_down += 1

        self.image_down = rgb
        return

    def callback_imu(self, imu):
        self.f_imu.write("{},{},{},{},{},{},{}\n".format(rospy.get_time(), \
                                                                     imu.linear_acceleration.x,
                                                                     imu.linear_acceleration.y,
                                                                     imu.linear_acceleration.z, \
                                                                     imu.angular_velocity.x, imu.angular_velocity.y,
                                                                     imu.angular_velocity.z))
        return

    def callback_magnetometer(self, magnetometer):
        self.f_mag.write("{},{},{},{}\n".format(rospy.get_time() \
                                                , magnetometer.magnetic_field.x, magnetometer.magnetic_field.y,
                                                magnetometer.magnetic_field.z))
        return

    def callback_pose(self, pose_static):
        pose = pose_static.transforms[5]
        self.f_gt.write("{},{},{},{},{},{},{},{}\n".format(rospy.get_time(), \
                                                           pose.transform.translation.x, pose.transform.translation.y,
                                                           pose.transform.translation.z, \
                                                           pose.transform.rotation.x, pose.transform.rotation.y,
                                                           pose.transform.rotation.z, pose.transform.rotation.w))
        self.path_record_planner.add_position_node(
            np.array([pose.transform.translation.x, pose.transform.translation.y, pose.transform.translation.z], \
                     dtype=np.float32))

        self.last_translational_pose[0] = pose.transform.translation.x
        self.last_translational_pose[1] = pose.transform.translation.y
        self.last_translational_pose[2] = pose.transform.translation.z

        self.last_rotational_pose_quaternion[0] = pose.transform.rotation.w
        self.last_rotational_pose_quaternion[1] = pose.transform.rotation.x
        self.last_rotational_pose_quaternion[2] = pose.transform.rotation.y
        self.last_rotational_pose_quaternion[3] = pose.transform.rotation.z

        self.last_rotational_pose[0], self.last_rotational_pose[1], self.last_rotational_pose[
            2] = euler_from_quaternion(pose.transform.rotation.x, pose.transform.rotation.y, pose.transform.rotation.z,
                                       pose.transform.rotation.w)

    def callback_camera_info(self, camera_info):
        return

    def callback_depth_info(self, depth_info):
        return

    def is_similar_path(self, pos_1_local_translation, pos_2_local_translation, tolerance=1.0):

        dist_pos_1 = np.sqrt(np.sum(np.square(pos_1_local_translation)))
        dist_pos_2 = np.sqrt(np.sum(np.square(pos_2_local_translation)))
        dist = min(dist_pos_1, dist_pos_2)

        pos_1_local_translation = (dist / dist_pos_1) * pos_1_local_translation
        pos_2_local_translation = (dist / dist_pos_2) * pos_2_local_translation

        distance_between_paths = np.sqrt(np.sum(np.square(pos_1_local_translation - pos_2_local_translation)))
        return distance_between_paths < tolerance

    def get_paths_from_depth(self, depth, image, std_thr=0.4):
        depth = np.where(np.isfinite(depth), depth, 10.0)

        mean = np.average(depth)
        std = np.std(depth)

        thr = np.where(depth > mean + std_thr * std, 255, 0).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thr)
        if nb_components > 1 and std > 0.5:
            depth_paths = []
            depth_horizontal_range = 87.0 * (np.pi / 180.0)
            depth_vertical_range = 58.0 * (np.pi / 180.0)
            for index, centre in enumerate(centroids[1:]):
                if stats[index + 1][cv2.CC_STAT_AREA] > 300:
                    if index < len(self.linspace_colours):
                        cv2.circle(image, (int(centre[0]), int(centre[1])), 10, (
                            self.linspace_colours[index][0], self.linspace_colours[index][1],
                            self.linspace_colours[index][2]), 2)

                    dist = depth[int(centre[1])][int(centre[0])] * 0.5
                    if dist > self.OCCUPANCY_DISTANCE:
                        dist = self.OCCUPANCY_DISTANCE
                    theta = ((depth.shape[1] / 2.0 - centre[0]) / depth.shape[
                        1]) * depth_horizontal_range
                    gamma = ((depth.shape[0] / 2.0 - centre[1]) / depth.shape[
                        0]) * depth_vertical_range

                    x_y = dist * np.cos(gamma)
                    z = dist * np.sin(gamma)

                    x = x_y * np.cos(theta)
                    y = x_y * np.sin(theta)
                    if dist > 0.5:
                        depth_paths.append(np.array([x, y, z]))

            return depth_paths
        else:
            return []

    def depth_to_image(self, depth, max_dist=10.0):
        image = depth * (255.0 / max_dist)
        image = np.where(image > 255.0, 255.0, image).astype(np.uint8)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image_rgb

    def get_occupance_point_cloud_from_depth(self, depth_front):
        # print(depth_front.shape, self.horizontal_map.shape, self.vertical_map.shape)
        depth_front = cv2.resize(depth_front, (self.depth_shape[1], self.depth_shape[0]), interpolation=cv2.INTER_AREA)
        depth_front = np.where(np.isfinite(depth_front), depth_front, 10.0)

        x_y = depth_front * np.cos(self.vertical_map)
        z = depth_front * np.sin(self.vertical_map)

        x = x_y * np.cos(self.horizontal_map)
        y = x_y * np.sin(self.horizontal_map)
        point_cloud = np.dstack([x.flatten(), y.flatten(), z.flatten()])[0]

        distances = np.sqrt(np.sum(np.square(point_cloud), axis=1))
        point_cloud = point_cloud[np.argwhere(np.isfinite(distances))]
        point_cloud = point_cloud.reshape((len(point_cloud), 3))

        return point_cloud

    def is_collision_detected(self, position, point_cloud):
        # print(position, point_cloud.shape)
        occupancy_x = [position[0] - self.bounding_box_size[0] / 2.0, position[0] + self.bounding_box_size[0] / 2.0]
        occupancy_y = [position[1] - self.bounding_box_size[1] / 2.0, position[1] + self.bounding_box_size[1] / 2.0]
        occupancy_z = [position[2] - self.bounding_box_size[2] / 2.0, position[2] + self.bounding_box_size[2] / 2.0]

        x_collisions = np.logical_and(point_cloud[:, 0] > occupancy_x[0], point_cloud[:, 0] < occupancy_x[1])
        y_collisions = np.logical_and(point_cloud[:, 1] > occupancy_y[0],
                                      point_cloud[:, 1] < occupancy_y[1])
        z_collisions = np.logical_and(point_cloud[:, 2] > occupancy_z[0],
                                      point_cloud[:, 2] < occupancy_z[1])

        xyz_collisions = np.logical_and(np.logical_and(x_collisions, y_collisions), z_collisions)
        return point_cloud[np.argwhere(xyz_collisions)].reshape((len(np.argwhere(xyz_collisions)), 3))

    def is_collision_to_path_detected(self, local_translation, point_cloud):
        positions = np.linspace([0.0, 0.0, 0.0], local_translation,
                                self.NUM_POSITIONS_CHECKED_FOR_COLLISIONS)
        for index, pos in enumerate(positions):
            points_collision_detected = self.is_collision_detected(pos, point_cloud)

            # print(pos, points_collision_detected)
            iterations = 0
            if len(points_collision_detected) > 0:
                adjustment = np.average(point_cloud, axis=0) - pos
                while iterations < self.max_collision_avoidance_iterations:
                    points_collision_detected = self.is_collision_detected(pos, point_cloud)
                    if len(points_collision_detected) == 0:
                        return True, pos

                    # print("Adjustment {}".format(adjustment))
                    pos = pos + 0.1 * adjustment
                    iterations += 1

                return True, pos

        return False, positions[len(positions) - 1]

    def get_unrestricted_path_to_local_translation(self, local_translation, depth_front):
        point_cloud = self.get_occupance_point_cloud_from_depth(depth_front)

        dist = np.sqrt(np.sum(np.square(local_translation)))
        # if dist > self.OCCUPANCY_DISTANCE:
        local_translation = (self.OCCUPANCY_DISTANCE / dist) * local_translation

        collision_detected, positions = self.is_collision_to_path_detected(local_translation, point_cloud)

        return collision_detected, positions

    def update_occupancy_point_cloud(self, depth_front):
        print(depth_front.shape, self.horizontal_map.shape, self.vertical_map.shape)
        depth_front = cv2.resize(depth_front, (self.depth_shape[1], self.depth_shape[0]), interpolation=cv2.INTER_AREA)
        x_y = depth_front * np.cos(self.vertical_map)
        z = depth_front * np.sin(self.vertical_map)

        x = x_y * np.cos(self.horizontal_map)
        y = x_y * np.sin(self.horizontal_map)
        point_cloud = np.dstack([x, y, z])

        if len(self.Point_Cloud_Occupancy) > 0:
            # Remove points further than 5 meters away
            dist = np.sqrt(np.sum(np.square(self.Point_Cloud_Occupancy - self.last_translational_pose), axis=1))
            # Add new points
            self.Point_Cloud_Occupancy = self.Point_Cloud_Occupancy[np.argwhere(dist < self.OCCUPANCY_DISTANCE)]
            self.Point_Cloud_Occupancy = np.reshape(self.Point_Cloud_Occupancy, (len(self.Point_Cloud_Occupancy), 3))

            indexes = np.argwhere(depth_front < 5.0)
            Point_Cloud = point_cloud[indexes[:, 0], indexes[:, 1]]
            Point_Cloud = np.reshape(Point_Cloud, (len(Point_Cloud), 3, 1))

            rotation_matrix = quaternion_rotation_matrix(self.last_rotational_pose_quaternion)

            Global_point_clouds = np.dot(rotation_matrix, Point_Cloud)
            new_point_clouds = np.reshape(
                np.dstack([Global_point_clouds[0], Global_point_clouds[1], Global_point_clouds[2]]),
                (len(Point_Cloud), 3)) + self.last_translational_pose

            self.Point_Cloud_Occupancy = np.append(self.Point_Cloud_Occupancy, new_point_clouds, axis=0)
            print(self.Point_Cloud_Occupancy.shape)
            return
        else:
            indexes = np.argwhere(depth_front < self.OCCUPANCY_DISTANCE)
            Point_Cloud = point_cloud[indexes[:, 0], indexes[:, 1]]
            Point_Cloud = np.reshape(Point_Cloud, (len(Point_Cloud), 3, 1))

            rotation_matrix = quaternion_rotation_matrix(self.last_rotational_pose_quaternion)

            Global_point_clouds = np.dot(rotation_matrix, Point_Cloud)
            self.Point_Cloud_Occupancy = np.reshape(
                np.dstack([Global_point_clouds[0], Global_point_clouds[1], Global_point_clouds[2]]),
                (len(Point_Cloud), 3))

            # print(self.Point_Cloud_Occupancy[0], np.dot(rotation_matrix, Point_Cloud[0]))
            # print(Global_point_clouds.shape, Point_Cloud.shape, self.Point_Cloud_Occupancy.shape)

    def global_translation_to_local_translation(self, global_translation, orientation_quaternion):
        inv_rotation_matrix = np.linalg.inv(quaternion_rotation_matrix(orientation_quaternion))

        global_translation = np.reshape(global_translation, (len(global_translation), 3, 1))

        local_translation = np.dot(inv_rotation_matrix, global_translation)
        local_translation = np.reshape(np.dstack([local_translation[0], local_translation[1], local_translation[2]]),
                                       (len(global_translation), 3))

        return local_translation

    def get_global_position_of_paths(self, paths, orientation_quaternion):
        rotation_matrix = quaternion_rotation_matrix(orientation_quaternion)
        paths = np.reshape(paths, (len(paths), 3, 1))
        Global_positions = np.dot(rotation_matrix, paths)
        Global_positions = np.reshape(np.dstack([Global_positions[0], Global_positions[1], Global_positions[2]]),
                                      (len(paths), 3))

        '''Local_positions = np.dot(np.linalg.inv(rotation_matrix), np.reshape(Global_positions, (len(Global_positions), 3, 1)))
        Local_positions = np.reshape(np.dstack([Local_positions[0], Local_positions[1], Local_positions[2]]),
                                      (len(paths), 3))
        paths = np.reshape(paths, (len(paths), 3))

        for index in range(len(paths)):
            print(paths[index], Global_positions[index], Local_positions[index])'''

        return Global_positions

    def get_paths_from_lidar(self, lidar, std_thr=0.7):
        lidar_dist = np.sqrt(np.square(lidar["x"]) + np.square(lidar["y"]) + np.square(lidar["z"]))

        self.Lidar_depth_image = self.depth_to_image(lidar_dist, 100.0)

        lidar_dist = np.where(np.isfinite(lidar_dist), lidar_dist, 100.0)

        mean = np.average(lidar_dist[lidar_dist.shape[0] // 4:3 * lidar_dist.shape[0] // 4,
                          lidar_dist.shape[1] // 4:3 * lidar_dist.shape[1] // 4])
        std = np.std(lidar_dist[lidar_dist.shape[0] // 4:3 * lidar_dist.shape[0] // 4,
                     lidar_dist.shape[1] // 4:3 * lidar_dist.shape[1] // 4])

        thr = np.where(lidar_dist > mean + std_thr * std, 255, 0).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thr)

        centroids = np.delete(centroids, 0, axis=0)
        stats = np.delete(stats, 0, axis=0)

        index = 0
        while index < len(stats):
            centre_x = centroids[index][0]
            if stats[index][4] < 500 or (
                    (centre_x < thr.shape[1] / 4.0 or centre_x > 3.0 * thr.shape[1] / 4.0) and not self.transverse):
                centroids = np.delete(centroids, index, axis=0)
                stats = np.delete(stats, index, axis=0)
            else:
                index += 1

        lidar_paths_detected = []
        if len(centroids) > 0:

            lidar_horizontal_range = 2.0 * np.pi
            lidar_vertical_range = np.pi / 6.0
            for index, centre in enumerate(centroids):
                if index < len(self.linspace_colours[index]):
                    cv2.circle(self.Lidar_depth_image, (int(centre[0]), int(centre[1])), 10, (
                    self.linspace_colours[index][0], self.linspace_colours[index][1], self.linspace_colours[index][2]),
                               2)
                dist = lidar_dist[int(centre[1])][int(centre[0])]
                if np.isnan(dist):
                    dist = 50

                z = lidar["z"][int(centre[1]), int(centre[0])] * 0.75

                x = lidar["x"][int(centre[1]), int(centre[0])] * 0.75
                y = lidar["y"][int(centre[1]), int(centre[0])] * 0.75

                dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                if np.all(np.isfinite([x, y, z])) and dist > 5.0:
                    lidar_paths_detected.append(np.array([x, y, z]))

        return lidar_paths_detected

    def set_cmd_vel_to_reach_pose(self, local_translation):
        dist = np.sqrt(np.sum(np.square(local_translation)))
        if dist > self.OCCUPANCY_DISTANCE:
            # print(self.OCCUPANCY_DISTANCE/dist, local_translation)
            local_translation = (self.OCCUPANCY_DISTANCE / dist) * np.array(local_translation)

        if local_translation[0] > 0.0:
            obsticle_in_sight, local_translation = self.get_unrestricted_path_to_local_translation(local_translation,
                                                                                                   self.depth_front)
            print("Obstacle in Sight: {}".format(obsticle_in_sight))
        print(local_translation)

        # print(local_translation)
        yaw = np.arctan2(local_translation[1], local_translation[0])
        # print("yaw: {}".format(yaw * 180.0 / np.pi))

        if self.dist_front >= self.MAX_VELOSITY_DISTANCE:
            max_velocity_x = self.MAX_X_VELOCITY
        elif self.dist_front < self.MAX_VELOSITY_DISTANCE and self.dist_front > self.MAX_DIST_1_LIMIT:
            m = ((self.MAX_X_VELOCITY - self.VEL_L1) / (self.MAX_VELOSITY_DISTANCE - self.MAX_DIST_1_LIMIT))
            max_velocity_x = m * self.dist_front + self.MAX_X_VELOCITY - m * self.MAX_VELOSITY_DISTANCE

        elif self.dist_front < self.MAX_DIST_1_LIMIT and self.dist_front > self.MAX_DIST_2_LIMIT:
            m = ((self.VEL_L1 - self.VEL_L2) / (self.MAX_DIST_1_LIMIT - self.MAX_DIST_2_LIMIT))
            max_velocity_x = m * self.dist_front + self.VEL_L1 - m * self.MAX_DIST_1_LIMIT
        else:
            max_velocity_x = 0.0

        max_velocity_x *= np.abs(local_translation[0]) / (
                np.abs(local_translation[0]) + np.abs(local_translation[1]) + np.abs(local_translation[2]))

        if self.dist_up + self.dist_down < 2.0:
            self.move_cmd.linear.z = self.pid_velocity_z.calc(5.0 * (self.dist_up - self.dist_down), 0.0)
            max_velocity_x *= 2.0 / (self.dist_up + self.dist_down)
        else:
            self.move_cmd.linear.z = self.pid_velocity_z.calc(local_translation[2], 0.0)

        if self.dist_up + self.dist_down < 2.0:
            # print(self.dist_down - self.dist_up)
            self.move_cmd.linear.z = self.pid_velocity_z.calc(5.0 * (self.dist_down - self.dist_up), 0.0)
            max_velocity_x *= 2.0 / (self.dist_up + self.dist_down)
        elif self.dist_up < 1.0:
            # print(4.0*(self.dist_up - 1.0))
            self.move_cmd.linear.z = self.pid_velocity_z.calc(5.0 * (self.dist_up - 1.0), 0.0)
            if self.move_cmd.linear.z > -0.5:
                self.move_cmd.linear.z = -0.5
            max_velocity_x *= self.dist_up
        elif self.dist_down < 1.0:
            # print(4.0*(1.0 - self.dist_down))
            self.move_cmd.linear.z = self.pid_velocity_z.calc(5.0 * (1.0 - self.dist_down), 0.0)
            if self.move_cmd.linear.z < 0.5:
                self.move_cmd.linear.z = 0.5
            max_velocity_x *= self.dist_down
        else:
            self.move_cmd.linear.z = self.pid_velocity_z.calc(local_translation[2], 0.0)

        # print(max_velocity_x)
        # print(local_translation)

        self.move_cmd.linear.x += 0.05*(self.pid_velocity_x.calc(local_translation[0], 0.0) - self.move_cmd.linear.x)
        if self.move_cmd.linear.x > max_velocity_x:
            self.move_cmd.linear.x = max_velocity_x
            self.pid_velocity_x.calc(0.0, 0.0)

        if self.move_cmd.linear.x < 0.0:
            self.move_cmd.linear.x = 0.0
            self.move_cmd.linear.z = 0.0

        self.move_cmd.angular.z = self.pid_rotation_z.calc(yaw, 0.0)
        local_translation[1] *= 0.5

        if local_translation[1] > 0.5:
            local_translation[1] = 0.5
        elif local_translation[1] < -0.5:
            local_translation[1] = -0.5

        self.move_cmd.linear.y = local_translation[1]

        self.move_cmd.linear.y = 0.0

        return

    # def local_poses_to_global_poses(self):

    def get_possible_paths(self, include_lidar=True, include_front_depth=True, include_up_depth=True,
                           include_down_depth=True):
        lidar_paths = self.get_paths_from_lidar(self.lidar_scan)
        lidar_paths_global = self.get_global_position_of_paths(lidar_paths,
                                                               self.last_rotational_pose_quaternion) + self.last_translational_pose
        print("Lidar:")
        for path in lidar_paths_global:
            print("\tx: {:.2f}\ty: {:.2f}\tz: {:.2f}\tExplored:{}".format(path[0], path[1], path[2],
                                                                          self.path_record_planner.has_path_been_explored(
                                                                              path)))

        depth_front_paths = self.get_paths_from_depth(self.depth_front, self.image_front)
        depth_front_paths_global = self.get_global_position_of_paths(depth_front_paths,
                                                                     self.last_rotational_pose_quaternion) + self.last_translational_pose
        print("Depth Front:")
        for path in depth_front_paths_global:
            print("\tx: {:.2f}\ty: {:.2f}\tz: {:.2f}\tExplored:{}".format(path[0], path[1], path[2],
                                                                          self.path_record_planner.has_path_been_explored(
                                                                              path)))
        depth_up_paths = self.get_paths_from_depth(self.depth_up, self.image_up)
        depth_up_paths_global = self.get_global_position_of_paths(depth_up_paths,
                                                                  quaternion_multiply([0.707, 0.0, -0.707, 0.0],
                                                                                      self.last_rotational_pose_quaternion)) + self.last_translational_pose
        print("Depth Up:")
        for path in depth_up_paths_global:
            print("\tx: {:.2f}\ty: {:.2f}\tz: {:.2f}\tExplored:{}".format(path[0], path[1], path[2],
                                                                          self.path_record_planner.has_path_been_explored(
                                                                              path)))

        depth_down_paths = self.get_paths_from_depth(self.depth_down, self.image_down)
        depth_down_paths_global = self.get_global_position_of_paths(depth_down_paths,
                                                                    quaternion_multiply([0.707, 0.0, 0.707, 0.0],
                                                                                        self.last_rotational_pose_quaternion)) + self.last_translational_pose
        print("Depth down:")
        for path in depth_down_paths_global:
            print("\tx: {:.2f}\ty: {:.2f}\tz: {:.2f}\tExplored:{}".format(path[0], path[1], path[2],
                                                                          self.path_record_planner.has_path_been_explored(
                                                                              path)))

        possible_local_paths = []
        possible_global_pose = []

        if include_lidar:
            possible_local_paths.extend(lidar_paths)
            possible_global_pose.extend(lidar_paths_global)

        if include_front_depth:
            possible_local_paths.extend(depth_front_paths)
            possible_global_pose.extend(depth_front_paths_global)

        if include_up_depth:
            possible_local_paths.extend(depth_up_paths)
            possible_global_pose.extend(depth_up_paths_global)

        if include_down_depth:
            possible_local_paths.extend(depth_down_paths)
            possible_global_pose.extend(depth_down_paths_global)
        # possible_paths.extend(depth_up_paths)
        # possible_paths.extend(depth_down_paths)

        return possible_local_paths, possible_global_pose

    def random_explore(self):
        if self.path_record_planner.has_path_been_explored(self.last_translational_pose):
            self.num_positions_explored += 1
            print("Position Explored Previously {}".format(self.num_positions_explored))
        else:
            self.num_positions_explored = 0
            print("Position Not Explored Previously {}".format(self.num_positions_explored))

        if self.num_positions_explored > self.NUM_POSITIONS_EXPLORE_TO_TRAVERSE or ( False and (
                self.i_rgb > self.limit_index and self.limit_Dataset_genetation)):
            # following explored path
            self.last_followed_path = None
            self.num_positions_explored = 0
            self.random_exploration = False
            self.transverse = True
            self.time_search_started = rospy.get_time()
            self.num_unexplored_positions = 0
            return

        possible_paths, possible_paths_global = self.get_possible_paths(include_up_depth=False,
                                                                        include_down_depth=False)

        depth_front_paths = self.get_paths_from_depth(self.depth_front, self.image_front)
        lidar_paths = self.get_paths_from_lidar(self.lidar_scan)

        paths_not_previously_explored = []
        for index, path in enumerate(possible_paths):
            if not self.path_record_planner.has_path_been_explored(possible_paths_global[index]):
                paths_not_previously_explored.append(path)

        if len(depth_front_paths) > 0 and len(lidar_paths) > 0:
            for index_depth, path_depth in enumerate(depth_front_paths):
                for index_lidar, path_lidar in enumerate(lidar_paths):
                    if self.is_similar_path(path_depth,
                                            path_lidar) and np.any(self.last_followed_path != None) and (
                    self.is_similar_path(path_lidar,
                                         self.last_followed_path)):
                        print("Exploring unexplored path visible in both front depth depth and lidar")
                        self.last_followed_path = path_depth
                        self.set_cmd_vel_to_reach_pose(path_depth)
                        return

        if len(paths_not_previously_explored) > 0:
            print("Exploring unexplored path")
            if len(lidar_paths) > 1:
                indexes_seen_again = []
                for path in reversed(
                        self.get_global_position_of_paths(lidar_paths, self.last_rotational_pose_quaternion)):
                    not_in_recents = True
                    for index, explored in enumerate(self.Recent_Exploration_added_nodes):
                        if self.is_similar_path(explored, path, 2.5):
                            not_in_recents = False
                            if not index in indexes_seen_again:
                                indexes_seen_again.append(index)

                    if not self.path_record_planner.has_path_been_explored(
                            path + self.last_translational_pose) and not_in_recents:
                        if self.limit_Dataset_genetation:
                            if self.i_rgb < 0.75*self.limit_index:
                                self.path_record_planner.add_exploration_node(path + self.last_translational_pose)
                        else:
                            self.path_record_planner.add_exploration_node(path + self.last_translational_pose)
                        self.Recent_Exploration_added_nodes.append(path)

                    for index in reversed(range(len(self.Recent_Exploration_added_nodes))):
                        if not index in indexes_seen_again:
                            del self.Recent_Exploration_added_nodes[index]

                '''for path in lidar_paths:
                    if self.is_similar_path(path, self.last_followed_path, 1.5):
                        self.last_followed_path = path
                        self.set_cmd_vel_to_reach_pose(self.last_followed_path)

                        return

                index = None
                while index == None:
                    index = input("Select Path To Follow: ")
                    try:
                        index = int(index)
                        if index > len(lidar_paths) - 1:
                            index = None
                    except:
                        index = None

                self.last_followed_path = lidar_paths[index]
                self.set_cmd_vel_to_reach_pose(self.last_followed_path)'''

            for path in paths_not_previously_explored:
                if np.any(self.last_followed_path != None):
                    if self.is_similar_path(self.last_followed_path, path, 3.0):
                        self.last_followed_path = path

                        self.set_cmd_vel_to_reach_pose(path)
                        return

            self.last_followed_path = paths_not_previously_explored[0]
            self.set_cmd_vel_to_reach_pose(paths_not_previously_explored[0])

        else:
            if len(possible_paths) > 0:
                print("Exploring previously explored path path")
                if len(possible_paths) > 1:
                    close_nodes = [0.0] * len(possible_paths)
                    for index, path in enumerate(possible_paths):
                        close_nodes[index] = self.path_record_planner.get_closest_distance_in_map(path)

                    # Explore node with the lowest number of explored nodes in the vicinity
                    self.last_followed_path = possible_paths[np.argmax(close_nodes)]
                    self.set_cmd_vel_to_reach_pose(possible_paths[np.argmax(close_nodes)])
                else:
                    self.last_followed_path = possible_paths[0]
                    self.set_cmd_vel_to_reach_pose(possible_paths[0])

            else:
                # No paths visible
                '''self.random_exploration = False
                self.search_path = True
                self.time_search_started = rospy.get_time()'''
                self.last_followed_path = None
                self.num_positions_explored = 0
                self.random_exploration = False
                self.transverse = True
                self.time_search_started = rospy.get_time()
                self.num_unexplored_positions = 0
            return

        return

    def search_for_path(self):
        print("Time passed: {:.2f}".format(rospy.get_time() - self.time_search_started))

        if rospy.get_time() - self.time_search_started < np.pi/(self.MAX_X_VELOCITY * (0.3 / 4.0)):
            self.move_cmd.linear.x = 0.0
            self.move_cmd.linear.y = 0.0
            self.move_cmd.linear.z = 0.0
            self.move_cmd.angular.z = np.pi/self.MAX_X_VELOCITY * (0.3 / 4.0)
        else:
            self.search_path = False
            # self.transverse = True
            self.random_exploration = True

        '''if rospy.get_time() - self.time_search_started < 1.0:
            self.move_cmd.linear.x = 0.0
            self.move_cmd.linear.y = 0.0
            self.move_cmd.linear.z = 0.0
            self.move_cmd.angular.z = 0.0

            self.pid_velocity_x = PIDArduino(sampletime=self.sample_time, kp=-0.4, ki=-0.1, kd=-0.001,
                                             out_min=0.0,
                                             out_max=self.MAX_X_VELOCITY, time=rospy.get_time)
            self.pid_velocity_z = PIDArduino(sampletime=self.sample_time, kp=-0.25, ki=-0.1, kd=-0.0010,
                                             out_min=-self.MAX_X_VELOCITY * (1.5 / 4.0),
                                             out_max=self.MAX_X_VELOCITY * (1.5 / 4.0), time=rospy.get_time)
            self.pid_rotation_z = PIDArduino(sampletime=self.sample_time, kp=-0.5, ki=-0.1, kd=-0.0010,
                                             out_min=-self.MAX_X_VELOCITY * (0.3 / 4.0),
                                             out_max=self.MAX_X_VELOCITY * (0.3 / 4.0), time=rospy.get_time)
        elif rospy.get_time() - self.time_search_started > 1.0 and rospy.get_time() - self.time_search_started < 4.0:
            self.move_cmd.angular.z = np.pi / 3.0
        else:
            self.search_path = False
            # self.transverse = True
            self.random_exploration = True
        elif rospy.get_time() - self.time_search_started > 2.0 and rospy.get_time() - self.time_search_started < 6.5:
            self.move_cmd.angular.z = -np.pi / 3.0
        elif rospy.get_time() - self.time_search_started > 6.0:
            self.search_path = False
            #self.transverse = True
            self.random_exploration = True

        local_paths, global_paths = self.get_possible_paths(include_up_depth=False, include_down_depth=False)
        for index in range(len(local_paths)):
            if self.path_record_planner.has_path_been_explored(
                    global_paths[index]) == False and rospy.get_time() - self.time_search_started < 3.0:
                self.move_cmd.angular.z = 0.0
                self.random_exploration = True
                self.search_path = False'''

        return

    def traverse_to_unexplored_path(self):
        if not self.path_record_planner.has_path_been_explored(self.last_translational_pose):
            self.num_unexplored_positions += 1
            print("Position Not Explored Previously {}".format(self.num_unexplored_positions))
        else:
            self.num_unexplored_positions = 0
            print("Position Explored Previously {}".format(self.num_unexplored_positions))

        if self.num_unexplored_positions > self.THR_UNEXPLORED:
            self.Transversal_Path = None
            self.transverse = False
            self.random_exploration = True

        if np.any(self.Transversal_Path == None) or ((self.i_rgb > self.limit_index and self.limit_Dataset_genetation) and not self.Path_Traversed_to_origin):
            if (self.i_rgb > self.limit_index and self.limit_Dataset_genetation) :
                self.Path_Traversed_to_origin = True
            self.taverse_slips = 0
            print(self.taverse_slips)
            self.Transversal_Path = self.path_record_planner.get_transverse_path_to_closest_exploration_node(
                self.taverse_slips, (self.i_rgb > self.limit_index and self.limit_Dataset_genetation))
        print("{} Nodels left in transversal path".format(len(self.Transversal_Path)))
        print("Following positiono : \tx: {:.2f}\ty: {:.2f}\tz: {:.2f}".format(self.Transversal_Path[0][0],
                                                                               self.Transversal_Path[0][1],
                                                                               self.Transversal_Path[0][2]))

        distances = np.sqrt(np.sum(np.square(np.array(self.Transversal_Path) - self.last_translational_pose), axis=1))
        indexes = np.argwhere(distances < self.THRESHOLD_REMOVE_FROM_PATH)
        if len(self.Transversal_Path) > 2:
            if len(indexes) > 0:
                index = np.max(indexes)
                print(distances[index])
                for i in range(index):
                    if len(self.Transversal_Path) > 2:
                        del self.Transversal_Path[0]
            else:
                if np.min(distances) > 20.0:
                    print(self.taverse_slips)
                    # self.taverse_slips += 1
                    self.Transversal_Path = self.path_record_planner.get_transverse_path_to_closest_exploration_node(
                        self.taverse_slips)

        else:
            # Turn to unexplored path and go to random explore
            self.last_followed_path = self.global_translation_to_local_translation(
                [self.Transversal_Path[len(self.Transversal_Path) - 1] - self.last_translational_pose],
                self.last_rotational_pose_quaternion)[0]
            self.set_cmd_vel_to_reach_pose(self.last_followed_path)
            self.move_cmd.linear.x = 0.0
            self.move_cmd.linear.y = 0.0

            if np.abs(self.move_cmd.angular.z) < 0.1:
                self.Transversal_Path = None
                self.transverse = False
                self.random_exploration = True

            return

        '''distances = np.sqrt(np.sum(np.square(self.Transversal_Path - self.last_translational_pose), axis=1))
        min_index = min(np.argmin(distances) + 1, len(distances) - 1)
        print(self.global_translation_to_local_translation([self.Transversal_Path[min_index] - self.last_translational_pose], self.last_rotational_pose_quaternion)[0])
        self.set_cmd_vel_to_reach_pose(self.global_translation_to_local_translation([self.Transversal_Path[min_index] - self.last_translational_pose], self.last_rotational_pose_quaternion)[0])
        '''
        local_paths, global_paths = self.get_possible_paths(include_up_depth=False, include_down_depth=False)
        depth_front_paths = self.get_paths_from_depth(self.depth_front, self.image_front)
        dist = [0.0] * len(global_paths)
        dist_index = [0] * len(global_paths)
        if len(global_paths) > 0:
            for index, path in enumerate(global_paths):
                distances = np.sqrt(
                    np.sum(np.square(self.Transversal_Path[0:max(len(self.Transversal_Path), 2400)] - path), axis=1))
                ind = np.argmin(distances)
                dist_index[index] = ind
                dist[index] = distances[ind]

            dist = np.array(dist)
            dist_index = np.array(dist_index)
            print(dist)

            if len(np.argwhere(dist < 20.0)) > 1:
                indexes = np.argwhere(dist < 20.0)
                # print(indexes)
                for index in indexes:
                    if index == len(distances) - 1:
                        self.Transversal_Path = self.Transversal_Path[
                                                len(self.Transversal_Path) - 2: len(self.Transversal_Path)]
                    if dist_index[index] == np.max(dist_index[indexes]):
                        # print(index)
                        followed_path = local_paths[index[0]]
                        self.last_followed_path = followed_path
                        self.set_cmd_vel_to_reach_pose(followed_path)
            elif np.min(dist) < 15.0:
                index = np.argmin(dist)
                followed_path = local_paths[index]
                self.last_followed_path = followed_path
                self.set_cmd_vel_to_reach_pose(followed_path)
            else:
                distances = np.sqrt(np.sum(np.square(self.Transversal_Path - self.last_translational_pose), axis=1))
                min_index = min(np.argmin(distances) + 1, len(distances) - 1)
                print(self.global_translation_to_local_translation(
                    [self.Transversal_Path[min_index] - self.last_translational_pose],
                    self.last_rotational_pose_quaternion)[0])
                self.set_cmd_vel_to_reach_pose(self.global_translation_to_local_translation(
                    [self.Transversal_Path[min_index] - self.last_translational_pose],
                    self.last_rotational_pose_quaternion)[0])
        else:
            distances = np.sqrt(np.sum(np.square(self.Transversal_Path - self.last_translational_pose), axis=1))
            min_index = min(np.argmin(distances) + 1, len(distances) - 1)
            print(self.global_translation_to_local_translation(
                [self.Transversal_Path[min_index] - self.last_translational_pose],
                self.last_rotational_pose_quaternion)[0])
            self.set_cmd_vel_to_reach_pose(self.global_translation_to_local_translation(
                [self.Transversal_Path[min_index] - self.last_translational_pose],
                self.last_rotational_pose_quaternion)[0])
        return

    def timer_vel_publisher_callback(self, event=None):
        os.system('clear')
        # self.update_occupancy_point_cloud(self.depth_front)
        print("Time:", rospy.get_time())

        print("\nDistances:")
        print("Front: ", self.dist_front)
        print("Up: ", self.dist_up, "AVE: ", self.ave_dist_up)
        print("Down: ", self.dist_down, "AVE: ", self.ave_dist_down)
        print("\n")

        if self.random_exploration:
            print("State: Random Exploration")

        if self.search_path:
            print("State: Searching for path")

        if self.transverse:
            print("State: Transversing to path")

        print("Position:")
        print("\tTranslation[m]: \tx: {:.2f}\ty: {:.2f}\tz: {:.2f}".format(self.last_translational_pose[0],
                                                                           self.last_translational_pose[1],
                                                                           self.last_translational_pose[2]))
        print(
            "\tRotation[deg]: \tx: {:.2f}\ty: {:.2f}\tz: {:.2f}".format(self.last_rotational_pose[0] * (180.0 / np.pi),
                                                                        self.last_rotational_pose[1] * (180.0 / np.pi),
                                                                        self.last_rotational_pose[2] * (180.0 / np.pi)))

        if not self.path_record_planner.has_path_been_explored(self.last_translational_pose):
            self.num_positions_not_exploured_in_previous_runs += 1

        if self.i_rgb > 0:
            print("{:.2f} % of positions not explored previously".format(
                100.0 * self.num_positions_not_exploured_in_previous_runs / self.i_rgb))
            print("distance to origin {:.2f}".format(np.sqrt(np.sum(np.square(self.last_translational_pose - self.path_record_planner.root_node.pose)))))

        if self.limit_Dataset_genetation and ((False and self.i_rgb >= 2.2 * self.limit_index) or (self.i_rgb >= self.limit_index and (np.sqrt(np.sum(np.square(self.last_translational_pose - self.path_record_planner.root_node.pose))) < 10.0))):
            print("Terminating process ...........", (1.0*self.num_positions_not_exploured_in_previous_runs)/self.i_rgb)
            self.flag_exit_program = True

            self.lidar_subscriber.unregister()
            self.depth_front_subcriber.unregister()
            self.depth_up_subcriber.unregister()
            self.depth_down_subcriber.unregister()

            self.image_front_sunscriber.unregister()
            self.image_up_sunscriber.unregister()
            self.image_down_sunscriber.unregister()

            self.pose_subscriber.unregister()
            self.magnetometer_subscriber.unregister()
            self.imu_subscriber.unregister()

            if not self.PATH_WRITTEN:
                self.PATH_WRITTEN = True
                if (1.0*self.num_positions_not_exploured_in_previous_runs)/self.i_rgb < 0.1 :
                    shutil.rmtree(self.Save_Path)
                else:
                    self.path_record_planner.save_map_to_file(self.map_saved_file)

            if not len(self.path_record_planner.exploration_nodes) > 0:
                if os.path.isfile(self.map_saved_file):
                    os.remove(self.map_saved_file)

        self.path_record_planner.display_exploration_nodes()
        if self.random_exploration:
            self.random_explore()

        elif self.search_path:
            self.search_for_path()

        elif self.transverse:
            self.traverse_to_unexplored_path()
        else:
            raise Exception("Unknown State Reached")

        self.cmd_vel_Publisher.publish(self.move_cmd)
        print(self.move_cmd)

        img_display = cv2.hconcat([self.image_up, self.image_front, self.image_down])
        img_depth = cv2.hconcat([self.depth_to_image(self.depth_up), self.depth_to_image(self.depth_front),
                                 self.depth_to_image(self.depth_down)])

        if np.any(self.Lidar_depth_image == None):
            cv2.imshow("frame", cv2.vconcat([img_display, img_depth]))
        else:
            img_lidar = cv2.resize(self.Lidar_depth_image, (img_depth.shape[1], self.Lidar_depth_image.shape[0] * 3),
                                   interpolation=cv2.INTER_AREA)
            cv2.imshow("frame", cv2.vconcat([img_display, img_depth, img_lidar]))

        cv2.waitKey(5)


if __name__ == '__main__':
    x = dataset_collector()