#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import cv2
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from utils.math import concatenate_translation, polar_to_xy, quaternion_to_rotation_matrix, xyz_to_homogenous

class LiDARCameraProjectionNode:

    def __init__(self):
        rospy.init_node('lidar_camera_projection_node', anonymous=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.image_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.lidar_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.timer = rospy.Timer(rospy.Duration(0.03), self.projection_callback)
        self.bridge = CvBridge()

        self.laser_angle_increment = rospy.get_param('~laser_angle_increment', 0.1)  # radian
        self.baseline = rospy.get_param('~baseline', 0.05)  # 5 cm

        self.latest_image = None
        self.latest_scan = None

        self.base_tf_name = "base_link"
        self.camera_tf_name = "camera_link"
        self.camera_optical_tf_name = "camera_rgb_optical_frame"
        self.LiDAR_tf_name = "base_scan"

        self.camera_intrinsic_matrix = np.array([
            1696.80268, 0.0, 960.5,
            0.0, 1696.80268, 540.5,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)

        self.initialize = False

        self.T_camera_LIDAR = None
        self.projection_matrix = None

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def scan_callback(self, scan_msg):
        try:
            self.laser_angle_increment = scan_msg.angle_increment
            self.latest_scan = scan_msg.ranges
        except Exception as e:
            rospy.logerr(f"Failed to process scan data: {e}")

    def lookup_transform(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time())
            return trans.transform.translation, trans.transform.rotation
        except tf2_ros.LookupException as e:
            rospy.logwarn(f"Could not transform {source_frame} to {target_frame}: {e}")
            return None, None

    def projection_callback(self, event):
        if not self.initialize:
            trans, rot = self.lookup_transform(self.camera_tf_name, self.LiDAR_tf_name)
            if trans is not None and rot is not None:
                self.T_LiDAR_camera = concatenate_translation(
                    quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w),
                    np.array([trans.x, trans.y, trans.z])
                )

            trans, rot = self.lookup_transform(self.camera_optical_tf_name, self.base_tf_name)
            if trans is not None and rot is not None:
                T_camera_base = concatenate_translation(
                    quaternion_to_rotation_matrix(rot.x, rot.y, rot.z, rot.w),
                    np.array([trans.x, trans.y, trans.z])
                )
                T_camera_base[:3, 3] = 0  # Set translation to zero
                self.projection_matrix = self.camera_intrinsic_matrix @ T_camera_base[:3, :4]

            if self.T_LiDAR_camera is not None and self.projection_matrix is not None:
                self.initialize = True

        if self.initialize and self.latest_image is not None and self.latest_scan is not None:
            debug_image = self.latest_image.copy()

            for i, scan_point in enumerate(self.latest_scan):
                if scan_point < 0.1:
                    continue
                P_scan = polar_to_xy(self.laser_angle_increment * i, scan_point)
                P_camera = self.T_LiDAR_camera @ xyz_to_homogenous(P_scan[0], P_scan[1], 0)

                if P_camera[0, 0] < 0:
                    continue
                P_image = self.projection_matrix @ P_camera

                image_x = P_image[0, 0] / P_image[2, 0]
                image_y = P_image[1, 0] / P_image[2, 0]

                if 0 <= image_x < self.latest_image.shape[1] and 0 <= image_y < self.latest_image.shape[0]:
                    circle_r = 6 - int(scan_point / 3.0 * 3)
                    cv2.circle(debug_image, (int(image_x), int(image_y)), circle_r, (0, 0, 255), -1)

                    cv2.putText(debug_image, f'{scan_point:.2f}m', (int(image_x), int(image_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            resize_image = cv2.resize(debug_image, (debug_image.shape[1] // 2, debug_image.shape[0] // 2))
            cv2.imshow('Camera Image', resize_image)
            cv2.waitKey(1)

def main():
    try:
        node = LiDARCameraProjectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

