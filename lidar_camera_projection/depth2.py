import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformListener, Buffer
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
from utils.math import concatenate_translation, polar_to_xy, quaternion_to_rotation_matrix, xyz_to_homogenous

class LiDARCameraProjectionNode(Node):

    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.image_subscriber = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, qos_profile_sensor_data)
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.03, self.projection_callback)
        self.bridge = CvBridge()

        self.laser_angle_increment = self.declare_parameter('laser_angle_increment', 0.1).value  # radian
        self.baseline = self.declare_parameter('baseline', 0.05).value  # 5 cm

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
            self.get_logger().error(f"Failed to convert image: {e}")

    def scan_callback(self, scan_msg):
        try:
            self.laser_angle_increment = scan_msg.angle_increment
            self.latest_scan = scan_msg.ranges
        except Exception as e:
            self.get_logger().error(f"Failed to process scan data: {e}")

    def lookup_transform(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return trans.transform.translation, trans.transform.rotation
        except Exception as e:
            self.get_logger().warn(f"Could not transform {source_frame} to {target_frame}: {e}")
            return None, None

    def projection_callback(self):
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

def main(args=None):
    rclpy.init(args=args)
    node = LiDARCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

