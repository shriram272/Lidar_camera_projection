import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import numpy as np
import cv2
from geometry_msgs.msg import TransformStamped
from depth_anything.blocks import FeatureFusionBlock, _make_scratch
import torch
from depth_anything.util.math import concatenate_translation, polar_to_xy, quaternion_to_rotation_matrix, xyz_to_homogenous
from torch import nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import torch.nn.functional as F

# DepthAnything model definition
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()

        assert encoder in ['vits', 'vitb', 'vitl']

        if localhub:
            self.pretrained = torch.hub.load('src/depth_anything_ros/src/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)

        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)


class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


# LiDARCameraProjectionNode definition
class LiDARCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_subscriber = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.timer = self.create_timer(0.03, self.projection_callback)
        self.bridge = CvBridge()

        self.laser_angle_increment = self.get_parameter('laser_angle_increment').value  # radian
        self.baseline = self.get_parameter('baseline').value  # 5 cm

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

        # Load DepthAnything model
        self.depth_anything_model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitb14")
        self.depth_anything_model.eval()

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
        except tf2_ros.LookupException as e:
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

            # Convert the debug_image to a tensor and pass it through the DepthAnything model
            input_image = torch.from_numpy(debug_image).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                depth_map = self.depth_anything_model(input_image)

            # Overlay the depth map on the original image
            depth_map_resized = cv2.resize(depth_map.squeeze(0).cpu().numpy(), (debug_image.shape[1], debug_image.shape[0]))
            depth_map_normalized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX)
            depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            combined_image = cv2.addWeighted(debug_image, 0.7, depth_map_colored, 0.3, 0)

            resize_image = cv2.resize(combined_image, (combined_image.shape[1] // 2, combined_image.shape[0] // 2))
            cv2.imshow('Camera Image with Depth', resize_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LiDARCameraProjectionNode()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
