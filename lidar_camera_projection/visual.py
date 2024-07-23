import roslib
roslib.load_manifest("gaussian_markers")
import rospy
import PyKDL
from visualization_msgs.msg import Marker, MarkerArray
import visualization_msgs
from geometry_msgs.msg import Point
import numpy as np
from plyfile import PlyData

def read_ply_file(file_path):
    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'], vertex['opacity']]).T
    scales = np.vstack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']]).T
    rotations = np.vstack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']]).T
    return points, colors, scales, rotations

def create_marker(id, position, orientation, scale, color, marker_type=Marker.SPHERE):
    marker = Marker()
    marker.header.frame_id = "/root"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "ply_visualization"
    marker.id = id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.a = color[3]  # Use opacity for alpha
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    return marker

def main():
    rospy.init_node('ply_visualization', anonymous=True)
    marker_pub = rospy.Publisher("visualization_markers", Marker, queue_size=10)
    rate = rospy.Rate(10)

    ply_file_path = "path/to/your/ply/file.ply"  # Update this path
    points, colors, scales, rotations = read_ply_file(ply_file_path)

    id = 0
    for point, color, scale, rotation in zip(points, colors, scales, rotations):
        position = point
        orientation = rotation
        marker = create_marker(id, position, orientation, scale, color)
        marker_pub.publish(marker)
        id += 1
        rate.sleep()

    rospy.spin()

if __name__ == "__main__":
    main()
