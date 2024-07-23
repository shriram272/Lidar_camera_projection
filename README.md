This project was aimed at integrating a 2D Lidar and images from monocular camera to get better depth estimation.
Normally most depth estimation like Depth Anything determine the relative depth and metric depth maps require more training or separate models like Zoedepth.
This problem can be solved in a simpler way by using a 2D lidar laser scan data in a single plane and calculate a scaling factor to determine the metric depth map from relative depth map.

The LiDAR scan points are projected onto the camera image through several steps involving transformations and projections. Here's a detailed breakdown of the process:

    Transform Lookup:
        The node first looks up the transform between the LiDAR frame and the camera frame using lookup_transform. This transform provides the translation and rotation between the two frames.
        The rotation is converted into a rotation matrix using quaternion_to_rotation_matrix.
        The translation and rotation are then concatenated into a transformation matrix T_LiDAR_camera.

    Projection Matrix:
        The node also looks up the transform between the camera optical frame and the base frame, and constructs a transformation matrix T_camera_base.
        The projection matrix is computed as the product of the camera intrinsic matrix and the T_camera_base transformation matrix.

    Projection Process:
        For each point in the LiDAR scan, the polar coordinates (angle and distance) are converted into Cartesian coordinates using polar_to_xy.
        These coordinates are then transformed into the camera frame using the transformation matrix T_LiDAR_camera.
        The 3D point in the camera frame is projected onto the image plane using the projection matrix. This involves converting the 3D coordinates to homogeneous coordinates with xyz_to_homogenous and then applying the projection matrix.

    Pixel Coordinates Calculation:
        The projected 2D point on the image plane is obtained by normalizing the homogeneous coordinates.
        The x and y coordinates of the projected point are then checked to ensure they fall within the bounds of the image.

    Drawing on Image:
        If the projected point is within the image bounds, a circle is drawn on the image at that location.
        The disparity between the projected x-coordinate and the original x-coordinate is used to estimate the depth. This depth is annotated on the image near the corresponding circle.

        

https://github.com/user-attachments/assets/7ce4db84-49d8-416f-9b72-a6059d5d16ff



INTEGTATION WITH DEPTH ANYTHING
The relative depth map was published as a image topic.
The relative depth values and the depth of real depth of corresponding points as calculated by the lidar was used to get a average scaling factor.
This was then used to multiply the relative depths and get a metric depth map.
![WhatsApp Image 2024-07-23 at 23 31 34](https://github.com/user-attachments/assets/c35972ab-b5dd-462b-b4e8-e5d51f05bd1b)



   
