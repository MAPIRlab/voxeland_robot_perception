# Voxeland Robot Perception

The robot perception package is used by [Voxeland](https://github.com/MAPIRlab/Voxeland) to provide the input point clouds. Specifically, it is responsible for synchronizing the color and depth images, (optionally) interfacing with the semantic segmentation provider, and generating a 3D point cloud from the resulting information.

See [the launch files](launch) for the parameters. 

To use without semantic information (and, thus, without depending on an external classifier), select `pointcloud_type` which does not contain semantic information (`XYZ` or `RGB`).

To use WITH semantics, see the readme of the main [Voxeland repo](https://github.com/MAPIRlab/Voxeland), which links to compatible semantic segmentation packages.

The [requirements] file included in this repo lists the (non-ROS) external dependencies. The ROS packages that this project needs are listed in the standard way, through the `package.xml` file, so that they can be installed with `rosdep`.