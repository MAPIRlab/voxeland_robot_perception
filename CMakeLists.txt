cmake_minimum_required(VERSION 3.8)
project(voxeland_robot_perception)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(segmentation_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(modules)

# Install Python executables
install(PROGRAMS
  voxeland_robot_perception/robot_perception_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME})

ament_package()