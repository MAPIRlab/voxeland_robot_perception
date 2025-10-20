"""
Universal Semantic Mapping Launch File for Voxeland

This unified launch file supports TALOS, Detectron2, and YOLOE detectors.
Simply change the object_detector parameter to switch between them:

Examples:
  # For TALOS (open vocabulary)
  ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=talos

  # For Detectron2 (COCO categories)  
  ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=detectron

  # For YOLOE (open vocabulary)
  ros2 launch voxeland_robot_perception semantic_mapping.launch.py object_detector:=yoloe

The launch file automatically configures the appropriate service names,
parameters, and features based on the selected detector.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.frontend.parse_substitution import parse_substitution


def launch_arguments():
    return [
        # Core parameters
        DeclareLaunchArgument("dataset", default_value="other"),
        DeclareLaunchArgument("object_detector", default_value="detectron", 
                            description="Detector type: 'talos', 'yoloe', or 'detectron'"),

        # Camera and sensor topics
        DeclareLaunchArgument("topic_camera_info", default_value="/camera/camera_info"),
        DeclareLaunchArgument("topic_rgb_image", default_value="/camera/rgb"),
        DeclareLaunchArgument("topic_depth_image", default_value="/camera/depth"),
        DeclareLaunchArgument("topic_localization", default_value="/amcl_pose"),

        # Frame IDs
        DeclareLaunchArgument("map_frame_id", default_value="map"),
        DeclareLaunchArgument("robot_frame_id", default_value="camera"),
        DeclareLaunchArgument("camera_frame_id", default_value="camera"),

        # Mapping configuration
        DeclareLaunchArgument("mapping_mode", default_value="XYZSemantics"),
        DeclareLaunchArgument("semantic_segmentation_mode", default_value="service"),
        DeclareLaunchArgument("topic_sem_seg", default_value="/ViMantic/Detections"),
        
        # Open vocabulary specific parameters (used only with TALOS)
        DeclareLaunchArgument("initial_categories", default_value="",
                            description="Initial categories for open vocabulary (empty for pure open vocab)"),
        DeclareLaunchArgument("save_categories_file", default_value="/tmp/voxeland_categories.json",
                            description="File to save discovered categories"),
        
        # Display options
        DeclareLaunchArgument("use_xterm", default_value="true",
                            description="Use xterm for robot_perception_node output"),
        
        # RViz configuration
        DeclareLaunchArgument("rviz_config", 
            default_value=PathJoinSubstitution([
                FindPackageShare('voxeland_robot_perception'),
                'rviz',
                'rgbd_nn_map.rviz'
            ]),
            description="Path to RViz configuration file"
        ),
    ]


def launch_setup(context, *args, **kwargs):
    # Get detector type from context
    detector_type = context.launch_configurations['object_detector'].lower()
    use_xterm = context.launch_configurations['use_xterm'].lower() == "true"
    
    # Configure detector-specific parameters
    if detector_type == "talos":
        service_name = "/talos/segment"
        object_detector_name = "TALOS"
        # TALOS-specific parameters
        extra_params = {
            "initial_categories": LaunchConfiguration('initial_categories'),
            "save_categories_file": LaunchConfiguration('save_categories_file'),
            "filter_semantics": True,
        }
    elif detector_type == "yoloe":
        service_name = "/yoloe/segment"
        object_detector_name = "YOLOE"
        # YOLOE supports open vocabulary
        extra_params = {
            "initial_categories": LaunchConfiguration('initial_categories'),
            "save_categories_file": LaunchConfiguration('save_categories_file'),
            "filter_semantics": True,
        }
    else:  # detectron2 or any other
        service_name = "/detectron/segment"
        object_detector_name = "Detectron2"
        # Detectron2 doesn't need these parameters
        extra_params = {}
    
    # Base parameters common to all detectors
    base_parameters = {
        "dataset": LaunchConfiguration('dataset'),
        "object_detector": object_detector_name,
        
        # Input topics
        "topic_camera_info": LaunchConfiguration('topic_camera_info'),
        "topic_rgb_image": LaunchConfiguration('topic_rgb_image'),
        "topic_depth_image": LaunchConfiguration('topic_depth_image'),
        "topic_localization": LaunchConfiguration('topic_localization'),
        
        # Segmentation configuration
        "semantic_segmentation_mode": LaunchConfiguration('semantic_segmentation_mode'),
        "service_sem_seg": service_name,
        "topic_sem_seg": LaunchConfiguration('topic_sem_seg'),
        
        # Image message types
        "rgb_image_type": "Image",
        "depth_image_type": "Image",
        
        # Camera calibration (intrinsics from topic)
        "intrinsics_from_topic": True,
        "width": 1920,
        "height": 1080,
        "cx": 959.5,
        "cy": 539.5,
        "fx": 1371.022,
        "fy": 1371.022,
        "camera_max_depth": 10.0,
        
        # Depth limits
        "limit_reliable_depth": False,
        "min_reliable_depth": 0.01,
        "max_reliable_depth": 3.00,
        
        # Frame IDs
        "map_frame_id": LaunchConfiguration('map_frame_id'),
        "robot_frame_id": LaunchConfiguration('robot_frame_id'),
        "camera_frame_id": LaunchConfiguration('camera_frame_id'),
        
        # Output configuration
        "pointcloud_type": LaunchConfiguration('mapping_mode'),
        "topic_pointcloud_output": "cloud_in",
    }
    
    # Merge base parameters with detector-specific ones
    all_parameters = {**base_parameters, **extra_params}
    
    # Robot perception node
    robot_perception_node = Node(
        package="voxeland_robot_perception",
        executable="robot_perception_node.py",
        name="robot_perception_node",
        prefix="xterm -hold -e" if use_xterm else "",
        output="screen",
        parameters=[all_parameters]
    )
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')]
    )
    
    return [
        robot_perception_node,
        rviz_node,
    ]


def generate_launch_description():
    launch_description = [
        # Set environment variables for better logging
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
    ]
    
    # Add launch arguments
    launch_description.extend(launch_arguments())
    
    # Add launch setup function
    launch_description.append(OpaqueFunction(function=launch_setup))
    
    return LaunchDescription(launch_description)