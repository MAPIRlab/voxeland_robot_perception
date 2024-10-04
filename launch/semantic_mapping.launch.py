import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,SetLaunchConfiguration,IncludeLaunchDescription,SetEnvironmentVariable,OpaqueFunction,GroupAction
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from launch.frontend.parse_substitution import parse_substitution

#===========================
def launch_arguments():
    return [
        DeclareLaunchArgument("dataset", default_value="other"),
        DeclareLaunchArgument("object_detector", default_value="Detectron2"),

        DeclareLaunchArgument("topic_camera_info", default_value="/camera/camera_info"),
        DeclareLaunchArgument("topic_rgb_image", default_value="/camera/rgb"),
        DeclareLaunchArgument("topic_depth_image", default_value="/camera/depth"),
        DeclareLaunchArgument("topic_localization", default_value="/amcl_pose"),

        DeclareLaunchArgument("map_frame_id", default_value="map"),
        DeclareLaunchArgument("robot_frame_id", default_value="camera"),
        DeclareLaunchArgument("camera_frame_id", default_value="camera"),

        DeclareLaunchArgument("mapping_mode", default_value="XYZSemantics"),
   ]
#==========================

def launch_setup(context, *args, **kwargs):
    node = Node(
        package="voxeland_robot_perception",
        executable="robot_perception_node.py",
        prefix ="xterm -hold -e",
        parameters=[
           {"dataset": parse_substitution("$(var dataset)")},
           {"object_detector": parse_substitution("$(var object_detector)")},

            #input topics
           {"topic_camera_info": parse_substitution("$(var topic_camera_info)")},
           {"topic_rgb_image": parse_substitution("$(var topic_rgb_image)")},
           {"topic_depth_image": parse_substitution("$(var topic_depth_image)")},
           {"topic_localization": parse_substitution("$(var topic_localization)")},

            #Segmentation
           {"semantic_segmentation_mode": "service"},
           {"service_sem_seg": "/detectron/segment"},
           {"topic_sem_seg": "/ViMantic/Detections"},

            #Message type (Image, CompressedImage)
           {"rgb_image_type": "Image"},
           {"depth_image_type": "Image"},

            #Camera calibration. If intrinsics_from_topic==true, the rest of the params are ignored
		    {"intrinsics_from_topic" : True},
            {"width" :1920 },
			{"height" :1080 },
			{"cx" :959.5 },
			{"cy" :539.5 },
			{"fx" :1371.022 },
			{"fy" :1371.022 },
            {"camera_max_depth" :10.0 },

            #Depth Limits
            {"limit_reliable_depth" : False },
            {"min_reliable_depth" : 0.01 },
            {"max_reliable_depth" : 3.00 },

            #Frame IDs
            {"map_frame_id" : parse_substitution("$(var map_frame_id)") },
            {"robot_frame_id" : parse_substitution("$(var robot_frame_id)") },
            {"camera_frame_id" : parse_substitution("$(var camera_frame_id)") },

            #Output configuration
            {"pointcloud_type" :"XYZSemantics" },					# Possible values: "XYZ", "XYZRGB" and "XYZSemantics"
            {"topic_pointcloud_output" :"cloud_in" },

        ],
    )
    return [
        node,
    ]


def generate_launch_description():

    launch_description = [
       # Set env var to print messages to stdout immediately
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
   ]
   
    launch_description.extend(launch_arguments())
    launch_description.append(OpaqueFunction(function=launch_setup))
   
    return  LaunchDescription(launch_description)