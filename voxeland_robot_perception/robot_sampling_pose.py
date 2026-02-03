#!/usr/bin/env python3

# System Libraries
import threading
from copy import deepcopy
import time

# Third-party libraries
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Own libraries
from modules.transformations import Transformations
from modules.pointclouds import Semantic_PointCloud_Utils

# ROS-related libraries
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import threading
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# ROS messages/services
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseWithCovariance, PoseStamped, PoseArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2

from std_msgs.msg import Header

from segmentation_msgs.msg import SemanticPointCloud
from segmentation_msgs.srv import SegmentImage

########################################################################################################################
################################################ MAPPING NODE DEFINITION ###############################################
########################################################################################################################

class PoseSampler(Node):

    def __init__(self):

        super().__init__('pose_sampler')

        self.get_logger().warn("[VOXELAND] Initializing Sampling Pose Node...")

        # OBJECTS INITIALIZATION
        self.transformations = Transformations()

        # MAPPING PARAMETER CONFIGURATION
        self.n_samples_map = self.load_param("n_samples",1)

        self.data_queue = []

        # TOPICS AND SERVICES CONFIGURATION
        
        self.create_subscription(SemanticPointCloud, self.load_param('topic_pose_to_sample', '/pose_to_sample'), self.sampling_cb, 1)
    
        # OUTPUT CONFIGURATION
        self.pointcloud_pub = self.create_publisher(SemanticPointCloud, self.load_param('sampling_results', "cloud_in"), 1)
        self.pointcloud_pub1 = self.create_publisher(PointCloud2, "cloud_in_local", 1)
        self.sampled_pose_pub = self.create_publisher(PoseArray,self.load_param("sampled_pose_echo","sampled_pose"),1)
        self.sensor_pose = self.create_publisher(PoseWithCovarianceStamped,self.load_param("sensor_pose","remap/giraff/amcl_pose"),1)
        
        self.get_logger().warn("[VOXELAND] Everything ready to map!")

    def run(self):

        thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        thread.start()

        while rclpy.ok(): 

            if len(self.data_queue) == 0:
                continue

            cloud_msg = self.data_queue.pop(0)

            p,v_angles,cv_matrix=self.transformations.msg_to_pv_cvmatrix(cloud_msg.pose)

            # self.get_logger().info("Pose: {} \n v_angles: {} \n cv_mat: {}".format(p,v_angles,cv_matrix)) 

            sample_points = self.transformations.sample_distribution(p,v_angles,cv_matrix,self.n_samples_map) 

            # self.get_logger().info("Sampled Results: {} \n ".format(sample_points)) 

            poses_Stamped=PoseArray()
            poses_Stamped.header = cloud_msg.header
            poses_Stamped.header.frame_id = "map"
           
            sensor_pose_Stamped = PoseWithCovarianceStamped()
            sensor_pose_Stamped.pose = cloud_msg.pose
            sensor_pose_Stamped.header = poses_Stamped.header


            for i in range(sample_points.shape[0]):
  
                pose_msg = self.transformations.pv_to_msg(sample_points[i,0:3],sample_points[i,3:6],cv_matrix) # P = sample_points[0:3];
                                                                                                    # V_ang = sample_point[3:6]
                cloud_msg.pose = pose_msg
                self.pointcloud_pub.publish(cloud_msg)
                self.pointcloud_pub1.publish(cloud_msg.cloud)
                
                poses_Stamped.poses.append(cloud_msg.pose.pose)

            sensor_pose_Stamped.pose = cloud_msg.pose
    
            self.sampled_pose_pub.publish(poses_Stamped)
            self.sensor_pose.publish(sensor_pose_Stamped)
            self.get_logger().info("Published sampled pose results: {} poses sampled".format(len(poses_Stamped.poses)))
                             
                             
    ####################################################################################################################
    ##################################################### Callbacks ####################################################
    ####################################################################################################################

    def sampling_cb(self,cloud_msg):

        if len(self.data_queue) > 0:
            return

        self.data_queue.append(cloud_msg)

        return
    
    
    ####################################################################################################################
    ################################################# Additional Methods ###############################################
    ####################################################################################################################

    def load_param(self, param, default=None):
        new_param = self.declare_parameter(param, default).value
        self.get_logger().info("[VOXELAND] {}: {}".format(param, new_param))
        return new_param

########################################################################################################################
########################################################## MAIN ########################################################
########################################################################################################################

def main(args=None):

    rclpy.init(args=args)
    node = PoseSampler()
    node.run()


if __name__ == '__main__':
    main()