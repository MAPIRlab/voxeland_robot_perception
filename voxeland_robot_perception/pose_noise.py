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
from modules.uncertainity import Uncertainity_Ops

# ROS-related libraries
import rclpy
from rclpy.node import Node

import threading


# ROS messages/services
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseWithCovariance, PoseStamped, PoseArray

from std_msgs.msg import Header

from segmentation_msgs.msg import SemanticPointCloud
from segmentation_msgs.srv import SegmentImage

########################################################################################################################
################################################ MAPPING NODE DEFINITION ###############################################
########################################################################################################################

class AddNoise(Node):

    def __init__(self):

        super().__init__('NoiseAdder')

        self.get_logger().warn("[VOXELAND] Initializing Sampling Pose Node...")

        # OBJECTS INITIALIZATION
        self.transformations = Transformations()
        self.uncertainity_handler = Uncertainity_Ops()

        # MAPPING PARAMETER CONFIGURATION
        self.data_queue = []

        # TOPICS AND SERVICES CONFIGURATION
        self.create_subscription(PoseWithCovarianceStamped, self.load_param('Pose_GT', '/amcl_pose'), self.pose_recv, 1)
            
        # OUTPUT CONFIGURATION
        
        self.noisy_pose_pub = self.create_publisher(PoseWithCovarianceStamped,self.load_param("pose_noise","noisy_pose"),1)
        
        self.get_logger().warn("[VOXELAND] Everything ready to map!")

    def run(self):

        thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        thread.start()
        
        while rclpy.ok(): 

            if len(self.data_queue) == 0:
                continue

            gt_pose = self.data_queue.pop(0)

            gt_pose.pose.covariance = self.uncertainity_handler.rand_cv_Matrix().flatten()  

            self.noisy_pose_pub.publish(gt_pose)

                             
    ####################################################################################################################
    ##################################################### Callbacks ####################################################
    ####################################################################################################################

    def pose_recv(self,pose_msg):

        if len(self.data_queue) > 0:
            return

        self.data_queue.append(pose_msg)

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
    node = AddNoise()
    node.run()


if __name__ == '__main__':
    main()