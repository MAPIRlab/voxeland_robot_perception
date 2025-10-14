import numpy as np

from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge, CvBridgeError
from modules.transformations import Transformations
from tf2_geometry_msgs import transform_covariance as tf2_transform_covariance


class Uncertainity_Ops(object):

    #Extrinsics = TransformedStamped

    def __init__(self):

        self.transformation_tools = Transformations()

        return

    def pose_transform(self,se3_pose, cv_matrix, se3_transform):

        '''
        Description

        - Inputs

        :se3_pose: Pose in SE3 form 
        :cv_Matrix: Covariance Matrix
        :se3_transform: Pose tranformation to be applied

        - Outputs

        :pose_result: PoseWithCovariance msg   

        
        '''
        transform = self.transformation_tools.se3_to_TransformStamped(se3_transform)

        pose_result = self.transformation_tools.se3_to_msg(se3_pose @ se3_transform,tf2_transform_covariance(cv_matrix,transform))

        return pose_result
        



      

    