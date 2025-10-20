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
        Applies the transformation se3_transform to a se3_pose and propagates uncertainity associated to said pose

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
    

    def rand_cv_Matrix(self):

        mask = np.array([
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1]
            ], dtype=float)

        
        A = np.random.uniform(-0.01, 0.01, (6, 6))
 
        A_cov = A @ A.T
   
        A_cov = A_cov / np.max(A_cov) * np.random.uniform(0.02, 0.04)

        
        A_cov = (A_cov + A_cov.T) / 2

        for i in range(6):
                if A_cov[i, i] == 0:
                    A_cov[i, i] = np.random.uniform(0.02, 0.04)
          
        A_cov = A_cov * mask.T
        return A_cov
        



      

    