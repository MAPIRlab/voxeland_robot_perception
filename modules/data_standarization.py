from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

from modules.semantics import Semantics

class DataStandarization(object):
    
    def __init__(self, dataset, object_detector, object_categories_txt = "coco-classes.txt"):

        self.dataset = dataset
        self.object_detector = object_detector
        self.bridge = CvBridge()

    def standarize_rgb(self, rgb_msg):

        if self.dataset == "MAPIRlab":
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            standard_rgb = cv2.imdecode(np_arr, -1)
            #standard_rgb = cv2.cvtColor(standard_rgb, cv2.COLOR_RGB2BGR)
        
        elif self.dataset == "RobotAtVirtualHome":
            standard_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            standard_rgb = cv2.cvtColor(standard_rgb, cv2.COLOR_RGB2BGR)

        elif self.dataset == "OneObservation":
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            standard_rgb = cv2.imdecode(np_arr, -1)
        
        elif self.dataset == "MAPIRlab_olfaction":
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            standard_rgb = cv2.imdecode(np_arr, -1)
        
        elif self.dataset == "uHumans2":
            standard_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")

        elif self.dataset == "sceneNN":
            standard_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            standard_rgb = cv2.cvtColor(standard_rgb, cv2.COLOR_RGB2BGR)

        elif self.dataset == "ROS-Unity":
            standard_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            standard_rgb = cv2.cvtColor(standard_rgb, cv2.COLOR_RGB2BGR)
        
        else:
            standard_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            standard_rgb = cv2.cvtColor(standard_rgb, cv2.COLOR_RGB2BGR)


        return standard_rgb
    
    def standarize_depth(self, depth_msg):

        if self.dataset == "MAPIRlab":
            img_depth = np.clip(self.bridge.imgmsg_to_cv2(depth_msg), 0, 10.0)
            img_depth = img_depth * 65535/10.0
            img_depth = np.array(img_depth, dtype = np.uint16)
            standard_depth = np.divide(img_depth, 65535.0)
            standard_depth *= 10.0

        elif self.dataset == "RobotAtVirtualHome":

            standard_depth =  self.bridge.imgmsg_to_cv2(depth_msg, "64FC1")

        elif self.dataset == "OneObservation":

            standard_depth = self.bridge.imgmsg_to_cv2(depth_msg)
            standard_depth = np.divide(standard_depth, 1000.0)

        elif self.dataset == "MAPIRlab_olfaction":
            buf = np.ndarray(shape=(1, len(depth_msg.data)),
                         dtype=np.uint8, buffer=depth_msg.data)
            standard_depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(float) * 0.001
        
        elif self.dataset == "uHumans2":
            standard_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        
        elif self.dataset == "sceneNN":
            standard_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            standard_depth = np.divide(standard_depth, 1000.0)

        elif self.dataset == "ROS-Unity":
            standard_depth = self.bridge.imgmsg_to_cv2(depth_msg, "mono16")
            standard_depth = np.divide(standard_depth, 1000.0)
        
        else:
            standard_depth =  self.bridge.imgmsg_to_cv2(depth_msg)

            
        return standard_depth

    def standarize_semantics(self, semantic_msg):

        if len(semantic_msg.instances) == 0:
                return Semantics(0,0)

        standard_semantics = Semantics(semantic_msg.instances[0].mask.width, semantic_msg.instances[0].mask.height)

        if self.object_detector == "Detectron2":
            
            standard_semantics.add_objects(semantic_msg.instances)

        return standard_semantics
