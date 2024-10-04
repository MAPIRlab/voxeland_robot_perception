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
from modules.camera import Camera
from modules.data_standarization import DataStandarization
from modules.pointclouds import Semantic_PointCloud_Utils

# ROS-related libraries
import rclpy
import message_filters
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import threading
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# ROS messages/services
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from segmentation_msgs.msg import SemanticPointCloud
from segmentation_msgs.srv import SegmentImage

########################################################################################################################
################################################ MAPPING NODE DEFINITION ###############################################
########################################################################################################################

class MinimalMapper(Node):

    def __init__(self):

        super().__init__('voxeland_perception_node')

        self.get_logger().warn("[VOXELAND] Initializing Robot Perception Node...")

        # OBJECTS INITIALIZATION
        self.camera = Camera()
        self.transformations = Transformations()

        # This is hardcoded... change to read it from a file
        categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                        'toothbrush']
        
        self.valid_classes = ["bed", "chair", "couch", "dining table", "book", "refrigerator", "tv", "toilet", "handbag", "unknown"]
        self.valid_classes = categories

         # OBJECTS AND HANDLERS

        self.pointcloud_utils = Semantic_PointCloud_Utils(categories)
        self.standarization = DataStandarization(dataset=self.load_param('dataset', "VirtualGallery"),
                                                 object_detector=self.load_param('object_detector', "Detectron2"))
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # CAMERA INFO: INTRINSICS AND SPECS
        intrinsics_from_topic = self.load_param('intrinsics_from_topic', True)
        if not intrinsics_from_topic:

            width = self.load_param('width', 1920)
            height = self.load_param('height', 1080)
            cx = self.load_param('cx', 959.5)
            cy = self.load_param('cy', 539.5)
            fx = self.load_param('fx', 1371.022)
            fy = self.load_param('fy', 1371.022)
            
            self.camera.set_intrinsics(width, height, cx, cy, fx, fy)

            self.standarization.set_image_dimension(width, height)

        self.limit_depth = self.load_param('limit_reliable_depth', False)
        if self.limit_depth:

            self.camera.set_limit_depth_range(self.load_param('min_reliable_depth', None), self.load_param('max_reliable_depth', None))

        # TOPICS AND SERVICES CONFIGURATION
        if intrinsics_from_topic:
            self.create_subscription(CameraInfo, self.load_param('topic_camera_info', '/camera/camera_info'), self.camera_info_cb, 1)
            
        subscriptions = [
        message_filters.Subscriber(self, PoseWithCovarianceStamped, 
                                   self.load_param('topic_localization', '/amcl_pose')),        # Robot Pose
        message_filters.Subscriber(self, eval(self.load_param('rgb_image_type', "Image")), 
                                   self.load_param('topic_rgb_image', "/camera/rgb")),          # RGB Image
        message_filters.Subscriber(self, eval(self.load_param('depth_image_type', "Image")), 
                                   self.load_param('topic_depth_image', "/camera/depth")),       # Depth Image
        ]

        self.segmentation_from = self.load_param('semantic_segmentation_mode', "service")
        if self.segmentation_from == "topic":
            subscriptions.append(message_filters.Subscriber(self, Image, "/camera/segmentation"))
        elif self.segmentation_from == "service": 
            self.cli = self.create_client(SegmentImage, '/detectron/segment')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Semantic segmentation service not available, waiting...')

        message_filter = message_filters.ApproximateTimeSynchronizer(subscriptions, 1, 0.1)
        message_filter.registerCallback(self.new_incoming_observation_cb)

        # FRAME IDs
        self.map_frame_id = self.load_param('map_frame_id', "map")
        self.robot_frame_id = self.load_param('robot_frame_id', "base_link")
        self.camera_frame_id = self.load_param('camera_frame_id', "camera")

        # SEMANTIC MAPPING VARIABLES
        self.filter_semantics = self.load_param('filter_semantics', True)
        self.data_queue = []
        
        # OUTPUT CONFIGURATION
        self.pointcloud_type = self.load_param('pointcloud_type', "XYZRGB")
        self.pointcloud_pub = self.create_publisher(SemanticPointCloud, self.load_param('topic_pointcloud_output', "cloud_in"), 1)
        self.pointcloud_pub1 = self.create_publisher(PointCloud2, "cloud_in_local", 1)

        # Timers
        self._start_time = 0  # Time when the code starts
        self.opinions_time = 0
        self.opinions_k = 0

        self.get_logger().warn("[VOXELAND] Everything ready to map!")

    def run(self):

        thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        thread.start()

        while rclpy.ok(): 

            if len(self.data_queue) == 0:
                continue

            st = time.time()

            processing_observation = self.data_queue.pop(0)

            x, y, z = self.camera.xyz_from_depth(processing_observation["img_depth"])

            if self.limit_depth:

                mask_depth_limits = self.camera.get_limited_depth_mask(z)
                x = x[mask_depth_limits]
                y = y[mask_depth_limits]
                z = z[mask_depth_limits]
            
            xyz_cloud = np.array([z.reshape(-1), x.reshape(-1), y.reshape(-1)]).T
            

            if "RGB" in self.pointcloud_type:
                if self.limit_depth:
                    rgb_colors = processing_observation["img_rgb"].reshape((-1,3))[mask_depth_limits.flatten()]
                else:
                    rgb_colors = processing_observation["img_rgb"].reshape((-1,3))
            if "Semantics" in self.pointcloud_type:
                if self.limit_depth:
                    semantic_ids = processing_observation["semantics"].semantic_image[mask_depth_limits].reshape(-1)
                else:
                    semantic_ids = processing_observation["semantics"].semantic_image.reshape(-1)

                if self.filter_semantics:
                    
                    semantic_ids = processing_observation["semantics"].filter_valid_classes(semantic_ids, self.valid_classes)

                    unique_ids = np.unique(semantic_ids)
                    for id in unique_ids:
                        st_time = time.time()
                        if id == 0:
                            continue
                        xyz_obj = xyz_cloud[semantic_ids == id]
                        semantic_labels = self.pointcloud_utils.remove_background_from_object_pointcloud(xyz_obj, id)
                        semantic_ids[semantic_ids == id] = semantic_labels
                        #pcd_original = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz_obj))
                        #o3d.visualization.draw_geometries([pcd_original])
                        #o3d.io.write_point_cloud("test_pointcloud.pcd", pcd_original)
                        #pcd_filtered = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz_cloud[semantic_ids == id]))
                        #pcd_original.paint_uniform_color([1, 0, 0])
                        #pcd_filtered.paint_uniform_color([0, 1, 0])
                        #o3d.visualization.draw_geometries([pcd_original, pcd_filtered])

                        ##### TODO: JL Esto de abajo es como segmentaba antes y funcionaba pero muy lento, revisar!!
                        """
                        xyz_obj = xyz_cloud[semantic_ids == id]
                        #pcd_original = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz_obj))
                        semantic_obj = semantic_ids[semantic_ids == id]
                        clustering = DBSCAN(eps=0.05, min_samples=20).fit(xyz_obj)
                        labels = clustering.labels_.astype(np.float_)
                        labels_unique, counts = np.unique(labels, return_counts=True)
                        majority_label = labels_unique[np.argmax(counts)]
                        semantic_obj[(labels != majority_label)] = 0
                        semantic_ids[semantic_ids == id] = semantic_obj
                        #pcd_filtered = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz_cloud[semantic_ids == id]))
                        #o3d.visualization.draw_geometries([pcd_filtered])
                        """
                        #self.get_logger().info("elapsed time filtering semantics: {}".format((time.time() - st_time)*1000.))
         

            """
            for id in np.unique(semantic_ids):

                if id == 1:
                    continue
                
                pcd_local = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz_cloud))
                xyz_id =

            labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
            non_noise_labels = labels[labels != -1]
            unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            xyz_mask = labels == most_common_label
            """
            
            self.opinions_time += time.time() - st
            self.opinions_k += 1
            global_pose = processing_observation["pose"] @ self.camera.extrinsics

            #a = time.time()
            #pcd1 = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(xyz_cloud))
            #pcd1 = pcd1.transform(global_pose)
            #xyz_cloud = np.array(pcd1.points)
            #self.get_logger().info("{}".format(time.time() - a))
            
            pose_msg = self.transformations.se3_to_msg(global_pose, processing_observation["covariance"])
            #pose_msg = self.transformations.se3_to_msg(np.eye(4), processing_observation["covariance"])


            if self.pointcloud_type == "XYZ":
                cloud_msg = self.pointcloud_utils.create_point_cloud_msg(xyz_cloud, 
                                                                         sensor_pose = pose_msg,
                                                                         cloud_frame_reference = self.camera_frame_id, 
                                                                         timestamp = processing_observation["timestamp"])
            elif self.pointcloud_type == "XYZRGB":
                cloud_msg = self.pointcloud_utils.create_point_cloud_msg(xyz_cloud, 
                                                                         colors = rgb_colors,
                                                                         sensor_pose = pose_msg,
                                                                         cloud_frame_reference = self.camera_frame_id, 
                                                                         timestamp = processing_observation["timestamp"])

            elif self.pointcloud_type == "XYZSemantics":
                cloud_msg = self.pointcloud_utils.create_point_cloud_msg(xyz_cloud, 
                                                                         semantics_ids = semantic_ids,
                                                                         semantics_instances=processing_observation["semantics"].objects,
                                                                         sensor_pose = pose_msg,
                                                                         cloud_frame_reference = self.camera_frame_id, 
                                                                         timestamp = processing_observation["timestamp"])
            
            elif self.pointcloud_type == "XYZRGBSemantics":       
                cloud_msg = self.pointcloud_utils.create_point_cloud_msg(xyz_cloud, 
                                                                         colors = rgb_colors,
                                                                         semantics_ids = semantic_ids,
                                                                         semantics_instances=processing_observation["semantics"].objects,
                                                                         sensor_pose = pose_msg,
                                                                         cloud_frame_reference = self.camera_frame_id, 
                                                                         timestamp = processing_observation["timestamp"])


            self.pointcloud_pub.publish(cloud_msg)
            self.pointcloud_pub1.publish(cloud_msg.cloud)

            if "Semantics" in self.pointcloud_type:
                self.get_logger().info("Observation processed! ||  {} objects detected.".format(processing_observation["semantics"].n_objects))

            self.get_logger().info("Average opinions generation time: {} ms".format(1000.*self.opinions_time / float(self.opinions_k)))


    ####################################################################################################################
    ###################################################### Services ####################################################
    ####################################################################################################################
    
    def request_segmentation(self, observation):
        req = SegmentImage.Request()
        req.image = self.bridge.cv2_to_imgmsg(observation["img_rgb"], encoding="passthrough")
        future = self.cli.call_async(req)
        future.add_done_callback(lambda response: self.process_service_result(observation, response))

    def process_service_result(self, observation, response):

        observation["semantics"] = self.standarization.standarize_semantics(response.result())

        #if observation["semantics"].n_objects > 0:
        self.data_queue.append(observation)


    ####################################################################################################################
    ##################################################### Callbacks ####################################################
    ####################################################################################################################

    def camera_info_cb(self, camera_info):

        if not self.camera.intrinsics_initialized:

            self.camera.set_intrinsics(camera_info.width, camera_info.height, camera_info.k[2], 
                                       camera_info.k[5], camera_info.k[0], camera_info.k[4])
            
            self.standarization.set_image_dimension(camera_info.width, camera_info.height)

    def new_incoming_observation_cb(*args):
        
        if args[0].segmentation_from == "topic":
            self, pose_msg, rgb_msg, depth_msg, sem_msg = args
        else:
            self, pose_msg, rgb_msg, depth_msg  = args


        if self.camera.intrinsics_initialized and self.camera.extrinsics_initialized:

            img_rgb = self.standarization.standarize_rgb(rgb_msg)
            img_depth = self.standarization.standarize_depth(depth_msg)

            #try:
            pose_se3 = self.transformations.msg_to_se3(pose_msg.pose.pose)
            #except:
            #    self.get_logger().warn("ERROR EN LA POSE!")
            #    return
            pose_covariance = pose_msg.pose.covariance         

            new_observation = {"pose": pose_se3,
                               "covariance": pose_covariance,
                               "img_rgb": img_rgb, 
                               "img_depth": img_depth, 
                               "semantics": None,
                               "timestamp": rgb_msg.header.stamp}

            if "Semantics" in self.pointcloud_type:

                if args[0].segmentation_from == "topic":
                    new_observation["semantics"] = self.standarization.standarize_semantics(sem_msg)
                    #if new_observation["semantics"].n_objects > 0:
                    self.data_queue.append(new_observation)

                else:
                    self.request_segmentation(new_observation)
            
            else:
                self.data_queue.append(new_observation)

        
        elif not self.camera.extrinsics_initialized:

            try:
                extrinsics = self.tf_buffer.lookup_transform(self.robot_frame_id,
                                                             self.camera_frame_id,
                                                             rclpy.time.Time())
                
                # Uncomment for robotatvirtualhome
                #extrinsics.transform.rotation.y = 0.0871557
                #extrinsics.transform.rotation.w = 0.9961947

                self.camera.set_extrinsics(self.transformations.msg_to_se3(extrinsics))
                self.get_logger().warn("{}".format(extrinsics))


            except:
                # Uncomment for uHumans2
                #extrinsics = TransformStamped()
                #extrinsics.transform.translation.x = 0.
                #extrinsics.transform.translation.y = 0.05
                #extrinsics.transform.translation.z = 0.
                #extrinsics.transform.rotation.x = 0.
                #extrinsics.transform.rotation.y = 0.
                #extrinsics.transform.rotation.z = 0.
                #extrinsics.transform.rotation.w = 1.
                #self.camera.set_extrinsics(self.transformations.msg_to_se3(extrinsics))

                self.get_logger().warn("Transformation from {} [CAMERA FRAME] to {} [ROBOT FRAME] not found!".format(self.robot_frame_id,
                                                                                                                   rgb_msg.header.frame_id))


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
    node = MinimalMapper()
    node.run()


if __name__ == '__main__':
    main()