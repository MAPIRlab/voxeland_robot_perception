import numpy as np

from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge, CvBridgeError

class Semantics(object):

    def __init__(self, image_width, image_height):

        self.bridge = CvBridge()

        self.n_objects = 0
        self.objects = []
        self.semantic_image = np.zeros((image_height, image_width), dtype=np.uint8)

        self.add_unknown_category()

    def add_unknown_category(self):
        
        unknown = Detection2D()
        unknown.id = str(0)
        unknown.results = []
        
        hyp = ObjectHypothesisWithPose()
        
        hyp.hypothesis.class_id = "unknown"
        hyp.hypothesis.score = 0.05

        unknown.results.append(hyp)

        self.objects.append(unknown)

    def add_objects(self, objects_list):

        next_object_id = len(self.objects)

        for id, obj in enumerate(objects_list):
            #["person", "chair", "dining table", "tv", "laptop", "microwave"]
            obj.detection.id = str(next_object_id + id)
            self.objects.append(obj.detection)
            self.semantic_image[self.bridge.imgmsg_to_cv2(obj.mask) == 255] = next_object_id + id
            """
            if obj.detection.results[0].hypothesis.class_id in ["chair"]:
                
                obj.detection.id = str(next_object_id + id)
                self.objects.append(obj.detection)
                self.semantic_image[self.bridge.imgmsg_to_cv2(obj.mask) == 255] = next_object_id + id
            
            else:
                obj.detection.id = str(next_object_id + id)
                self.objects.append(obj.detection)
                self.semantic_image[self.bridge.imgmsg_to_cv2(obj.mask) == 255] = 0
            """

        self.n_objects = len(self.objects) - 1


        
    def filter_valid_classes(self, points_ids, valid_classes):
        
        updated_points_ids = points_ids.copy()

        for obj in self.objects:

            if obj.results[0].hypothesis.class_id not in valid_classes:

                updated_points_ids[updated_points_ids == int(obj.id)] = 0

        return updated_points_ids

