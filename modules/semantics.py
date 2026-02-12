import numpy as np

from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge, CvBridgeError

class Semantics(object):

    def __init__(self, image_width, image_height, category_manager=None):

        self.bridge = CvBridge()
        self.category_manager = category_manager

        self.n_objects = 0
        self.objects = []
        self.semantic_image = np.zeros((image_height, image_width), dtype=np.uint8)

        self.add_unknown_category()
    
    def add_category(self, category_name):
        """Add a category dynamically if using category manager."""
        if self.category_manager is not None:
            return self.category_manager.add_category(category_name)
        return 0  # Default to unknown category index

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
            # Add category to manager if using open vocabulary
            if self.category_manager is not None:
                for result in obj.detection.results:
                    self.add_category(result.hypothesis.class_id)
            
            obj.detection.id = str(next_object_id + id)
            self.objects.append(obj.detection)
            self.semantic_image[self.bridge.imgmsg_to_cv2(obj.mask) == 255] = next_object_id + id

        self.n_objects = len(self.objects) - 1


        
    def filter_valid_classes(self, points_ids, valid_classes):
        
        updated_points_ids = points_ids.copy()

        for obj in self.objects:

            if obj.results[0].hypothesis.class_id not in valid_classes:

                updated_points_ids[updated_points_ids == int(obj.id)] = 0

        return updated_points_ids

