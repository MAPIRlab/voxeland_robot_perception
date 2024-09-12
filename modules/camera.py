import numpy as np

class Camera(object):

    def __init__(self):

        # Intrinsics
        self.width = None
        self.height = None
        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None
        self.intrinsics_initialized = False

        # Extrinsics
        self.extrinsics = None
        self.extrinsics_initialized = False

        # Grid to convert 2D pixels to 3D points
        self.c_values = None
        self.r_values = None

        # Depth Specs
        self.max_reliable_depth = None
        self.min_reliable_depth = None

    def set_intrinsics(self, width, height, cx, cy, fx, fy):

        self.width = width
        self.height = height
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
    
        # Initialize grid to convert 2D pixels to 3D points
        image_c, image_r = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
        self.c_values = (cx - image_c) / fx
        self.r_values = (cy - image_r) / fy

        self.intrinsics_initialized = True
        
    def set_extrinsics(self, extrinsics):

        self.extrinsics = extrinsics
        self.extrinsics_initialized = True

    def set_limit_depth_range(self, min_reliable_depth, max_reliable_depth):

        self.min_reliable_depth = min_reliable_depth
        self.max_reliable_depth = max_reliable_depth
        
    def xyz_from_depth(self, depth_img):

        x = self.c_values * depth_img
        y = self.r_values * depth_img

        return x, y, depth_img

    def get_limited_depth_mask(self, depth_img):

        if self.min_reliable_depth is not None and self.max_reliable_depth is not None:

            return (depth_img > self.min_reliable_depth) & (depth_img < self.max_reliable_depth)

        elif self.min_reliable_depth is not None:

            return (depth_img > self.min_reliable_depth)

        elif self.max_reliable_depth is not None:

            return (depth_img < self.max_reliable_depth)

        
