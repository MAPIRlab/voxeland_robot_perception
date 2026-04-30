import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sensor_msgs_py import point_cloud2 as pc2

from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from segmentation_msgs.msg import SemanticPointCloud


class Semantic_PointCloud_Utils(object):
  
    def __init__(self, known_categories, category_manager=None):
        """
        Initialize with known categories and optional category manager.
        
        Args:
            known_categories: Initial list of known categories (for backward compatibility)
            category_manager: OpenVocabularyCategoryManager instance for dynamic category handling
        """
        self.category_manager = category_manager
        
        if self.category_manager is not None:
            # Use dynamic categories
            self.categories = self.category_manager.get_all_categories()
        else:
            # Fallback to static categories for backward compatibility
            self.categories = known_categories.copy()
            if "unknown" not in self.categories:
                self.categories.append("unknown")
    
    def get_categories(self):
        """Get current list of categories."""
        if self.category_manager is not None:
            self.categories = self.category_manager.get_all_categories()
        return self.categories.copy()

    def create_point_cloud_msg(self, xyz_points, sensor_pose, cloud_frame_reference, timestamp, colors = None, semantics_ids = None, semantics_instances = None):
   
        msg = SemanticPointCloud()

        header = Header()
        header.stamp = timestamp
        header.frame_id = cloud_frame_reference
        msg.header = header
        msg.pose = sensor_pose
        xyz_points = np.core.records.fromarrays(xyz_points.T, names='x, y, z', formats='f4, f4, f4')

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = [xyz_points]
        offset = 12

        if colors is not None:
            colors = colors[:, 2] * 2 ** 16 + colors[:, 1] * 2 ** 8 + colors[:, 0]
            colors = np.array(colors, dtype=[("rgb", "u4")])
            fields.append(PointField(name='rgb', offset=offset, datatype=PointField.UINT32, count=1))
            cloud_data.append(colors)
            offset += 4

        if semantics_ids is not None:
            semantics_ids = np.array(semantics_ids.astype(np.int32), dtype=[("instance_id", "i4")])
            fields.append(PointField(name='instance_id', offset=offset, datatype=PointField.INT32, count=1))
            cloud_data.append(semantics_ids)
            msg.instances = [obj for obj in semantics_instances]
            for obj in msg.instances:
                obj.header = header
            offset += 4
            
        cloud_data = self.join_struct_arrays(cloud_data)
        msg.cloud = pc2.create_cloud(header, fields, cloud_data)
        
        # Always use current categories (may have grown dynamically)
        current_categories = self.get_categories()
        msg.categories = current_categories
        
        return msg
    
    @staticmethod
    def join_struct_arrays(arrays):
        sizes = np.array([a.itemsize for a in arrays])
        offsets = np.r_[0, sizes.cumsum()]
        n = len(arrays[0])
        joint = np.empty((n, offsets[-1]), dtype=np.uint8)
        for a, size, offset in zip(arrays, sizes, offsets):
            joint[:,offset:offset+size] = a.view(np.uint8).reshape(n,size)
        dtype = sum((a.dtype.descr for a in arrays), [])
        return joint.ravel().view(dtype)
    
    @staticmethod
    def remove_background_from_object_pointcloud(xyz_obj, id):
        # Downsample by voxelization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_obj)
        downsampled_pcd = pcd.voxel_down_sample(0.05)
        downsampled_points = np.asarray(downsampled_pcd.points)

        clustering_distance = 0.1

        # Apply clustering to segment object from background
        clustering = DBSCAN(eps=0.1, min_samples=15, n_jobs=-1).fit(downsampled_points)
        labels = clustering.labels_

        if len(labels) == 0:
            return np.zeros(len(xyz_obj), dtype=int)
        labels_unique, counts = np.unique(labels, return_counts=True)
        majority_label = labels_unique[np.argmax(counts)]

        if majority_label == -1:
            return np.zeros(len(xyz_obj), dtype=int)
        
        # Build KD-Tree for the downsampled points
        # downsampled_tree = KDTree(downsampled_points)

        # Filter points that belong to the majority cluster
        majority_cluster_points = downsampled_points[labels == majority_label]
        # Build KD-Tree for the original points
        original_tree = KDTree(xyz_obj)
        # Create segmentation labels
        segmentation_labels = np.zeros(len(xyz_obj), dtype=int)

        # Query the original points to assign the label
        for point in majority_cluster_points:
            idx = original_tree.query_radius(point.reshape(1, -1), clustering_distance)[0]
            segmentation_labels[idx] = id
            
        return segmentation_labels