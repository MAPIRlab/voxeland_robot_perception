import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from sensor_msgs_py import point_cloud2 as pc2

from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from segmentation_msgs.msg import SemanticPointCloud


class Semantic_PointCloud_Utils(object):

    def __init__(self, known_categories):

        self.categories = known_categories
        self.categories.append("unknown")
    
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

            offset += 4
        
        cloud_data = self.join_struct_arrays(cloud_data)
        msg.cloud = pc2.create_cloud(header, fields, cloud_data)

        msg.categories = self.categories

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
    def remove_background_from_object_pointcloud_v0(xyz_obj, id):

        # Downsample by voxelization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_obj)
        downsampled_pcd = pcd.voxel_down_sample(0.05)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Apply clustering to segment object from background
        clustering = DBSCAN(eps=0.05*np.sqrt(2)+0.001, min_samples=9).fit(downsampled_points)
        labels = clustering.labels_.astype(np.float_)
        labels_unique, counts = np.unique(labels, return_counts=True)
        majority_label = labels_unique[np.argmax(counts)]
        #labels[(labels != majority_label)] = 0
        #labels[(labels == majority_label)] = id

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        segmentation_labels = np.zeros(len(xyz_obj), dtype=int)

        if majority_label == -1:
            return segmentation_labels
            
        for i, point in enumerate(downsampled_points):
                if labels[i] == majority_label:
                    [_, idx, _] = pcd_tree.search_radius_vector_3d(point,0.05*np.sqrt(2)+0.001)
                    for j in idx:
                        segmentation_labels[j] = id
        
        if False:
            # Backpropagation of labels to original point clouds
            tree = cKDTree(downsampled_points)
            _, indices = tree.query(xyz_obj, k=1)
            segmentation_labels = labels[indices]
        
        return segmentation_labels

    @staticmethod
    def remove_background_from_object_pointcloud(xyz_obj, id):
        # Downsample by voxelization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_obj)
        downsampled_pcd = pcd.voxel_down_sample(0.05)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Apply clustering to segment object from background
        clustering = DBSCAN(eps=0.05 * np.sqrt(2) + 0.001, min_samples=9).fit(downsampled_points)
        labels = clustering.labels_

        if len(labels) == 0:
            return np.zeros(len(xyz_obj), dtype=int)

        labels_unique, counts = np.unique(labels, return_counts=True)
        majority_label = labels_unique[np.argmax(counts)]

        if majority_label == -1:
            return np.zeros(len(xyz_obj), dtype=int)

        # Build KD-Tree for the downsampled points
        downsampled_tree = cKDTree(downsampled_points)
        
        # Filter points that belong to the majority cluster
        majority_cluster_points = downsampled_points[labels == majority_label]

        # Build KD-Tree for the original points
        original_tree = cKDTree(xyz_obj)

        # Create segmentation labels
        segmentation_labels = np.zeros(len(xyz_obj), dtype=int)

        # Query the original points to assign the label
        for point in majority_cluster_points:
            idx = original_tree.query_ball_point(point, 0.05 * np.sqrt(2) + 0.001)
            segmentation_labels[idx] = id

        return segmentation_labels

