#!/usr/bin/env python3

"""
Automatic Semantic Map Saver Node

This node saves the semantic point cloud data every 30 seconds to the map_output directory.
It subscribes to /cloud_in topic and saves the accumulated semantic information.
"""

import rclpy
from rclpy.node import Node
from segmentation_msgs.msg import SemanticPointCloud
import json
import os
import time
from datetime import datetime
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from collections import defaultdict
import threading

class AutomaticMapSaver(Node):
    def __init__(self):
        super().__init__('automatic_map_saver')
        
        self.get_logger().info("Starting Automatic Semantic Map Saver...")
        
        # Configuration
        self.save_interval = 30.0  # seconds
        self.output_dir = "/home/ubuntu/ros2_ws/map_output"
        self.current_map_file = os.path.join(self.output_dir, "current_semantic_map.json")
        self.current_pcd_file = os.path.join(self.output_dir, "current_semantic_map.pcd")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage
        self.accumulated_points = []
        self.accumulated_semantics = []
        self.accumulated_instances = []
        self.categories = []
        self.lock = threading.Lock()
        
        # Statistics
        self.total_points_received = 0
        self.last_update_time = time.time()
        
        # Subscription to semantic point clouds
        self.subscription = self.create_subscription(
            SemanticPointCloud,
            '/cloud_in',
            self.semantic_cloud_callback,
            10
        )
        
        # Timer for automatic saving
        self.save_timer = self.create_timer(self.save_interval, self.save_current_map)
        
        self.get_logger().info(f"Automatic map saver initialized. Saving every {self.save_interval} seconds to {self.output_dir}")

    def semantic_cloud_callback(self, msg):
        """Process incoming semantic point cloud data."""
        with self.lock:
            try:
                # Extract point cloud data
                points = list(pc2.read_points(msg.cloud, skip_nans=True))
                
                if not points:
                    return
                
                # Process points and extract semantic information
                for point in points:
                    # Extract XYZ coordinates
                    x, y, z = point[0], point[1], point[2]
                    
                    # Extract RGB if available
                    rgb = None
                    if len(point) > 3:
                        rgb = point[3] if hasattr(point[3], '__iter__') else point[3]
                    
                    # Extract instance ID if available
                    instance_id = None
                    if len(point) > 4:
                        instance_id = int(point[4]) if point[4] is not None else 0
                    
                    # Store point data
                    point_data = {
                        'x': float(x),
                        'y': float(y), 
                        'z': float(z),
                        'rgb': int(rgb) if rgb is not None else 0,
                        'instance_id': int(instance_id) if instance_id is not None else 0,
                        'timestamp': float(time.time())
                    }
                    
                    self.accumulated_points.append(point_data)
                
                # Store semantic instances information
                for instance in msg.instances:
                    if hasattr(instance, 'results') and len(instance.results) > 0:
                        instance_data = {
                            'id': str(instance.id),
                            'class_id': str(instance.results[0].hypothesis.class_id),
                            'confidence': float(instance.results[0].hypothesis.score),
                            'bbox': {
                                'center_x': float(instance.bbox.center.position.x),
                                'center_y': float(instance.bbox.center.position.y),
                                'size_x': float(instance.bbox.size_x),
                                'size_y': float(instance.bbox.size_y)
                            } if hasattr(instance, 'bbox') else None,
                            'timestamp': float(time.time())
                        }
                        self.accumulated_instances.append(instance_data)
                
                # Update categories
                if hasattr(msg, 'categories'):
                    self.categories = list(msg.categories)
                
                # Update statistics
                self.total_points_received += len(points)
                self.last_update_time = time.time()
                
                # Log periodic updates
                if self.total_points_received % 1000 == 0:
                    self.get_logger().info(f"Accumulated {self.total_points_received} points, {len(self.accumulated_instances)} instances")
                
            except Exception as e:
                self.get_logger().error(f"Error processing semantic cloud: {e}")

    def save_current_map(self):
        """Save the current accumulated map data."""
        with self.lock:
            if not self.accumulated_points:
                self.get_logger().warn("No map data to save yet...")
                return
            
            try:
                # Prepare map data
                map_data = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_points': len(self.accumulated_points),
                        'total_instances': len(self.accumulated_instances),
                        'categories': self.categories,
                        'save_interval_seconds': self.save_interval,
                        'last_update_time': self.last_update_time
                    },
                    # 'points': self.accumulated_points,
                    'instances': self.accumulated_instances,
                    'categories_info': {
                        'available_categories': self.categories,
                        'unique_detected_classes': list(set([inst['class_id'] for inst in self.accumulated_instances]))
                    }
                }
                
                # Save JSON file (overwrite previous)
                with open(self.current_map_file, 'w') as f:
                    json.dump(map_data, f, indent=2)
                
                # Save simplified PCD format
                self.save_as_pcd()
                
                # Log save operation
                unique_classes = set([inst['class_id'] for inst in self.accumulated_instances])
                self.get_logger().info(
                    f"✅ Map saved! Points: {len(self.accumulated_points)}, "
                    f"Instances: {len(self.accumulated_instances)}, "
                    f"Unique classes: {len(unique_classes)} {list(unique_classes)[:10]}{'...' if len(unique_classes) > 10 else ''}"
                )
                
                # Log file locations
                self.get_logger().info(f"📁 JSON: {self.current_map_file}")
                self.get_logger().info(f"📁 PCD: {self.current_pcd_file}")
                
            except Exception as e:
                self.get_logger().error(f"❌ Error saving map: {e}")

    def save_as_pcd(self):
        """Save point cloud in PCD format for visualization."""
        try:
            with open(self.current_pcd_file, 'w') as f:
                # PCD header
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z rgb instance_id\n")
                f.write("SIZE 4 4 4 4 4\n")
                f.write("TYPE F F F U U\n")
                f.write("COUNT 1 1 1 1 1\n")
                f.write(f"WIDTH {len(self.accumulated_points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(self.accumulated_points)}\n")
                f.write("DATA ascii\n")
                
                # Point data
                for point in self.accumulated_points:
                    rgb_value = point.get('rgb', 0) if point.get('rgb') is not None else 0
                    instance_id = point.get('instance_id', 0) if point.get('instance_id') is not None else 0
                    f.write(f"{point['x']:.6f} {point['y']:.6f} {point['z']:.6f} {rgb_value} {instance_id}\n")
                    
        except Exception as e:
            self.get_logger().error(f"Error saving PCD file: {e}")

    def save_final_map(self):
        """Save final map with timestamp when node shuts down."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_json = os.path.join(self.output_dir, f"final_semantic_map_{timestamp}.json")
        final_pcd = os.path.join(self.output_dir, f"final_semantic_map_{timestamp}.pcd")
        
        with self.lock:
            if self.accumulated_points:
                try:
                    # Copy current map as final version
                    import shutil
                    shutil.copy2(self.current_map_file, final_json)
                    shutil.copy2(self.current_pcd_file, final_pcd)
                    
                    unique_classes = set([inst['class_id'] for inst in self.accumulated_instances])
                    self.get_logger().info(
                        f"🎯 Final map saved! "
                        f"Points: {len(self.accumulated_points)}, "
                        f"Instances: {len(self.accumulated_instances)}, "
                        f"Classes: {len(unique_classes)}"
                    )
                    self.get_logger().info(f"📁 Final JSON: {final_json}")
                    self.get_logger().info(f"📁 Final PCD: {final_pcd}")
                    
                except Exception as e:
                    self.get_logger().error(f"Error saving final map: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = AutomaticMapSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down automatic map saver...")
        node.save_final_map()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()