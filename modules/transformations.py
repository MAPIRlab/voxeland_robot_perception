import numpy as np
from math import sin, cos, sqrt

from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Transform, TransformStamped, Quaternion, Vector3, Point, PoseWithCovariance, TwistWithCovariance
from nav_msgs.msg import Odometry

class Transformations(object):

    def pose_to_pq(self, msg):
        """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([msg.orientation.x, msg.orientation.y,
                      msg.orientation.z, msg.orientation.w])
        return p, q

    def pose_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.pose_to_pq(msg.pose)

    def transform_to_pq(self,msg):
        """Convert a C{geometry_msgs/Transform} or {geometry_msgs/TransformStamped} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        if isinstance(msg,TransformStamped): msg = msg.transform 

        p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y,
                      msg.rotation.z, msg.rotation.w])
        return p, q

    def transform_stamped_to_pq(self,msg):
        """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

        :param msg: ROS message to be converted
        :return:
          - p: position as a np.array
          - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.transform_to_pq(msg.transform)

    @staticmethod
    def tf_to_pose_with_covariance_stamped(msg):
        """Convert a TFMessage into pose with covariance stamped message

                :param msg: ROS message to be converted
                :return:
                  - pose_msg: pose with covariance stamped message including tf data, but with null covariance
                """

        pose_msg = PoseWithCovarianceStamped()
        point = Point()
        point.x = msg.transform.translation.x
        point.y = msg.transform.translation.y
        point.z = msg.transform.translation.z
        pose_msg.header = msg.header
        pose_msg.pose.pose.position = point
        pose_msg.pose.pose.orientation = msg.transform.rotation
        pose_msg.pose.covariance = 36 * [0.]

        return pose_msg

    @staticmethod
    def tf_to_odometry(msg):
        """Convert a TFMessage into Odometry message

                :param msg: ROS message to be converted
                :return:
                  - odom_msg: odometry message including tf data, but with null covariance
                """

        odom_msg = Odometry()
        odom_msg.header = msg.header
        odom_msg.child_frame_id = msg.child_frame_id

        pose = PoseWithCovariance()
        point = Point()
        point.x = msg.transform.translation.x
        point.y = msg.transform.translation.y
        point.z = msg.transform.translation.z
        pose.pose.position = point
        pose.pose.orientation = msg.transform.rotation
        pose.covariance = 36 * [0.]

        odom_msg.pose = pose

        return odom_msg
    
    def se3_to_msg(self, se3, covariance = None):

        """Convert a SE3 representation into a Pose/PoseWithCovariance msg
            - Input

            :se3: SE3 matrix representation of a pose/transformation
            :covariance: (Default - None) Covariance Matrix associated to the given SE3 matrix pose/transformation

            - Output
                  - pose_msg: Pose | PoseWithCovariance msg if covariance is None | not None
                """

        
        pose = Pose()
        pose.position.x = se3[0,3]
        pose.position.y = se3[1,3]
        pose.position.z = se3[2,3]
        pose.orientation = self.quaternion_from_matrix(se3)

        if covariance is not None:
            msg = PoseWithCovariance()
            msg.pose = pose
            msg.covariance = covariance
            
            return msg

        else:
            return pose


    def msg_to_se3(self,msg):
        """Conversion from geometric ROS messages into SE(3)

        :param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
        C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
        :return: a 4x4 SE(3) matrix as a numpy array
        @note: Throws TypeError if we receive an incorrect type.
        """
        if isinstance(msg, Pose):
            p, q = self.pose_to_pq(msg)
        elif isinstance(msg, PoseStamped):
            p, q = self.pose_stamped_to_pq(msg)
        elif isinstance(msg, Transform):
            p, q = self.transform_to_pq(msg)
        elif isinstance(msg, TransformStamped):
            p, q = self.transform_stamped_to_pq(msg)
        else:
            raise TypeError("Invalid type for conversion to SE(3)")
        norm = np.linalg.norm(q)
        if np.abs(norm - 1.0) > 1e-2:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                    str(q), np.linalg.norm(q)))
        else:
        #elif np.abs(norm - 1.0) > 1e-6:
            q = q / norm
        g = self.quaternion_matrix(q)
        g[0:3, -1] = p
        return g

    @staticmethod
    def quaternion_multiply(quaternion0, quaternion1):
        """

        :param quaternion0: quaternion 0
        :param quaternion1: quaternion 1
        :return: multiplication of quaternion 0 and 1
        """
        w0 = quaternion0.w
        x0 = quaternion0.x
        y0 = quaternion0.y
        z0 = quaternion0.z

        w1 = quaternion1.w
        x1 = quaternion1.x
        y1 = quaternion1.y
        z1 = quaternion1.z

        quat = Quaternion()
        quat.x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        quat.y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        quat.z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        quat.w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        return quat

    @staticmethod
    def x_rotation(theta):
        """

        :param theta: rotation angle in X-axis
        :return: matrix of transformation representing the rotation of theta in X-axis
        """
        return np.asarray([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])

    @staticmethod
    def y_rotation(theta):
        """

        :param theta: rotation angle in Y-axis
        :return: matrix of transformation representing the rotation of theta in Y-axis
        """
        return np.asarray([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

    @staticmethod
    def z_rotation(theta):
        """

        :param theta: rotation angle in Z-axis
        :return: matrix of transformation representing the rotation of theta in Z-axis
        """
        return np.asarray([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

    @staticmethod
    def quaternion_matrix(quaternion):
        """
                Covert a quaternion into a full three-dimensional rotation matrix.

                Input
                :param quaternion: A 4 element array representing the quaternion (q0,q1,q2,q3)

                Output
                :return: A 3x3 element matrix representing the full 3D rotation matrix.
                         This rotation matrix converts a point in the local reference
                         frame to a point in the global reference frame.

        """

        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < np.finfo(float).eps * 4.0:
            return np.identity(4)
        q *= sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    
    @staticmethod
    def quaternion_from_matrix(matrix):

        q = np.empty((4, ), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / sqrt(t * M[3, 3])

        q_msg = Quaternion()
        q_msg.x = q[0]
        q_msg.y = q[1]
        q_msg.z = q[2]
        q_msg.w = q[3]

        return q_msg
    
    def msg_to_pv_cvmatrix(self,pose_msg):  

        """Method to transform a Pose/PoseWithCovariance/PoseWithCovarianceStamped msg into position vector and Euler angles
        
        - Input: 
        
        :pose_msg: msg received from topic
        
        - Output:

        :p: Position Vector [x,y,z]
        :v_ang: Euler Angles Vector [pitch, roll, yaw] = [u,v,w]
        :cv_matrix: Covariance Matrix from pose      
        
        """  

        p,q,cv_matrix=self.pose_to_pq_cvmatrix(pose_msg) # P, Covariance Matrix and Quaternion extracted from msg
        v_ang = self.quaternion_to_euler(q) # Quaternion to Euler angles conversion

        return p, v_ang, cv_matrix
    
    def pv_to_msg(self,p,v_ang,cv_matrix):

        """Method to transform Position and Euler Angles vectors to PoseWithCovariance msg 

        - Input: 
        
        :p: Position Vector [x,y,z]
        :v_ang: Euler Angles Vector [pitch, roll, yaw] = [u,v,w]
        :cv_matrix: Covariance Matrix  
        
        - Output:

        :msg: PoseWithCovariance msg   
        
        """  

        msg = PoseWithCovariance()

        q = self.euler_to_quaternion(v_ang)   
        se3=self.quaternion_matrix(q)
        se3[0,3]=p[0]
        se3[1,3]=p[1]
        se3[2,3]=p[2]

        covariance = cv_matrix.flatten()
        
        msg = self.se3_to_msg(se3,covariance)

        return msg

    @staticmethod
    def quaternion_to_euler(q):

        ''' 
        Method to convert quaternion to roll, pitch and yaw angles
        
        - Input: q = quaternion array [x,y,z,w] (xi + yj + zk + w)

        - Output ang_v = [y,p,r] : [yaw , pitch , roll]
        
        '''

        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]

        disc = w*y-x*z

        if np.abs(disc) < 0.5:  
            yaw = np.arctan2(2*(z*w+x*y),1-2*(y**2*z**2))
            pitch = np.arcsin(2*disc)
            roll = np.arctan2(2*(w*x+y*z),1-2*(x**2+y**2))

            return [yaw,pitch,roll]
        
        elif disc == 0.5:
            
            yaw = -2 *np.atan2(x/w)
            pitch = np.pi/2
        
        else:
            yaw = 2*np.atan2(x,w)
            pitch = -np.pi/2

        return [yaw,pitch,0]
    
    @staticmethod 
    def euler_to_quaternion(v_ang):

        """:Euler to Quaternion: 
        
        - Input:
            :v_angle: 3x1 array containing roll, pitch and yaw angles [u,v,w]

        - Output
            :q: Quaternion array, [x,y,z,w]
        
        """

        q=[0,0,0,0]

        u = v_ang[0]
        v = v_ang[1]
        w = v_ang[2]

        q[0] = np.sin(u/2)*np.cos(v/2)*np.cos(w/2)-np.cos(u/2)*np.sin(v/2)*np.sin(w/2)   #x
        q[1] = np.cos(u/2)*np.sin(v/2)*np.cos(w/2)+np.sin(u/2)*np.cos(v/2)*np.sin(w/2)   #y
        q[2] = np.cos(u/2)*np.cos(v/2)*np.sin(w/2)-np.sin(u/2)*np.sin(v/2)*np.cos(w/2)   #z
        q[3] = np.cos(u/2)*np.cos(v/2)*np.cos(w/2)+np.sin(u/2)*np.sin(v/2)*np.sin(w/2)   #w

        return q

    @staticmethod
    def pose_to_pq_cvmatrix(pose_msg):

        '''Transform Pose/PosewithCovariance/PoswWithCovarianceStamped msgs to p, q, and covariance matrix
        
        - Input: 
                :pose_msg: geometry_msg/ (a. Pose, b. PoseWithCovariance, c. PoseWithCovarianceStamped)

        - Output: 
                :p: Position vector: [x,y,z]
                :q: Quaternion vector: [x,y,z,w]
                :cv_matrix: 6x6 covariance matrix          

        '''

        if isinstance(pose_msg,Pose):
            
            p = [pose.position.x, pose.position.y, pose.position.z] 
            q = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
            cv_matrix = None

            return p,q,cv_matrix

        if isinstance(pose_msg,PoseWithCovarianceStamped):
            pose = pose_msg.pose.pose
            cv_matrix=pose_msg.pose.covariance.reshape(6,6)
        elif isinstance(pose_msg,PoseWithCovariance):
            pose = pose_msg.pose
            cv_matrix=pose_msg.covariance.reshape(6,6)
        
        p=[pose.position.x, pose.position.y, pose.position.z]
        q=[pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]

        return p,q,cv_matrix
    

    def se3_to_TransformStamped(self, se3):

        '''Converts a se3 pose transformation into a TransformStamped msg, required type by TF2 in order to propagate uncertainity
        
        
         - Input: 
                :se3: SE3 Transformation Matrix 

        - Output: 
                :transform: TransformStamped msg: TransformStamped geometry.msg used by TF2         
        '''

        transform = TransformStamped()


        pose = self.se3_to_msg(se3)

        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z

        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w

        return transform

 
    


    @staticmethod
    def sample_distribution(p,v_ang,cv_matrix,n_samples):

        '''Sample
        
        - Input: 
                :p: Position vector [x,y,z]
                :v_ang: Euler Angle Vector: [roll, pitch, yaw]
                :cv_matrix: 6x6 Covariance Matrix
                :n_sample: Desired number of Pose samples

        - Output: 
                :sample_results: n_samples x 6 matrix [x_i y_i z_i roll_i pitch_i yaw_i]                                                            
        '''

        mean_v = np.concatenate([p,v_ang]) 

        rng = np.random.RandomState(12)
        sampled_results=rng.multivariate_normal(mean=mean_v,cov=cv_matrix,size=n_samples)
        
        return sampled_results
        