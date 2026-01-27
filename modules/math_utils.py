import numpy as np

class MathUtils(object):

    @staticmethod
    def quat_inverse(q: np.ndarray):
        conjugate = np.array([q[0], -q[1], -q[2], -q[3]])
        norm = np.linalg.norm(q)
        return conjugate / norm

    # IMPORTANT: assumes unit quaternions
    @staticmethod
    def quat_distance(a: np.ndarray, b: np.ndarray):
        angle = np.arccos(2 * np.dot(a,b)**2 -1)
        return angle