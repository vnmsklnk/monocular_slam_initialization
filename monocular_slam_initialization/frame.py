import cv2
import numpy as np

from .utils import add_ones, inv_T, poseRt


class Frame:
    def __init__(self, img, feature_manager, K, D):
        self.img = img
        self.Tcw = None
        self.Rcw = None
        self.tcw = None
        self.Rwc = None
        self.Ow = None
        self.Twc = None
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.Kinv = np.array(
            [
                [1 / self.fx, 0, -self.cx / self.fx],
                [0, 1 / self.fy, -self.cy / self.fy],
                [0, 0, 1],
            ]
        )
        self.D = D
        self.kps, self.des = feature_manager.detectAndCompute(img)

        kps_data = np.array(
            [[x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps],
            dtype=np.float32,
        )
        self.kps = kps_data[:, :2]
        self.octaves = np.uint32(kps_data[:, 2])
        self.kpsu = self.undistort_points(self.kps)
        self.kpsn = self.unproject_points(self.kpsu)
        self.points = np.array([None] * len(self.kpsu))
        self.kid = None
        self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool) # actually it's not used

    def set_point_match(self, p, idx):
        self.points[idx] = p

    def undistort_points(self, uvs):
        uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
        uvs_undistorted = cv2.undistortPoints(
            uvs_contiguous, self.K, self.D, None, self.K
        )
        return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)

    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]

    def update_translation(self, tcw):
        new_pose = poseRt(self.Rcw, tcw)
        self.update_pose(new_pose)

    def update_pose(self, pose):
        self.Tcw = pose
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, 3]
        self.Rwc = self.Rcw.T
        self.Ow = -(self.Rwc @ self.tcw)  # origin of camera frame w.r.t world
        self.Twc = inv_T(pose)

    def project(self, xcs):
        projs = self.K @ xcs.T
        zs = projs[-1]
        projs = projs[:2] / zs
        return projs.T, zs

    def project_points(self, points):
        Rcw = self.Rcw
        tcw = self.tcw.reshape((3, 1))
        return self.project((Rcw @ points.T + tcw).T)

    def compute_points_median_depth(self, points3d=None):
        Rcw2 = self.Rcw[2, :3]  # just 2-nd row
        tcw2 = self.tcw[2]  # just 2-nd row
        if len(points3d) > 0:
            z = np.dot(Rcw2, points3d[:, :3].T) + tcw2
            z = sorted(z)
            return z[(len(z) - 1) // 2]
        else:
            raise ValueError("frame.compute_points_median_depth() with no points")


def match_frames(f1: Frame, f2: Frame, feature_matcher):
    matching_result = feature_matcher.match(
        f1.img, f2.img, f1.des, f2.des, f1.kps, f2.kps
    )
    return matching_result
