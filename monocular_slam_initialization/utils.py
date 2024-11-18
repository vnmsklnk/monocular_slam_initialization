import cv2
import numpy as np


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret


def inv_T(T):
    ret = np.eye(4)
    R_T = T[:3, :3].T
    t = T[:3, 3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ t
    return ret


def triangulate_normalized_points(pose_1w, pose_2w, kpn_1, kpn_2):
    P1w = pose_1w[:3, :]
    P2w = pose_2w[:3, :]

    point_4d_hom = cv2.triangulatePoints(P1w, P2w, kpn_1.T, kpn_2.T)
    good_pts_mask = np.where(point_4d_hom[3] != 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3]

    points_3d = point_4d[:3, :].T
    return points_3d, good_pts_mask


def add_ones_1D(x):
    return np.array([x[0], x[1], 1])


def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1.0e-10:
        return v, norm
    return v / norm, norm
