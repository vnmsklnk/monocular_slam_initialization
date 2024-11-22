import cv2
import torch

from monocular_slam_initialization.thirdparty.SuperPointPretrainedNetwork.demo_superpoint import (
    SuperPointFrontend,
)


class SuperPointOptions:
    def __init__(self, do_cuda=True):
        self.weights_path = "monocular_slam_initialization/thirdparty/SuperPointPretrainedNetwork/superpoint_v1.pth"
        self.nms_dist = 4
        self.conf_thresh = 0.015
        self.nn_thresh = 0.7
        use_cuda = torch.cuda.is_available() and do_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        print("SuperPoint using ", device)
        self.cuda = use_cuda


def convert_superpts_to_keypoints(pts, size=1):
    kps = []
    if pts is not None:
        kps = [cv2.KeyPoint(p[0], p[1], size=size, response=p[2]) for p in pts]
    return kps


def transpose_des(des):
    if des is not None:
        return des.T
    else:
        return None


class SuperPointFeature2D:
    def __init__(self, do_cuda=True):
        self.opts = SuperPointOptions(do_cuda)
        self.fe = SuperPointFrontend(
            weights_path=self.opts.weights_path,
            nms_dist=self.opts.nms_dist,
            conf_thresh=self.opts.conf_thresh,
            nn_thresh=self.opts.nn_thresh,
            cuda=self.opts.cuda,
        )
        self.pts = []
        self.kps = []
        self.des = []
        self.heatmap = []
        self.frame = None
        self.frameFloat = None
        self.keypoint_size = 20

    def detectAndCompute(self, frame, mask=None):
        self.frame = frame
        self.frameFloat = frame.astype("float32") / 255.0
        self.pts, self.des, self.heatmap = self.fe.run(self.frameFloat)
        self.kps = convert_superpts_to_keypoints(self.pts.T, size=self.keypoint_size)
        return self.kps, transpose_des(self.des)
