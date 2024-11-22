import sys

import cv2
import torch

sys.path.append("monocular_slam_initialization/thirdparty/accelerated_features")

from monocular_slam_initialization.thirdparty.accelerated_features.modules.xfeat import (
    XFeat,
)


def convert_superpts_to_keypoints(pts, scores, size=1):
    kps = []
    if pts is not None:
        kps = [cv2.KeyPoint(int(p[0]), int(p[1]), size=size, response=1) for p in pts]
    return kps


class CVWrapper:
    def __init__(self, mtd):
        self.mtd = mtd

    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(
            torch.tensor(x).unsqueeze(0).float().unsqueeze(0)
        )[0]


class XFeat2D:
    def __init__(self, num_features=2000):
        self.xfeat = XFeat(top_k=num_features)
        self.pts = []
        self.kps = []
        self.des = []
        self.heatmap = []
        self.frame = None
        self.frameFloat = None
        self.keypoint_size = 30

    def detectAndCompute(self, frame, mask=None):
        current = CVWrapper(self.xfeat).detectAndCompute(frame)
        kpts, descs = (
            current["keypoints"].cpu().numpy(),
            current["descriptors"].cpu().numpy(),
        )
        self.pts, self.des = kpts, descs
        self.kps = convert_superpts_to_keypoints(
            self.pts, scores=1, size=self.keypoint_size
        )
        return self.kps, self.des
