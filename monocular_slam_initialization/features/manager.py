import math

import cv2
import numpy as np

from .superpoint import SuperPointFeature2D
from .types import FeatureTypes
from .xfeat import XFeat2D as XfeatFeature2D


class FeatureManager:
    def __init__(self, num_features: int, detector_type: FeatureTypes):
        self.detector_type = detector_type
        self.num_features = num_features
        self.scale_factor = 1.2
        self.sigma_level0 = 1.0
        self.kNumLevelsInitSigma = 40
        if self.detector_type == FeatureTypes.SUPERPOINT:
            self._feature_detector = SuperPointFeature2D()
        elif self.detector_type == FeatureTypes.XFEAT:
            self._feature_detector = XfeatFeature2D(self.num_features)
        else:
            raise ValueError("Unknown feature detector %s" % self.detector_type)
        self.init_sigma_levels()

    def init_sigma_levels(self):
        num_levels = self.kNumLevelsInitSigma
        self.inv_scale_factor = 1.0 / self.scale_factor
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale_factor)

        self.scale_factors[0] = 1.0
        self.level_sigmas2[0] = self.sigma_level0 * self.sigma_level0
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i - 1] * self.scale_factor
            self.level_sigmas2[i] = (
                self.scale_factors[i] * self.scale_factors[i] * self.level_sigmas2[0]
            )
            self.level_sigmas[i] = math.sqrt(self.level_sigmas2[i])
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0 / self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0 / self.level_sigmas2[i]

    def detectAndCompute(self, frame, mask=None):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kps, des = self._feature_detector.detectAndCompute(frame, mask)
        return kps, des
