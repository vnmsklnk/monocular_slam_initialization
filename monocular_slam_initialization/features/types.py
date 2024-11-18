from enum import Enum


class FeatureTypes(Enum):
    SUPERPOINT = 0  # [end-to-end] joint detector-descriptor - "SuperPoint: Self-Supervised Interest Point Detection and Description"
    XFEAT = 1  # [end-to-end] joint detector-descriptor - "XFeat: Accelerated Features for Lightweight Image Matching"


class FeatureMatcherTypes(Enum):
    XFEAT = 0  # "XFeat: Accelerated Features for Lightweight Image Matching"
    LIGHTGLUE = 1  # "LightGlue: Local Feature Matching at Light Speed"
