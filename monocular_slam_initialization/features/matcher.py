import torch

from monocular_slam_initialization.thirdparty.accelerated_features.modules.xfeat import (
    XFeat,
)
from monocular_slam_initialization.thirdparty.LightGlue.lightglue import LightGlue

from .types import FeatureMatcherTypes, FeatureTypes


def feature_matcher_factory(
    matcher_type: FeatureMatcherTypes, detector_type: FeatureTypes
):
    if matcher_type == FeatureMatcherTypes.XFEAT:
        if detector_type != FeatureTypes.XFEAT:
            raise ValueError(
                f"Error. XFeat matcher should be used only with XFeat features"
            )
        return XFeatMatcher()
    elif matcher_type == FeatureMatcherTypes.LIGHTGLUE:
        if detector_type != FeatureTypes.SUPERPOINT:
            raise ValueError(
                f"Error. LightGlue matcher should be used only with SuperPoint features"
            )
        return LightGlueMatcher()
    else:
        raise ValueError(f"Unsupported matcher type: {matcher_type.name}")


class FeatureMatchingResult:
    def __init__(self):
        self.kps1 = None  # all reference keypoints (numpy array Nx2)
        self.kps2 = None  # all current keypoints   (numpy array Nx2)
        self.des1 = None  # all reference descriptors (numpy array NxD)
        self.des2 = None  # all current descriptors (numpy array NxD)
        self.idxs1 = None  # indices of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs2 = None  # indices of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)


class FeatureMatcher:
    def __init__(self, matcher_type=FeatureMatcherTypes.LIGHTGLUE):
        self.matcher_type = matcher_type
        self.matcher = None

    def match(self, img1, img2, des1, des2, kps1=None, kps2=None):
        result = FeatureMatchingResult()
        result.des1 = des1
        result.des2 = des2
        result.kps1 = kps1
        result.kps2 = kps2
        if self.matcher_type == FeatureMatcherTypes.LIGHTGLUE:
            if kps1 is None and kps2 is None:
                return result
            img1_shape = img1.shape[0:2]
            d0 = {
                "keypoints": torch.tensor(kps1, device=self.torch_device).unsqueeze(0),
                "descriptors": torch.tensor(des1, device=self.torch_device).unsqueeze(
                    0
                ),
                "image_size": torch.tensor(
                    img1_shape, device=self.torch_device
                ).unsqueeze(0),
            }
            d1 = {
                "keypoints": torch.tensor(kps2, device=self.torch_device).unsqueeze(0),
                "descriptors": torch.tensor(des2, device=self.torch_device).unsqueeze(
                    0
                ),
                "image_size": torch.tensor(
                    img1_shape, device=self.torch_device
                ).unsqueeze(0),
            }
            matches01 = self.matcher({"image0": d0, "image1": d1})
            idx0 = matches01["matches"][0][:, 0].cpu().tolist()
            idxs1 = matches01["matches"][0][:, 1].cpu().tolist()
            result.idxs1 = idx0
            result.idxs2 = idxs1
        elif self.matcher_type == FeatureMatcherTypes.XFEAT:
            d1_tensor = torch.tensor(des1, dtype=torch.float32)
            d2_tensor = torch.tensor(des2, dtype=torch.float32)
            min_cossim = 0.82  # default in xfeat code
            idx0, idxs1 = self.matcher.match(
                d1_tensor, d2_tensor, min_cossim=min_cossim
            )
            result.idxs1 = idx0.cpu()
            result.idxs2 = idxs1.cpu()
        return result


class XFeatMatcher(FeatureMatcher):
    def __init__(self):
        super().__init__(matcher_type=FeatureMatcherTypes.XFEAT)
        self.matcher = XFeat()
        self.matcher_name = "XFeatFeatureMatcher"
        print(f"matcher: {self.matcher_name}")


class LightGlueMatcher(FeatureMatcher):
    def __init__(self):
        super().__init__(matcher_type=FeatureMatcherTypes.LIGHTGLUE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = device
        self.matcher = LightGlue(features="superpoint", n_layers=2).eval().to(device)
        self.matcher_name = "LightGlueFeatureMatcher"
        print("device: ", self.torch_device)
        print(f"matcher: {self.matcher_name}")
