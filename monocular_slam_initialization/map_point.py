from typing import Dict, List, Tuple

import numpy as np

from .frame import Frame
from .utils import normalize_vector


class MapPointBase:
    _id = 0

    def __init__(self, id=None):
        if id is not None:
            self.id = id
        else:
            self.id = MapPointBase._id
            MapPointBase._id += 1

        self.map = None
        self._observations: Dict[Frame, int] = dict()
        self._frame_views = dict()
        self._is_bad = False
        self._num_observations = 0

    def observations(self) -> List[Tuple[Frame, int]]:
        return list(
            self._observations.items()
        )  # https://www.python.org/dev/peps/pep-0469/

    def add_observation(self, keyframe, idx):
        if keyframe not in self._observations:
            keyframe.set_point_match(self, idx)  # add point association in keyframe
            self._observations[keyframe] = idx
            self._num_observations += 1
            return True
        else:
            return False

    @property
    def is_bad(self):
        return self._is_bad


class MapPoint(MapPointBase):

    def __init__(
        self,
        position,
        color,
        feature_manager,
        keyframe: Frame = None,
        idxf=None,
        id=None,
    ):
        super().__init__(id)
        self._pt = np.array(position)

        self.color = color
        self.feature_manager = feature_manager

        self.des = None
        self._min_distance, self._max_distance = 0, float("inf")
        self.normal = np.array([0, 0, 1])

        self.kf_ref = keyframe
        self.first_kid = -1
        if keyframe is not None:
            self.first_kid = keyframe.kid
            if idxf is not None:
                self.des = keyframe.des[idxf]
            po = self._pt - self.kf_ref.Ow
            self.normal, dist = normalize_vector(po)
            if idxf is not None:
                level = keyframe.octaves[idxf]
            level_scale_factor = feature_manager.scale_factors[level]
            self._max_distance = dist * level_scale_factor
            self._min_distance = self._max_distance / feature_manager.scale_factors[0]

        self.num_observations_on_last_update_des = 1
        self.num_observations_on_last_update_normals = 1

    @property
    def pt(self):
        return self._pt

    def update_position(self, position):
        self._pt = position

    # update normal and depth representations
    def update_normal_and_depth(self, frame=None, idxf=None, force=False):
        skip = False
        if self._is_bad:
            return
        if (
            self._num_observations > self.num_observations_on_last_update_normals
            or force
        ):
            self.num_observations_on_last_update_normals = self._num_observations
            observations = list(self._observations.items())
            kf_ref = self.kf_ref
            idx_ref = self._observations[kf_ref]
            position = self._pt.copy()
        else:
            skip = True
        if skip or len(observations) == 0:
            return

        normals = np.array(
            [normalize_vector(position - kf.Ow)[0] for kf, idx in observations]
        )
        normal = normalize_vector(np.mean(normals, axis=0))[0]

        level = kf_ref.octaves[idx_ref]
        level_scale_factor = self.feature_manager.scale_factors[level]
        dist = np.linalg.norm(position - kf_ref.Ow)

        self._max_distance = dist * level_scale_factor
        self._min_distance = self._max_distance / self.feature_manager.scale_factors[0]
        self.normal = normal

    def update_best_descriptor(self, force=False):
        skip = False
        if self._is_bad:
            return
        if self._num_observations > self.num_observations_on_last_update_des or force:
            self.num_observations_on_last_update_des = self._num_observations
            observations = list(self._observations.items())
        else:
            skip = True
        if skip or len(observations) == 0:
            return
        descriptors = [kf.des[idx] for kf, idx in observations]
        N = len(descriptors)
        if N > 2:
            median_distances = [
                np.median(
                    self.feature_manager.descriptor_distances(
                        descriptors[i], descriptors
                    )
                )
                for i in range(N)
            ]
            self.des = descriptors[np.argmin(median_distances)].copy()

    def update_info(self):
        self.update_normal_and_depth()
        self.update_best_descriptor()
