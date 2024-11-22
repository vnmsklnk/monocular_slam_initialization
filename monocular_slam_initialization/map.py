import cv2
import g2o
import numpy as np
from ordered_set import OrderedSet

from .frame import Frame
from .map_point import MapPoint
from .optimizer_g2o import bundle_adjustment
from .utils import add_ones

kChi2Mono = 5.991
kScaleConsistencyFactor = 1.5
kLargeBAWindow = 20
kColorPatchDelta = 1


class Map:
    def __init__(self, feature_manager):
        self.points = set()
        self.keyframes = OrderedSet()
        self.max_point_id = 0
        self.max_keyframe_id = 0
        self.feature_manager = feature_manager

    def add_point(self, point):
        ret = self.max_point_id
        point.id = ret
        point.map = self
        self.max_point_id += 1
        self.points.add(point)
        return ret

    def add_keyframe(self, keyframe: Frame):
        ret = self.max_keyframe_id
        keyframe.kid = ret
        keyframe.map = self
        self.keyframes.add(keyframe)
        self.max_keyframe_id += 1
        return ret

    def add_points(
        self,
        points3d,
        mask_pts3d,
        kf1: Frame,
        kf2: Frame,
        idxs1,
        idxs2,
        img1,
        do_check,
        cos_max_parallax,
    ):
        assert points3d.shape[0] == len(idxs1)
        assert len(idxs2) == len(idxs1)

        idxs1 = np.array(idxs1)
        idxs2 = np.array(idxs2)

        added_points = []
        out_mask_pts3d = np.full(points3d.shape[0], False, dtype=bool)
        if mask_pts3d is None:
            mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)

        if do_check:
            # project points
            uvs1, proj_depths1 = kf1.project_points(points3d)
            bad_depths1 = proj_depths1 <= 0
            uvs2, proj_depths2 = kf2.project_points(points3d)
            bad_depths2 = proj_depths2 <= 0

            is_stereo1 = np.zeros(len(idxs1), dtype=bool)
            is_mono1 = np.logical_not(is_stereo1)
            is_stereo2 = np.zeros(len(idxs2), dtype=bool)
            is_mono2 = np.logical_not(is_stereo2)

            # compute back-projected rays (unit vectors)
            rays1 = np.dot(kf1.Rwc, add_ones(kf1.kpsn[idxs1]).T).T
            norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)
            rays1 /= norm_rays1
            rays2 = np.dot(kf2.Rwc, add_ones(kf2.kpsn[idxs2]).T).T
            norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)
            rays2 /= norm_rays2

            # compute dot products of rays
            cos_parallaxs = np.sum(rays1 * rays2, axis=1)

            recovered3d_from_stereo = np.zeros(len(idxs1), dtype=bool)
            bad_cos_parallaxs = np.logical_and(
                np.logical_or(cos_parallaxs < 0, cos_parallaxs > cos_max_parallax),
                np.logical_not(recovered3d_from_stereo),
            )

            # compute reprojection errors and check chi2
            # compute mono reproj errors on kf1
            errs1_mono_vec = uvs1 - kf1.kpsu[idxs1]
            errs1 = np.where(
                is_mono1[:, np.newaxis], errs1_mono_vec, np.zeros(2)
            )  # mono errors
            errs1_sqr = np.sum(errs1 * errs1, axis=1)  # squared reprojection errors
            kps1_levels = kf1.octaves[idxs1]
            invSigmas2_1 = self.feature_manager.inv_level_sigmas2[kps1_levels]
            chis2_1_mono = errs1_sqr * invSigmas2_1  # chi square
            bad_chis2_1 = chis2_1_mono > kChi2Mono

            # compute mono reproj errors on kf1
            errs2_mono_vec = uvs2 - kf2.kpsu[idxs2]  # mono errors
            errs2 = np.where(
                is_mono2[:, np.newaxis], errs2_mono_vec, np.zeros(2)
            )  # mono errors
            errs2_sqr = np.sum(errs2 * errs2, axis=1)  # squared reprojection errors
            kps2_levels = kf2.octaves[idxs2]
            invSigmas2_2 = self.feature_manager.inv_level_sigmas2[kps2_levels]
            chis2_2_mono = errs2_sqr * invSigmas2_2  # chi square
            bad_chis2_2 = (
                chis2_2_mono > kChi2Mono
            )  # chi-square 2 DOFs  (Hartley Zisserman pg 119)

            # scale consistency check
            ratio_scale_consistency = (
                kScaleConsistencyFactor * self.feature_manager.scale_factor
            )
            scale_factors_x_depths1 = (
                self.feature_manager.scale_factors[kps1_levels] * proj_depths1
            )
            scale_factors_x_depths1_x_ratio_scale_consistency = (
                scale_factors_x_depths1 * ratio_scale_consistency
            )
            scale_factors_x_depths2 = (
                self.feature_manager.scale_factors[kps2_levels] * proj_depths2
            )
            scale_factors_x_depths2_x_ratio_scale_consistency = (
                scale_factors_x_depths2 * ratio_scale_consistency
            )
            bad_scale_consistency = np.logical_or(
                (
                    scale_factors_x_depths1
                    > scale_factors_x_depths2_x_ratio_scale_consistency
                ),
                (
                    scale_factors_x_depths2
                    > scale_factors_x_depths1_x_ratio_scale_consistency
                ),
            )

            # combine all checks
            bad_points = (
                bad_cos_parallaxs
                | bad_depths1
                | bad_depths2
                | bad_chis2_1
                | bad_chis2_2
                | bad_scale_consistency
            )

        img_coords = np.rint(kf1.kps[idxs1]).astype(np.intp)
        delta = kColorPatchDelta
        patch_extension = 1 + 2 * delta
        img_pts_start = img_coords - delta
        img_pts_end = img_coords + delta
        img_ranges = np.linspace(
            img_pts_start, img_pts_end, patch_extension, dtype=np.intp
        )[:, :].T

        def img_range_elem(ranges, i):
            return ranges[:, i]

        for i, p in enumerate(points3d):
            if not mask_pts3d[i]:
                continue

            idx1_i = idxs1[i]
            idx2_i = idxs2[i]

            if do_check:
                if bad_points[i]:
                    continue
            try:
                img_range = img_range_elem(img_ranges, i)
                color_patch = img1[img_range[1][:, np.newaxis], img_range[0]]
                color = cv2.mean(color_patch)[:3]

            except IndexError:
                print("color out of range")
                color = (255, 0, 0)

            # add the point to this map
            mp = MapPoint(p[0:3], color, self.feature_manager, kf2, idx2_i)
            self.add_point(mp)
            mp.add_observation(kf1, idx1_i)
            mp.add_observation(kf2, idx2_i)
            mp.update_info()
            out_mask_pts3d[i] = True
            added_points.append(mp)
        return len(added_points), out_mask_pts3d, added_points

    def get_keyframes(self):
        return self.keyframes.copy()

    def get_points(self):
        return self.points.copy()

    def optimize(
        self,
        local_window=kLargeBAWindow,
        verbose=False,
        rounds=10,
        use_robust_kernel=False,
        abort_flag=g2o.Flag(),
    ):
        err = bundle_adjustment(
            self.get_keyframes(),
            self.get_points(),
            self.feature_manager,
            local_window=local_window,
            verbose=verbose,
            rounds=rounds,
            use_robust_kernel=use_robust_kernel,
            abort_flag=abort_flag,
        )
        return err
