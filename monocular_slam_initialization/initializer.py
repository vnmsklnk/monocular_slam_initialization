import cv2
import numpy as np

from .frame import Frame, match_frames
from .map import Map
from .utils import inv_T, poseRt, triangulate_normalized_points

kRansacThresholdNormalized = (
    0.0004  # metric threshold used for normalized image coordinates
)
kRansacProb = 0.999


class Initializer:
    def __init__(
        self, num_fin_features, use_robust_kernel, feature_manager, feature_matcher
    ):
        self.mask_match = None
        self.feature_manager = feature_manager
        self.feature_matcher = feature_matcher
        self.num_min_features = num_fin_features
        self.use_robust_kernel = use_robust_kernel

    def estimatePose(self, kpn_ref, kpn_cur):
        try:
            ransac_method = cv2.USAC_MSAC
        except:
            ransac_method = cv2.RANSAC
            # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )
        E, self.mask_match = cv2.findEssentialMat(
            kpn_cur,
            kpn_ref,
            focal=1,
            pp=(0.0, 0.0),
            method=ransac_method,
            prob=kRansacProb,
            threshold=kRansacThresholdNormalized,
        )
        _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0.0, 0.0))
        return poseRt(R, t.T)

    def initialize(self, f_ref: Frame, f_cur: Frame):
        is_ok = False
        if (
            len(f_ref.kps) < self.num_min_features
            or len(f_cur.kps) < self.num_min_features
        ):
            print("Inializer: ko - not enough features!")
            return is_ok
        else:
            print(f"Number ref kps: {len(f_ref.kps)}, number cur kps: {len(f_cur.kps)}")

        matching_result = match_frames(f_cur, f_ref, self.feature_matcher)
        idxs_cur, idxs_ref = np.asarray(matching_result.idxs1), np.asarray(
            matching_result.idxs2
        )

        print("# keypoint matches: ", len(idxs_cur))

        Trc = self.estimatePose(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur])
        Tcr = inv_T(Trc)
        f_ref.update_pose(np.eye(4))
        f_cur.update_pose(Tcr)

        # remove outliers from keypoint matches by using the mask computed with inter frame pose estimation
        mask_idxs = self.mask_match.ravel() == 1
        self.num_inliers = sum(mask_idxs)
        print("# keypoint inliers: ", self.num_inliers)
        idx_ref_inliers = idxs_ref[mask_idxs]
        idx_cur_inliers = idxs_cur[mask_idxs]

        map = Map(self.feature_manager)
        map.add_keyframe(f_ref)
        map.add_keyframe(f_cur)

        do_check = True
        pts3d, mask_pts3d = triangulate_normalized_points(
            f_cur.Tcw,
            f_ref.Tcw,
            f_cur.kpsn[idx_cur_inliers],
            f_ref.kpsn[idx_ref_inliers],
        )

        new_pts_count, mask_points, _ = map.add_points(
            pts3d,
            mask_pts3d,
            f_cur,
            f_ref,
            idx_cur_inliers,
            idx_ref_inliers,
            f_cur.img,
            do_check=do_check,
            cos_max_parallax=0.99998,
        )
        print("# triangulated points: ", new_pts_count)

        if new_pts_count > self.num_min_features:
            err = map.optimize(verbose=False, rounds=20, use_robust_kernel=self.use_robust_kernel)
            print("# init optimization error^2: %f" % err)

            num_map_points = len(map.points)
            print("# map points:   %d" % num_map_points)
            is_ok = num_map_points > self.num_min_features

            # set scene median depth to equal desired_median_depth
            pts = pts3d[mask_points]
            desired_median_depth = 20
            median_depth = f_cur.compute_points_median_depth(pts)
            if median_depth <= 0:
                is_ok = False
                return is_ok
            depth_scale = desired_median_depth / median_depth
            print(
                "forcing current median depth ",
                median_depth,
                " to ",
                desired_median_depth,
            )
            pts[:, :3] = pts[:, :3] * depth_scale  # scale points
            tcw = f_cur.tcw * depth_scale  # scale initial baseline
            f_cur.update_translation(tcw)

        if is_ok:
            print("Inializer: ok!")
        else:
            print("Inializer: ko!")
        return map
