import numpy as np

from .frame import match_frames
from .optimizer_g2o import pose_optimization
from .utils import triangulate_normalized_points


def propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur):
    idx_ref_out = []
    idx_cur_out = []

    # populate f_cur with map points by propagating map point matches of f_ref;
    # to this aim, we use map points observed in f_ref and keypoint matches between f_ref and f_cur
    num_matched_map_pts = 0
    for i, idx in enumerate(idxs_ref):  # iterate over keypoint matches
        p_ref = f_ref.points[idx]
        if (
            p_ref is None
        ):  # we don't have a map point P for i-th matched keypoint in f_ref
            continue
        if (
            f_ref.outliers[idx] or p_ref.is_bad
        ):  # do not consider pose optimization outliers or bad points
            continue
        idx_cur = idxs_cur[i]

        if p_ref.add_observation(
            f_cur, idx_cur
        ):  # => P is matched to the i-th matched keypoint in f_cur
            num_matched_map_pts += 1
            idx_ref_out.append(idx)
            idx_cur_out.append(idx_cur)

    return num_matched_map_pts, idx_ref_out, idx_cur_out


def track_reference_frame(
    f_ref, f_cur, feature_manager, feature_matcher, map, insert_keyframe=False
):
    matching_result = match_frames(f_cur, f_ref, feature_matcher)
    idxs_cur, idxs_ref = np.asarray(matching_result.idxs1), np.asarray(
        matching_result.idxs2
    )
    num_matched_kps = idxs_cur.shape[0]
    print("# keypoints matched: %d " % num_matched_kps)

    # propagate map point matches from kf_ref to f_cur  (do not override idxs_ref, idxs_cur)
    num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop = (
        propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur)
    )
    print("# matched map points in prev frame: %d " % num_found_map_pts_inter_frame)

    # f_cur pose optimization using last matches with kf_ref:
    # here, we use first guess of f_cur pose and propated map point matches from f_ref (matched keypoints)
    _, _, num_valid_points = pose_optimization(f_cur, feature_manager)
    if insert_keyframe:
        pts3d, mask_pts3d = triangulate_normalized_points(
            f_cur.Tcw,
            f_ref.Tcw,
            f_cur.kpsn[idxs_cur],
            f_ref.kpsn[idxs_ref],
        )
        new_pts_count, _, _ = map.add_points(
            pts3d,
            mask_pts3d,
            f_cur,
            f_ref,
            idxs_cur,
            idxs_ref,
            f_cur.img,
            do_check=True,
            cos_max_parallax=0.9999,
        )
    return num_valid_points
