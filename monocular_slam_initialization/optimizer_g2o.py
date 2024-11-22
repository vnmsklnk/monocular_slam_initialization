import math
from typing import Iterable

import g2o
import numpy as np

from .features import FeatureManager
from .frame import Frame
from .map_point import MapPoint
from .utils import poseRt


def bundle_adjustment(
    keyframes: Iterable[Frame],
    points: Iterable[MapPoint],
    feature_manager,
    local_window,
    fixed_points=False,
    verbose=False,
    rounds=10,
    use_robust_kernel=False,
    abort_flag=g2o.Flag(),
):
    if local_window is None:
        local_frames = keyframes
    else:
        local_frames = keyframes[-local_window:]

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)
    opt.set_force_stop_flag(abort_flag)

    thHuberMono = math.sqrt(5.991)

    graph_keyframes, graph_points = {}, {}

    # add frame vertices to graph
    for kf in local_frames if fixed_points else keyframes:
        # if points are fixed then consider just the local frames, otherwise we need all frames or at least two frames for each point
        se3 = g2o.SE3Quat(kf.Rcw, kf.tcw)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(kf.kid == 0 or kf not in local_frames)  # (use f.kid here!)
        opt.add_vertex(v_se3)

        graph_keyframes[kf] = v_se3

    num_edges = 0

    # add point vertices to graph
    for p in points:
        assert p is not None
        if p.is_bad:  # do not consider bad points
            continue
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(p.id * 2 + 1)  # odd ids
        v_p.set_estimate(p.pt[0:3])
        v_p.set_marginalized(True)
        v_p.set_fixed(fixed_points)
        opt.add_vertex(v_p)
        graph_points[p] = v_p

        # add edges
        for kf, idx in p.observations():
            if kf not in graph_keyframes:
                continue

            invSigma2 = feature_manager.inv_level_sigmas2[kf.octaves[idx]]

            edge = g2o.EdgeSE3ProjectXYZ()
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_keyframes[kf])
            edge.set_measurement(kf.kpsu[idx])

            edge.set_information(np.eye(2) * invSigma2)
            if use_robust_kernel:
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx = kf.fx
            edge.fy = kf.fy
            edge.cx = kf.cx
            edge.cy = kf.cy

            opt.add_edge(edge)
            num_edges += 1

    if verbose:
        opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(rounds)

    # put frames back
    for kf in graph_keyframes:
        est = graph_keyframes[kf].estimate()
        R = est.rotation().matrix()
        t = est.translation()
        kf.update_pose(poseRt(R, t))

    # put points back
    if not fixed_points:
        for p in graph_points:
            p.update_position(np.array(graph_points[p].estimate()))
            p.update_normal_and_depth(force=True)

    mean_squared_error = opt.active_chi2() / max(num_edges, 1)

    return mean_squared_error

def pose_optimization(frame: Frame, feature_manager: FeatureManager, rounds=10):
    is_ok = True

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)

    thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS

    point_edge_pairs = {}
    num_point_edges = 0

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw))
    v_se3.set_id(0)
    v_se3.set_fixed(False)
    opt.add_vertex(v_se3)

    # add point vertices to graph
    for idx, p in enumerate(frame.points):
        if p is None:
            continue

        # reset outlier flag
        frame.outliers[idx] = False

        # add edge
        invSigma2 = feature_manager.inv_level_sigmas2[frame.octaves[idx]]

        edge = g2o.EdgeSE3ProjectXYZOnlyPose()

        edge.set_vertex(0, opt.vertex(0))
        edge.set_measurement(frame.kpsu[idx])

        edge.set_information(np.eye(2) * invSigma2)
        edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

        edge.fx = frame.fx
        edge.fy = frame.fy
        edge.cx = frame.cx
        edge.cy = frame.cy
        edge.Xw = p.pt[0:3]

        opt.add_edge(edge)

        point_edge_pairs[p] = (edge, idx)  # one edge per point
        num_point_edges += 1

    if num_point_edges < 3:
        print('pose_optimization: not enough correspondences!')
        is_ok = False
        return 0, is_ok, 0

    # perform 4 optimizations:
    # after each optimization we classify observation as inlier/outlier;
    # at the next optimization, outliers are not included, but at the end they can be classified as inliers again
    chi2Mono = 5.991  # chi-square 2 DOFs
    num_bad_point_edges = 0

    for it in range(4):
        v_se3.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw))
        opt.initialize_optimization()
        opt.optimize(rounds)

        num_bad_point_edges = 0

        for p, edge_pair in point_edge_pairs.items():
            edge, idx = edge_pair
            if frame.outliers[idx]:
                edge.compute_error()

            chi2 = edge.chi2()

            chi2_check_failure = chi2 > chi2Mono
            if chi2_check_failure:
                frame.outliers[idx] = True
                edge.set_level(1)
                num_bad_point_edges += 1
            else:
                frame.outliers[idx] = False
                edge.set_level(0)

            if it == 2:
                edge.set_robust_kernel(None)

        if len(opt.edges()) < 10:
            print('pose_optimization: stopped - not enough edges!')
            is_ok = False
            break

    print('pose optimization: available ', num_point_edges, ' points, found ', num_bad_point_edges, ' bad points')
    if num_point_edges == num_bad_point_edges:
        print('pose_optimization: all the available correspondences are bad!')
        is_ok = False

        # update pose estimation
    if is_ok:
        est = v_se3.estimate()
        R = est.rotation().matrix()
        t = est.translation()
        frame.update_pose(poseRt(R, t))

    # since we have only one frame here, each edge corresponds to a single distinct point
    num_valid_points = num_point_edges - num_bad_point_edges

    mean_squared_error = opt.active_chi2() / max(num_valid_points, 1)
    return mean_squared_error, is_ok, num_valid_points

