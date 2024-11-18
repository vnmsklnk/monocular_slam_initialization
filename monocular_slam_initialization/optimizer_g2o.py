import math
from typing import Iterable

import g2o
import numpy as np

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
