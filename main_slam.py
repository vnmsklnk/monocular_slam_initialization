import cv2
import numpy as np
import open3d as o3d

import monocular_slam_initialization as msi

from pathlib import Path
from tqdm import tqdm

detector = msi.features.FeatureTypes.XFEAT
matcher = msi.features.FeatureMatcherTypes.XFEAT

# OR
# detector = msi.features.FeatureTypes.SUPERPOINT
# matcher = msi.features.FeatureMatcherTypes.LIGHTGLUE

feature_manager = msi.features.FeatureManager(num_features=2000, detector_type=detector)
feature_matcher = msi.features.feature_matcher_factory(
    matcher_type=matcher, detector_type=detector
)
num_fin_features = 150  # minimum features for successful initialization
use_robust_kernel = True

# Uncomment for ICL
# K = np.array([[481.20, 0, 319.50],
#              [0, -480.00, 239.50],
#              [0, 0, 1]])
# ref_img = cv2.cvtColor(cv2.imread('rgb_icl/0.png'), cv2.COLOR_BGR2RGB)
# cur_img = cv2.cvtColor(cv2.imread('rgb_icl/47.png'), cv2.COLOR_BGR2RGB)

# Params for TUM RGB-D fr3
# K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])

K = np.array([[726.94, 0, 591.265], [0, 726.425, 522.91], [0, 0, 1]])
images = sorted(
    Path("/home/ivan/Documents/Datasets/AMtown01/queries_resized/").iterdir()
)
ref_img = cv2.cvtColor(cv2.imread(str(images[100])), cv2.COLOR_BGR2RGB)
cur_img = cv2.cvtColor(cv2.imread(str(images[160])), cv2.COLOR_BGR2RGB)
D = np.array([0, 0, 0, 0, 0])  # ICL and TUM (fr3) are undistorted

initializer = msi.Initializer(
    num_fin_features, use_robust_kernel, feature_manager, feature_matcher
)
reference = msi.Frame(ref_img, feature_manager, K, D)
current = msi.Frame(cur_img, feature_manager, K, D)
map = initializer.initialize(reference, current)

if len(map.points) < num_fin_features:
    print("Not enough points for initialization!")
    exit()

images = images[161:]
next_coord_frames = []
last_pose = current.Tcw
for i, image in enumerate(tqdm(images)):
    next_img = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    next_frame = msi.Frame(next_img, feature_manager, K, D)

    next_frame.update_pose(last_pose)
    insert_keyframe = i % 5 == 0 and i != 0
    num_valid_points = msi.track_reference_frame(
        current,
        next_frame,
        feature_manager,
        feature_matcher,
        map,
        insert_keyframe=insert_keyframe,
    )
    if num_valid_points < 10:
        break
    next_coord_frames.append(
        o3d.geometry.TriangleMesh()
        .create_coordinate_frame()
        .paint_uniform_color([0, 0, 1])
        .transform(next_frame.Twc)
    )
    if insert_keyframe:
        current = next_frame
    last_pose = next_frame.Tcw

ref_coord_frame = (
    o3d.geometry.TriangleMesh().create_coordinate_frame().paint_uniform_color([1, 0, 0])
)
cur_coord_frame = (
    o3d.geometry.TriangleMesh()
    .create_coordinate_frame()
    .paint_uniform_color([0, 1, 0])
    .transform(current.Twc)
)

positions = []
colors = []
for point in map.points:
    positions.append(list(point.pt))
    colors.append(list(point.color))

positions = np.array(positions)
colors = np.array(colors) / 255
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(positions)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries(
    [point_cloud] + [ref_coord_frame, cur_coord_frame] + next_coord_frames
)
