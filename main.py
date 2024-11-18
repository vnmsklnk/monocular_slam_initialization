import cv2
import numpy as np
import open3d as o3d

import monocular_slam_initialization as msi

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
K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
ref_img = cv2.cvtColor(cv2.imread("rgb_tum/1341847980.722988.png"), cv2.COLOR_BGR2RGB)
cur_img = cv2.cvtColor(cv2.imread("rgb_tum/1341847981.994808.png"), cv2.COLOR_BGR2RGB)

D = np.array([0, 0, 0, 0, 0])  # ICL and TUM (fr3) are undistorted

initializer = msi.Initializer(
    num_fin_features, use_robust_kernel, feature_manager, feature_matcher
)
reference = msi.Frame(ref_img, feature_manager, K, D)
current = msi.Frame(cur_img, feature_manager, K, D)
points = initializer.initialize(reference, current)

if len(points) < num_fin_features:
    print("Not enough points for initialization!")
    exit()

positions = []
colors = []
for point in points:
    positions.append(list(point.pt))
    colors.append(list(point.color))

positions = np.array(positions)
colors = np.array(colors) / 255
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(positions)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([point_cloud])
