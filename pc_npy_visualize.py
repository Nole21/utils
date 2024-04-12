import numpy as np
from open3d import *

pc_file = "224_downsampled.npy"
pc = np.load(pc_file)
point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(pc)
# down_sampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=0.15)
# downsampled_pc = np.asarray(down_sampled_point_cloud.points)
# np.save("224_downsampled.npy", downsampled_pc)
# print(downsampled_pc.shape)
open3d.visualization.draw_geometries([point_cloud])
