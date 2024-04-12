# open3d visualization of PLY format point cloud data
import open3d as o3d

file = "waymo_input.ply"
cloud = o3d.io.read_point_cloud(file) # Read point cloud
o3d.visualization.draw_geometries([cloud])    # Visualize point cloud   