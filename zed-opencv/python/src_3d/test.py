import numpy as np
import open3d as o3d

if __name__ == "__main__":
    # Visualize Point Cloud
    p = 'C:/00_work/05_src/data/frm_t/20201015155835'
    # p = "data/toYOU_20201021_/20201015155835"
    f = f"{p}/pcd_extracted.ply"
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(f)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
    # Voxel downsampling
    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd])
    #  Vertex normal estimation
    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])
    # Access estimated vertex normal
    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :])
    print("")
    # Crop point cloud
    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair])
    print("")
    # Paint point cloud
    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([chair])
    print("")