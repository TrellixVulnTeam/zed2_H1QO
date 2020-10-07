import open3d as o3d

for i in range(1, 3, 1):
    pcd_file = 'C:/00_work/05_src/zed2/zed-opencv/python/Depth_%d.ply' % (i)
    print("Reading %s..." % (pcd_file))
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd_file_out = 'C:/00_work/05_src/zed2/zed-opencv/python/Depth_%d.pcd' % (i)
    o3d.io.write_point_cloud(pcd_file_out, pcd)