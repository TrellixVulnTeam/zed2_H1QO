import open3d
import numpy as np
if __name__ == "__main__":
    path='C:/00_work/05_src/zed2/zed-opencv/python/data_20201021_1/'
    prefix_reconstruction = "reconstruction"
    cameras=['ZED_21888201','ZED_22378008']
    cameras=['ZED_22378008']
    for name_cam in cameras:
        for count_save in range(8):
            filename = path + prefix_reconstruction + "-%06d.cloud" % (count_save) + '-' + name_cam+'.ply'
            ply = open3d.io.read_point_cloud(filename=filename)
            filename = path + prefix_reconstruction + "-%06d.pose" % (count_save) + '-' + name_cam+ '.csv'
            transmatrix=np.loadtxt(filename)
            pyln = ply.transform(transmatrix)
            filename=path + 'pose_' +prefix_reconstruction + "-%06d.cloud" % (count_save) + '-' + name_cam+'.ply'
            open3d.io.write_point_cloud(filename, pyln)