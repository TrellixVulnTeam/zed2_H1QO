import open3d

def transform_ply(ply,tx,ty,tz):
    transform = [
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ]
    pyln = ply.transform(transform)
    return pyln
if __name__ == "__main__":
    tx, ty, tz=0.2,0,0
    transform = [
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ]
    ply = open3d.io.read_point_cloud(filename='reconstruction-000000.pcd-ZED_21888201.ply')
    pyln = ply.transform(transform)
    open3d.io.write_point_cloud('reconstruction-000000.pcd-ZED_21888201_test.ply', pyln)