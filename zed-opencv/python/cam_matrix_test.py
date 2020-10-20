'''
I'm trying to get the camera matrix P for the left frame (I would like to project 3-D
points from fused point cloud to the left camera frame). This should be given by P = K[R | t],
being K the camera intrinsic matrix,
R the camera rotation and t = -RC where C is the camera center in the world coordinates.
K = np.array([[cam_params.left_cam.fx,                      0, cam_params.left_cam.cx],
              [                     0, cam_params.left_cam.fy, cam_params.left_cam.cy],
              [                     0,                      0,                      1]])
R = pose.get_rotation_matrix(sl.Rotation()).r.T
t = pose.get_translation(sl.Translation()).get()
world2cam = np.hstack((R, np.dot(-R, t).reshape(3,-1)))
P = np.dot(K, world2cam)
'''
import numpy as np
K = np.array([[200,    0,  245],
              [0,      200,128],
              [0,      0,  1]])
R = np.array([[200,    0,  245],
              [0,      200,128],
              [0,      0,  1]])
t=np.array([1,2,3])
world2cam=np.hstack((R, np.dot(-R, t).reshape(3,-1)))
P = np.dot(K, world2cam)
transform=np.vstack((P, [0,0,0,1]))
kkk=0
