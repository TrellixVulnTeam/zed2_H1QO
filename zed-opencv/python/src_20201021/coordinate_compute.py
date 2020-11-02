import numpy  as np
#カメラセット（左）5.30 			120			10			1.50
#カメラセット（右）6.30 			240			10			2.00
r,z,angy=5.3,1.5,(2*np.pi)/3.0 #120
def coordinate_xyz(r,z,angy):
    y=r*np.cos(angy)
    ang_z0=np.arcsin(z/r)
    z0=r*np.cos(ang_z0)
    ang_x=np.arcsin(z/z0)
    x=z0*np.cos(ang_x)
    return [x,y,z]
cam_left=coordinate_xyz(r,z,angy)

r,z,angy=6.3,2.0,(4*np.pi)/3.0 #240
cam_right=coordinate_xyz(r,z,angy)

print(cam_left)
print(cam_right)
#cam_left [4.856953777832357, -2.6499999999999986, 1.5]
#cam_right [5.629387178015028, -3.1500000000000026, 2.0]