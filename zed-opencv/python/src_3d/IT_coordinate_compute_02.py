import numpy  as np
#カメラセット（左）5.30 			120			10			1.50
#カメラセット（右）6.30 			240			10			2.00

def coordinate_xyz(r,y,angy):
    a=3*np.pi/2-angy
    OS=y/np.sin(a)
    x=-OS*np.cos(a)
    z=-np.sqrt(r**2-x**2-y**2)
    return [x,y,z]

# r,z,angy=5.3,1.5,(2*np.pi)/3.0 #120/180
# cam_left=coordinate_xyz(r,z,angy)

r,y,angy=6.3,2.0,(4*np.pi)/3.0 #240
cam_right=coordinate_xyz(r,y,angy)

# print(cam_left)
print(cam_right)

#cam_left [4.402272140611028, -2.541653005427766, 1.5]
#cam_right [-5.17373172864616, -2.987055406248771, 2.0]