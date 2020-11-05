import numpy  as np
#カメラセット（左）5.30 			120			10			1.50
#カメラセット（右）6.30 			240			10			2.00

def coordinate_xyz(r,z,angy):
    AngQOP=np.arcsin(z/r)
    QO=r*np.cos(AngQOP) #ポイントからXY平面の投影から原点の距離
    AngXY=np.pi-angy
    x=np.abs(QO*np.cos(AngXY))
    y=np.abs(QO*np.sin(AngXY))
    return [x,y,z]

r,z,angy=5.3,1.5,(2*np.pi)/3.0 #120/180
cam_left=coordinate_xyz(r,z,angy)

r,z,angy=6.3,2.0,(4*np.pi)/3.0 #240
cam_right=coordinate_xyz(r,z,angy)

print(cam_left)
print(cam_right)

#cam_left [4.402272140611028, -2.541653005427766, 1.5]
#cam_right [-5.17373172864616, -2.987055406248771, 2.0]