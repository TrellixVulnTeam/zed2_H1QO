import open3d
import numpy as np
from sklearn.linear_model import LinearRegression
def make_pcd(points, colors):
 pcd = open3d.geometry.PointCloud()
 pcd.points = open3d.utility.Vector3dVector(points)
 pcd.colors = open3d.utility.Vector3dVector(colors)
 return pcd
#function name:add_frame2pcd(pcd, rate,n)
#**************************************************************
#pcd: point cloud
#rate:frameの比率での開始と終了位置 【x軸開始,y軸開始,x軸終了,y軸終了】
#      例：[0.2, 0.2, 0.5, 0.5]の場合、frameの比率[0.2, 0.2, 0.5, 0.5]で枠を作成
#      例：image:400x600  開始位置【400*0.2,600*0.2,400*0.5,600*0.5】
#      開始位置(80,120) 終了位置(200,300)
#n:線のポイント数
def add_frame2pcd(pcd, rate,n):
  red=np.array([1, 0, 0]).astype(np.float64)
  wr0,hr0,wr1,hr1=rate
  points =np.array(pcd.points)
  colors=np.array(pcd.colors)
  x0,x1=np.min(points[:,0]),np.max(points[:,0])
  y0,y1=np.min(points[:,1]),np.max(points[:,1])

  #ポイントの開始と終了位置を計算
  h0=x0+(x1-x0)*hr0
  w0=y0+(y1-y0)*wr0
  h1=x0+(x1-x0)*hr1
  w1=y0+(y1-y0)*wr1

  # x 軸　y 軸　ポイント数を計算
  ph0,ph1,pw0,pw1=int(h0*n),int(h1*n),int(w0*n),int(w1*n)
  def train_dep(train_points):
      train_points=np.array(train_points)
      lr = LinearRegression()
      train_x = train_points[:, :2]
      train_y = train_points[:, 2]
      lr.fit(train_x, train_y)
      return lr
  def get_dep(lr,x,y):
    dep=lr.predict([[x,y]])[0]
    return dep
  points=list(points)
  colors=list(colors)
  # ３次元上の平面に対する法線ベクトルを取得
  train_points=list(filter(lambda x: x[0]> h0 and x[0]< h1 and x[1]>w0 and x[1]<w1 , points))
  lr=train_dep(train_points)
  # 枠線を作成
  for i in range(ph0,ph1):
      dep=get_dep(lr,i/n,w0)
      points.append([i/n,w0,dep])
      colors.append(red)
      dep=get_dep(lr,i/n,w1)
      points.append([i/n,w1,dep])
      colors.append(red)
  for i in range(pw0,pw1):
      dep=get_dep(lr,h0,i/n)
      points.append([h0,i/n,dep])
      colors.append(red)
      dep=get_dep(lr,h1,i/n)
      points.append([h1,i/n,dep])
      colors.append(red)
  pcd = make_pcd(np.array(points), colors)
  return pcd
if __name__ == "__main__":
    ply = open3d.io.read_point_cloud(filename='reconstruction-000000.pcd-ZED_21888201.ply')
    rate = [0.3, 0.3, 0.8, 0.5]
    pyln = add_frame2pcd(ply, rate,100)
    open3d.io.write_point_cloud('reconstruction-000000.pcd-ZED_21888201_frame.ply', pyln)