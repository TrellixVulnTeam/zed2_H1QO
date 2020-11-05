import open3d
import numpy as np
import struct,itertools
from sklearn.linear_model import LinearRegression
from datetime import datetime
def fill_nanmean(m):
    m=np.array(m)
    mv = np.nanmean(m,axis=2)
    return mv
def make_pcd(points, colors):
 pcd = open3d.geometry.PointCloud()
 pcd.points = open3d.utility.Vector3dVector(points)
 pcd.colors = open3d.utility.Vector3dVector(colors)
 return pcd
#function name:add_frame2pcd(pcd, frame,n)
#**************************************************************
#pcd: point cloud
#frame:frameの比率での開始と終了位置 【x軸開始,y軸開始,x軸終了,y軸終了】
#      例：[0.2, 0.2, 0.5, 0.5]の場合、frameの比率[0.2, 0.2, 0.5, 0.5]で枠を作成
#      例：image:400x600  開始位置【400*0.2,600*0.2,400*0.5,600*0.5】
#      開始位置(80,120) 終了位置(200,300)
#n:線のポイント数
def add_frame2ply(pcd, frame,n):
  red=np.array([1, 0, 0]).astype(np.float64)
  wr0,hr0,wr1,hr1=frame
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
  def predict_reg(lr,x,y):
    dep=lr.predict([[x,y]])[0]
    return dep
  points=list(points)
  colors=list(colors)
  # ３次元上の平面に対する法線ベクトルを取得
  train_points=list(filter(lambda x: x[0]> h0 and x[0]< h1 and x[1]>w0 and x[1]<w1 , points))
  lr=train_dep(train_points)
  # 枠線を作成
  for i in range(ph0,ph1):
      dep=predict_reg(lr,i/n,w0)
      points.append([i/n,w0,dep])
      colors.append(red)
      dep=predict_reg(lr,i/n,w1)
      points.append([i/n,w1,dep])
      colors.append(red)
  for i in range(pw0,pw1):
      dep=predict_reg(lr,h0,i/n)
      points.append([h0,i/n,dep])
      colors.append(red)
      dep=predict_reg(lr,h1,i/n)
      points.append([h1,i/n,dep])
      colors.append(red)
  pcd = make_pcd(np.array(points), colors)
  return pcd
def zed_depthfloat_to_abgr(f):
  """
    ZED pcd data format:
      ----------------------------------------------------------
      https://www.stereolabs.com/docs/depth-sensing/using-depth/
      ----------------------------------------------------------
      The point cloud stores its data on 4 channels using 32-bit
      float for each channel.
      The last float is used to store color information, where
      R, G, B, and alpha channels (4 x 8-bit) are concatenated
      into a single 32-bit float.
  """
  # https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex/38879403
  h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
  return [eval('0x'+a+b) for a, b in zip(h[::2], h[1::2]) if a+b != '0x']
def get_info_zed2pcd(zed_pcd,frame):
  x0,y0,x1,y1=frame
  zed_points = zed_pcd[:,:,:3]
  zed_colors = zed_pcd[:,:,3]
  point_frame=zed_points[x0:x1,y0:y1,:]
  points, colors = [], []
  for x, y in itertools.product(
    [a for a in range(zed_colors.shape[0])],
    [a for a in range(zed_colors.shape[1])]):
    if np.isinf(zed_points[x, y]).any() or  np.isnan(zed_points[x, y]).any():
      continue
    # color
    tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
    tmp = [tmp[3], tmp[2], tmp[1]] # ABGR to RGB
    tmp = np.array(tmp).astype(np.float64) / 255. # for ply (color is double)
    colors.append(tmp)
    # point
    tmp = np.array(zed_points[x, y]).astype(np.float64)
    points.append(tmp)
  return points,colors,point_frame
def add_frame2pcd2(zed2pcd, frame):
  red=np.array([1, 0, 0]).astype(np.float64)
  points, colors, point_frame=get_info_zed2pcd(zed2pcd,frame)

  #ポイントの開始と終了位置を計算
  point_frame =point_frame.reshape((-1, 3))
  h0,h1=np.min(point_frame[:,0]),np.max(point_frame[:,0])
  w0,w1=np.min(point_frame[:,1]),np.max(point_frame[:,1])
  n=100
  # x 軸　y 軸　ポイント数を計算
  ph0,ph1,pw0,pw1=int(h0*n),int(h1*n),int(w0*n),int(w1*n)
  def train_dep(train_points):
      train_points=np.array(train_points)
      lr = LinearRegression()
      train_x = train_points[:, :2]
      train_y = train_points[:, 2]
      lr.fit(train_x, train_y)
      return lr
  def predict_reg(lr,x,y):
    dep=lr.predict([[x,y]])[0]
    return dep
  points=list(points)
  colors=list(colors)
  # ３次元上の平面に対する法線ベクトルを取得
  train_points=list(filter(lambda x: x[0]> h0 and x[0]< h1 and x[1]>w0 and x[1]<w1 , points))
  lr=train_dep(train_points)
  # 枠線を作成
  def add_point(lr,x,y,points,colors):
      dep=predict_reg(lr,x,y)
      points.append([x,y,dep])
      colors.append(red)
      return points,colors
  for i, j in itertools.product(
    [a for a in range(ph0,ph1)],
    [a for a in range(pw0,pw1)]):
      points, colors=add_point(lr,i/n,w0,points,colors)
      points, colors=add_point(lr,i/n,w1,points,colors)
      points, colors=add_point(lr,h0,j/n,points,colors)
      points, colors=add_point(lr,h1,j/n,points,colors)
  pcd = make_pcd(np.array(points), colors)
  return pcd

def zed_abgr_to_depthfloat(abgr):
    # abgr='ff0000ff' #[255, 0, 0, 255]
    #https://stackoverflow.com/questions/1592158/convert-hex-to-float
    return struct.unpack('!f', bytes.fromhex(abgr))[0]
def add_frame2pcd(zed2pcd, frame):
  abgr = 'ff0000ff'
  abgr_f=zed_abgr_to_depthfloat(abgr)
  x0, y0, x1, y1 = frame
  def train_dep(train_points):
      train_points=np.array(train_points)
      lr = LinearRegression()
      train_x = train_points[:, :2]
      train_y = train_points[:, 2]
      lr.fit(train_x, train_y)
      return lr
  def predict_reg(lr,pointxy):
    pre=lr.predict(pointxy)
    pre=np.expand_dims(pre,axis=1)
    return pre

  # ３次元上の平面に対する法線ベクトルを取得
  print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 'add_frame2pcd train lst')
  train_dt_x,train_dt_y,train_dt_z=[],[],[]

  for i, j in itertools.product(
    [a for a in range(x0,x1)],
    [a for a in range(y0,y1)]):
      train_dt_x.append([i,j,zed2pcd[i,j,0]])
      train_dt_y.append([i,j,zed2pcd[i,j,1]])
      train_dt_z.append([i,j,zed2pcd[i,j,2]])
  print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 'add_frame2pcd train exc')
  lrx=train_dep(train_dt_x)
  lry=train_dep(train_dt_y)
  lrz=train_dep(train_dt_z)
  # 枠線を作成

  # 赤色を設定
  print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 'add_frame2pcd color set')
  zed2pcd[x0:x1, y0, 3] = abgr_f
  zed2pcd[x0:x1, y1, 3] = abgr_f
  zed2pcd[x0, y0:y1, 3] = abgr_f
  zed2pcd[x1, y0:y1, 3] = abgr_f

  #枠リスト
  print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 'add_frame2pcd xyz set')
  x01y0=[[x,y0] for x in range(x0,x1) ]
  x01y1=[[x,y1] for x in range(x0,x1) ]
  x0y01=[[x0,x] for x in range(y0,y1) ]
  x1y01=[[x1,x] for x in range(y0,y1) ]

  #xyz再計算
  zed2pcd[x0:x1, y0, 0:3] = np.concatenate([predict_reg(lrx, x01y0),predict_reg(lry, x01y0),predict_reg(lrz, x01y0)],axis=1)
  zed2pcd[x0:x1, y1, 0:3] = np.concatenate([predict_reg(lrx, x01y1),predict_reg(lry, x01y1),predict_reg(lrz, x01y1)],axis=1)
  zed2pcd[x0, y0:y1, 0:3] = np.concatenate([predict_reg(lrx, x0y01),predict_reg(lry, x0y01),predict_reg(lrz, x0y01)],axis=1)
  zed2pcd[x1, y0:y1, 0:3] = np.concatenate([predict_reg(lrx, x1y01),predict_reg(lry, x1y01),predict_reg(lrz, x1y01)],axis=1)

  return zed2pcd
def pcd_to_ply(zed_pcd):
    zed_points = zed_pcd[:, :, :3]
    zed_colors = zed_pcd[:, :, 3]
    points, colors = [], []
    for x, y in itertools.product(
            [a for a in range(zed_colors.shape[0])],
            [a for a in range(zed_colors.shape[1])]):
        if np.isinf(zed_points[x, y]).any() or np.isnan(zed_points[x, y]).any():
            continue
        # color
        tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
        tmp = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
        tmp = np.array(tmp).astype(np.float64) / 255.  # for ply (color is double)
        colors.append(tmp)
        # point
        tmp = np.array(zed_points[x, y]).astype(np.float64)
        points.append(tmp)
    ply = make_pcd(np.array(points), colors)
    return ply
if __name__ == "__main__":
    fn='C:/00_work/05_src/data/20201015155844/pcd.npy'
    pcd = np.load(fn)
    frame = [100, 200, 1000, 1000]
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'add_frame2pcd start')
    zed_pcd = add_frame2pcd(pcd, frame)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'add_frame2pcd end')
    pyln=pcd_to_ply(zed_pcd)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'pcd_to_ply end')
    open3d.io.write_point_cloud('frame2_4.ply', pyln)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'write_point_cloud end')