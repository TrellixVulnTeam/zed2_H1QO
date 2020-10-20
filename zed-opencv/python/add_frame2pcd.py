import open3d
import numpy as np
import struct,itertools
from sklearn.linear_model import LinearRegression
from datetime import datetime
from easydict import EasyDict
def make_pcd(points, colors):
 pcd = open3d.geometry.PointCloud()
 pcd.points = open3d.utility.Vector3dVector(points)
 pcd.colors = open3d.utility.Vector3dVector(colors)
 return pcd

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
  train_dt_x,train_dt_y,train_dt_z=[],[],[]
  for i, j in itertools.product(
    [a for a in range(x0,x1)],
    [a for a in range(y0,y1)]):
      train_dt_x.append([i,j,zed2pcd[i,j,0]])
      train_dt_y.append([i,j,zed2pcd[i,j,1]])
      train_dt_z.append([i,j,zed2pcd[i,j,2]])
  lrx=train_dep(train_dt_x)
  lry=train_dep(train_dt_y)
  lrz=train_dep(train_dt_z)

  # 枠線を作成
  # 赤色を設定
  zed2pcd[x0:x1, y0, 3] = abgr_f
  zed2pcd[x0:x1, y1, 3] = abgr_f
  zed2pcd[x0, y0:y1, 3] = abgr_f
  zed2pcd[x1, y0:y1, 3] = abgr_f

  #枠リスト
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
def get_center_point(zed2pcd,frame):
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
  train_dt_x,train_dt_y,train_dt_z=[],[],[]
  for i, j in itertools.product(
    [a for a in range(x0,x1)],
    [a for a in range(y0,y1)]):
      train_dt_x.append([i,j,zed2pcd[i,j,0]])
      train_dt_y.append([i,j,zed2pcd[i,j,1]])
      train_dt_z.append([i,j,zed2pcd[i,j,2]])
  lrx=train_dep(train_dt_x)
  lry=train_dep(train_dt_y)
  lrz=train_dep(train_dt_z)
  def get_center(lr,train_points):
      train_points=np.array(train_points)
      train_x = train_points[:, :2]
      xlst=predict_reg(lr, train_x)
      return np.mean(np.array(xlst))
  #枠リスト
  x=get_center(lrx,train_dt_x)
  y=get_center(lry,train_dt_y)
  z=get_center(lrz,train_dt_z)
  return x,y,z
def _test_add_frame_main(fn,frame,fnout):
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'s add_frame_main')
    pcd = np.load(fn)
    zed_pcd = add_frame2pcd(pcd, frame)
    pyln=pcd_to_ply(zed_pcd)
    open3d.io.write_point_cloud(fnout, pyln)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'e add_frame_main')
def move_pcds_by_frame(pcds,frames):
    pcds_n=[]
    def pcd_mv_frame_center(zed_pcd,mv):
        zed_pcd[:, :, :3]=np.add(zed_pcd[:, :, :3],mv)
        return zed_pcd
    pos0 = get_center_point(pcds[0], frames[0])
    for i,(pcd,frame) in enumerate(zip(pcds,frames)):
        pcd_f = add_frame2pcd(pcd, frame)
        if i==0:
            pcds_n.append(pcd_f)
            continue
        posi = get_center_point(pcd, frame)
        pos_mv = np.array(pos0) - np.array(posi)
        pcd_mv=pcd_mv_frame_center(pcd_f,pos_mv)
        pcds_n.append(pcd_mv)
    return pcds_n
if __name__ == "__main__":
    ids=['20201015155844','20201015155835']
    frame = EasyDict({})
    #image.jpg中のオブジェクト（新聞用紙の枠を選定）
    frame['20201015155844'] = [737, 1012, 979, 1500]  #20201015155844 [1012, 737, 1500, 979]
    frame['20201015155835'] = [945, 963, 1087, 1564]  #20201015155835 [963, 945, 1564, 1087]
    pcds, frames = [], []
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'s load pcd')
    for id in ids:
        fn = 'C:/00_work/05_src/data/' + id + '/pcd.npy'
        pcd = np.load(fn)
        pcds.append(pcd)
        frames.append(frame[id])
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'e load pcd')
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'s move_pcds_by_frame')
    pcds_mv=move_pcds_by_frame(pcds, frames)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'e move_pcds_by_frame')
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'s pcd_to_ply')
    for pcd_mv,id in zip(pcds_mv,ids):
        pyln=pcd_to_ply(pcd_mv)
        fn='frame_'+id+'_mv.ply'
        open3d.io.write_point_cloud(fn, pyln)
    print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'e pcd_to_ply')
    # fn = 'C:/00_work/05_src/data/' + ids[1] + '/pcd.npy'
    # fnout = 'frame_'+ids[1]+'.ply'
    # _test_add_frame_main(fn, frame, fnout)