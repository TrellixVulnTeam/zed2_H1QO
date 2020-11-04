import open3d
import numpy as np
import struct,itertools
from sklearn.linear_model import LinearRegression
from datetime import datetime
from easydict import EasyDict
from add_frame2pcd import pcd_to_ply,add_frame2pcd

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
def move_pcds_by_frame(pcds,frames):
    pcds_n=[]
    def pcd_mv_frame_center(zed_pcd,mv):
        zed_pcd[:, :, :3]=np.add(zed_pcd[:, :, :3],mv)
        return zed_pcd
    pos0 = get_center_point(pcds[0], frames[0])
    for i,(pcd,frame) in enumerate(zip(pcds,frames)):
        # pcd_f = pcd#add_frame2pcd(pcd, frame)
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
        fn = 'C:/00_work/05_src/data/frm_t/' + id + '/pcd.npy'
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