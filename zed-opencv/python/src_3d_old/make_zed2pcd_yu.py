from PIL import Image
from easydict import EasyDict
import numpy as np
import  json, open3d
import itertools
dtpath='C:/00_work/05_src/zed2/zed-opencv/python/'


def make_pcd(points, colors):
 pcd = open3d.geometry.PointCloud()
 pcd.points = open3d.utility.Vector3dVector(points)
 pcd.colors = open3d.utility.Vector3dVector(colors)
 return pcd
'''
if (GetRValue(vColor) >= 200) and
         (GetRValue(vColor) <= 255) and
         (GetGValue(vColor) >= 0) and
         (GetGValue(vColor) <= 50) and
         (GetBValue(vColor) >= 0) and
         (GetBValue(vColor) <= 50) then
'''
def get_red_pixel(pix):
 ret=False
 if pix[0]>=200 and pix[0]<=255 and \
  pix[1] >= 200 and pix[1] <= 255 and \
  pix[2] >= 200 and pix[2] <= 255 :
  ret=True
 return ret
def get_center_point(pts):
  ptsx,ptsy,ptsz=pts[:,0],pts[:,1],pts[:,2]
  def get_center(ptsk):
   m=np.mean(ptsk)
   # lk=ptsk[]

def add_frame2pcd1(pcd, window_frame):
  red=np.array([1, 0, 0]).astype(np.float64)
  points =np.array(pcd.points)
  dep=np.mean(points[:,2])
  colors=np.array(pcd.colors)
  h0,w0,h1,w1 = window_frame
  for i in range(h0,h1):
      ind=i*w0
      points[ind] = np.append(points[ind][:2], dep)
      colors[ind] = red
      ind=i*w1
      points[ind] = np.append(points[ind][:2], dep)
  for i in range(w0,w1):
      ind=i*h0
      points[ind] = np.append(points[ind][:2], dep)
      colors[ind] = red
      ind=i*h1
      points[ind] = np.append(points[ind][:2], dep)
  pcd = make_pcd(np.array(points), colors)
  return pcd

def add_frame2pcd(pcd, rate):
  red=np.array([1, 0, 0]).astype(np.float64)
  points =np.array(pcd.points)
  # dep=np.mean(points[:,2])
  dep=np.min(points[:,2])
  colors=np.array(pcd.colors)
  x0,x1=np.min(points[:,0]),np.max(points[:,0])
  y0,y1=np.min(points[:,1]),np.max(points[:,1])
  h0,w0,h1,w1 =np.divide(np.array([x0,y0,x1,y1]),rate)
  n=1000
  hi0=int(h0*n)
  hi1=int(h1*n)
  wi0=int(w0*n)
  wi1=int(w1*n)
  points=list(points)
  colors=list(colors)
  for i in range(hi0,hi1):
      points.append([i/n,w0,dep])
      colors.append(red)
      points.append([i/n,w1,dep])
      colors.append(red)
  for i in range(wi0,wi1):
      points.append([h0,i/n,dep])
      colors.append(red)
      points.append([h1,i/n,dep])
      colors.append(red)
  pcd = make_pcd(np.array(points), colors)
  return pcd
def transform_ply(ply, t):
 tx, ty, tz=t
 transform = [
  [1, 0, 0, tx],
  [0, 1, 0, ty],
  [0, 0, 1, tz],
  [0, 0, 0, 1],
 ]
 pyln = ply.transform(transform)
 return pyln
def transform_ply2(ply, t):
 pyln = ply.transform(t)
 return pyln
def calcurate_xyz(color, depth, intr):
 ws, hs = ([a for a in range(depth.shape[n])] for n in [0, 1])
 red_points,points, colors = [], [],[]
 for u, v in itertools.product(ws, hs):
  z = float(depth[u, v])
  if np.isnan(z) or np.isinf(z):
   continue
  x = float((u - intr.cx) * z / intr.fx)
  y = float((v - intr.cy) * z / intr.fy)
  point = np.asarray([x, y, z])
  points.append(point)
  rgb = (color[u, v][:3] / 255).astype(np.float64)
  redp=get_red_pixel(color[u, v])
  if redp:
   red_points.append(point)
  colors.append(rgb)
  # colors.append((color[u, v][:3]).astype(np.uint8))

 window_frame=[100,100,200,200]
 pcd = make_pcd(np.array(points), colors)
 return pcd  # if not verbose else pcd, points, colors
# VGAでの撮影。

camlist=['ZED_22378008','ZED_21888201']
img_cn=3
for i in range(img_cn):
 for cam in camlist:
  files = [
   "data_1013_1/reconstruction-%06d.color-%s.jpg"%(i,cam),
   "data_1013_1/reconstruction-%06d.depth-%s.png"%(i,cam),
   "data_1013_1/reconstruction-%06d.tow-%s.csv"%(i,cam),
   "data_1013_1/%s-camera-intrinsics.txt"%(cam),
   "data_1013_1/box.jpg",
   "data_1013_1/reconstruction-%06d.pose-%s.txt"%(i,cam),
  ]
  # boxcolor = np.array(Image.open(dtpath+files[4]))
  color = np.array(Image.open(dtpath+files[0]))
  depth = np.array(Image.open(dtpath+files[1]))
  window_frame=[100,100,300,300]
  # color, depth, depth_med=add_frame2img(color, depth, window_frame)
  k=np.loadtxt(dtpath+files[3])
  # wk=np.loadtxt(dtpath+files[5])
  wk=np.loadtxt(dtpath+files[2])
  intr=EasyDict({})
  intr.fx=k[0,0]
  intr.fy=k[1,1]
  intr.cx=k[0,2]
  intr.cy=k[1,2]
  pcd = calcurate_xyz(color, depth, intr)
  open3d.io.write_point_cloud('reconstruction-%06d.pcd-%s.ply'%(i,cam), pcd)
  # pcdn=transform_ply(pcd, wk[:3])
  # pcdn=transform_ply2(pcd, wk)
  pcdn=add_frame2pcd(pcd, 4,100)
  open3d.io.write_point_cloud('tran2_reconstruction-%06d.pcd-%s.ply'%(i,cam), pcdn)
