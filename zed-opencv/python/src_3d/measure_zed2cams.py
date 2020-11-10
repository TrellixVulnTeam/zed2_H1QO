import os, sys, time
import pyzed.sl as sl
import numpy as np
# from easydict import EasyDict
# from enum import IntEnum
from zed2cams import init,take
from  multiprocessing import Pool
import cv2

root_dir =  "dt/output/"
save_dir_fmt = root_dir + "/cam{}/"
# def __looptake_cbd(arg):
#     pass
#
# def __looptake2(cam_id):
#     menu=init(None, cam_id)
#     print(f'menu.save_dir: {menu.save_dir}')
#     j=0
#     while True:
#         take(menu)
#         print('process :%d, loop:%d is started' % (cam_id, j))
#         time.sleep(0.2)
#         j = j + 1

def mp_f(cam_id):
  return cam_id
class multi_take:
  def __init__(self, interval, pron):
    self.pool = Pool(processes=pron)
    self.interval=interval
    self.menus=[]
    self.info="take started"
  def start(self,work,cam_ids,cbk):
    for i in cam_ids:
        self.pool.apply_async(func=work,
                              args=(i,),
                              callback=cbk)
        time.sleep(0.1)
        print("process: camera-%d is started!"%(i))

  def looptake_cbd(self,cam_id):
    menu=init(None, cam_id)
    self.menus.append(menu)
    print(f'menu.save_dir: {menu.save_dir}')
    j=0
    while True:
        take(menu)
        # print('process :%d, loop:%d is started' % (cam_id, j))
        self.info = 'process :%d, loop:%d is started' % (cam_id, j)
        time.sleep(self.interval)
        j = j + 1
  def terminate(self):
    for menu in self.menus:
      menu.zed.cam.close()
    self.pool.close()
    self.pool.terminate()
def take_data(root_dir):
    mp = multi_take(0.2, 5)
    root_dir_tmp = input(f'Please set data save directory. default[{root_dir}] :')
    if root_dir_tmp != '':
      root_dir = root_dir_tmp
      if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        # save_dir_fmt = root_dir + "/cam{}/"
        print(f"{root_dir} is created.")

    cameras = sl.Camera.get_device_list()
    cam_ids=[]
    for cam_id, cam in enumerate(cameras):
      cam_ids.append(cam_id)
    print(f'available devices:{cameras}')
    # while True:
    #   comm = input('Please enter command(t: take data, q:quit: ')
    #   if not comm in ['t', 'q']:
    #     continue
    #   if comm == 't':
    #     mp.start(mp_f, cam_ids, mp.looptake_cbd)
    #   elif comm == 'q':
    #     print('finish script...')
    #     break

    print('Please enter command(t: take data, q:quit: ')
    im = np.zeros((100, 300), np.uint8)
    cv2.imshow('Keypressed', im)
    while(1):
        k = cv2.waitKey(0)
        im_c = im.copy()
        cv2.putText(
            im_c,
            f'{chr(k)} -> {k}\n%s'%(mp.info),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2)
        cv2.imshow('Keypressed', im_c)
        if k == 27:  # Esc key to stop
            print('finish script...')
            break
        elif k == 116 or  k == 84:#t
            mp.start(mp_f, cam_ids, mp.looptake_cbd)
        else:
            continue
    mp.terminate()
    sys.exit(1)

if __name__ == "__main__":
  take_data(root_dir)
