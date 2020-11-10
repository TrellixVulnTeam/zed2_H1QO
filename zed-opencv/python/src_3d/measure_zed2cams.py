import os, sys, time
import pyzed.sl as sl
# import numpy as np
# from easydict import EasyDict
# from enum import IntEnum
from zed2cams import init,take
from  multiprocessing import Pool

root_dir =  "dt/output/"
save_dir_fmt = root_dir + "/cam{}/"
def __take_cbk(arg):
    pass

def __looptake(cam_id):
    menu=init(None, cam_id)
    print(f'menu.save_dir: {menu.save_dir}')
    j=0
    while True:
        take(menu)
        print('process :%d, loop:%d is started' % (cam_id, j))
        time.sleep(0.2)
        j = j + 1
class multi_take:
  def __init__(self, interval, pron):
    self.pool = Pool(processes=pron)
    self.interval=interval
  def start(self,work,cam_ids,cbk):
    for i in cam_ids:
        self.pool.apply_async(func=work,
        # self.pool.apply_async(func=self.looptake,
                              args=(i,),
                              callback=cbk)
        time.sleep(0.1)
        print("process: camera-%d is started!"%(i))

  def looptake(self,cam_id):
    self.menu = init(None, cam_id)
    print(f'menu.save_dir: {self.menu.save_dir}')
    j = 0
    while True:
      take(self.menu)
      print('process :%d, loop:%d is started' % (cam_id, j))
      time.sleep(0.2)
      j = j + 1
  def terminate(self):
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
    #   menu=init(None, cam_id)
    #   menus.append([menu,cam.serial_number])
    #   print(f'menu.save_dir: {menu.save_dir}')
    print(f'available devices:{cameras}')
    while True:
      comm = input('Please enter command(t: take data, q:quit: ')
      if not comm in ['t', 'q']:
        continue
      if comm == 't':
        mp.start(__looptake, cam_ids, __take_cbk)
        # mp.start(mp.looptake, cam_ids, __take_cbk)
      elif comm == 'q':
        print('finish script...')
        break
    mp.pool.close()
    mp.pool.terminate()
    sys.exit(1)

if __name__ == "__main__":
  take_data(root_dir)