import os, sys, time
import pyzed.sl as sl
from zed2cams import init,take
from  multiprocessing import Pool

root_dir =  "dt/output/"
save_dir_fmt = root_dir + "/cam{}/"
def __looptake_cbd(arg):
    pass
def __looptake(cam_id,r_dir):
    menu=init(None, cam_id)
    save_dir_fmt_t = r_dir + "/cam{cam_id}/"
    menu.save_dir=save_dir_fmt_t
    print(f'menu.save_dir: {menu.save_dir}')
    j=0
    while True:
        take(menu)
        print('process :%d, loop:%d is started' % (cam_id, j))
        time.sleep(0.2)
        j = j + 1

def mp_f(cam_id):
  return cam_id
class multi_take:
  def __init__(self, interval, pron):
    self.pool = Pool(processes=pron)
    self.interval=interval
  def start(self,work,cam_ids,r_dir,cbk):
    for i in cam_ids:
        self.pool.apply_async(func=work,
                              args=(i,r_dir,),
                              callback=cbk)
        time.sleep(0.1)
        print("process: camera-%d is started!"%(i))

  def terminate(self):
    self.pool.close()
    self.pool.terminate()
def take_data():
    global root_dir
    mp = multi_take(0.2, 5)
    root_dir_tmp = input(f'Please set data save directory. default[{root_dir}] :')
    if root_dir_tmp != '':
      root_dir = root_dir_tmp
      if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        print(f"{root_dir} is created.")
    cameras = sl.Camera.get_device_list()
    cam_ids=[]
    for cam_id, cam in enumerate(cameras):
      cam_ids.append(cam_id)
    print(f'available devices:{cameras}')
    while True:
      comm = input('Please enter command(t: take data, q:quit: ')
      if not comm in ['t', 'q']:
        continue
      if comm == 't':
        mp.start(__looptake, cam_ids,root_dir, __looptake_cbd)
      elif comm == 'q':
        print('finish script...')
        break

    mp.terminate()
    sys.exit(1)

if __name__ == "__main__":
  take_data()
