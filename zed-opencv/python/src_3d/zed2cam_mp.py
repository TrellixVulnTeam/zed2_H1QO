import os, sys, time, json, atexit, functools, datetime, platform
import pyzed.sl as sl
import numpy as np
from easydict import EasyDict
from enum import IntEnum
import multiprocessing

# reference URL : <https://stackoverflow.com/questions/60569791/zed-camera-api-example-is-causing-mysterious-segfault-mutex-lock-fault>

# https://learn.adafruit.com/adafruit-sensorlab-magnetometer-calibration/magnetic-calibration-with-jupyter

verbose = True
root_dir =  "dt/output/"
save_dir_fmt = root_dir + "/cam{}/"
save_dir = None

class TakeMode(IntEnum):
  ALL = 0
  RGB = 1
  RGBD = 2
  PCD = 3

class TakeSensorMode(IntEnum):
  ALL = 0
  ONLY_POSE = 1
  RAW_SENSOR = 2

cam_reso = EasyDict({
  "size2name":{
    "2208x1242" : "HD2K",
    "1920x1080" : "HD1080",
    "1280x720"  : "HD720",
    "672x376"  : "VGA"},
  "name2size":{
    "HD2K"   : "2208x1242",
    "HD1080" : "1920x1080",
    "HD720"  : "1280x720",
    "VGA"    : "672x376"
  },
  "HD2K": { "fx": 1048.5426025390625, "fy": 1048.5426025390625, "cx": 1041.6142578125, "cy": 644.5799560546875 },
  "HD1080": { "fx": 1050.95166015625, "fy": 1050.95166015625, "cx": 902.114501953125, "cy": 562.9620971679688 },
  "HD720": {"fx": 523.443359375, "fy": 523.443359375, "cx": 596.2228393554688, "cy": 371.38214111328125 },
  "VGA": { "fx": 262.5944519042969, "fy": 262.5944519042969, "cx": 304.97894287109375, "cy": 194.1575469970703 }
})

mode = EasyDict({})
mode.sensing = [sl.SENSING_MODE(a) for a in range(len(sl.SENSING_MODE.__members__))]
mode.depth = [sl.DEPTH_MODE(a) for a in range(len(sl.DEPTH_MODE.__members__))]
mode.unit = [sl.UNIT(a) for a in range(len(sl.UNIT.__members__))]
mode.resolution = [sl.RESOLUTION(a) for a in range(len(sl.RESOLUTION.__members__))]
mode.coordinate_system = [sl.COORDINATE_SYSTEM(a) for a in range(len(sl.COORDINATE_SYSTEM.__members__))]

def all_done():
  global menu
  if menu.zed.cam:
    if menu.zed.cam.is_opened():
       menu.zed.cam.close()
  del menu
  print('proc done.')
# atexit.register(all_done)

def stop_watch(func) :
  @functools.wraps(func)
  def wrapper(*args, **kargs) :
    start = time.time()
    result = func(*args,**kargs)
    if verbose:
      print(f"{func.__name__}() elaplsed: {time.time() - start:.03f} [sec]")
    return result
  return wrapper

def init_menu():
  menu = EasyDict({})
  menu.init = False
  menu.save_dir = save_dir
  menu.take = {}
  menu = switch_take_mode(menu, TakeMode.ALL)
  menu.mode = {}
  menu.mode.sensing = 1 # default: FILL  (mode.sensing's index)
  menu.mode.depth =  3  # default: ULTRA (mode.depth's index)
  menu.unit = 2         # default: METER (mode.unit's index)
  #menu.coordinate_system = 0 # default: IMAGE
  menu.coordinate_system = 5 # default: RIGHT-HANDED Z-UP X Forward
  menu.depth_range = {}
  menu.depth_range.max = 20.
  menu.depth_range.min = 0.4
  menu.cam = {}
  menu.cam.resolution =  0 # default: HD2K
  r = str(mode.resolution[menu.cam.resolution]).split('.')[1]
  [menu.cam.width, menu.cam.height] = cam_reso.name2size[r].split("x")
  #os.makedirs(menu.save_dir, exist_ok=True)
  menu.init = True
  return menu

def switch_take_mode(menu, mode):
  assert str(mode).split(".")[1] in TakeMode.__members__.keys()
  m = menu.take
  m.color = True if mode in [TakeMode.ALL, TakeMode.RGB, TakeMode.RGBD] else False
  m.depth = True if mode in [TakeMode.ALL, TakeMode.RGBD] else False
  m.point_cloud = True if mode in [TakeMode.ALL, TakeMode.PCD] else False
  return menu

def init_params(menu,cam_id=0):
  global save_dir
  zed = EasyDict({})
  zed.cam = sl.Camera()
  zed.mat = EasyDict({
    'pose' : sl.Pose(),
    'translation': sl.Translation(),
    'transform' : sl.Transform(),
    'image' : sl.Mat(), # image_map
    'depth' : sl.Mat(), # depth_map
    'point_cloud' : sl.Mat(),
    'sensors' : sl.SensorsData() # sensors_data
  })
  
  zed.param = EasyDict({
    'init' : sl.InitParameters(
              camera_resolution = mode.resolution[menu.cam.resolution],
              depth_mode = mode.depth[menu.mode.depth],
              coordinate_units = mode.unit[menu.unit],
              coordinate_system=mode.coordinate_system[menu.coordinate_system],
              depth_minimum_distance = menu.depth_range.min,
              depth_maximum_distance = menu.depth_range.max,
              sdk_verbose = verbose),
    'runtime' : sl.RuntimeParameters(
              sensing_mode = mode.sensing[menu.mode.sensing]),
    'tracking' : sl.PositionalTrackingParameters(zed.mat.transform)
  })
  
  #######
  zed.param.init.set_from_camera_id(cam_id)
  save_dir = save_dir_fmt.format(cam_id)
  return zed

# menu = init_menu()

def open_cam(menu):
  zed = menu.zed
  if not zed.cam.is_opened():
    status = zed.cam.open(zed.param.init)
    zed.cam.enable_positional_tracking(zed.param.tracking)
    if status != sl.ERROR_CODE.SUCCESS :
      print(repr(status))
      zed.cam.close()
      exit(1)
  menu.zed = zed
  return menu

def reset_cam(menu, cam_id):
  if 'zed' in menu.keys():
    if 'cam' in menu.zed.keys():
      menu.zed.cam.close()
  menu = init_menu()
  menu.save_dir = save_dir_fmt.format(cam_id).replace('//','/')
  menu.zed = init_params(menu,cam_id)
  menu = open_cam(menu)
  return menu

@stop_watch
def init(menu, cam_id=0):
  menu = init_menu()
  menu.save_dir = save_dir_fmt.format(cam_id).replace('//','/')
  menu.zed = init_params(menu,cam_id)
  menu = open_cam(menu)
  return menu

@stop_watch
def get_pose_transform_matrix(menu):
  if menu.zed.cam.grab(menu.zed.param.runtime) ==  sl.ERROR_CODE.SUCCESS:
    while menu.zed.cam.get_position(menu.zed.mat.pose) != \
      sl.POSITIONAL_TRACKING_STATE.OK:
      stat = menu.zed.cam.grab(menu.zed.param.runtime)
      if menu.zed.cam.get_position(menu.zed.mat.pose) == sl.POSITIONAL_TRACKING_STATE.OK:
        break
      if time.time()-st > 1.:
        break
  if menu.zed.cam.get_position(menu.zed.mat.pose) == \
      sl.POSITIONAL_TRACKING_STATE.OK:
    transform = menu.zed.mat.pose.pose_data(menu.zed.mat.transform)
    rotation = menu.zed.mat.pose.get_rotation_vector()
    translation = menu.zed.mat.pose.get_translation(menu.zed.mat.translation).get()
  return EasyDict({'transform':transform.m, 'rot':rotation, 'translation':translation})

@stop_watch
def take(menu):
  st = time.time()
  zed = menu.zed
  mat = zed.mat
  # TODO: raw_sensor値を取得する実装
  #   MAG calibration: https://learn.adafruit.com/adafruit-sensorlab-magnetometer-calibration/magnetic-calibration-with-motioncal
  #   
  if zed.cam.grab(zed.param.runtime) ==  sl.ERROR_CODE.SUCCESS:
    status = [
      zed.cam.get_sensors_data(mat.sensors, sl.TIME_REFERENCE.CURRENT),
      # only left cam has depth map, so image should be same
      zed.cam.retrieve_image(mat.image, sl.VIEW.LEFT) if menu.take.color else sl.ERROR_CODE.SUCCESS, 
      zed.cam.retrieve_measure(mat.depth, sl.MEASURE.DEPTH)  if menu.take.depth else sl.ERROR_CODE.SUCCESS,
      zed.cam.retrieve_measure(mat.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)  if menu.take.point_cloud else sl.ERROR_CODE.SUCCESS
    ]
    if not all([s == sl.ERROR_CODE.SUCCESS for s in status]):
      print('error take_data()')
      return menu
    if verbose:
      print(f"data get: {(time.time() - st)}[sec]")
    # --------- retain data --------------
    zed.dat = EasyDict({'imu': {}, 'mag':{}, 'temp':{}, 'baro':{}})
    zed.pose = get_pose_transform_matrix(menu)
    d = zed.mat.sensors
    # imu
    data = d.get_imu_data()
    zed.dat.imu.orientation = list(data.get_pose().get_orientation().get())
    zed.dat.imu.linear_acceleration = data.get_linear_acceleration()
    zed.dat.imu.angular_velocity = data.get_angular_velocity()
    zed.dat.imu.timestamp_ns = data.timestamp.get_nanoseconds()
    # mag
    data = d.get_magnetometer_data()
    zed.dat.mag.magnet_field = list(data.get_magnetic_field_calibrated())
    zed.dat.mag.timestamp_ns = data.timestamp.get_nanoseconds()
    # air pressure
    data = d.get_barometer_data()
    zed.dat.baro.pressure = data.pressure
    zed.dat.baro.timestamp_ns = data.timestamp.get_nanoseconds()
    # temperature
    data = d.get_temperature_data()
    zed.dat.temp.temperature = [data.get(
      eval('sl.SENSOR_LOCATION.{}'.format(str(m)))) \
        for m in sl.SENSOR_LOCATION.__members__][:-1]
    zed.dat.temp.temperature_ave = sum(zed.dat.temp.temperature) \
      / (len(sl.SENSOR_LOCATION.__members__) - 1)
  menu.zed = zed
  save(menu)
  return menu

def get_windows_path(p):
  if 'Win' in platform.system():
    if '/mnt/c/' in p:
      p = p.replace('/mnt/c/', 'c:\\')
    p = p.replace('/','\\')
  return p

@stop_watch
def save(menu):
  zed = menu.zed
  now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  p = get_windows_path(f"{menu.save_dir}/{now}")
  os.makedirs(p, exist_ok=True)
  if 'dat' in zed:
    json.dump(zed.dat, open(f"{p}/sensor.json", "w"), indent=2)
  if 'pose' in zed:
    np.save(f"{p}/transform.npy", zed.pose.transform)
    np.save(f"{p}/rotation.npy", zed.pose.rot)
    np.save(f"{p}/translation.npy", zed.pose.translation)
  if menu.take.color:
    np.save(f"{p}/image.npy", zed.mat.image.get_data())
  if menu.take.depth:
    np.save(f"{p}/depth.npy", zed.mat.depth.get_data())
  if menu.take.point_cloud:
    np.save(f"{p}/pcd.npy", zed.mat.point_cloud.get_data())



from  multiprocessing import Pool
def __take_cbk(arg):
    print(arg)
def __looptake(i,menu_cam):
    j = 0
    menu, cam_serid = menu_cam
    # take(menu)
    while True:
        take(menu)
        print('process :%d, loop:%d is started' % (i, j))
        time.sleep(0.2)
        j = j + 1
def take_data(root_dir):
    class multi_take:
      def __init__(self, interval, pron):
        self.pool = Pool(processes=pron)
        self.interval=interval
      def looptake(self,work,i):
        while True:
            work(i)
            time.sleep(self.interval)
            print(i)
        pass
      def start(self,work,menus,cbk):
        for i,menu_cam in enumerate(menus):
            self.pool.apply_async(func=work,
                                  args=(i,menu_cam,),
                                  callback=cbk)
            time.sleep(0.1)
            print("process: %d is started!"%(i))
      def terminate(self):
        self.pool.close()
        self.pool.terminate()

    mp = multi_take(0.2, 5)
    root_dir_tmp = input(f'Please set data save directory. default[{root_dir}] :')
    if root_dir_tmp != '':
      root_dir = root_dir_tmp
      if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        save_dir_fmt = root_dir + "/cam{}/"
        print(f"{root_dir} is created.")

    cameras = sl.Camera.get_device_list()
    menus=[]
    for cam_id, cam in enumerate(cameras):
      menu=init(None, cam_id)
      menus.append([menu,cam.serial_number])
      print(f'menu.save_dir: {menu.save_dir}')
      print(f'available devices:{cameras}')
    while True:
      comm = input('Please enter command(t: take data, q:quit: ')
      if not comm in ['t', 'q']:
        continue
      if comm == 't':
        # menu = take(menu)
        print("take start")
        # for i,menu_cam in enumerate(menus):
          # menu,cam_ser=menu_cam
          # __looptake(i,menu_cam)
        mp.start(__looptake, menus, __take_cbk)
      elif comm == 'q':
        # print(f'available devices:{menu.zed.cam.get_device_list()}')
        print('finish script...')
        break
      # else:
      #   menu = reset_cam(menu, int(comm))
      #   print(f'menu.save_dir: {menu.save_dir}')
        # print(f'available devices:{menu.zed.cam.get_device_list()}')
    for i,menu_cam in enumerate(menus):
        menu,cam_ser=menu_cam
        menu.zed.cam.close()
    mp.pool.close()
    mp.terminate()
    sys.exit(1)

if __name__ == "__main__":
  take_data(root_dir)
