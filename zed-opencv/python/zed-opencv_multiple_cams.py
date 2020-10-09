import sys
import csv
import pandas as pd
import pyzed.sl as sl
import cv2
import numpy as np
import threading
import time
import signal
from transforms3d.quaternions import quat2mat, mat2quat

help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
prefix_reconstruction = "reconstruction"
path = "./data/"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"
key_flg=False
key_last_no=0

zed_pose_list=[]
zed_sensors_list= []
zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
image_size_list=[]
image_zed_list=[]
depth_image_zed_list=[]
stop_signal = False
def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()

def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001) #1ms
    zed_list[index].close()
def translations_quaternions_to_transform(pose):
    t = pose[:3]
    q = pose[3:]

    T = np.eye(4)
    T[:3, :3] = quat2mat(q)
    T[:3, 3] = t
    return T
def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 
def get_pos_dt(zed, zed_pose, zed_sensors):
    #https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    zed_imu = zed_sensors.get_imu_data()  # Display the translation and timestamp
    py_translation = sl.Translation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
    # print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

    # Display the orientation quaternion
    py_orientation = sl.Orientation()
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
    pose_lst = [tx, ty, tz, ow, ox, oy, oz]
    return pose_lst

def export_list_csv(export_list, csv_dir):

    with open(csv_dir, "w") as f:
        writer = csv.writer(f, lineterminator='\n')

        if isinstance(export_list[0], list): #多次元の場合
            writer.writerows(export_list,delimiter=' ')

        else:
            writer.writerow(export_list)
def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)

def save_left_image(zed, filename) :
    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()
    cv2.imwrite(filename, image_cv_left)
    return image_cv_left
def get_camera_intrintic_info(zed):
    cx = zed.get_camera_information().calibration_parameters.left_cam.cx
    cy = zed.get_camera_information().calibration_parameters.left_cam.cy
    fx = zed.get_camera_information().calibration_parameters.left_cam.fx
    fy = zed.get_camera_information().calibration_parameters.left_cam.fy
    distortion = zed.get_camera_information().calibration_parameters.left_cam.disto
    k=[
        [fx,0, cx],
        [0, fy,cy],
        [0, 0,  1]
    ]
    return k
def process_key_event(zed, key,zed_pose, zed_sensors,name_cam):
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68: #d
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:#N
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:#p
        filename=path + prefix_point_cloud + str(count_save)
        save_point_cloud(zed, filename)
        pose_lst=get_pos_dt(zed, zed_pose,zed_sensors)
        export_list_csv(pose_lst, filename + '.csv')
        count_save += 1
    elif key == 109 or key == 77:#m
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:#h
        print(help_string)
    elif key == 114 or  key == 82:#R

        if count_save >100:
            key_flg=False
            key_last_no=0
            count_save=0
        else:
            key_last_no=key
            key_flg=True
            print(count_save)
            pose_lst = get_pos_dt(zed, zed_pose,zed_sensors)
            translation = translations_quaternions_to_transform(pose_lst)
            df = pd.DataFrame(translation)
            filename = path + name_cam+'-'+prefix_reconstruction + "-%06d.pose" % (count_save)
            df.to_csv(filename + '.txt', sep=' ', header=None, index=None)
            filename = path + name_cam+'-'+ prefix_reconstruction + "-%06d.depth" % (count_save)
            save_depth(zed, filename)
            filename = path + name_cam+'-'+ prefix_reconstruction + "-%06d.color" % (count_save)
            image_ocv_left = save_left_image(zed, filename + ".jpg")
            # cv2.imshow("Image", image_ocv_left)
            count_save += 1
            time.sleep(0.5)
    elif key == 115:#f4
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".jpg")
        count_save += 1
    else:
        pass

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'r' to save reconstruction data")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")


def main() :
    global image_zed_list
    global depth_image_zed_list
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    global zed_pose_list
    global zed_sensors_list
    signal.signal(signal.SIGINT, signal_handler)
    # List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    # init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_resolution = sl.RESOLUTION.VGA
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER

    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED_{}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        # Create a ZED camera object
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        zed_pose_list.append(sl.Pose())
        zed_sensors_list.append(sl.SensorsData())
        timestamp_list.append(0)
        last_ts_list.append(0)
        # Open the camera
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        #tracing enable
        py_transform = sl.Transform()
        tracking_parameters = sl.PositionalTrackingParameters(init_pos=py_transform)
        err = zed_list[index].enable_positional_tracking(tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            zed_list[index].close()
            exit(1)
        index = index + 1

    #Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()

    #https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py

    # py_translation = sl.Translation()
    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    index = 0
    for cam in cameras:
        image_size = zed_list[index].get_camera_information().camera_resolution
        image_size.width = image_size.width /2
        image_size.height = image_size.height /2# Declare your sl.Mat matrices
        image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        point_cloud = sl.Mat()
        image_size_list.append(image_size)
        image_zed_list.append(image_zed)
        depth_image_zed_list.append(depth_image_zed)
        ########
        cam_intr=get_camera_intrintic_info(zed_list[index])
        filename = path + name_list[index]+"-camera-intrinsics.txt"
        df = pd.DataFrame(cam_intr)
        df.to_csv(filename , sep=' ', header=None, index=None)
        index += 1


    key = ' '
    while key != 113 :
        def get_cam_color_depth(zed):
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS :
                # Retrieve the left image, depth image in the half-resolution
                zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
                zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
                # Retrieve the RGBA point cloud in half resolution
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

                # To recover data from sl.Mat to use it with opencv, use the get_data() method
                # It returns a numpy array that can be used as a matrix with opencv
                image_ocv = image_zed.get_data()
                depth_image_ocv = depth_image_zed.get_data()
            return err,image_ocv,depth_image_ocv
        index=0
        image_ocv_cat=None
        depth_image_ocv_cat=None
        for cam in cameras:
            err, image_ocv, depth_image_ocv=get_cam_color_depth(zed_list[index])
            if err==sl.ERROR_CODE.SUCCESS and image_ocv_cat is None:
                image_ocv_cat=image_ocv
                depth_image_ocv_cat=depth_image_ocv
            else:
                image_ocv_cat=np.hstack([image_ocv_cat,image_ocv])
                depth_image_ocv_cat=np.hstack([depth_image_ocv_cat,depth_image_ocv])
            index+=1

        cv2.imshow("Image", image_ocv_cat)
        cv2.imshow("Depth", depth_image_ocv_cat)

        key = cv2.waitKey(10)
        if key_flg:
            key=key_last_no
        index=0
        for cam in cameras:
            process_key_event(zed_list[index], key,zed_pose_list[index], zed_sensors_list[index],name_list[index])
            index+=1

    cv2.destroyAllWindows()
    index = 0
    for cam in cameras:
        zed_list[index].close()
        index+=1
    print("\nFINISH")

if __name__ == "__main__":
    main()
