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
from easydict import EasyDict
import os, datetime

# https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py
# https://github.com/stereolabs/zed-multi-camera/blob/master/python/multi_camera.py
basePath = "./data/"
stop_signal = False


def signal_handler(signal, frame):
    global stop_signal
    stop_signal = True
    time.sleep(0.5)
    exit()


def grab_run(cams, index):
    global stop_signal

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = cams.zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cams.zed_list[index].retrieve_image(cams.left_list[index], sl.VIEW.LEFT)
            cams.zed_list[index].retrieve_measure(cams.depth_list[index], sl.MEASURE.DEPTH)
            cams.timestamp_list[index] = cams.zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001)  # 1ms
    cams.zed_list[index].close()


def translations_quaternions_to_transform(pose):
    t = pose[:3]
    q = pose[3:]

    T = np.eye(4)
    T[:3, :3] = quat2mat(q)
    T[:3, 3] = t
    return T


def get_translate(cams, index, cnt_r=10):
    err = cams.zed_list[index].grab(cams.runtime_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed, zed_pose = cams.zed_list[index], cams.pose_list[index]

    # https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    py_translation = cams.py_translation_list[index]
    tx = round(zed_pose.get_translation(py_translation).get()[0], cnt_r)
    ty = round(zed_pose.get_translation(py_translation).get()[1], cnt_r)
    tz = round(zed_pose.get_translation(py_translation).get()[2], cnt_r)

    return [tx, ty, tz]


def get_imu_pose(cams, index, cnt_r=10):
    err = cams.zed_list[index].grab(cams.runtime_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed = cams.zed_list[index]
    zed_pose = cams.pose_list[index]
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    py_translation = sl.Translation()  # cams.py_translation_list[index]

    tx = round(zed_pose.get_translation(py_translation).get()[0], cnt_r)
    ty = round(zed_pose.get_translation(py_translation).get()[1], cnt_r)
    tz = round(zed_pose.get_translation(py_translation).get()[2], cnt_r)

    zed_sensors = cams.zed_sensors_list[index]
    zed_imu = zed_sensors.get_imu_data()
    zed_imu_pose = cams.transform_list[index]  # sl.Transform()
    ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], cnt_r)
    oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], cnt_r)
    oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], cnt_r)
    ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], cnt_r)
    imu_pose_lst = [tx, ty, tz, ow, ox, oy, oz]
    return imu_pose_lst


def get_pos_dt(cams, index, cnt_r=10):
    err = cams.zed_list[index].grab(cams.runtime_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    zed, zed_pose, zed_sensors = cams.zed_list[index], cams.pose_list[index], cams.zed_sensors_list[index]

    # https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    # zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    # zed_imu = zed_sensors.get_imu_data()  # Display the translation and timestamp
    py_translation = cams.py_translation_list[index]
    py_orientation = cams.py_orientation_list[index]
    # py_orientation = sl.Orientation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], cnt_r)
    ty = round(zed_pose.get_translation(py_translation).get()[1], cnt_r)
    tz = round(zed_pose.get_translation(py_translation).get()[2], cnt_r)
    # print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

    # Display the orientation quaternion

    ox = round(zed_pose.get_orientation(py_orientation).get()[0], cnt_r)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], cnt_r)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], cnt_r)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], cnt_r)
    pose_lst = [tx, ty, tz, ow, ox, oy, oz]
    # get camera transform data
    R = zed_pose.get_rotation_matrix(sl.Rotation()).r.T / 1000
    t = zed_pose.get_translation(sl.Translation()).get()
    world2cam = np.hstack((R, np.dot(-R, t).reshape(3, -1)))
    K, distortion = get_camera_intrintic_info(zed)

    P = np.dot(K, world2cam)
    transform = np.vstack((P, [0, 0, 0, 1]))
    return pose_lst, transform


def export_list_csv(export_list, csv_dir):
    with open(csv_dir, "w") as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=' ', )

        if isinstance(export_list[0], list):  # 多次元の場合
            writer.writerows(export_list)

        else:
            writer.writerow(export_list)


def save_point_cloud(cams, index, filename):
    err = cams.zed_list[index].grab(cams.runtime_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed = cams.zed_list[index]
    zed_pose = cams.pose_list[index]
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    # sl_mat_tmp = sl.Mat()
    sl_mat_tmp = cams.pointcloud_list[index]
    zed.retrieve_measure(sl_mat_tmp, sl.MEASURE.XYZRGBA)
    fn = filename + '.ply'
    saved = (sl_mat_tmp.write(fn) == sl.ERROR_CODE.SUCCESS)
    if saved:
        print("point cloud save Done:", fn)
    else:
        print("Failed... Please check that you have permissions to write on disk")


def save_depth(cams, index, filename):
    err = cams.zed_list[index].grab(cams.runtime_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed = cams.zed_list[index]
    zed_pose = cams.pose_list[index]
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    sl_mat_tmp = cams.depth_list[index]
    zed.retrieve_image(sl_mat_tmp, sl.VIEW.DEPTH, sl.MEM.CPU)
    depth_image_ocv = sl_mat_tmp.get_data()
    np.save(f'{filename}.npy', depth_image_ocv)
    # cv2.imwrite(f'{filename}.png',depth_image_ocv)


def save_left_image(cams, index, filename):
    err = cams.zed_list[index].grab(cams.runtime_list[index])
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed = cams.zed_list[index]
    zed_pose = cams.pose_list[index]
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    sl_mat_tmp = cams.left_list[index]
    zed.retrieve_image(sl_mat_tmp, sl.VIEW.LEFT)
    image_cv_left = sl_mat_tmp.get_data()
    cv2.imwrite(filename + '.jpg', image_cv_left)


def get_camera_intrintic_info(zed):
    cx = zed.get_camera_information().calibration_parameters.left_cam.cx
    cy = zed.get_camera_information().calibration_parameters.left_cam.cy
    fx = zed.get_camera_information().calibration_parameters.left_cam.fx
    fy = zed.get_camera_information().calibration_parameters.left_cam.fy
    distortion = zed.get_camera_information().calibration_parameters.left_cam.disto
    k = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]
    return k, distortion


def get_cam_data(cams, index, take_cam_id):
    name_cam = cams.name_list[index]
    take_id = '%06d' % (take_cam_id)
    fd_cam = f'{basePath}/{name_cam}/{take_id}'
    os.makedirs(fd_cam, exist_ok=True)
    print("take_cam_id:", take_cam_id)
    # get transform matrix
    posetransform = get_pose_transform_matrix(cams, index)
    np.savetxt(f'{fd_cam}/trans_m.csv', posetransform)

    # get pose data
    pose_lst, transform = get_pos_dt(cams, index)
    np.savetxt(f'{fd_cam}/pose_ori.csv', pose_lst)
    np.savetxt(f'{fd_cam}/trans_rtkp.csv', transform)

    imu_pose = get_imu_pose(cams, index)
    np.savetxt(f'{fd_cam}/pose_imu.csv', imu_pose)

    trans = translations_quaternions_to_transform(pose_lst)
    np.savetxt(f'{fd_cam}/trans_qua.csv', trans)

    save_depth(cams, index, f'{fd_cam}/depth')
    save_left_image(cams, index, f'{fd_cam}/color')
    save_point_cloud(cams, index, f'{fd_cam}/ply')
    np.save(f"{fd_cam}/pcd.npy", cams.pointcloud_list[index].get_data())
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')
    print(f'{fd_cam}/{now}.csv')
    np.savetxt(f'{fd_cam}/{now}.csv', [0])


def get_pose_transform_matrix(cams, index):
    # https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%203%20-%20depth%20sensing/python/depth_sensing.py
    # mirror_ref = cams.transform_list[index]
    # py_translation=cams.py_translation_list[index]
    # translate=get_translate(cams, index)
    # mirror_ref.set_translation(py_translation(translate))
    # mirror_ref.set_translation(translate)

    zed, zed_pose, zed_sensors = cams.zed_list[index], cams.pose_list[index], cams.zed_sensors_list[index]
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    return cams.transform_list[index].m  # sl.transform.m


def print_help():
    print(" Press 'r' to save reconstruction data")


def get_cam_color_depth(cams, index):
    zed = cams.zed_list[index]
    image_zed = cams.image_zed_list[index]
    depth_image_zed = cams.depth_image_zed_list[index]
    image_size = cams.image_size_list[index]
    # Retrieve the left image, depth image in the half-resolution
    zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
    zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
    # Retrieve the RGBA point cloud in half resolution
    # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

    # To recover data from sl.Mat to use it with opencv, use the get_data() method
    # It returns a numpy array that can be used as a matrix with opencv
    image_ocv = image_zed.get_data()
    depth_image_ocv = depth_image_zed.get_data()
    return image_ocv, depth_image_ocv


def main():
    # global stop_signal
    # signal.signal(signal.SIGINT, signal_handler)
    # List and open cameras
    cameras = sl.Camera.get_device_list()
    index = 0
    cams = EasyDict({})

    cams.pose_list = []
    cams.zed_sensors_list = []
    cams.zed_list = []
    cams.left_list = []
    cams.depth_list = []
    cams.pointcloud_list = []
    cams.timestamp_list = []
    cams.image_size_list = []
    cams.image_zed_list = []
    cams.depth_image_zed_list = []
    cams.name_list = []
    cams.name_list = []
    cams.py_translation_list = []
    cams.py_orientation_list = []
    cams.transform_list = []
    cams.runtime_list = []
    # Set configuration parameters

    '''
    https://www.stereolabs.com/docs/positional-tracking/using-tracking/
    Positional tracking uses image and depth information to estimate the position of the camera in 3D space. 
    To improve tracking results, use high FPS video modes such as HD720 and WVGA.
    '''
    init = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,#HD720
        # camera_resolution=sl.RESOLUTION.HD2K,#HD720
                             # coordinate_units=sl.UNIT.METER,
                             coordinate_units=sl.UNIT.MILLIMETER,#精度アップのため
                             depth_mode=sl.DEPTH_MODE.PERFORMANCE,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        cams.name_list.append("ZED_{}".format(cam.serial_number))
        print("Opening {}".format(cams.name_list[index]))
        # Create a ZED camera object
        cams.zed_list.append(sl.Camera())
        cams.left_list.append(sl.Mat())
        cams.depth_list.append(sl.Mat())
        cams.pointcloud_list.append(sl.Mat())
        cams.pose_list.append(sl.Pose())
        cams.zed_sensors_list.append(sl.SensorsData())
        cams.timestamp_list.append(0)
        cams.py_translation_list.append(sl.Translation())
        cams.transform_list.append(sl.Transform())
        cams.py_orientation_list.append(sl.Orientation())

        # Open the camera
        status = cams.zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            cams.zed_list[index].close()
            exit(1)
        # tracing enable
        py_transform = cams.transform_list[index]
        tracking_parameters = sl.PositionalTrackingParameters(init_pos=py_transform)
        err = cams.zed_list[index].enable_positional_tracking(tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            cams.zed_list[index].close()
            exit(1)
        runtime = sl.RuntimeParameters()
        cams.runtime_list.append(runtime)
        index = index + 1

    # Start camera threads
    # for index in range(0, len(cams.zed_list)):
    #     if cams.zed_list[index].is_opened():
    #         thread_list.append(threading.Thread(target=grab_run, args=(cams,index,)))
    #         thread_list[index].start()

    # https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py

    # py_translation = sl.Translation()
    # Display help in console
    print_help()

    # Prepare new image size to retrieve half-resolution images
    for index, cam in enumerate(cameras):
        fd_cam = f'{basePath}/{cams.name_list[index]}'
        os.makedirs(fd_cam, exist_ok=True)
        image_size = cams.zed_list[index].get_camera_information().camera_resolution
        image_size.width = image_size.width / 2
        image_size.height = image_size.height / 2  # Declare your sl.Mat matrices
        # image_zed = cams.left_list[index](image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        # depth_image_zed = cams.depth_list[index](image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        cams.image_size_list.append(image_size)
        cams.image_zed_list.append(image_zed)
        cams.depth_image_zed_list.append(depth_image_zed)
        ########
        cam_intr, distortion = get_camera_intrintic_info(cams.zed_list[index])
        filename = f'{fd_cam}/camera-intrinsics.csv'
        np.savetxt(filename, cam_intr)
        filename = f'{fd_cam}/camera-distortion.csv'
        np.savetxt(filename, distortion)

    # *******************************************************************
    take_by_keyinput(cameras, cams)
    # take_by_keyinput_camera_view(cameras, cams)
    # *******************************************************************
    index = 0
    for cam in cameras:
        cams.zed_list[index].close()
        index += 1
    print("\nFINISH")


def take_by_keyinput_camera_view(cameras, cams):
    key = ' '
    take_cam_id = 0
    while key != 113:
        index = 0
        image_ocv_cat = None
        depth_image_ocv_cat = None
        for cam in cameras:
            image_ocv, depth_image_ocv = get_cam_color_depth(cams, index)
            if image_ocv_cat is None:
                image_ocv_cat = image_ocv
                depth_image_ocv_cat = depth_image_ocv
            else:
                image_ocv_cat = np.hstack([image_ocv_cat, image_ocv])
                depth_image_ocv_cat = np.hstack([depth_image_ocv_cat, depth_image_ocv])
            index += 1

        cv2.imshow("Image", image_ocv_cat)
        cv2.imshow("Depth", depth_image_ocv_cat)

        key = cv2.waitKey(10)
        if key == 114 or key == 82:  # R
            index = 0
            for cam in cameras:
                get_cam_data(cams, index, take_cam_id)
                index += 1
            take_cam_id += 1
        if key == 113 or key == 81:  # q
            index = 0
            for cam in cameras:
                cams.zed_list[index].close()
                index += 1
            print('finish script...')
            sys.exit(1)

    cv2.destroyAllWindows()


def take_by_keyinput(cameras, cams):
    take_cam_id = 0
    while True:
        comm = input('Please enter command(r: take data, q:quit): ')
        if not comm in ['r', 'q']:
            continue
        if comm == 'r':
            index = 0
            for cam in cameras:
                get_cam_data(cams, index, take_cam_id)
                index += 1
            take_cam_id += 1
        elif comm == 'q':
            index = 0
            for cam in cameras:
                cams.zed_list[index].close()
                index += 1
            print('finish script...')
            sys.exit(1)


if __name__ == "__main__":
    main()
