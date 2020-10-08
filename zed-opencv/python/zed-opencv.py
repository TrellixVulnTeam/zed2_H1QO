import sys
import numpy as np
import pyzed.sl as sl
import cv2
import csv
import pandas as pd
import time
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

# def save_tracing_dt(zed,filename,camera_pose,py_translation) :
#     tracking_state = zed.get_position(camera_pose)
#     if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
#         # rotation = camera_pose.get_rotation_vector()
#         rotation = camera_pose.get_orientation()
#         rx=rotation[0]
#         ry=rotation[1]
#         rz=rotation[3]
#         ro=rotation[4]
#         translation = camera_pose.get_translation(py_translation)
#         tx = translation.get()[0]
#         ty = translation.get()[1]
#         tz = translation.get()[2]
#         pose_lst=[tx,ty,tz,rx,ry,rz,ro]
#         df=pd.DataFrame(pose_lst)
#         df.to_csv(filename+'.csv',header=None, index=None)

def get_pos_dt(zed, zed_pose, sl):
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    # zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    # zed_imu = zed_sensors.get_imu_data()  # Display the translation and timestamp
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
def process_key_event(zed, key,zed_pose, sl,image_zed,depth_image_zed,point_cloud,image_size):
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
        pose_lst=get_pos_dt(zed, zed_pose, sl)
        export_list_csv(pose_lst, filename + '.csv')
        count_save += 1
    elif key == 109 or key == 77:#m
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:#h
        print(help_string)
    elif key == 114 or  key == 82:#R
        print("create reconstruction datadd")
        for count_save in range(100):
            print(count_save)
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()

            pose_lst=get_pos_dt(zed, zed_pose, sl)
            translation=translations_quaternions_to_transform(pose_lst)
            df=pd.DataFrame(translation)
            filename=path+prefix_reconstruction+"-%06d.pose"%(count_save)
            df.to_csv(filename+'.txt',sep=' ',header=None,index=None)
            filename=path+prefix_reconstruction+"-%06d.depth"%(count_save)
            save_depth(zed,filename)
            filename=path+prefix_reconstruction+"-%06d.color"%(count_save)
            save_left_image(zed,filename + ".jpg")
            cv2.imshow("Image", image_ocv)
            time.sleep(1)
        count_save=0
    elif key == 115:#f4
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".jpg")
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'r' to save reconstruction data")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")


def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    # init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_resolution = sl.RESOLUTION.VGA
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER
    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)
    #https://github.com/stereolabs/zed-examples/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/python/positional_tracking.py
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    zed_pose = sl.Pose()
    zed_sensors = sl.SensorsData()
    # py_translation = sl.Translation()
    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()
    ########
    cam_intr=get_camera_intrintic_info(zed)
    filename = path + "camera-intrinsics.txt"
    df = pd.DataFrame(cam_intr)
    df.to_csv(filename , sep=' ', header=None, index=None)
    key = ' '
    while key != 113 :
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

            cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)
            process_key_event(zed, key,zed_pose, sl,image_zed,depth_image_zed,point_cloud,image_size)

    cv2.destroyAllWindows()
    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
