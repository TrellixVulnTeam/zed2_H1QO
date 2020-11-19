import pandas as pd
import numpy as np
from PIL import ImageOps
from PIL import Image
import cv2,os,datetime
import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt

def convert_bgra2rgba(img):
  # ZED2 numpy color is BGRA.
  rgba = np.zeros(img.shape).astype(np.uint8)
  rgba[:,:,0] = img[:,:,2] # R
  rgba[:,:,1] = img[:,:,1] # G
  rgba[:,:,2] = img[:,:,0] # B
  rgba[:,:,3] = img[:,:,3] # A
  return rgba
def load_image_from_npy(p, convert=True):
  color_org = np.load(p)
  color = convert_bgra2rgba(color_org) if convert else color_org
  # pil_img = Image.fromarray(color.astype(np.uint8))
  # pil_img.save(f'{p}/image.png')
  return color[:,:,:3],color_org
def load_image(fn, input_size):
    # color = cv2.imread(fn)
    # image = Image.fromarray(color.astype(np.uint8))

    color,color_org=load_image_from_npy(fn)
    # color_npy=np.load(fn)
    # color=color_npy[:,:,:3]
    image = Image.fromarray(color[:,:,:3].astype(np.uint8))
    old_size = image.size  # old_size is in (width, height) format
    desired_ratio = input_size[0] / input_size[1]
    old_ratio = old_size[0] / old_size[1]

    if old_ratio < desired_ratio:  # '<': cropping, '>': padding
        new_size = (old_size[0], int(old_size[0] / desired_ratio))
    else:
        new_size = (int(old_size[1] * desired_ratio), old_size[1])

    # print(new_size, old_size)

    # Cropping the original image to the desired aspect ratio
    delta_w = new_size[0] - old_size[0]
    delta_h = new_size[1] - old_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    cropped_image = ImageOps.expand(image, padding)
    # Resize the cropped image to the desired model size
    resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1
    return image_for_prediction, cropped_image, color


def Load_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    # get image size - converting from BHWC to WH
    input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
    return interpreter, input_details, input_size


def run_inference(interpreter, input_details, image_for_prediction, cropped_image):
    # Invoke the interpreter to run inference.
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    interpreter.invoke()

    # Retrieve the raw output map.
    raw_prediction = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()

    width, height = cropped_image.size
    seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
    return seg_map


def load_labels_info(path):
    ade20k_labels_info = pd.read_csv(path)
    labels_list = list(ade20k_labels_info['Name'])
    ade20k_labels_info.head()
    labels_list.insert(0, 'others')

    full_label_map = np.arange(len(labels_list)).reshape(len(labels_list), 1)
    full_color_map = label_to_color_image(full_label_map)
    return full_color_map, np.asarray(labels_list)


def create_ade20k_label_colormap():
    return np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_ade20k_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map, FULL_COLOR_MAP, LABEL_NAMES):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

def load_image3(fn, input_size):
    color_npy=np.load(fn)
    color=color_npy[:,:,:3]
    image = Image.fromarray(color.astype(np.uint8))
    resized_image = image.convert('RGB').resize(input_size, Image.BILINEAR)

    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1
    return image_for_prediction, resized_image, color


def main():
    # Load the model.
    tflite_path = 'dt/lite-model_deeplabv3-xception65-ade20k_1_default_2.tflite'
    fn = 'dt/image_1.png'
    path = 'dt/objectInfo150.csv'

    full_color_map, labels_list = load_labels_info(path)
    interpreter, input_details, input_size = Load_model(tflite_path)

    image_for_prediction, cropped_image = load_image(fn, input_size)
    seg_map = run_inference(interpreter, input_details, image_for_prediction, cropped_image)

    vis_segmentation(cropped_image, seg_map, full_color_map, labels_list)


def vis_segmentation3(image, seg_map, FULL_COLOR_MAP, LABEL_NAMES):
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)

    plt.show()
def image_mask_merge(img_org,seg_map):
  img_org=Image.fromarray(img_org)
  # seg_map_resize1=Image.fromarray(seg_map)
  seg_map_color = label_to_color_image(seg_map).astype(np.uint8)
  seg_map_color_img=Image.fromarray(seg_map_color)
  # mask = Image.new("L", img_org.size, 128)
  im = Image.blend(img_org, seg_map_color_img, 0.6)
  return im

def get_absolute_file_paths(directory,ext=".npy",fn='image.npy'):
   fils_list=[]
   fn_list=[]
   dir_list=[]
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           if  f.find(fn)>=0:
               fils_list.append( os.path.abspath(os.path.join(dirpath, f)))
               fn_list.append( f)
               dir_list.append(dirpath)
   return fils_list,fn_list,dir_list
def segmentation_resize(img_org,seg_map):
    seg_map_im = np.expand_dims(seg_map, axis=2)
    seg_map_imk = np.concatenate([seg_map_im, seg_map_im, seg_map_im], axis=2)
    array = np.array(seg_map_imk, dtype='uint8')
    w, h = np.array(img_org).shape[:2]
    seg_map_resize = cv2.resize(array, (h, w))
    return seg_map_resize
def semantic_segmentation_from_npy_all_file(path,path_mod,path_lable):
    # img, detection_dt = get_image_inferenced_result(path, path_category_index,path_mod ,th=0.5)

    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"load_model start")
    full_color_map, labels_list = load_labels_info(path_lable)
    interpreter, input_details, input_size = Load_model(path_mod)
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"load_model end")


    fils_list,fn_list,dir_list=get_absolute_file_paths(path, ext=".npy", fn='image.npy')
    fpdir=np.concatenate([np.expand_dims(fils_list,axis=1),np.expand_dims(dir_list,axis=1)],axis=1)
    l = list(fpdir)
    l.sort(key=lambda x: x[0])
    fpdir = np.array(l)
    cnt=len(fils_list)
    i=0
    for i,fpdiri in enumerate(fpdir):
        path,fdir=fpdiri
        print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"detection:",i,cnt,path)
        image_for_prediction, cropped_image, img_org = load_image3(path, input_size)
        seg_map = run_inference(interpreter, input_details, image_for_prediction, cropped_image)
        seg_map_resize=segmentation_resize(img_org,seg_map)
        seg_map_resize[seg_map_resize != 13] = 1
        out_img=image_mask_merge(img_org, seg_map_resize[:,:,0])
        path_o=f'{fdir}/human_segmentation_3'
        cv2.imwrite(path_o + '.png', np.array(out_img))

        image_for_prediction, cropped_image, img_org = load_image(path, input_size)
        seg_map = run_inference(interpreter, input_details, image_for_prediction, cropped_image)
        seg_map=np.array(seg_map)
        seg_map[seg_map != 13] = 1
        out_img=image_mask_merge(np.array(cropped_image), seg_map)
        path_o=f'{fdir}/human_segmentation_crop_3'
        cv2.imwrite(path_o + '.png', np.array(out_img))

if __name__ == "__main__":
    tflite_path = 'dt/lite-model_deeplabv3-xception65-ade20k_1_default_2.tflite'
    path = 'C:/00_work/05_src/data/fromito/data/fromWATA'
    path_lable = 'dt/objectInfo150.csv'
    semantic_segmentation_from_npy_all_file(path,tflite_path,path_lable)
