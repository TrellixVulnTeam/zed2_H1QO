import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import warnings
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import cv2,os,datetime
warnings.simplefilter('ignore')

def load_model(path_mod,path_category_index):
    category_index = np.load(path_category_index, allow_pickle='TRUE').item()
    detector = hub.load(path_mod)
    return detector,category_index
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
def load_image(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return np.expand_dims(img,axis=0)
def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      # display_str = "{}: {}%".format(class_names[i].decode("ascii"),
      display_str = "{}: {}%".format(class_names[i],
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image
def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)

def show_image(img,detection_dt):
    image = np.array(img[0])
    image_with_boxes = draw_boxes(
        image, detection_dt["detection_boxes"],
        detection_dt["detection_classes_name"], detection_dt["detection_scores"])
    display_image(Image.fromarray(image_with_boxes))

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin
def object_detection_inference_result(image_tensor,detector,category_index,th=0.5):
    detector_output = detector(image_tensor)
    result = {key: value.numpy() for key, value in detector_output.items()}
    detection_dt = {}
    bboxlst, clslst, cls_scorelst, keypointlst, keypoint_scorelst, clsnamelst = [], [], [], [], [], []
    for bbox, cls, cls_score in zip(result['detection_boxes'][0], result['detection_classes'][0],
                                    result['detection_scores'][0]):
        if cls_score < th:
            continue
        name = category_index.get(cls).get('name')
        bboxlst.append(bbox)
        clslst.append(cls)
        cls_scorelst.append(cls_score)
        clsnamelst.append(name)
    detection_dt['detection_boxes'] = np.array(bboxlst)
    detection_dt['detection_classes'] = np.array(clslst)
    detection_dt['detection_classes_name'] = np.array(clsnamelst)
    detection_dt['detection_scores'] = np.array(cls_scorelst)
    return detection_dt
def object_detection_inference_keypoints(path,detector,category_index,th=0.5):
    img = load_image(path)
    image_tensor = tf.image.convert_image_dtype(img, tf.uint8)
    detector_output = detector(image_tensor)
    result = {key: value.numpy() for key, value in detector_output.items()}
    detection_dt = {}
    bboxlst, clslst, cls_scorelst, keypointlst, keypoint_scorelst, clsnamelst = [], [], [], [], [], []
    for bbox, cls, cls_score, keypoint, keypoint_score in zip(result['detection_boxes'][0],
                                                              result['detection_classes'][0],
                                                              result['detection_scores'][0],
                                                              result['detection_keypoints'][0],
                                                              result['detection_keypoint_scores'][0]):
        if cls_score < th:
            continue
        name = category_index.get(cls).get('name')
        bboxlst.append(bbox)
        clslst.append(cls)
        cls_scorelst.append(cls_score)
        keypointlst.append(keypoint)
        keypoint_scorelst.append(keypoint_score)
        clsnamelst.append(name)
    detection_dt['detection_boxes'] = np.array(bboxlst)
    detection_dt['detection_classes'] = np.array(clslst)
    detection_dt['detection_classes_name'] = np.array(clsnamelst)
    detection_dt['detection_scores'] = np.array(cls_scorelst)
    detection_dt['detection_keypoints'] = np.array(keypointlst)
    detection_dt['detection_keypoint_scores'] = np.array(keypoint_scorelst)
    return img,detection_dt


def object_detection_inference_bbox(path, detector, category_index, th=0.5):
    img = load_image(path)
    im_width, im_height = img[0].shape[:2]
    image_tensor = tf.image.convert_image_dtype(img, tf.uint8)
    detector_output = detector(image_tensor)
    result = {key: value.numpy() for key, value in detector_output.items()}
    detection_dt = {}
    bboxlst, clslst, cls_scorelst, clsnamelst = [], [], [], [],
    for bbox, cls, cls_score in zip(result['detection_boxes'][0], result['detection_classes'][0],
                                    result['detection_scores'][0]):
        if cls_score < th:
            continue
        name = category_index.get(cls).get('name')

        ymin, xmin, ymax, xmax = tuple(bbox)
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        bboxlst.append([left, right, top, bottom])
        clslst.append(cls)
        cls_scorelst.append(cls_score)
        clsnamelst.append(name)
    detection_dt['detection_boxes'] = np.array(bboxlst, dtype=np.int)
    detection_dt['detection_classes'] = np.array(clslst)
    detection_dt['detection_classes_name'] = np.array(clsnamelst)
    detection_dt['detection_scores'] = np.array(cls_scorelst)
    return img, detection_dt
# output array image and detection ressult
def get_image_inferenced_result(path_img,path_category_index,path_mod,th=0.5):
    category_index = np.load(path_category_index, allow_pickle='TRUE').item()
    detector = hub.load(path_mod)
    img, detection_dt= object_detection_inference_bbox(path_img, detector, category_index, th=th)
    return img,detection_dt

# output the bbox drawed image to destinition file path
def draw_bbox_on_image_inferenced_result(path_img,detector,category_index,path_img_out_id,th=0.5):
    # img = load_image(path_img)
    img,color_org = load_image_from_npy(path_img)
    image_tensor = tf.image.convert_image_dtype(img, tf.uint8)
    image_tensor=tf.expand_dims(image_tensor,axis=0)
    detection_dt= object_detection_inference_result(image_tensor, detector, category_index, th=th)
    image = np.array(img)
    image_with_boxes = draw_boxes(
        image, detection_dt["detection_boxes"],
        detection_dt["detection_classes_name"], detection_dt["detection_scores"])
    cv2.imwrite(path_img_out_id+'.png',cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    if 'person' in detection_dt["detection_classes_name"]:
        human_npy_fp=path_img_out_id+'.npy'
        color_org[:,:,:3]=image_with_boxes
        np.save(human_npy_fp,color_org)
    # return cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

def object_detection_from_npy_all_file(path,path_mod,path_category_index):
    # img, detection_dt = get_image_inferenced_result(path, path_category_index,path_mod ,th=0.5)

    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"load_model start")
    detector, category_index = load_model(path_mod, path_category_index)
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"load_model end")

    # draw_bbox_on_image_inferenced_result(path, detector, category_index, path_o, th=0.5)
    fils_list,fn_list,dir_list=get_absolute_file_paths(path, ext=".npy", fn='image.npy')
    fpdir=np.concatenate([np.expand_dims(fils_list,axis=1),np.expand_dims(dir_list,axis=1)],axis=1)
    l = list(fpdir)
    l.sort(key=lambda x: x[0])
    fpdir = np.array(l)
    cnt=len(fils_list)
    i=0
    for i,fpdiri in enumerate(fpdir):
        path,fdir=fpdiri
        path_o=f'{fdir}/human_detection'
        print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"detection:",i,cnt,path)
        draw_bbox_on_image_inferenced_result(path, detector, category_index, path_o, th=0.5)
if __name__ == "__main__":
  #path = 'C:/00_work/05_src/data/fromito/data/fromWATA'
  path='/home/user1/20201111_shinyoko_4f'
  path_category_index = 'dt/category_index.npy'
  # "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1"
  # "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1"
  path_mod = "https://tfhub.dev/tensorflow/efficientdet/d4/1"
  object_detection_from_npy_all_file(path,path_mod,path_category_index)
