import datetime, numpy as np,os,cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from skimage.measure import compare_ssim
from object_detection_draw_v6 import load_model,get_absolute_file_paths,human_bbox_on_image_inferenced_result,load_image_from_npy
def get_same_time_image_from_cameras(pathi,camlst=['cam0','cam1','cam2'],th=5000000):
    files_list,fn_list,dir_list=get_absolute_file_paths(pathi, ext=".npy", fn='image.npy')
    # fpdir=np.concatenate([np.expand_dims(files_list,axis=1),np.expand_dims(dir_list,axis=1)],axis=1)
    lst_file=[]
    for fn_full, fn,fd in zip(files_list,fn_list,dir_list):
        time_str=os.path.basename(os.path.normpath(fd))
        path_cam=fd.replace(time_str,'')
        cam=os.path.basename(os.path.normpath(path_cam))
        lst_file.append([fn_full,fn,cam,time_str])
    df_all=pd.DataFrame(lst_file,columns=['file_full','file_name','camera','times'])
    df_cam0 = df_all[df_all['file_full'].str.contains(camlst[0])].reset_index(drop=True)
    same_time_cam_lst=[]
    for dt in df_cam0.values.tolist():
        fn_full, fn, cam, time_str=dt
        time_0=int(time_str[6:])
        subtract_lst=[]
        # subtract_lst.append([fn_full, fn, cam, time_str ,0])
        subtract_lst.append(time_str)
        flg_same_time=True
        for cam in camlst[1:]:
            df_cami = df_all[df_all['file_full'].str.contains(cam)].reset_index(drop=True)
            #df_cami['times'].replace('_', '', inplace=True)
            
            df_cami["times_c"]=df_cami["times"].str.replace('_', '')
            df_cami["times_t"]=df_cami["times_c"].str[6:]
            
            df_cami[["times_t"]] = df_cami[["times_t"]].apply(pd.to_numeric)
            i=np.argmin(abs(df_cami.loc[:,'times_t']-time_0))
            fn_full_i, fn_i, cam_i, time_,times_c,time_str_i = df_cami.iloc[i,:]
            if abs(int(time_str_i)-time_0)>th:
                flg_same_time=False
                break
            # subtract_lst.append([fn_full_i, fn_i, cam_i, time_str_i ,abs(int(time_str_i)-time_0)])
            subtract_lst.extend([time_])
        if flg_same_time:
            same_time_cam_lst.append(subtract_lst)
    return same_time_cam_lst
def get_same_object_from_multiple_visual(imgs_same_time,bboxes,sh=(30,30)):
    imgs_same_time_new=[]
    img_bboxes_new=[]
    imgs_same_time_new.append(imgs_same_time[0])
    img_bboxes_new.append(bboxes[0])
    obj0=None
    for i ,img_i in enumerate(imgs_same_time):
        objs=[]
        for j,object in enumerate(img_i):
            object_resize=cv2.resize(object,tuple(sh))
            object_vec=object_resize[:,:,0].reshape(-1)
            objs.append(object_vec)
        if i==0:
            obj0=objs
            continue
        sim=cosine_similarity(obj0,objs)
        sim_inds=np.argmax(sim,axis=0)
        objs_new=[]
        bboxes_new=[]
        for sim_ind in sim_inds:
            objs_new.append(img_i[sim_ind])
            bboxes_new.append(bboxes[i][sim_ind])
        imgs_same_time_new.append(objs_new)
        img_bboxes_new.append(bboxes_new)
    return imgs_same_time_new,img_bboxes_new

def get_img_bboxes(image,bboxes):
    human_imgs=[]
    for k, box in enumerate(bboxes):
        xmin, xmax, ymin, ymax = tuple(box)
        image_box = image[ymin:ymax, xmin:xmax, :]
        human_imgs.append(image_box)
    return human_imgs
def object_detection_from_npy_all_file(pathi,patho,path_mod,path_category_index,camlst = ['cam0', 'cam1','cam2']):
    human_dt_lst=[]
    same_time_cam_lst=get_same_time_image_from_cameras(pathi,camlst=camlst) 
    same_time_cam_lst.sort(key=lambda x: x[0])   
    print(same_time_cam_lst)
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"load_model start")
    detector, category_index = load_model(path_mod, path_category_index)
    print(datetime.datetime.now().strftime('%Y%m%d%H%M%S'),"load_model end")
    fn_same_time_cam_lst=f'{patho}/same_time_cam_lst.npy'
    np.save(fn_same_time_cam_lst,same_time_cam_lst)
    for i,same_times in enumerate(same_time_cam_lst):
        merge_id='_%04d_'%(i)
        human_imgs_same_time=[]
        for j,time_id in enumerate(same_times):
            cam=camlst[j]
            path_id=f'{pathi}/{cam}/{time_id}/'
            fnpath=f'{path_id}/image.npy'
            fdir_out=path_id.replace(pathi,patho)
            
            path_o=f'{fdir_out}/human_detection_merge_{merge_id}'
            print(fnpath)
            bboxes_ret,image=human_bbox_on_image_inferenced_result(fnpath, detector, category_index,  th=0.5)
            fn_bboxes_ret = f'{fdir_out}/bboxes.npy'
            np.save(fn_bboxes_ret, bboxes_ret)
            human_imgs=[]
            if len(bboxes_ret)>0:
                os.makedirs(fdir_out, exist_ok=True)
            else:
                continue
            img_objs=get_img_bboxes(image,bboxes_ret)
            for k, img in enumerate(img_objs):
                human_imgs.append(img)
                fn_human=f'{path_o}_%02d.png'%(k)
                print(fn_human)
                cv2.imwrite(fn_human,img)

            human_imgs_same_time.append(human_imgs)
        # human_img0=human_imgs_same_time[0]
        #(score, diff) = compare_ssim(before_gray, after_gray, full=True)
        # ret=get_same_object_from_multiple_visual(human_imgs_same_time)
        # imgs_objs,imgs_bboxes=get_same_object_from_multiple_visual(human_imgs_same_time, bboxes_ret, sh=[30, 30])
    return human_dt_lst


def object_crop_for_merge(pathi, patho, camlst=['cam0', 'cam1', 'cam2']):
    fn_same_time_cam_lst = f'{patho}/same_time_cam_lst.npy'
    same_time_cam_lst=np.load(fn_same_time_cam_lst)
    human_dt_lst=None
    for i, same_times in enumerate(same_time_cam_lst):
        merge_id = '_%04d_' % (i)
        human_imgs_same_time = []
        for j, time_id in enumerate(same_times):
            cam = camlst[j]
            path_id = f'{pathi}/{cam}/{time_id}/'
            fnpath = f'{path_id}/image.npy'
            fdir_out = path_id.replace(pathi, patho)

            path_o = f'{fdir_out}/human_detection_merge_{merge_id}'
            # print(fnpath)
            # bboxes_ret, image = human_bbox_on_image_inferenced_result(fnpath, detector, category_index,  th=0.5)
            fn_bboxes_ret = f'{fdir_out}/bboxes.npy'
            bboxes_ret=np.load(fn_bboxes_ret)
            human_imgs = []
            if len(bboxes_ret) > 0:
                os.makedirs(fdir_out, exist_ok=True)
            else:
                continue
            image, color_org = load_image_from_npy(fnpath)
            img_objs = get_img_bboxes(image, bboxes_ret)
            for k, img in enumerate(img_objs):
                human_imgs.append(img)
                # fn_human = f'{path_o}_%02d.png' % (k)
                # print(fn_human)
                # cv2.imwrite(fn_human, img)

            human_imgs_same_time.append(human_imgs)
        # human_img0=human_imgs_same_time[0]
        # (score, diff) = compare_ssim(before_gray, after_gray, full=True)
        # ret=get_same_object_from_multiple_visual(human_imgs_same_time)
        imgs_objs,imgs_bboxes=get_same_object_from_multiple_visual(human_imgs_same_time, bboxes_ret, sh=[30, 30])
    return human_dt_lst
def main():
  path = 'C:/00_work/05_src/data/fromito/data/fromWATA'
  patho = 'C:/00_work2/05_src/data/fromito/data/fromWATA'

  # path = '/home/user1/yu_develop/202011161_sample/image/'
  # patho = '/home/user1/yu_develop/202011161_sample/out/'
  path_category_index = 'dt/category_index.npy'
  # "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1"
  # "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1"

  # category_index = np.load(path_category_index, allow_pickle='TRUE').item()
  path_mod = "https://tfhub.dev/tensorflow/efficientdet/d4/1"
  # human_dt_lst=object_detection_from_npy_all_file(path,patho,path_mod,path_category_index,camlst = ['cam0', 'cam1'])
  #human_dt_lst=object_detection_from_npy_all_file(path,patho,path_mod,path_category_index)
  human_dt_lst=object_crop_for_merge(path, patho, camlst=['cam0', 'cam1'])
if __name__ == "__main__":

    pass