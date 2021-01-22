#!/usr/bin/env python3
import sys
%cd {bp}/contextdesc
LOCAL_PATH = '../'
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)
import os
from models import get_model
from models.reg_model import RegModel
UTILS_PATH="{bp}/contextdesc/"
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)
os.environ['PYTHONPATH'] = '/env/python:{bp}/contextdesc/'
# get_model('reg_model')
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

UTILS_PATH = 'utils'
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)
from opencvhelper import MatcherWrapper

from easydict import EasyDict

UTILS_PATH = '../models'
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)
from models import get_model



FLAGS = EasyDict({})
FLAGS.loc_model='pretrained/contextdesc++'
FLAGS.reg_model='pretrained/retrieval_model'
FLAGS.img1_path='imgs/rgb_cam0.png'
FLAGS.img2_path='imgs/rgb_cam1.png'
FLAGS.n_sample=2048
FLAGS.model_type='pb'
FLAGS.dense_desc=False
FLAGS.ratio_test=False
FLAGS.cross_check=False


def load_imgs(img_paths):
    rgb_list = []
    gray_list = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
    return rgb_list, gray_list


def extract_regional_features(rgb_list, model_path):
    reg_feat_list = []
    model = get_model('reg_model')(model_path)
    for _, val in enumerate(rgb_list):
        reg_feat = model.run_test_data(val)
        reg_feat_list.append(reg_feat)
    model.close()
    return reg_feat_list


def extract_local_features(gray_list, model_path):
    cv_kpts_list = []
    loc_info_list = []
    loc_feat_list = []
    sift_feat_list = []
    model = get_model('loc_model')(model_path, **{'sift_desc': True,
                                                  'n_sample': FLAGS.n_sample,
                                                  'peak_thld': 0.04,
                                                  'dense_desc': FLAGS.dense_desc,
                                                  'upright': False})
    for _, val in enumerate(gray_list):
        loc_feat, kpt_mb, normalized_xy, cv_kpts, sift_desc = model.run_test_data(val)
        raw_kpts = [np.array((i.pt[0], i.pt[1], i.size, i.angle, i.response)) for i in cv_kpts]
        raw_kpts = np.stack(raw_kpts, axis=0)
        loc_info = np.concatenate((raw_kpts, normalized_xy, loc_feat, kpt_mb), axis=-1)
        cv_kpts_list.append(cv_kpts)
        loc_info_list.append(loc_info)
        sift_feat_list.append(sift_desc)
        loc_feat_list.append(loc_feat / np.linalg.norm(loc_feat, axis=-1, keepdims=True))
    model.close()
    return cv_kpts_list, loc_info_list, loc_feat_list, sift_feat_list


def extract_augmented_features(reg_feat_list, loc_info_list, model_path):
    aug_feat_list = []
    model = get_model('aug_model')(model_path, **{'quantz': False})
    assert len(reg_feat_list) == len(loc_info_list)
    for idx, _ in enumerate(reg_feat_list):
        aug_feat, _ = model.run_test_data([reg_feat_list[idx], loc_info_list[idx]])
        aug_feat_list.append(aug_feat)
    model.close()
    return aug_feat_list
