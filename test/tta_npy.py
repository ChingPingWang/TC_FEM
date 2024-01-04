import numpy as np
from skimage.io import imread, imshow, imsave
from mrcnn import model_luo as modellib

from mrcnn.config import Config
from keras import backend as K
import scipy.io as sio 
import cv2
import glob
import matplotlib.pyplot as plt
import os
import sys
import threading
from queue import Queue
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import dilation,disk, closing, square, opening, erosion
from skimage import measure
import time
import pandas as pd 
import pdb

def make_label_(im_,label_): 
    im_label = im_.copy()
    for number_cell in range(label_.shape[0]):
        class_id   = label_[number_cell,0]
        axis_title = label_[number_cell,1]
        mask_bound = label_[number_cell,2].astype(np.uint8)
        c = cv2.findContours(mask_bound,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #only accepted binary image
        bound = (c[1][0][:,0,1],c[1][0][:,0,0]) # position (x,y)
        bound_mask= np.zeros(mask_bound.shape) 
        bound_mask[bound] = 255 
        bound_mask = dilation(bound_mask,disk(2)) 
        bound  = np.where(bound_mask == 255) 
        new_bound = (bound[0] + axis_title[0],bound[1] + axis_title[2])
        new_bound[0][np.where(new_bound[0] >= im_.shape[0])] = im_.shape[0] - 1 
        new_bound[1][np.where(new_bound[1] >= im_.shape[1])] = im_.shape[1] - 1 
        if class_id == 1: #Red:model prediction (positive)
            im_label[:,:,0][new_bound] = 255
            im_label[:,:,1][new_bound] = 0
            im_label[:,:,2][new_bound] = 0
        elif class_id == 2: #Blue:model prediction (negative)
            im_label[:,:,0][new_bound] = 0
            im_label[:,:,1][new_bound] = 0
            im_label[:,:,2][new_bound] = 255
    return im_label

def overlap_preposes(input_):
    result = input_[0]
    size   = input_[1]
    th = 0.5
    result = np.array(result)
    mask_overlap = np.zeros(size)
    delete = []
    for l in tqdm(range(result.shape[0])):
        mask_out = np.zeros(size).astype(np.int32)
        mask_out[result[l,1][0] : result[l,1][1], result[l,1][2] :result[l,1][3]] = result[l,2]
        mask_overlap = mask_overlap + mask_out * (l+1)
        if mask_overlap.max() > (l+1):
            pixel_number = result[int(l),2].sum()
            number = mask_overlap[np.where(mask_overlap > (l+1))]
            uni_number = np.unique(number)
            for uni in uni_number:
                pixel_number2 = result[int(uni-l-1-1),2].sum()
                overlap  = len(np.where(number == uni)[0])/pixel_number
                overlap2 = len(np.where(number == uni)[0])/pixel_number2
                if overlap < th:
                    if overlap2 > th:
                        mask_overlap[np.where(mask_overlap == uni)] = (l+1)
                        mask_overlap[np.where(mask_overlap == (uni - l - 1))]   = 0
                        delete.append(int(uni-l-1-1))
                    else:
                        mask_overlap[np.where(mask_overlap == uni)] = (l+1)
                if overlap > th:
                    if pixel_number2 >= pixel_number:
                        mask_overlap[np.where(mask_overlap == uni)] = (uni - l - 1)
                        mask_overlap[np.where(mask_overlap == (l+1))]    = 0
                        delete.append(l)
                    if pixel_number2 < pixel_number:
                        mask_overlap[np.where(mask_overlap == uni)] = (l+1)
                        mask_overlap[np.where(mask_overlap == (uni - l - 1))]   = 0
                        delete.append(int(uni-l-1-1))
                    if pixel_number2 < 50:
                        delete.append(int(uni-l-1-1))
                    
    if delete != []:
        result = np.delete(result,delete,0)
    return result.tolist()
def th_overlap_preposes(result,size):
    result_size = len(result)//8
    pool = multiprocessing.Pool(processes= 8)
    start = len(result)
    print('start = ' +  str(start))
    print(len(result[0::8]) + len(result[1::8]) + len(result[2::8]) +
          len(result[3::8]) +len(result[4::8]) +len(result[5::8]) +
          len(result[6::8]) +len(result[7::8]))
    tasks = [(result[0::8],size),
             (result[1::8],size),
             (result[2::8],size),
             (result[3::8],size),
             (result[4::8],size),
             (result[5::8],size),
             (result[6::8],size),
             (result[7::8],size),]
    result_mult = pool.map(overlap_preposes, tasks)
    result = result_mult[0] + result_mult[1] + \
                  result_mult[2] + result_mult[3] + \
                  result_mult[4] + result_mult[5] + \
                  result_mult[6] + result_mult[7]
    pool.close()
    pool.join()
    del result_mult
    end1 = len(result)
    print('end1 = ' + str(end1))
    result = overlap_preposes((result,size))
    end2 = len(result)
    print('end2 = ' + str(end2))
    return result

class NucleusConfig(Config):
    NAME = "nucleus"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2 
    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 1
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 4096
    POST_NMS_ROIS_TRAINING = 2048
    POST_NMS_ROIS_INFERENCE = 3072
    MEAN_PIXEL = np.array([217.81, 198.47, 217.69])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (28, 28)
    TRAIN_ROIS_PER_IMAGE = 1024
    MAX_GT_INSTANCES = 1024
    DETECTION_MAX_INSTANCES = 2048
    ROI_POSITIVE_RATIO = 0.33
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7
    # FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    Data_Set_Nms_max  = 0.7
    Data_Set_Nms_min  = 0.3

class NucleusInferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_RESIZE_MODE = "pad64"
    IMAGE_RESIZE_MODE = "none"
    RPN_NMS_THRESHOLD = 0.7

def box_filp_trans(box_list,mode,h,w):
    if mode=='fliplr':
      flip_matrix = np.array([[-1,0],[0,1]])
    if mode=='flipud':
      flip_matrix = np.array([[1,0],[0,-1]])
    if mode=='fliplrud':
      flip_matrix = np.array([[-1,0],[0,-1]])
    new_box_list = []
    for ii in range(len(box_list)):
      bx1, by1, bx2, by2 = box_list[ii][0], box_list[ii][1], box_list[ii][2], box_list[ii][3]
      p1, p2 = np.array([bx1,by1]), np.array([bx2,by2])
      if mode=='fliplr':
        nbx1, nby1 = flip_matrix.dot(p2)[0]+w, flip_matrix.dot(p1)[1]
        nbx2, nby2 = flip_matrix.dot(p1)[0]+w, flip_matrix.dot(p2)[1]
      if mode=='flipud':
        nbx1, nby1 = flip_matrix.dot(p1)[0], flip_matrix.dot(p2)[1]+h
        nbx2, nby2 = flip_matrix.dot(p2)[0], flip_matrix.dot(p1)[1]+h
      if mode=='fliplrud':
        nbx1, nby1 = flip_matrix.dot(p2)[0]+w, flip_matrix.dot(p2)[1]+h
        nbx2, nby2 = flip_matrix.dot(p1)[0]+w, flip_matrix.dot(p1)[1]+h
      new_box_list.append([nbx1, nby1, nbx2, nby2])
    return new_box_list

def mask_filp_trans(mask_list,mode):
    new_mask_list = []
    if mode=='fliplr':
      for ii in range(mask_list.shape[2]):
        img_mask = mask_list[:,:,ii]      
        new_mask = np.fliplr(img_mask)
        new_mask_list.append(new_mask)
    if mode=='flipud':
      for ii in range(mask_list.shape[2]):
        img_mask = mask_list[:,:,ii]
        new_mask = np.flipud(img_mask)
        new_mask_list.append(new_mask)
    if mode=='fliplrud':
      for ii in range(mask_list.shape[2]):
        img_mask = mask_list[:,:,ii]
        new_mask = np.fliplr(np.flipud(img_mask))
        new_mask_list.append(new_mask)
    if mode=='ori':
      for ii in range(mask_list.shape[2]):
          img_mask = mask_list[:,:,ii]
          new_mask_list.append(img_mask)
    return new_mask_list


def overlap_pred(model,image,name,size=(512,512),overlap =(256,256)):
    score_ls, box_ls, mask_ls, class_ls = [], [], [], []
    step_number = 0
    for i in range((image.shape[0]//overlap[0]) + 1):
        for j in range((image.shape[1]//overlap[1]) + 1):
            x_title = i*overlap[0]
            y_title = j*overlap[1]
            x_end   = x_title + size[0]
            y_end   = y_title + size[1]
            if y_end > image.shape[1]: #if y_end exceed the width of image
                y_end = image.shape[1] #choose the longest width of image
            if x_end > image.shape[0]:
                x_end = image.shape[0]
            pre_image = image[x_title:x_end,y_title:y_end,:] #crop the image 
            rangemask = np.zeros(size) #default = 512*512
            rangemask[0:pre_image.shape[0],0:pre_image.shape[1]] = np.ones(pre_image.shape[0:2]) #
            pre_image_pad = np.ones(size + (3,))*255
            pre_image_pad[0:pre_image.shape[0],0:pre_image.shape[1],:] = pre_image            
            
            result0 = model.detect([pre_image_pad], verbose=0)[0] #predict cells in assigned area
            result1 = model.detect([(np.fliplr(pre_image_pad))], verbose=0)[0] #predict cells in assigned area
            result2 = model.detect([np.flipud(pre_image_pad)], verbose=0)[0] #predict cells in assigned area
            result3 = model.detect([np.fliplr(np.flipud(pre_image_pad))], verbose=0)[0] #predict cells in assigned area

            r0, r1, r2, r3 = result0['rois'].tolist(), box_filp_trans(result1['rois'],'fliplr',512,512), box_filp_trans(result2['rois'],'flipud',512,512), box_filp_trans(result3['rois'],'fliplrud',512,512)
            m0, m1, m2, m3 = mask_filp_trans(result0['masks'],'ori'), mask_filp_trans(result1['masks'],'fliplr'), mask_filp_trans(result2['masks'],'flipud'), mask_filp_trans(result3['masks'],'fliplrud')
            s0, s1, s2, s3 = result0['scores'].tolist(), result1['scores'].tolist(), result2['scores'].tolist(), result3['scores'].tolist()
            c0, c1, c2, c3 = result0['class_ids'].tolist(), result1['class_ids'].tolist(), result2['class_ids'].tolist(), result3['class_ids'].tolist()
                           
            score_ls = s0+s1+s2+s3
            box_ls = r0+r1+r2+r3
            mask_ls = m0+m1+m2+m3
            class_ls = c0+c1+c2+c3
            #pdb.set_trace() 
    return score_ls, box_ls, mask_ls, class_ls


def model_cell_load(weight_path):
    config = NucleusInferenceConfig()
    model  = modellib.MaskRCNN(mode="inference", config=config,model_dir=weight_path)
    model.load_weights(weight_path, by_name=True)
    return model

#tumor
def read_and_nor(im,mode='std'):
    im = im.astype(np.float)
    if im.shape[-1] >= 3:
        im = im[:,:,0:3]
    if mode == 'nor':
        im = im/255.
        # print('image divided by 255')
    if mode == 'std':
        #im = (im - im.mean())/im.std()
        im[:,:,0] = (im[:,:,0] - np.mean(im[:,:,0]))/np.std(im[:,:,0])
        im[:,:,1] = (im[:,:,1] - np.mean(im[:,:,1]))/np.std(im[:,:,1])
        im[:,:,2] = (im[:,:,2] - np.mean(im[:,:,2]))/np.std(im[:,:,2])
        # print('image in std normalize')
    else :
        print('The image may not be normalized')
    return im

def crop_list_fn(im_,crop_range_):
    image_enpty_list = []
    crop_list  = ((im_.shape[0]//crop_range_[0]) + 1 , (im_.shape[1]//crop_range_[1]) + 1 )
    for i in range(crop_list[0]):
        for j in range(crop_list[1]):
            i_range = (i*512,(i+1)*512)
            j_range = (j*512,(j+1)*512)
            if (i+1)*512 > im_.shape[0]:
                i_range = (im_.shape[0]-crop_range_[0], im_.shape[0])
            if (j+1)*512 > im_.shape[1]:
                j_range = (im_.shape[1]-crop_range_[1], im_.shape[1])
            image_enpty_list.append(im_[i_range[0]:i_range[1],j_range[0]:j_range[1],:])
    return image_enpty_list

#------------------------------------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#------------------------------------------------------------------------------------------------------------------------
file_path = '/mnt/disk6/nancy/paip_2023/data/PAIP_val/'
save_path = '/mnt/disk6/nancy/paip_2023/result/tta/val_67_luo/'
model = model_cell_load('/mnt/disk6/nancy/paip_2023/logs/mask_rcnn_nucleus_0067.h5')

path_list = os.listdir(file_path)
file_ls = list()
for file_n in path_list:
    if file_n.endswith('.png'):
        full_path = file_path + file_n
        file_ls.append(full_path)

total_ls= []
for ii in file_ls[0:1]:
    id_n = ii.split('/')[-1].split('.')[0]
    im = imread(ii)[:,:,0:3]
    sc, bb, mk, cl = overlap_pred(model,im,id_n,size=(512,512),overlap =(400,400))
    #total_ls.append([sc, bb, mk, cl])
    pdb.set_tmkrace()
    np.save(save_path + id_n + '_sc.npy', sc)
    np.save(save_path + id_n + '_bb.npy', bb)
    np.save(save_path + id_n + '_mk.npy', mk)
    np.save(save_path + id_n + '_cl.npy', cl)
    