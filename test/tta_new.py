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

def overlap_pred(model,image,name,size=(512,512),overlap =(256,256)):
    masks_new  = []
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
            result = model.detect([pre_image_pad], verbose=0)[0] #predict cells in assigned area

            scores = result['scores']
            class_ids = result['class_ids']
            masks     = result['masks']
            rois      = result['rois']

            del result
            cls_ls, box_ls, mask_ls, sc_ls = [], [], [], []
            for num, c in enumerate(class_ids):
                sc = scores[num]
                mask = (masks[:,:,num] * rangemask).astype(np.bool) #cell mask (binary)
                x0 = rois[num,:][0] # coordinate of cells
                x1 = rois[num,:][2] 
                y0 = rois[num,:][1] 
                y1 = rois[num,:][3]
                if x1 > pre_image.shape[0]:
                    x1 = pre_image.shape[0] 
                    rois[num,:][2] = rois[num,:][2] - (rois[num,:][2] - pre_image.shape[0])
                if y1 > pre_image.shape[1]:
                    y1 = pre_image.shape[1]
                    rois[num,:][3] = rois[num,:][3] - (rois[num,:][3] - pre_image.shape[1])
                x0 = rois[num,:][0] + x_title 
                x1 = rois[num,:][2] + x_title
                y0 = rois[num,:][1] + y_title
                y1 = rois[num,:][3] + y_title
                if mask.max() != 0:
                    roi_new = [x0, x1, y0, y1]
                    masks_new.append([c, roi_new, mask[rois[num,:][0]:rois[num,:][2] , rois[num,:][1]:rois[num,:][3]], sc])                    
    return masks_new

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
        #print('image divided by 255')
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
    
def pred_to_list(pred):
  cl, bb, mk, sc = [], [] ,[], []
  for ii in range(len(pred)):
    cl.append(pred[ii][0])
    bb.append(pred[ii][1])
    mk.append(pred[ii][2])
    sc.append(pred[ii][3])   
  return cl, bb, mk, sc 
  
def box_filp_trans(box_list,mode,h,w):
    if mode=='fliplr':
        flip_matrix = np.array([[1,0],[0,-1]])
    if mode=='flipud':
        flip_matrix = np.array([[-1,0],[0,1]])
    if mode=='fliplrud':
        flip_matrix = np.array([[-1,0],[0,-1]])
    new_box_list = []
    for ii in range(len(box_list)):
        bx1, bx2, by1, by2 = box_list[ii][0], box_list[ii][1], box_list[ii][2], box_list[ii][3]
        p1, p2 = np.array([bx1,by1]), np.array([bx2,by2])
        if mode=='fliplr':
            nbx1, nby1 = flip_matrix.dot(p2)[0], flip_matrix.dot(p1)[1]+w
            nbx2, nby2 = flip_matrix.dot(p1)[0], flip_matrix.dot(p2)[1]+w
        if mode=='flipud':
            nbx1, nby1 = flip_matrix.dot(p1)[0]+h, flip_matrix.dot(p2)[1]
            nbx2, nby2 = flip_matrix.dot(p2)[0]+h, flip_matrix.dot(p1)[1]
        if mode=='fliplrud':
            nbx1, nby1 = flip_matrix.dot(p1)[0]+h, flip_matrix.dot(p1)[1]+w
            nbx2, nby2 = flip_matrix.dot(p2)[0]+h, flip_matrix.dot(p2)[1]+w
        new_box_list.append([nbx2, nbx1, nby2, nby1])
    return new_box_list

def mask_filp_trans(mask_list,mode):
    new_mask_list = []
    if mode=='fliplr':
        for ii in range(len(mask_list)):
            img_mask = mask_list[ii]
            new_mask = np.fliplr(img_mask)
            new_mask_list.append(new_mask)
    if mode=='flipud':
        for ii in range(len(mask_list)):
            img_mask = mask_list[ii]
            new_mask = np.flipud(img_mask)
            new_mask_list.append(new_mask)
    if mode=='fliplrud':
        for ii in range(len(mask_list)):
            img_mask = mask_list[ii]
            new_mask = np.fliplr(np.flipud(img_mask))
            new_mask_list.append(new_mask)
    if mode=='ori':
        for ii in range(len(mask_list)):
            img_mask = mask_list[ii]
            new_mask_list.append(img_mask)
    return new_mask_list
    
#------------------------------------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#------------------------------------------------------------------------------------------------------------------------
file_path = '/mnt/disk6/nancy/paip_2023/data/for_ppt/'
save_path = '/mnt/disk6/nancy/paip_2023/result/test_phase/mrcnn/test_160_luo/'
model = model_cell_load('/mnt/disk6/nancy/paip_2023/logs/mask_rcnn_nucleus_0111.h5')

path_list = os.listdir(file_path)
file_ls = list()
for file_n in path_list:
    if file_n.endswith('.png'):
        full_path = file_path + file_n
        file_ls.append(full_path)

total_ls= []
for ii in file_ls[:]:
    id_n = ii.split('/')[-1].split('.')[0]
    print(id_n)
    im = imread(ii)[:,:,0:3]
    im_flr = np.fliplr(im)
    im_fud = np.flipud(im)
    im_flrud = np.fliplr(np.flipud(im))
    
    pred0 = overlap_pred(model,im,id_n,size=(512,512),overlap =(400,400))
    pred1 = overlap_pred(model,im_flr,id_n,size=(512,512),overlap =(400,400))
    pred2 = overlap_pred(model,im_fud,id_n,size=(512,512),overlap =(400,400))
    pred3 = overlap_pred(model,im_flrud,id_n,size=(512,512),overlap =(400,400))
    
    np.save('/mnt/disk6/nancy/paip_2023/data/for_ppt/tr_c002_1_R1.npy',pred0)
    np.save('/mnt/disk6/nancy/paip_2023/data/for_ppt/tr_c002_1_R2.npy',pred1)
    np.save('/mnt/disk6/nancy/paip_2023/data/for_ppt/tr_c002_1_R3.npy',pred2)
    np.save('/mnt/disk6/nancy/paip_2023/data/for_ppt/tr_c002_1_R4.npy',pred3)
    
    # cl0, bb0, mk0, sc0 = pred_to_list(pred0)
    # cl1, bb1, mk1, sc1 = pred_to_list(pred1)
    # cl2, bb2, mk2, sc2 = pred_to_list(pred2)
    # cl3, bb3, mk3, sc3 = pred_to_list(pred3)
      
    # bb1, mk1 = box_filp_trans(bb1,'fliplr',1024,1024), mask_filp_trans(mk1,'fliplr')    
    # bb2, mk2 = box_filp_trans(bb2,'flipud',1024,1024), mask_filp_trans(mk2,'flipud')    
    # bb3, mk3 = box_filp_trans(bb3,'fliplrud',1024,1024), mask_filp_trans(mk3,'fliplrud')
    
    # #pdb.set_trace()
    # cl_total = cl0+cl1+cl2+cl3
    # bb_total = bb0+bb1+bb2+bb3
    # mk_total = mk0+mk1+mk2+mk3
    # sc_total = sc0+sc1+sc2+sc3
    
    # np.save(save_path+id_n+'_cl.npy',cl_total)
    # np.save(save_path+id_n+'_bb.npy',bb_total)
    # np.save(save_path+id_n+'_mk.npy',mk_total)
    # np.save(save_path+id_n+'_sc.npy',sc_total)

    
    
    

    
    
    
    
    