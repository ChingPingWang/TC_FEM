# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:37:26 2021

@author: B1-410

To display colorful cells in the figure(include annotated cells)

Input: 
1. original image
2. prediction (npy), annotation (npy)
"""

import numpy as np
from skimage.io import imread, imshow, imsave
import cv2
import glob
from skimage.morphology import dilation, disk, closing, square, erosion
import pdb
import os
import matplotlib.pyplot as plt

def refine_box_axis(axis_title,L):
    if axis_title[0]>L or axis_title[1]>L or axis_title[2]>L or axis_title[3]>L:
            # print('Box(before)',axis_title)
            for i in range(4):
                if axis_title[0]<L and axis_title[1]==L and axis_title[2]==L and axis_title[3]>L:
                    axis_title[2]=0
                    axis_title[3]=axis_title[3]-L

                elif axis_title[i]>L:
                    axis_title[i] = axis_title[i]-L

                elif axis_title[0]==L and axis_title[1]>L:
                    axis_title[0]=0
                    axis_title[1]=axis_title[1]-L

                elif axis_title[1]==L and axis_title[0]<L:
                    axis_title[1]==L

                elif axis_title[3]==L and axis_title[2]<L:
                    axis_title[3]==L

                elif axis_title[i]==L:
                    axis_title[i] = 0
            # print('Box(After)',axis_title)
    else:
        axis_title = axis_title
    return axis_title

def contour_m(mask_bound,axis_title,im_shape):
    # bound_out = closing(mask_bound,disk(4)) 
    # bound_in = erosion(mask_bound,disk(2)) 
    # out_contour = bound_out-bound_in
    c = cv2.findContours(mask_bound,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    bound = (c[0][0][:,0,1],c[0][0][:,0,0])
    bound_mask= np.zeros(mask_bound.shape) 
    bound_mask[bound] = 255 
    bound_mask = dilation(bound_mask,disk(1)) 
    bound  = np.where(bound_mask == 255)
    out_contour = (bound[0] + axis_title[0],bound[1] + axis_title[2])
    out_contour[0][np.where(out_contour[0] >= im_shape[0])] = im_shape[0] - 1 
    out_contour[1][np.where(out_contour[1] >= im_shape[1])] = im_shape[1] - 1
    return out_contour

def contour_class_map(class_id,image,contour_map):
    if class_id == 1: #Yellow:model prediction (positive) (tumor)
        image[:,:,0][contour_map] = 255
        image[:,:,1][contour_map] = 255
        image[:,:,2][contour_map] = 0
        # plt.imshow(image)
        # plt.show()
    elif class_id == 2: #Blue:model prediction (negative) (non-tumor)
        image[:,:,0][contour_map] = 0
        image[:,:,1][contour_map] = 0
        image[:,:,2][contour_map] = 255
        # plt.imshow(image)
        # plt.show()
    return image

def check_no_ovl_npy_list(im_,label_): 
    im_label = im_.copy()
    im_shape = im_.shape
    bound_map = np.zeros((512,512))
    count_num = 0
    t_count, nont_count = 0, 0
    count_num_list = []
    for number_cell in range(label_.shape[0]):
        count_num = count_num+1
        class_id   = label_[number_cell,0]
        axis_title = label_[number_cell,1]
        axis_title = refine_box_axis(axis_title,512)
        mask_bound = label_[number_cell,2].astype(np.uint8)
        num_bound = 0
        if count_num == 1:
            bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
            count_num_list.append(count_num-1)
        else:
            instance_map = np.zeros((512,512))
            instance_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound

            overlapping = instance_map+bound_map
            if 2 in overlapping:
                overlap_rate = np.sum(overlapping==2)/np.sum(mask_bound==1)
                if overlap_rate>0.5:
                    pass
                else:
                    bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
                    # print(count_num)
                    count_num_list.append(count_num-1)
            else:
                bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
                # print(count_num)
                count_num_list.append(count_num-1)
    return count_num_list

def make_label_ovl(im_,label_): 
    im_label = im_.copy()
    im_shape = im_.shape
    bound_map = np.zeros((512,512))
    count_num = 0
    t_count, nont_count = 0, 0
    for number_cell in range(label_.shape[0]):
        count_num = count_num+1
        class_id   = label_[number_cell,0]
        axis_title = label_[number_cell,1]
        # print(axis_title)
        axis_title = refine_box_axis(axis_title,512)

        mask_bound = label_[number_cell,2].astype(np.uint8)
        num_bound = 0
        if count_num == 1:
            bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
            contour_map = contour_m(mask_bound,axis_title,im_shape)
            result_contour = contour_class_map(class_id,im_label,contour_map)
            if class_id==1: t_count=t_count+1
            elif class_id==2: nont_count=nont_count+1
        else:
            instance_map = np.zeros((512,512))
            instance_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound

            overlapping = instance_map+bound_map
            if 2 in overlapping:
                overlap_rate = np.sum(overlapping==2)/np.sum(mask_bound==1)
                if overlap_rate>0.5:
                    pass
                else:
                    bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
                    contour_map = contour_m(mask_bound,axis_title,im_shape)
                    result_contour = contour_class_map(class_id,im_label,contour_map)
                    if class_id==1: t_count=t_count+1
                    elif class_id==2: nont_count=nont_count+1
            else:
                bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
                contour_map = contour_m(mask_bound,axis_title,im_shape)
                result_contour = contour_class_map(class_id,im_label,contour_map)
                if class_id==1: t_count=t_count+1
                elif class_id==2: nont_count=nont_count+1
    return result_contour, t_count, nont_count

def check_sc_npy_list(im_,label_): 
    im_label = im_.copy()
    im_shape = im_.shape
    bound_map = np.zeros((512,512))
    bound_pd_map = np.zeros((512,512))
    count_num = 0
    t_count, nont_count = 0, 0
    count_num_list = []
    for number_cell in range(label_.shape[0]):
        count_num = count_num+1
        class_id   = label_[number_cell,0]
        axis_title = label_[number_cell,1]
        axis_title = refine_box_axis(axis_title,512)
        mask_bound = label_[number_cell,2].astype(np.uint8)
        pb_score = label_[number_cell,3]
        num_bound = 0
        if count_num == 1:
            bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
            count_num_list.append(count_num-1)

            ## probability
            bound_pd_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound*pb_score
        else:
            instance_map = np.zeros((512,512))
            instance_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
            overlapping = instance_map+bound_map
            # plt.imshow(overlapping)
            # plt.show()

            ## probability
            inst_pb_map = np.zeros((512,512))
            inst_pb_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound*pb_score

            if 2 in overlapping:
                overlap_rate = np.sum(overlapping==2)/np.sum(mask_bound==1)
                print('if 2 in overlapping:')
                print(np.sum(overlapping==2))
                print(np.sum(mask_bound==2))
                print(overlap_rate)
                f, ax = plt.subplots(1,3)
                ax[0].imshow(bound_map) #first image
                ax[1].imshow(instance_map)
                ax[2].imshow(overlapping)
                plt.show()
                if overlap_rate>0.5 and np.sum(mask_bound==1)!=0:
                    print('if overlap_rate>0.5 and np.sum(mask_bound==1)!=0:')
                    print(np.sum(overlapping==2))
                    print(np.sum(mask_bound==2))
                    print(overlap_rate)
                    f, ax = plt.subplots(1,3)
                    ax[0].imshow(bound_map) #first image
                    ax[1].imshow(instance_map)
                    ax[2].imshow(overlapping)
                    plt.show()
                else:
                    bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
                    # print(count_num)
                    count_num_list.append(count_num-1)
            else:
                bound_map[axis_title[0]:axis_title[1],axis_title[2]:axis_title[3]] = mask_bound
                # print(count_num)
                count_num_list.append(count_num-1)
    return count_num_list


def make_label_(im_,label_): 
    im_label = im_.copy()
    bound_map = np.zeros((512,512))
    count_num = 0
    for number_cell in range(label_.shape[0]):
        count_num = count_num+1
        class_id   = label_[number_cell,0]
        axis_title = label_[number_cell,1]
        mask_bound = label_[number_cell,2].astype(np.uint8)
        # print(mask_bound.shape)
        # plt.imshow(mask_bound)
        # plt.show()
        c = cv2.findContours(mask_bound,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #only accepted binary image
        bound = (c[0][0][:,0,1],c[0][0][:,0,0]) # position (x,y)
        bound_mask= np.zeros(mask_bound.shape) 
        bound_mask[bound] = 255 
        bound_mask = dilation(bound_mask,disk(1)) 
        bound  = np.where(bound_mask == 255)
        new_bound = (bound[0] + axis_title[0],bound[1] + axis_title[2])
        new_bound[0][np.where(new_bound[0] >= im_.shape[0])] = im_.shape[0] - 1 
        new_bound[1][np.where(new_bound[1] >= im_.shape[1])] = im_.shape[1] - 1 
        # plt.imshow(new_bound)
        # plt.show()

        if class_id == 1: #Yellow:model prediction (positive) (tumor)
            im_label[:,:,0][new_bound] = 255
            im_label[:,:,1][new_bound] = 255
            im_label[:,:,2][new_bound] = 0
            # plt.imshow(im_label)
            # plt.show()
        elif class_id == 2: #Blue:model prediction (negative) (non-tumor)
            im_label[:,:,0][new_bound] = 0
            im_label[:,:,1][new_bound] = 0
            im_label[:,:,2][new_bound] = 255
            # plt.imshow(im_label)
            # plt.show()
    return im_label

def calculate_TC_value(prediction):
    total_cell_num = len(predict)
    tumor_num = 0
    for ii in range(total_cell_num):
        cell_cls = predict[ii][0]
        if cell_cls == 1:
            tumor_num = tumor_num+1
    tc_value = tumor_num/total_cell_num
    non_tumor_num = total_cell_num-tumor_num
    return round(tc_value,2)*100, tumor_num, non_tumor_num

def save_submission_result(im_,label_): ### For PAIP challenge 2023
    im_label = im_.copy()
    cell_result = np.zeros([im_.shape[0],im_.shape[1]])
    for number_cell in range(label_.shape[0]):
        class_id   = label_[number_cell,0]
        axis_title = label_[number_cell,1]
        mask_bound = label_[number_cell,2].astype(np.uint8)
        bound = np.where(mask_bound==1)
        new_bound = (bound[0] + axis_title[0],bound[1] + axis_title[2])
        for ii in range(len(new_bound[0])):
            cell_result[new_bound[0][ii],new_bound[1][ii]] = class_id
    return cell_result

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
            nbx1, nby1 = flip_matrix.dot(p2)[0]+w, flip_matrix.dot(p1)[1]
            nbx2, nby2 = flip_matrix.dot(p1)[0]+w, flip_matrix.dot(p2)[1]
        if mode=='flipud':
            nbx1, nby1 = flip_matrix.dot(p1)[0], flip_matrix.dot(p2)[1]+h
            nbx2, nby2 = flip_matrix.dot(p2)[0], flip_matrix.dot(p1)[1]+h
        if mode=='fliplrud':
            nbx1, nby1 = flip_matrix.dot(p2)[0]+w, flip_matrix.dot(p2)[1]+h
            nbx2, nby2 = flip_matrix.dot(p1)[0]+w, flip_matrix.dot(p1)[1]+h
        new_box_list.append([nbx1, nbx2, nby1, nby2])
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

def regenerate_npy(predict,no_ovl_list):
    total_npy = []
    for num in range(len(no_ovl_list)):
        pd_num = no_ovl_list[num]
        npy_info = predict[pd_num]
        total_npy.append(npy_info)
    return np.stack(total_npy, axis=0)

# all_path ='G:/PAIP_2023/data/PAIP2023_Test/' ##ori image
all_path = 'G:/challenge/PAIP_2023/data/PAIP_in_test/'
# all_path = 'G:/PAIP_2023/data/PAIP_img/'
file_list = os.listdir(all_path)

all_list = []
for file_name in file_list:
    if file_name.endswith('.png'):
        full_path = all_path + file_name
        # print(full_path)
        all_list.append(full_path)
        # print(all_list)

# TC_ls=[]
# import pdb
# for file in all_list[:]: #Windows began from 0, Linux began from 2
#     image = imread(file)
#     image_name = file.split('/')[-1].split('.')[0]
#     predict_name = file.split('/')[-1].split('.')[0] + '.npy'
#     predict = np.load('G:/PAIP_2023/result/test_phase/test_152_luo_knn/'+ image_name + '.npy',allow_pickle=True) #cell npy
#     # pdb.set_trace()
#     #### 1. Display prediction --------------------------------------------------
#     result = make_label_(image, predict) ## use it when displayed the labeling
#     imsave('G:/PAIP_2023/result/test_phase/test_152_luo_knn_png/'+ image_name +'.png',result)

#     #### 2. Calculate TC value --------------------------------------------------
#     TC, tcell, ntcell = calculate_TC_value(predict)
#     # print('TC_value:',TC)
#     TC_ls.append([image_name,TC,tcell,ntcell])

#     ### 3. Submission format (.png) --------------------------------------------------
#     sb_png = save_submission_result(image, predict)
#     cv2.imwrite('G:/PAIP_2023/result/submission/test_phase/mrcnn/test_152_luo_knn/'+ image_name +'.png',sb_png)

# ### 4. Save ID and TC value
import pandas as pd
# csv_list = pd.DataFrame(TC_ls,columns=['ID','TC','tumor_cell','non-tumor_cell'])
# csv_list.to_csv("G:/PAIP_2023/result/test_inside/excel/test_152_mrcnn_knn_info.csv",index=False)

TC_ls=[]
import pdb
# count = -1
for file in all_list[:]: #Windows began from 0, Linux began from 2
    # # count = count+1
    # # print('count-----------',count)
    image = imread(file)
    image_name = file.split('/')[-1].split('.')[0]
    # # print(image_name)
    # predict_name = file.split('/')[-1].split('.')[0] + '.npy'
    predict = np.load('C:/Users/Vanessa/OneDrive/桌面/ScientificReport/result/npy/ensemble/MRtta_AMR/'+ image_name + '.npy',allow_pickle=True) #cell npy
    # # pdb.set_trace()
    # noovl_npy_list = check_no_ovl_npy_list(image, predict)
    # # sc_npy_list = check_sc_npy_list(image, predict)
    # # print(noovl_npy_list)
    # # print(len(noovl_npy_list))
    # # pdb.set_trace()
    # result, tcell, ntcell = make_label_ovl(image, predict)
    # # plt.imshow(result)
    # # plt.show()
    # # imsave('C:/Users/Vanessa/OneDrive/桌面/PAIP_inside_test/overlap_without_sc/png/AMR65tta/'+ image_name +'.png',result)

    # TC = tcell/(tcell+ntcell) 
    # TC = int(round(TC,2)*100)
    # # print('TC',TC,'Tnum', tcell, 'Ntnum', ntcell)
    # TC_ls.append([image_name,TC,tcell,ntcell])
    # csv_list = pd.DataFrame(TC_ls,columns=['ID','TC','Tnum','Ntnum'])
    # # csv_list.to_csv("C:/Users/Vanessa/OneDrive/桌面/PAIP_inside_test/overlap_without_sc/AMR65tta_info.csv",index=False)

    # noovl_pred = regenerate_npy(predict,noovl_npy_list)
    # np.save('C:/Users/Vanessa/OneDrive/桌面/PAIP_inside_test/overlap_without_sc/npy/AMR65tta/'+ image_name + '.npy',noovl_pred)
    
    
    #### 1. Display prediction --------------------------------------------------
    result = make_label_(image, predict) ## use it when displayed the labeling
    # result, tcell, ntcell = make_label_ovl(image, predict)

    # TC = tcell/(tcell+ntcell) 
    # TC = int(round(TC,2)*100)
    # # print('TC',TC,'Tnum', tcell, 'Ntnum', ntcell)
    # print(ntcell)

    # plt.imshow(result)
    # plt.show()
    imsave('G:/challenge/PAIP_2023/ensemble/MRtta_AMR_png/'+ image_name +'.png',result)