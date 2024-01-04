import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import pdb

file_path = 'G:/PAIP_2023/data/PAIP_val/'

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
            new_box_list.append([nbx1, nbx2, nby1, nby2])
        if mode=='flipud':
            nbx1, nby1 = flip_matrix.dot(p1)[0]+h, flip_matrix.dot(p2)[1]
            nbx2, nby2 = flip_matrix.dot(p2)[0]+h, flip_matrix.dot(p1)[1]
            new_box_list.append([nbx1, nbx2, nby1, nby2])
        if mode=='fliplrud':
            nbx1, nby1 = flip_matrix.dot(p1)[0]+h, flip_matrix.dot(p1)[1]+w
            nbx2, nby2 = flip_matrix.dot(p2)[0]+h, flip_matrix.dot(p2)[1]+w
            new_box_list.append([nbx1, nbx2, nby1, nby2])
    return new_box_list

bb = [[87, 111, 197, 218],[136, 152, 9, 26],[270, 286, 412, 436]]
nb_ls = box_filp_trans(bb,'fliplr',1024,1024)
print(nb_ls)

path_list = os.listdir(file_path)
file_ls = list()
for file_n in path_list:
    if file_n.endswith('.png'):
        full_path = file_path + file_n
        file_ls.append(full_path)

total_ls= []
for ii in file_ls[0:1]:
    id_n = ii.split('/')[-1].split('.')[0]
    print(id_n)
    im = imread(ii)[:,:,0:3]
    plt.imshow(im)
    plt.show()

    # im_flr = np.fliplr(im)
    # plt.imshow(im_flr[bb[0][0]:bb[0][1],bb[0][2]:bb[0][3],:])
    # plt.show()
    # im_flr[bb[0][0]:bb[0][1],bb[0][2]:bb[0][3],1]=255
    # plt.imshow(im_flr)
    # plt.show()

    # im_fud = np.flipud(im)
    # plt.imshow(im_fud[bb[1][0]:bb[1][1],bb[1][2]:bb[1][3],:])
    # plt.show()
    # im_fud[bb[1][0]:bb[1][1],bb[1][2]:bb[1][3],1]=255
    # plt.imshow(im_fud)
    # plt.show()

    print(nb_ls[0][0],nb_ls[0][1],nb_ls[0][2],nb_ls[0][3])
    im[nb_ls[0][3]:nb_ls[0][2],nb_ls[0][1]:nb_ls[0][0],1]=255
    plt.imshow(im)
    plt.show()