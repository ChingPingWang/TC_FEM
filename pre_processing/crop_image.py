import numpy as np
from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt

def img_crop_position_list(scale_x,scale_y,crop_len,stride):
	pos_list = []
	for num_x in range(0,scale_x,stride):
		for num_y in range(0,scale_y,stride):
			len_x = num_x + crop_len
			len_y = num_y + crop_len
			##----Position x----##
			if len_x > scale_x:
				x1 = scale_x - crop_len
			else:
				x1 = num_x
			##----Position y----##
			if len_y > scale_y:
				y1 = scale_y - crop_len
			else:
				y1 = num_y
			each_pos = [x1,y1]
			pos_list.append(each_pos)
	return pos_list

rd_path = 'G:/PAIP_2023/data/PAIP_gt_custom/' ##Step1: Put resample image
fl_list = os.listdir(rd_path)

for fname in fl_list[:]:
	id_n = fname[:-4]
	im = imread(rd_path+fname)
	sizex, sizey = im.shape[0], im.shape[1]
	pos_ls = img_crop_position_list(sizex,sizey,crop_len=512, stride=448) ## stride=crop_len*0.875(7/8)

	count = 0
	for list_num in pos_ls[count:]:
		count = count+1
		pos_x0, pos_x1 = list_num[0], list_num[0]+512
		pos_y0, pos_y1 = list_num[1], list_num[1]+512
		crop_im = im[pos_y0:pos_y1, pos_x0:pos_x1]
		plt.imshow(crop_im)
		plt.show()
		imsave('G:/PAIP_2023/data/PAIP_gt_custom/'+ id_n +'_'+ str(count) +'.png',crop_im) ##Step2: Set the save path