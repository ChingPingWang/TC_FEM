from skimage.io import imread,imshow,imsave
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os

def tumor_dilation(img,tumor):
	kernel = np.ones((7,7), np.uint8)
	dilation = cv2.dilate(tumor, kernel, iterations = 1)
	final = dilation - tumor
	num = 1
	while num <= final.max() :
		area = np.where(final == num)
		for i in range(len(area[0])):
			w = area[0][i]
			h = area[1][i]
			# print(w,h)
			img[w,h,0] = 255
			img[w,h,1] = 255
			img[w,h,2] = 0
		num += 1
	return img

def non_tumor_dilation(img,non_tumor):
	kernel = np.ones((7,7), np.uint8)
	dilation = cv2.dilate(non_tumor, kernel, iterations = 1)
	final = dilation - non_tumor
	num = 1
	while num <= final.max() :
		area = np.where(final == num)
		for i in range(len(area[0])):
			w = area[0][i]
			h = area[1][i]
			# print(w,h)
			img[w,h,0] = 0
			img[w,h,1] = 0
			img[w,h,2] = 255
		num += 1
	return img

file_path = 'G:/PAIP_2023/data/PAIP_in_test/'
path_list = os.listdir(file_path)
im_file = list()
for file_n in path_list:
	if file_n.endswith('.png'):
		full_path = file_path + file_n
		im_file.append(full_path)

for image in im_file:
	name = image.split('/')[-1]
	picture = image.split('/')[-1].split('_')[0]+'_'+image.split('/')[-1].split('_')[1]
	# non_name = image.split('/')[-1].split('_')[0]+'_'+image.split('/')[-1].split('_')[1]+'_non'+image.split('/')[-1].split('_')[2]
	print(picture)
	img = imread(image)
	# print(name)
	tumor = imread('G:/PAIP_2023/data/PAIP_masks/tumor/'+ picture[:-4] +'_tumor.png')
	non_tumor =imread('G:/PAIP_2023/data/PAIP_masks/non_tumor/'+ picture[:-4] +'_nontumor.png')
	img = tumor_dilation(img,tumor)
	img = non_tumor_dilation(img,non_tumor)
	imsave('G:/PAIP_2023/document/extend_abstract/fig/gt/'+name,img)
		
# plt.imshow(img)
# plt.show()
# plt.imshow(dilation)
# plt.show()
# plt.imshow(img)
# plt.show()