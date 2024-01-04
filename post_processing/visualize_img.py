import numpy as np
import matplotlib.pyplot as plt # 需安裝 pillow 才能讀 JPEG
from skimage.io import imread
import os

all_path = 'G:/PAIP_2023/data/PAIP2023_Test/'
im_ls = os.listdir(all_path)
ls_name=[]

for ii in im_ls[:]:
	if ii.endswith('.png'):
		ls_name.append(ii)

for id_n in ls_name[0:1]:
	a = imread('G:/PAIP_2023/data/PAIP2023_Test/'+id_n)
	b = imread('G:/PAIP_2023/result/test_phase/test_111_luo_png/'+id_n)
	c = imread('G:/PAIP_2023/result/test_phase/test_111_luo_knn_png/'+id_n)
	d = imread('G:/PAIP_2023/tta/result/mrcnn/test_111_tta_luo_png/'+id_n)
	# tta_knn = imread(all_path+'tta_knn/'+id_n)
	# 顯示
	plt.figure(figsize=(12, 9))
	plt.subplot(221)
	plt.title(id_n)
	plt.imshow(a)
	plt.subplot(222)
	plt.title('ori')
	plt.imshow(b)
	plt.subplot(223)
	plt.title('knn')
	plt.imshow(c)
	plt.subplot(224)
	plt.title('tta')
	plt.imshow(d)
	plt.tight_layout()
	plt.show()

# for id_n in ls_name[:]:
	#MRCNN
	# a = imread('G:/PAIP_2023/result/others/ping/mrcnn/epoch5/final_test/ori/'+id_n)
	# b = imread('G:/PAIP_2023/result/others/ping/mrcnn/epoch67/final_test/ori/'+id_n)
	# # c = imread('G:/PAIP_2023/result/others/ping/mrcnn/epoch111/final_test/ori/'+id_n)
	# # #MRCNN_att
	# # d = imread('G:/PAIP_2023/result/others/ping/mrcnn_att/epoch65/final_test/ori/'+id_n)
	# # e = imread('G:/PAIP_2023/result/others/ping/mrcnn_att/epoch96/final_test/ori/'+id_n)
	# # f = imread('G:/PAIP_2023/result/others/ping/mrcnn_att/epoch105/final_test/ori/'+id_n)
	# # 顯示
	# plt.figure(figsize=(15, 9))
	# plt.subplot(231)
	# plt.title('mr_5_'+id_n)
	# plt.imshow(a)
	# plt.subplot(232)
	# plt.title('mr_67')
	# plt.imshow(b)
	# plt.subplot(233)
	# plt.title('mr_111')
	# plt.imshow(c)
	# plt.subplot(234)
	# plt.title('mrA_65')
	# plt.imshow(d)
	# plt.subplot(235)
	# plt.title('mrA_96')
	# plt.imshow(e)
	# plt.subplot(236)
	# plt.title('mrA_105')
	# plt.imshow(f)
	# plt.tight_layout()
	# plt.show()