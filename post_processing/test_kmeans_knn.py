import numpy as np
import os
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from natsort import natsorted


def blur(image,sigma):
	blurred = gaussian(image, sigma=(sigma, sigma), truncate=3.5)
	return blurred

def kmeans(image,K):
	w, h, d = tuple(image.shape)
	image_data = np.reshape(image, (w * h, d))

	# 將顏色分類為 K 種
	kmeans = MiniBatchKMeans(n_clusters=K, batch_size=5000)
	labels = kmeans.fit_predict(image_data)
	centers = kmeans.cluster_centers_

	# 根據分類將顏色寫入新的影像陣列
	image_compressed = np.zeros(image.shape)
	label_idx = 0
	for i in range(w):
		for j in range(h):
			image_compressed[i][j] = centers[labels[label_idx]]
			label_idx += 1
	binary_imgR = image_compressed[:,:,0]
	maxR, minR = binary_imgR.max(), binary_imgR.min()
	binary_imgR[binary_imgR==maxR] = 1
	binary_imgR[binary_imgR==minR] = 0
	return binary_imgR

def classify_group(cell_pred,area):
	aarea_cell = list() #tumor
	barea_cell = list() #non-tumor
	for num_cell in cell_pred:
		off_set = num_cell[1]
		off_set_x = off_set[0]
		off_set_y = off_set[2]
		cell_boundary = num_cell[2]
		cell_pos = np.where(cell_boundary)
		cell_pos_x = cell_pos[0] + off_set_x
		cell_pos_y = cell_pos[1] + off_set_y
		flag = 1
		for x,y in zip(cell_pos_x,cell_pos_y):
			flag = flag * area[x,y]
			if flag == 0:
				barea_cell.append(num_cell)
			else:
				aarea_cell.append(num_cell)
			break
	return aarea_cell, barea_cell

def KNN(value,O_list,npy,n_npy,i):
	cla_list, dis_list = [], []
	for k in range(value):
		cla = npy[O_list[k][1]][0]
		dis = O_list[k][0]
		cla_list.append(cla)
		dis_list.append(dis)
	for ii in range(1,value):
		dis_len = dis_list[ii]-dis_list[0]
		c_1, c_2 = cla_list.count(1), cla_list.count(2)
		if c_1>c_2:
			n_npy[i][0] = 1
		else:
			n_npy[i][0] = 2
	return n_npy

def Distance(npy,num):
	O_list = []
	x0, y0 = int((npy[num][1][1]+npy[num][1][0])/2), int((npy[num][1][3]+npy[num][1][2])/2)
	for i in range(len(npy)):
		if i == num :
			pass
		else :
			x1, y1 = int((npy[i][1][1]+npy[i][1][0])/2), int((npy[i][1][3]+npy[i][1][2])/2)
			o = (((x0-x1)**2)+((y0-y1)**2))**0.5
			O_list.append([o,i])
	O_list = natsorted(O_list)
	return O_list

def KNN_algorithm(value, pred):
	pred_c = pred.copy()
	if len(pred)<=value:
		pred_c = pred_c
		print('yes')
	else:
		for cell_num in range(len(pred)):
			O_list = Distance(pred,cell_num)
			pred_c = KNN(value,O_list,pred,pred_c,cell_num)
	return pred_c

def combined(knn_resulta,knn_resultb):
	total_ls=[]
	for i in range(len(knn_resulta)):
		clasa = knn_resulta[i][0]
		boxxa = knn_resulta[i][1]
		maska = knn_resulta[i][2]
		scora = knn_resulta[i][3]
		total_ls.append([clasa,boxxa,maska,scora])
	for j in range(len(knn_resultb)):
		clasb = knn_resultb[j][0]
		boxxb = knn_resultb[j][1]
		maskb = knn_resultb[j][2]
		scorb = knn_resultb[j][3]
		total_ls.append([clasb,boxxb,maskb,scorb])
	return np.array(total_ls)

all_path ='G:/PAIP_2023/data/PAIP2023_Test/' ##ori image
npy_path = 'G:/PAIP_2023/result/test_phase/test_152_luo/'
file_list = os.listdir(all_path)

all_list = []
for file_name in file_list:
	if file_name.endswith('.png'):
		full_path = all_path + file_name
		all_list.append(full_path)

import pdb
K=3
count = 0
for file in all_list[:]:
	image = imread(file)
	image_name = file.split('/')[-1].split('.')[0]
	print(image_name)
	im_blr = blur(image,30)
	kmeans_img = kmeans(im_blr,2)
	pred = np.load(npy_path+image_name+'.npy',allow_pickle=True)
	pred_a, pred_b = classify_group(pred,kmeans_img) ## only can use for kmeans_img ##If needed, users must to change the class for use
	# pdb.set_trace()
	knn_resulta = KNN_algorithm(K,np.array(pred_a))
	knn_resultb = KNN_algorithm(K,np.array(pred_b))
	knn_result = combined(knn_resulta,knn_resultb)
	np.save('G:/PAIP_2023/result/test_phase/test_152_luo_knn/'+image_name+'.npy',knn_result)

	# gt = imread('G:/PAIP_2023/data/PAIP_gt_custom/test/'+image_name+'.png.png')
	# ori_pred = imread('G:/PAIP_2023/result/test_inside/test_111_luo_png/'+image_name+'.png')
	# ori_knn_pred = imread('G:/PAIP_2023/result/test_inside/test_111_luo_3kk_png/'+image_name+'.png')
	# plt.figure(figsize=(12, 9))
	# plt.subplot(234)
	# plt.title('Original')
	# plt.imshow(image)
	# plt.subplot(232)
	# plt.title('blurred')
	# plt.imshow(im_blr)
	# plt.subplot(233)
	# plt.title('kmeans_img')
	# plt.imshow(kmeans_img)
	# plt.subplot(231)
	# plt.title('Ground Truth')
	# plt.imshow(gt)
	# plt.subplot(235)
	# plt.title('Original pred')
	# plt.imshow(ori_pred)
	# plt.subplot(236)
	# plt.title('knn pred')
	# plt.imshow(ori_knn_pred)
	# plt.show()

	# result = np.load('G:/PAIP_2023/result/test_phase/test_111_luo_knn/'+ image_name + '.npy',allow_pickle=True)
	# r = tumor_cell_only(result,kmeans_img)
	# np.save('G:/PAIP_2023/result/test_phase/test_111_luo_knn_classify/' + image_name + '.npy', r)
