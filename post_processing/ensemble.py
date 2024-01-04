import numpy as np 
# import glob
# from natsort import natsorted
import pandas as pd
from skimage.io import imsave
import os

model1_path = 'G:/challenge/PAIP_2023/tta/result/inside/mrcnn/epoch111_npy/'
model2_path = 'G:/challenge/PAIP_2023/ensemble/MRtta_AMR/'
npy_file = os.listdir(model1_path)
for file_name in npy_file[:] :
	img = np.zeros([512,512])
	gt_sc = np.zeros([512,512])
	npy_1 = np.load(model1_path+file_name, allow_pickle=True)
	gt = np.load(model2_path+file_name, allow_pickle=True)

	for i in range(len(gt)):
		cla = int(gt[i][0])
		score = gt[i][3]
		if cla == int(1):
			box = gt[i][1]
			x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
			if x1 > 512 or x2 > 512 or y1 > 512 or y2 > 512:
				pass
			else :
				x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
				mask = gt[i][2]*1
				sc = gt[i][2]*score
				img[x1:x2, y1:y2] = mask
				gt_sc[x1:x2, y1:y2] = sc
		else :
			box = gt[i][1]
			x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
			if x1 > 512 or x2 > 512 or y1 > 512 or y2 > 512:
				pass
			else :
				mask = gt[i][2]*10
				sc = gt[i][2]*score
				img[x1:x2, y1:y2] = mask
				gt_sc[x1:x2, y1:y2] = sc

	save_npy = []
	for i in range(len(npy_1)):
		c_img = img.copy()
		pre = np.zeros([512,512])
		pre_sc = np.zeros([512,512])
		cla = int(npy_1[i][0])
		score = npy_1[i][3]
		if cla == 1:
			box = npy_1[i][1]
			x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
			if x1 > 512 or x2 > 512 or y1 > 512 or y2 > 512:
				pass
			else :
				x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
				mask = npy_1[i][2]*1
				sc = npy_1[i][2]*1
				pre[x1:x2, y1:y2] = mask
				pre_sc[x1:x2, y1:y2] = sc

		else :
			box = npy_1[i][1]
			x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
			if x1 > 512 or x2 > 512 or y1 > 512 or y2 > 512:
				pass
			else :
				mask = npy_1[i][2]*10
				sc = npy_1[i][2]*1
				pre[x1:x2, y1:y2] = mask
				pre_sc[x1:x2, y1:y2] = sc
		final = pre*c_img
		final_sc = pre_sc*gt_sc
		if np.any(final == 1) or np.any(final == 100):
			save_npy.append(npy_1[i])
		elif np.any(final == 10):
			fsc = np.max(final_sc)
			if np.max(final_sc) > np.max(pre_sc):
				for j in range(len(gt)):
					if np.max(final_sc) == gt[j][3]:
						save_npy.append(gt[j])
				pass
			else :
				save_npy.append(npy_1[i])
		else : 
			pass

	# print(save_npy[2])
	# np.save('G:/challenge/PAIP_2023/ensemble/MRtta_AMR/'+file_name, save_npy)
	np.save('G:/challenge/PAIP_2023/ensemble/MRtta_AMR_MRtta/'+file_name, save_npy)