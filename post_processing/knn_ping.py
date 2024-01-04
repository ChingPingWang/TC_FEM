import numpy as np 
from natsort import natsorted
import os

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

# def KNN(value,O_list,npy,n_npy,i):
# 	cla_list = []
# 	for k in range(value):
# 		cla = npy[O_list[k][1]][0]
# 		cla_list.append(cla)
# 	if np.sum(cla_list) >= 10 or np.sum(cla_list) <= 5:
# 		if np.sum(cla_list) >= 10 :
# 			n_npy[i][0] = 2
# 		else :
# 			n_npy[i][0] = 1
			
# 	elif np.sum(cla_list) < 10 or np.sum(cla_list) > 5:
# 		cla = npy[i][0]
# 		sc = int(npy[i][3])
# 	else:
# 		pass
# 	return n_npy
import pdb

def KNN(value,O_list,npy,n_npy,i):
	n_npy = npy.copy()
	cla_list, dis_list = [], []
	for k in range(value):
		cla = npy[O_list[k][1]][0]
		dis = O_list[k][0]
		cla_list.append(cla)
		dis_list.append(dis)
	for ii in range(1,value):
		dis_len = dis_list[ii]-dis_list[0]
		# c_1, c_2 = cla_list.count(1), cla_list.count(2)
		# if c_1>c_2:
		# 	n_npy[i][0] = 1
		# else:
		# 	n_npy[i][0] = 2
		while dis_len>=100:
			n_npy[i][0] = cla_list[0]
			break
		else:
			c_1, c_2 = cla_list.count(1), cla_list.count(2)
			if c_1>c_2:
				n_npy[i][0] = 1
			else:
				n_npy[i][0] = 2
	return n_npy

npy_path = 'G:/PAIP_2023/result/test_phase/test_111_luo/'

npy_file = os.listdir(npy_path)
O_list = []
for file in npy_file[0:1]:
	pred = np.load(npy_path+file,allow_pickle=True)
	pred_c = pred.copy()
	for cell_num in range(len(pred)):
		O_list = Distance(pred,cell_num)
		knn_result = KNN(6,O_list,pred,pred_c,cell_num)
	np.save('G:/PAIP_2023/result/test_phase/test_111_luo_knn/'+file,knn_result)
