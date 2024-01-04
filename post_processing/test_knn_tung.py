import numpy as np 
from natsort import natsorted
import glob
# npy = np.load('E:\\PAIP2023\\test\\mask_atten_96\\npy\\tr_c001_4.npy',allow_pickle=True)
# n_npy = npy.copy()
# print(len(npy))
# t_sc = []
# nt_sc = []
# for i in range(len(npy)):
# 	cla = int(npy[i][0])
# 	sc = npy[i][3]
# 	if cla == 1:
# 		t_sc.append(sc)
# 	else:
# 		nt_sc.append(sc)

# print(t_sc)
# print(nt_sc)
# npy_file = sorted(glob.glob('E:\\PAIP2023\\test\\mask_atten_96\\npy\\*.npy'))


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
			# print(O_list)
	O_list = natsorted(O_list)
	# print(O_list)
	return O_list

def KNN_2(value,O_list,npy,n_npy,i):
	cla_list = []
	for k in range(value):
		# print(O_list[k][0])
		cla = npy[O_list[k][1]][0]
		# print(cla)
		# print(O_list[k][1])
		# if int(O_list[k][0]) < 50:
		cla_list.append(cla)

	if np.sum(cla_list) >= 25 or np.sum(cla_list) <= 16:
		if np.sum(cla_list) >= 25 :
			n_npy[i][0] = 2
		else :
			n_npy[i][0] = 1
	elif np.sum(cla_list) < 25 or np.sum(cla_list) > 16:
		cla = npy[i][0]
		sc = int(npy[i][3])
		# if npy == 1 and sc <= 0.84:
		# 	# n_sc = sc - (sc*0.29)
		# 	npy[i][0] = 2
		# elif npy == 2 and sc <= 0.84:
		# 	npy[i][0] = 1
	else:
		pass

	return n_npy

def KNN_1(value,O_list,npy,n_npy,i):
	cla_list = []
	# n_npy = npy.copy()
	for k in range(value):
		cla = npy[O_list[k][1]][0]
		# print(O_list[k][1])
		cla_list.append(cla)
	if np.sum(cla_list) >= 9 :
		if np.sum(cla_list) >= 14 :
			n_npy[i][0] = 2
		else :
			n_npy[i][0] = 1
	else:
		pass
	return n_npy

def KNN(value,O_list,npy,i):
	cla_list = []
	# n_npy = npy.copy()
	for k in range(value):
		cla = npy[O_list[k][1]][0]
		# print(O_list[k][1])
		cla_list.append(cla)
	if np.sum(cla_list) >=5 :
	# if np.sum(cla_list) >= 9 :
		n_npy[i][0] = 2
	else :
		n_npy[i][0] = 1
	# else:
	# 	pass
	return n_npy

def ch(npy,i):
	sc = int(npy[i][3])
	cla = int(npy[i][0])
	if sc < 0.84 and cla == 1:
		n_npy[i][0] = 2
	elif sc < 0.84 and cla == 2:
		n_npy[i][0] = 1
	else : 
		pass
	return n_npy
# npy_file = sorted(glob.glob('E:\\PAIP2023\\test\\mask_atten_96\\tta\\96\\epoch96_npy\\*.npy'))
# npy_file = sorted(glob.glob('E:\\PAIP2023\\test\\mask_atten_96\\new\\105\\*.npy'))
npy_file = sorted(glob.glob('E:\\PAIP2023\\test\\tta_65\\*.npy'))
count = 0
O_list = []
# print(npy_file)
for file in npy_file:
# for file in npy_file[0:1]:
	name = file.split('\\')[-1]
	print(name)
	# npy = np.load('E:\\PAIP2023\\test\\mask_atten_96\\tta\\96\\epoch96_npy\\'+name,allow_pickle=True)
	n_file = np.load('E:\\PAIP2023\\test\\tta_65\\'+name,allow_pickle=True)
	n_npy_1 = n_file.copy()
	# print(n_npy_1.shape, n_file.shape)
	# print(n_file[0][3] == n_npy_1[0][3])
	# print(n_npy_1[0][0])
	# print(n_npy_1 == n_file)
	for i in range(len(n_file)):
		O_list = Distance(n_file,i)
		n_npy_1 = KNN_2(15,O_list,n_file,n_npy_1,i)

	n_npy_2 = n_npy_1.copy()
	for i in range(len(n_file)):
		O_list = Distance(n_file,i)
		n_npy_2 = KNN_1(8,O_list,n_npy_1,n_npy_2,i)

	n_npy_3 = n_npy_2.copy()
	for i in range(len(n_file)):
		O_list = Distance(n_file,i)
		n_npy_3 = KNN_1(8,O_list,n_npy_2,n_npy_3,i)

		# n_npy_3 = KNN_2(15,O_list,n_npy_2,i)
		# n_npy = KNN(8,O_list,n_npy,i)
		# n_npy = KNN_1(15,O_list,n_npy,i)

	# np.save('E:\\PAIP2023\\test\\mask_atten_96\\tta\\96\\new_npy\\'+name,n_npy_2)
	# np.save('E:\\PAIP2023\\test\\mask_atten_96\\new\\6\\6\\new_npy\\'+name,n_npy_2)
	# np.save('E:\\PAIP2023\\test\\mask_atten_96\\new\\6\\6_3\\new_npy\\'+name,n_npy_3)
	np.save('E:\\PAIP2023\\mrcnn_attu_65\\tta_KNN\\new_npy\\'+name,n_npy_1)
