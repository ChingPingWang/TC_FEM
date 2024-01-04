import numpy as np
import matplotlib.pyplot as plt # 需安裝 pillow 才能讀 JPEG
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
import os
from scipy.ndimage import gaussian_filter,median_filter
from skimage.filters import gaussian

def blur(image,sigma):
	blurred = gaussian(image, sigma=(sigma, sigma), truncate=3.5)
	# im_gm = median_filter(im_g, size=6)
	return blurred
	
# K 值 (要保留的顏色數量)
K = 2

path = 'G:/PAIP_2023/data/PAIP2023_Test/'
all_list = os.listdir(path)

for n_img in all_list[:]:
	# 讀取圖片
	# image = imread(path+n_img) / 255
	img = imread(path+n_img)
	result1 = imread('G:/PAIP_2023/tta/result/mrcnn/test_111_tta_luo_png/'+n_img)
	result2 = imread('G:/PAIP_2023/result/test_phase/test_111_luo_knn_png/'+n_img)
	im_blr = blur(img,20)
	image = im_blr
	# plt.imshow(im_blr)
	# plt.show()
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

	# 儲存壓縮後的圖片
	#plt.imsave(r'C:\Users\使用者名稱\Downloads\compressed.jpg', image_compressed)

	# 顯示原圖跟壓縮圖的對照
	plt.figure(figsize=(12, 9))
	plt.subplot(231)
	plt.title('Original photo')
	plt.imshow(img)
	plt.subplot(232)
	plt.title('Original photo')
	plt.imshow(im_blr)
	plt.subplot(233)
	plt.title(f'Compressed to KMeans={K} colors')
	plt.imshow(image_compressed)
	plt.subplot(235)
	plt.title('mrcnn_att')
	plt.imshow(result1)
	plt.subplot(236)
	plt.title('mrcnn_att_knn')
	plt.imshow(result2)
	plt.tight_layout()
	plt.show()