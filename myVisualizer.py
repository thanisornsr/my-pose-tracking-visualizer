from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import math


class Posetrack_Visualizer:
	def __init__(self,dataset_name='Posetrack'):
		self.dataset_name = dataset_name
		self.input_shape = None
		self.output_shape = None
		self.imgs = []
		self.heatmapts = []
		self.valids = []

	def add_data(self,to_add_imgs=None,to_add_heatmaps=None,to_add_valids=None):
		if to_add_imgs is not None:
			if to_add_imgs.shape[-1] == 3:
				self.imgs = to_add_imgs
				self.input_shape = (to_add_imgs.shape[1],to_add_imgs.shape[2])
			else:
				print('Wrong type of data [imgs]')
		if to_add_heatmaps is not None:
			if to_add_heatmaps[-1] == 17:
				self.heatmapts = to_add_heatmaps
				self.output_shape = (to_add_heatmaps.shape[1],to_add_heatmaps.shape[2])
			else:
				print('Wrong type of data [heatmaps]')
		if to_add_valids is not None:
			if to_add_valids[-1] == 17:
				self.valids = to_add_valids
			else:
				print('Wrong type of data [valids]')
	def print_valid(self,idx_valid):
		print(self.valids[idx_valid,0,0,:])
	def show_heatmap(self,idx_hm):
		temp_img = np.reshape(self.imgs[idx_hm,:,:,:],(*self.input_shape,3))
		temp_heatmap = np.reshape(self.heatmaps[idx_hm,:,:,:],(*self.output_shape,17))
		crop_resize = resize(temp_img, self.output_shape)

		fig, axs = plt.subplots(3,6)
		if self.dataset_name == 'Posetrack':
			joint_name = ["nose", "head_bottom", "head_top", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
		else:
			joint_name = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
		for k in range(18):
			i = math.floor(k/6)
			j = k%6
			if k == 0:
				axs[i,j].imshow(crop_resize)
				axs[i,j].set_title('Cropped img', fontdict={'fontsize' : 8})
				axs[i,j].axis('off')
			else:
				axs[i,j].imshow(temp_heatmap[:,:,k-1],cmap='hot',interpolation ='nearest')
				axs[i,j].set_title(joint_name[k-1], fontdict={'fontsize' : 8})
				axs[i,j].axis('off')

	def show_keypoints(self,idx_kp):
		temp_img2 = np.reshape(self.imgs[idx_kp,:,:,:],(*self.input_shape,3))
		temp_heatmap2 = np.reshape(self.heatmaps[idx_kp,:,:,:],(*self.output_shape,17))
		crop_resize = resize(temp_img2, self.output_shape)
		kp_x = []
		kp_y = []
		for j in range(17):
			result = np.where(temp_heatmap2[:,:,j] == np.amax(temp_heatmap2[:,:,j]))
			ty = result [0][0]
			tx = result [1][0]
			kp_x.append(tx)
			kp_y.append(ty)
		implot = plt.imshow(crop_resize)
		plt.scatter(x = kp_x, y = kp_y,c = 'r',s = 15)
		plt.show()

	# def show_skeleton(self,idx):
	# 	temp_img3 = np.reshape(self.imgs[idx,:,:,:],(*self.input_shape,3))
	# 	temp_heatmap3 = np.reshape(self.heatmaps[idx,:,:,:],(*self.output_shape,17))
	# 	crop_resize = resize(temp_img3, self.output_shape)
	# 	kp_x = []
	# 	kp_y = []
	# 	for j in range(17):
	# 		result = np.where(temp_heatmap3[:,:,j] == np.amax(temp_heatmap3[:,:,j]))
	# 		ty = result [0][0]
	# 		tx = result [1][0]
	# 		kp_x.append(tx)
	# 		kp_y.append(ty)

	# 	if self.dataset_name == 'Posetrack':
	# 		skeleton_list = [(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(5,7),(6,8),(7,9),(8,10),(6,12),(5,11),(12,11),(12,14),(11,13),(14,16),(13,15)]
	# 	else:
	# 		skeleton_list = [(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(5,7),(6,8),(7,9),(8,10),(6,12),(5,11),(12,11),(12,14),(11,13),(14,16),(13,15)]

	# 	for sk in skeleton_list:
	# 		p1,p2 = sk
	# 		x1 = kp_x[p1]
	# 		y1 = kp_y[p1]
	# 		x2 = kp_x[p2]
	# 		y2 = kp_y[p2]