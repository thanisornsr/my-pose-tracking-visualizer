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
		self.heatmaps = []
		self.valids = []
		self.thres = 15

	def add_data(self,to_add_imgs=None,to_add_heatmaps=None,to_add_valids=None):
		if to_add_imgs is not None:
			if to_add_imgs.shape[-1] == 3:
				self.imgs = to_add_imgs
				self.input_shape = (to_add_imgs.shape[1],to_add_imgs.shape[2])
			else:
				print('Wrong type of data [imgs]')
		if to_add_heatmaps is not None:
			if to_add_heatmaps.shape[-1] == 17:
				self.heatmaps = to_add_heatmaps
				self.output_shape = (to_add_heatmaps.shape[1],to_add_heatmaps.shape[2])
			else:
				print('Wrong type of data [heatmaps]')
		if to_add_valids is not None:
			if to_add_valids.shape[-1] == 17:
				self.valids = to_add_valids
			else:
				print('Wrong type of data [valids]')
		else:
			if to_add_heatmaps is not None:
				self.valids = self.get_valids()
			else:
				print('Please add heatmaps')
	def get_valids(self):
		pthres = self.thres
		pheatmap = self.heatmaps
		pvalid = np.empty(pheatmap.shape)
		poutput_shape = self.output_shape
		for j in range(pvalid.shape[0]):
			for k in range(pvalid.shape[-1]):
				temp_max = np.max(pheatmap[j,:,:,k])
				if temp_max > pthres:
					pvalid[j,:,:,k] = np.ones(poutput_shape)
				else:
					pvalid[j,:,:,k] = np.zeros(poutput_shape)
		return pvalid

	def set_valid_threshold(self,new_thres):
		self.thres = new_thres

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
		temp_valid2 = self.valids[idx_kp,0,0,:]
		crop_resize = resize(temp_img2, self.output_shape)
		kp_x = []
		kp_y = []
		for j in range(17):
			if temp_valid2[j] > 0:
				result = np.where(temp_heatmap2[:,:,j] == np.amax(temp_heatmap2[:,:,j]))
				ty = result [0][0]
				tx = result [1][0]
			else:
				ty = -1
				tx = -1
			kp_x.append(tx)
			kp_y.append(ty)
		implot = plt.imshow(crop_resize)
		for xs,ys in zip(kp_x,kp_y):
			if xs >= 0 and ys >= 0:
				plt.scatter(x = xs, y = ys, c = 'r', s = 15)
		plt.show()

	def show_skeleton(self,idx):
		temp_img3 = np.reshape(self.imgs[idx,:,:,:],(*self.input_shape,3))
		temp_heatmap3 = np.reshape(self.heatmaps[idx,:,:,:],(*self.output_shape,17))
		temp_valid3 = self.valids[idx,0,0,:]
		crop_resize = resize(temp_img3, self.output_shape)
		kp_x = []
		kp_y = []
		for j in range(17):
			if temp_valid3[j] > 0:
				result = np.where(temp_heatmap3[:,:,j] == np.amax(temp_heatmap3[:,:,j]))
				ty = result [0][0]
				tx = result [1][0]
			else:
				ty = -1
				tx = -1
			kp_x.append(tx)
			kp_y.append(ty)

		if self.dataset_name == 'Posetrack':
			skeleton_list = [(0,1),(0,2),(0,3),(0,4),(5,6),(5,7),(6,8),(7,9),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]
			color_list = ['b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c']
		else:
			skeleton_list = [(0,3),(0,4),(1,2),(5,6),(5,7),(6,8),(7,9),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]
			color_list = ['b','g','r','c','m','y','b','g','r','c','m','y','b','g','r']

		ci = 0
		for sk in skeleton_list:
			p1,p2 = sk
			x1 = kp_x[p1]
			y1 = kp_y[p1]
			x2 = kp_x[p2]
			y2 = kp_y[p2]
			tc = color_list[ci]
			ci = ci + 1
			if (x1 >= 0) and (x2 >= 0) and (y1 >= 0) and (y2 >= 0):
				plt.plot([x1,x2],[y1,y2], color = tc, linewidth = 3)
		plt.imshow(crop_resize)
		plt.show()

