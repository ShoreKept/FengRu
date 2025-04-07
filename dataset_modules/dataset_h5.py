import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
import sys 
from PIL import Image, ImageOps
import h5py
import os

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		print('注意下面的图片！')
		type(img)
		return {'img': img, 'coord': coord}
class Whole_Slide_Bag_FP_fengru(Dataset):
	def __init__(self,
		file_path,
  		h5_path
    	):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.h5_path = h5_path
		self.file_path = file_path
		with h5py.File(self.h5_path, "r") as f:
			dset = f['coords']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.h5_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')

	def __getitem__(self, idx):
		with h5py.File(self.h5_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = Image.open(self.file_path+'/'+str(coord[0]//256)+'_'+str(coord[1]//256)+'.jpg')
		img = np.array(img)
		# img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		# print(img.shape)
		img = img.transpose(2, 0, 1)
		img = img.astype(np.float32)
		# sys.exit()
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP_TCGA_LUAD(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.length = len(os.listdir(file_path))
        print(self.file_path)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        coords = os.listdir(self.file_path)
        coord = np.array([int(i) for i in coords[idx][:-4].split('_')])
        img = Image.open(os.path.join(self.file_path, coords[idx]))
        # if img.size != (256, 256):
        #     print('imgsize= ', img.size)
        #     print('coord=', coord)
        #     print("file_path = ", self.file_path)
        # if img.size != (256, 256):
        #     assert img.size == (256, 256), 'man!!!!!!!!!!!!!!!!!!!!!!!!!'
        #     print("MAN BO !!!!!!!!!!!!!!!!!!!")
        # try:
        #     assert img.size == (256, 256), "图像尺寸不符合要求！"
        # except AssertionError as e:
        #     print(f"断言失败：{e}"    )
        #     print(f"当前图像尺寸为 {img.size}")
        if img.size!=(256, 256):
            target_size = (256, 256)
            width, height = img.size
            delta_w = target_size[0] - width
            delta_h = target_size[1] - height
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            img = ImageOps.expand(img, padding, fill="white")
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)
        return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	  
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




