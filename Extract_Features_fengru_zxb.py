import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP_fengru
from models import get_encoder


from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True #我加的
Image.MAX_IMAGE_PIXELS = None  # 将其设为 None，以解除限制（但慎用）

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):   #verbose冗长
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)  #不阻塞的方式会提高并行能力
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.tiff')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='conch_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=512)  #修改
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=256)  #修改
args = parser.parse_args()

classNames = ['Other', 'Wild']


if __name__ == '__main__':
	print('initializing dataset')
	feat_path = '/home/graduate2024/data_new/Extracted_Features_fengru'
	data_path = '/home/graduate2024/data_new/graduate2024_old/zxb_EGFR_Slices'
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)	
	_ = model.eval()
	model = model.to(device)
	for className in classNames:
		print(className+' begin!!!')
		cls_feat_path = os.path.join(feat_path, className, 'h5_files')
		cls_data_path = os.path.join(data_path, className)
		os.makedirs(cls_feat_path, exist_ok=True)
		slide_ids = os.listdir(cls_data_path)
		

		total = len(slide_ids)

		loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
		
		for idx in tqdm(range(total)):
			slide_id = slide_ids[idx]
			slide_feat_name = slide_id+'.h5'
			# h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			print('\nprogress: {}/{}'.format(idx, total))


			time_start = time.time()
	##修改生成dataset的函数
			slide_h5_path = os.path.join(cls_data_path, slide_id, slide_id+'_h5.h5')
			wsi_path = os.path.join(cls_data_path, slide_id)
			dataset = Whole_Slide_Bag_FP_fengru(file_path = wsi_path, 
												h5_path = slide_h5_path
												)

			loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
			output_file_path = compute_w_loader(os.path.join(cls_feat_path, slide_id+'.h5'), loader = loader, model = model, verbose = 1)

			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))




