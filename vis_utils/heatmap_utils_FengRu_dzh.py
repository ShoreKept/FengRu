import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.wsi_dataset import Wsi_Region
from utils.transform_utils import get_eval_transforms
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile  # 将注意力分数（score）映射到百分位数

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)  #从 slide_path 加载 WSI
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()  # OpenSlide 格式的 WSI, OpenSlide 可以解析多个病理扫描仪生成的专有 WSI 格式
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)  # 适合 1/32 下采样 的可视化层级（减少计算负担??）
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap  # coords：每个 Patch 在 WSI 上的坐标。
                    # scores：模型计算的 注意力分数，用于着色热图。
                    # vis_level：指定热图的分辨率。

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)  # 组织区域分割，去除背景噪声
    wsi_object.saveSegmentation(seg_mask_path)  #  加载 WSI 并进行组织分割
    return wsi_object

def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size'] 
    
    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):  #  roi：切片 Patch 图像数据。
        roi = roi.to(device)                                # coords：对应 Patch 在 WSI 中的位置。
        coords = coords.numpy()
        
        with torch.inference_mode():
            features = feature_extractor(roi)  # 计算 Patch 级特征

            if attn_save_path is not None:
                A = model(features, attention_only=True)
           
                if A.size(0) > 1: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1,1).cpu().numpy()

                if ref_scores is not None:
                   for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx].item(), ref_scores)[0]

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)  # 将 注意力分数 A 和 坐标 coords 存入 HDF5 文件。
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object
