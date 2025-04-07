from __future__ import print_function
# ... (previous imports remain the same)
from sklearn.metrics import f1_score, recall_score
# ... (keep all the code until the main section)
import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils1 import *  #修改
from math import floor
import matplotlib.pyplot as plt
#修改
from dataset_modules.dataset_generic_1 import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils_new_fengru import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'EGFR_subtyping', 'EGFR_Four_subtyping_fengru', 'EGFR_Five_subtyping_fengru'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'EGFR_subtyping':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/home/graduate2024/data/CLAM_csv/EGFR_CLAM.csv',
                            data_dir= os.path.join(args.data_root_dir, 'DataSet1'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Normal':0, 'L858R':1, 'Other':2, '19DEL':3, 'Wild':4},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'EGFR_Four_subtyping_fengru':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = '/home/graduate2024/data_new/CLAM_csv/EGFR_CLAM.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Normal':0, '19DEL':1, 'L858R':2, 'ELSE':3},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'EGFR_Five_subtyping_fengru':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/home/graduate2024/data_new/CLAM_csv/EGFR_CLAM_five.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Normal':0, '19DEL':1, 'L858R':2, 'Other':3, 'Wild':4},
                            patient_strat= False,
                            ignore=[])


else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}


if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    all_recall = []
    all_cm = []
    all_labels_b=[]
    all_probs=[]
    
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
            
        model, patient_results, test_error, Auc, f1, recall, df, cm, fold_label, fold_prob  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        from sklearn.preprocessing import label_binarize 
        # print(f"fold_label:{fold_label}")
        fold_label_b=label_binarize(fold_label,classes=[i for i in range(args.n_classes)])    #要修改
        # print(f"fold_label_b:{fold_label_b}")

        all_results.append(all_results)
        all_auc.append(Auc)
        all_acc.append(1-test_error)
        all_f1.append(f1)
        all_recall.append(recall)
        all_cm.append(cm)
        
        all_labels_b.append(fold_label_b)
        all_probs.append(fold_prob)
        
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
    
    
    # print(f"all_labels_b:{all_labels_b}")
    for i, label in enumerate(all_labels_b):  
        print(f"第 {i} 个 fold_label_b 的形状: {label.shape}")  
    #绘制ROC曲线
    label_3d = np.stack(all_labels_b, axis=0)
    prob_3d=np.stack(all_probs,axis=0)
    fpr_all = []
    tpr_all = []
    all_auc_values=[]
    from sklearn.metrics import roc_curve, auc
    from scipy.interpolate import interp1d
    n_folds=10
    n_classes=args.n_classes   
    if n_classes==4:
        colors = ['b', 'g', 'r', 'c']  
        label_dict = {0:'Normal', 1:'19DEL', 2:'L858R', 3:'ELSE'}  
    elif n_classes==5:
        colors = ['b', 'g', 'r', 'c', 'm']  # 蓝色, 绿色, 红色, 青色, 洋红色
        label_dict = {0:'Normal', 1:'19DEL', 2:'L858R', 3:'Other', 4:'Wild'}
    for fold in range(n_folds):
        fpr_fold = []
        tpr_fold = []
        fold_auc_values=[]
        for cls in range(n_classes):
            fpr, tpr, _ = roc_curve(label_3d[fold, :, cls], prob_3d[fold, :, cls], pos_label=1)
            # 为了能够平均不同折的ROC曲线，我们需要对它们进行插值到相同的FPR点集
            fpr_interp = np.linspace(0, 1, 100)
            tpr_interp = interp1d(fpr, tpr, fill_value="extrapolate")(fpr_interp)
            auc_value=auc(fpr,tpr)
            fpr_fold.append(fpr_interp)
            tpr_fold.append(tpr_interp)
            fold_auc_values.append(auc_value)
        fpr_all.append(np.array(fpr_fold).T)  # 转置以匹配类别维度
        tpr_all.append(np.array(tpr_fold).T)
        all_auc_values.append(fold_auc_values)
    # 将所有折的FPR和TPR平均
    mean_fpr = np.mean(fpr_all, axis=0)
    mean_tpr = np.mean(tpr_all, axis=0)
    all_auc_values_flat = [auc_value for fold_auc_values in all_auc_values for auc_value in fold_auc_values]  # 展平列表以计算总体标准差
    auc_means = [np.mean([all_auc_values[f][c] for f in range(n_folds)]) for c in range(n_classes)]  # 计算每个类别的AUC均值
    auc_stds = [np.std([all_auc_values[f][c] for f in range(n_folds)], ddof=0 if n_folds > 1 else 1) for c in range(n_classes)]  # 计算每个类别的AUC标准差
 
    # 计算TPR的标准差（注意：这里计算的是总体标准差）
    std_tpr = np.std(tpr_all, axis=0, ddof=0 if n_folds > 1 else 1)  # ddof=1用于样本标准差，但这里我们计算总体标准差
    plt.figure()
    for cls in range(n_classes):
        color = colors[cls] 
        mean_auc=auc_means[cls]
        auc_std=auc_stds[cls]
        plt.plot(mean_fpr[:, cls], mean_tpr[:, cls], label=f'{label_dict[cls]} (AUC = {mean_auc:.2f}±{auc_std:.2f})', color=color)
        plt.fill_between(mean_fpr[:, cls], mean_tpr[:, cls] - 1.96 * std_tpr[:, cls], mean_tpr[:, cls] + 1.96 * std_tpr[:, cls], alpha=0.2, color=color)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {n_classes}-Class Problem with 10-Fold CV for {args.task}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.save_dir, 'roc_curve.png'))


    #导出混淆矩阵
    average_cm = np.mean(all_cm, axis=0)
    with open(os.path.join(args.save_dir, 'confusion_matrices.csv'), 'w') as f:  
        for i, cm in enumerate(all_cm):  
            f.write(f'Confusion Matrix {i + 1}:\n')   
            np.savetxt(f, cm, delimiter=',', fmt='%d')  
            f.write('\n')  
        f.write('Average Confusion Matrix:\n')  
        np.savetxt(f, average_cm, delimiter=',', fmt='%.2f')
        f.write('\n') 
        cm_percentage = average_cm.astype('float') / average_cm.sum(axis=1)[:, np.newaxis] * 100  
        f.write('Confusion Matrix in Percentage:\n')  
        np.savetxt(f, cm_percentage, delimiter=',', fmt='%.2f')


    final_df = pd.DataFrame({
        'folds': folds, 
        'test_auc': all_auc, 
        'test_acc': all_acc,
        'test_f1': all_f1,
        'test_recall': all_recall
    })
    print(all_auc)
    # final_df.to_csv(os.path.join(args.save_dir, 'final_results.csv'), index=False)
    # 导出 F1 分数和 Recall
    final_f1_recall_df = pd.DataFrame({
        'folds': folds,
        'test_f1': all_f1,
        'test_recall': all_recall
    })

    # 保存到 CSV 文件
    final_f1_recall_df.to_csv(os.path.join(args.save_dir, 'final_f1_recall_results.csv'), index=False)
