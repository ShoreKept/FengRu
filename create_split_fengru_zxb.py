import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'EGFR_subtyping', 'EGFR_Four_subtyping_fengru', 'EGFR_Five_subtyping_fengru', 'RCC_fengru', 'LUAD_fengru'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'EGFR_subtyping':
    args.n_classes=5
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/graduate2024/data/CLAM_csv/EGFR_CLAM.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Normal':0, 'L858R':1, 'Other':2, '19DEL':3, 'Wild':4},
                            patient_strat= True,
                            patient_voting='max',
                            ignore=[])
elif args.task == 'EGFR_Four_subtyping_fengru':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/graduate2024/data_new/CLAM_csv/EGFR_CLAM.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Normal':0, '19DEL':1, 'L858R':2, 'ELSE':3},
                            patient_strat= True,
                            patient_voting='max',
                            ignore=[])
elif args.task == 'EGFR_Five_subtyping_fengru':
    args.n_classes=5
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/graduate2024/data_new/CLAM_csv/EGFR_CLAM_five.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Normal':0, '19DEL':1, 'L858R':2, 'Other':3, 'Wild':4},
                            patient_strat= True,
                            patient_voting='max',
                            ignore=[])
elif args.task == 'RCC_fengru':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/graduate2024/data_new/CLAM_csv/tcga_rcc.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2},
                            patient_strat= True,
                            patient_voting='max',
                            ignore=[])
elif args.task == 'LUAD_fengru':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/graduate2024/data_new/CLAM_csv/tcga_luad.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2},
                            patient_strat= True,
                            patient_voting='max',
                            ignore=[])
else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)
print("aaaaaaaaaaaaaaaaa", num_slides_cls, val_num, test_num)
if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:     
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            # print('++++', splits[].slide_data)
            save_splits(split_datasets=splits, column_keys=['train', 'val', 'test'], filename=os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



