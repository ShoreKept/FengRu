import pandas as pd
import numpy as np
import argparse
import os
import copy

parser = argparse.ArgumentParser(description='Create cross validation splits')
parser.add_argument('-d', '--dataset_csv',default = '/home/graduate2024/data_new/CLAM_csv/EGFR_CLAM.csv', type=str, help='Path to processed data')
parser.add_argument('-p', '--output_dir',default = '/home/graduate2024/data_new/splits', type=str, help='Path to output directory')
parser.add_argument('-n', '--num_splits', default=10, type=int, help='Number of splits to create')
parser.add_argument('-s', '--seed', default=114514, type=int, help='Random seed')
parser.add_argument('--label_col', default=None, type=str, 
                    help='Column name for label, if set, class balance will be maintained')
args = parser.parse_args()

print(f'Creating {args.num_splits} splits for {args.dataset_csv}')
print(f'Output directory: {args.output_dir}')
print(f'Random seed: {args.seed}')
print(f'Label column: {args.label_col}')
print()

if not os.path.exists(args.output_dir):
    print(f'Creating output directory {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

dataset_df = pd.read_csv(args.dataset_csv)
case_ids = dataset_df['case_id'].unique()
np.random.seed(args.seed)
np.random.shuffle(case_ids)

if args.label_col is None:
    fold_case_ids = np.array_split(case_ids, args.num_splits)
    # print(fold_case_ids)
else:
    case_label_dict = {}
    for case_id in case_ids:
        case_id_label = dataset_df[dataset_df['case_id'] == case_id][args.label_col]
        if case_id_label.shape[0] > 1:
            print(f'Case {case_id} has more than one label, set as {case_id_label.values[0]}')
        case_label_dict[case_id] = case_id_label.values[0]
    
    label_case_dict = {}
    for case_id, case_label in case_label_dict.items():
        if case_label not in label_case_dict:
            label_case_dict[case_label] = []
        label_case_dict[case_label].append(case_id)
    
    print('Class balance:')
    for label, case_ids in label_case_dict.items():
        print(f'{label}: {len(case_ids)} cases')
    print()
    
    fold_case_ids = [[] for _ in range(args.num_splits)]
    for label, label_case_ids in label_case_dict.items():
        np.random.shuffle(case_ids)
        fold_label_case_ids = np.array_split(label_case_ids, args.num_splits)
        for i, case_ids_split in enumerate(fold_label_case_ids):
            fold_case_ids[i].extend(case_ids_split)


for i, fold_case in enumerate(fold_case_ids):
    print(f'Fold {i}: {len(fold_case)} cases')
print()

fold_slide_ids = []
for fold_case_id in fold_case_ids:
    fold_slide_ids.append(dataset_df[dataset_df['case_id'].isin(fold_case_id)]['slide_id'].tolist())


for fold_idx in range(args.num_splits):
    train_list = []
    for i, fold_slide_id in enumerate(fold_slide_ids):
        if i != fold_idx:
            train_list.extend(fold_slide_id)

    val_list = copy.deepcopy(fold_slide_ids[fold_idx])
    val_list.extend(['']*(len(train_list) - len(val_list)))

    fold_df = {'train': train_list, 'val': val_list, 'test': val_list}
    pd.DataFrame(fold_df).to_csv(os.path.join(args.output_dir, f'split_{fold_idx}.csv'), index=False)

print('Finished')