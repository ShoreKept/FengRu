import h5py
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description='Add position information')
parser.add_argument('--feat_dir', type=str, default=None, 
                    help='data directory to all_features\'s parent')
parser.add_argument('--result_dir', type=str, default=None,
                    help='result directory')
args = parser.parse_args()
def returnIdx(source, target):
    source = np.array(source)
    target = np.array(target)
    return np.where((source == target).all(axis=1))[0]
for className in os.listdir(args.feat_dir):
    if className != 'all_features':
        continue
    path = os.path.join(args.feat_dir, className, 'h5_files')
    resultPath = os.path.join(args.result_dir, className, 'h5_files')
    os.makedirs(resultPath, exist_ok = True)
    for slide in os.listdir(path):
        if os.path.exists(os.path.join(resultPath, slide)):
            continue
        neighbors = []
        with h5py.File(os.path.join(path, slide), 'r') as f:
            feats = f['features'][:]
            coords = f['coords'][:]
        for coord in coords:
            neighbor = coord.copy()
            tmp = []
            neighbor[1] = neighbor[1]-256
            indices = returnIdx(coords, neighbor)
            if len(indices) > 0:
                tmp.append(indices[0])  
            else:
                tmp.append(-1)  
            neighbor = coord.copy()
            neighbor[1] = neighbor[1]+256
            indices = returnIdx(coords, neighbor)
            if len(indices) > 0:
                tmp.append(indices[0])  
            else:
                tmp.append(-1)  
            neighbor = coord.copy()
            neighbor[0] = neighbor[0]-256
            indices = returnIdx(coords, neighbor)
            if len(indices) > 0:
                tmp.append(indices[0])  
            else:
                tmp.append(-1)  
            neighbor = coord.copy()
            neighbor[0] = neighbor[0]+256
            indices = returnIdx(coords, neighbor)
            if len(indices) > 0:
                tmp.append(indices[0])  
            else:
                tmp.append(-1)  
            neighbors.append(tmp)
        with h5py.File(os.path.join(resultPath, slide), 'w') as f:
            f.create_dataset('pos', data = neighbors)
            f.create_dataset('features', data = feats)
            f.create_dataset('coords', data = coords)

            