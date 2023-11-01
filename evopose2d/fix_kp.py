import argparse
import os.path as osp
from dataset.coco import cn as cfg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import json
import shutil
import pandas as pd

SKELETON = [(0, 1), (1, 2), (12, 2), (12, 3), (3, 4), (4, 5), (6, 7),
            (7, 8), (8, 12), (12, 9), (9, 10), (10, 11), (12, 13)]
# skeleton = [[x[0]+1, x[1]+1] for x in skeleton]
NAME2IDX = {
    "Right ankle":0,
    "Right knee":1,
    "Right hip":2,
    "Left hip":3,
    "Left knee":4,
    "Left ankle":5,
    "Right wrist":6,
    "Right elbow":7,
    "Right shoulder":8,
    "Left shoulder":9,
    "Left elbow":10,
    "Left wrist":11,
    "thorax":12,
    "head top":13, 
}
IDX2NAME = {v:k for k,v in NAME2IDX.items()}
KP_NAMES  = list(IDX2NAME.values())
KP_LABELS = list(IDX2NAME.keys())

kp_df = pd.read_csv('csv/kps.csv')
kp_stat = kp_df.groupby(['kp_name']).agg([min, max, 'mean', 'median', 'std', pd.Series.mode])

def fix_kp(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    kps = np.array(data)
    cntx = 0; cnty = 0;
    for img_idx in tqdm(range(len(kps)), desc='Fixing keypoints ', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        kp_img = kps[img_idx]
        for kp_idx in range(14):
            kp = kp_img[kp_idx]
            kp_lim = kp_stat.loc[KP_NAMES[kp_idx]]
            x  = kp_lim.x
            y  = kp_lim.y
            if not (kp[0]>=x['min'] and kp[0]<=x['max']):
                kps[img_idx][kp_idx][0] = int(np.clip(kp[0],x['min'],x['max']))
                cntx+=1
            if not (kp[1]>=y['min'] and kp[1]<=y['max']):
                kps[img_idx][kp_idx][1] = int(np.clip(kp[1],y['min'],y['max']))
                cnty+=1

    print(f'  == Summary ==\ncntx : {cntx} | cnty : {cnty}')
    return kps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip-path', type=str, default='data/submission/proletarians_pseudo518.zip')
    parser.add_argument('--save-dir', type=str, default='data/fixed')
    args = parser.parse_args()
    zip_path = args.zip_path
    save_dir = args.save_dir
    filename = zip_path.split(os.sep)[-1].split('.zip')[0]
    save_path = osp.join(save_dir, filename)+'-fixed.zip'

    # unzip from zip path
    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f'Extracting {zip_path} to {save_dir}')
    command = f"unzip -q {zip_path} -d {save_dir}"
    os.system(command)

    sub_path = os.path.join(save_dir, 'preds.json')
    kps = fix_kp(sub_path)
    kps = kps.tolist()
    with open(sub_path, 'w') as f:
        json.dump(kps, f)
    print(f'Saved to {save_path}')
    command = f"zip -qr {save_path} {sub_path} -j"
    os.system(command)
    os.remove(sub_path)
    
        
