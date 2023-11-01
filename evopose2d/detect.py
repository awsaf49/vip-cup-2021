import argparse
from pycocotools.coco import COCO
import os.path as osp
import tensorflow as tf
from dataset.dataloader import preprocess
from dataset.coco import cn as cfg
import numpy as np
from validate import get_preds
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os

# KP_PAIRS = [[5, 6], [6, 12], [12, 11], [11, 5],
#             [5, 7], [7, 9], [11, 13], [13, 15],
#             [6, 8], [8, 10], [12, 14], [14, 16]]

SKELETON = [(0, 1), (1, 2), (12, 2), (12, 3), (3, 4), (4, 5), (6, 7),
            (7, 8), (8, 12), (12, 9), (9, 10), (10, 11), (12, 13)]
KP_PAIRS = [[x[0], x[1]] for x in SKELETON]
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
    "Thorax":12,
    "Head top":13, 
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='evopose2d_L.yaml')
    parser.add_argument('-d', '--data-dir', default='data/images', help='directory of images')
    parser.add_argument('-s', '--save-dir', default='data/prediction', help='directory to save prediction')
    parser.add_argument('--ckpt-path',type=str,default='models/evopose2d_L.h5')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--max-det', type=int, default=5, help='max number of detection')
    parser.add_argument('--ttas', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--interp', type=str, default='bilinear', help='interpolation method')
    args = parser.parse_args()

    # load the config .yaml file
    cfg.merge_from_file('configs/' + args.cfg)

    # load the trained model
    print('\nLoading Model:')
    ckpt_path = args.ckpt_path
    model = tf.keras.models.load_model(ckpt_path)
    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]
    cfg.DATASET.INPUT_SHAPE  = model.input_shape[1:]
    print('\nInput:',cfg.DATASET.INPUT_SHAPE)
    print('Output:',cfg.DATASET.OUTPUT_SHAPE)
    print('Original:',cfg.DATASET.ORIGINAL_SHAPE)

    # load the image paths
    image_paths = glob(osp.join(args.data_dir, '**/*'), recursive=True)
    image_paths = [x for x in image_paths if ('png' in x or 'jpg' in x or 'jpeg' in x)] # filter out other files
    np.random.shuffle(image_paths)
    print();
    cnt = 0
    for image_path in tqdm(image_paths, desc='Detecting keypoints ', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

        # get test image
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (cfg.DATASET.INPUT_SHAPE[0], cfg.DATASET.INPUT_SHAPE[1]),method=args.interp)

        # get bbox
        dim = np.array(img.shape[:2])
        bbox = [0.20*dim[0], 0.20*dim[1], 0.8*dim[0], 0.8*dim[1]]

        # preprocess
        kp = tf.constant([[0.0, 0.0]]) # dummy value
        norm_img = preprocess(0, img, bbox, kp, 0., cfg.DATASET, split='val', predict_kp=True, do_transform=False)
        ttas_hms = []
        for tta in range(max(args.ttas) + 1):
            # do tta
            if tta==0:
                tta_img = norm_img
            elif tta==1:
                tta_img = norm_img[:, ::-1, :]
            if tta!=0 & tta!=1:
                tta_img = tf.image.random_hue(norm_img, cfg.DATASET.HUE)
                tta_img = tf.image.random_saturation(tta_img, cfg.DATASET.SAT[0], cfg.DATASET.SAT[1])
                tta_img = tf.image.random_contrast(tta_img, cfg.DATASET.CONT[0], cfg.DATASET.CONT[1])
                tta_img = tf.image.random_brightness(tta_img, cfg.DATASET.BRI)
            # generate heatmap predictions
            tta_hms = model.predict(tf.expand_dims(tta_img, 0))
            # inv flip
            if tta==1:
                tta_hms = tta_hms[:, :, ::-1, :]
                tmp = tta_hms.copy()
                for i in range(len(cfg.DATASET.KP_FLIP)):
                    tta_hms[:, :, :, i] = tmp[:, :, :, cfg.DATASET.KP_FLIP[i]]
                    # shift to align features
                    tta_hms[:, :, 1:, :] = tta_hms[:, :, 0:-1, :].copy()
            ttas_hms.append(tta_hms)
        hms = np.mean(ttas_hms, axis=0)
        # print(hms.shape)
        # get keypoint predictions from heatmaps
        M = None
        preds = get_preds(hms, M, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)[0]

        # plot results
        img = img.numpy()[:, :, [2, 1, 0]]
        overlay = img.copy()

        cmap   = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(NAME2IDX) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        for i, p in enumerate(KP_PAIRS):
            overlay = cv2.line(overlay,
                            tuple(np.int32(np.round(preds[p[0], :2]))),
                            tuple(np.int32(np.round(preds[p[1], :2]))), color=colors[i], thickness=4)
        for i, (x, y, v) in enumerate(preds):
            overlay = cv2.circle(overlay, (int(np.round(x)), int(np.round(y))), radius=3, color=colors[i], thickness=4)

        img = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)
        
        filename = image_path.split(os.sep)[-1]
        study_id = image_path.split(os.sep)[-2]
        new_path = os.path.join(args.save_dir, study_id+'_'+filename)
        cv2.imwrite(new_path, img)
        cnt += 1
        if cnt>=args.max_det and args.max_det>0:
            print('Max Detection:',args.max_det)
            break