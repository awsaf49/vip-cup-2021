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
    parser.add_argument('-c', '--cfg', default='evopose2d_S.yaml')
    parser.add_argument('-p', '--coco-path', required=True,
                        help='Path to folder containing COCO images and annotation directories.')
    parser.add_argument('-i', '--img-id', type=int, default=785)
    parser.add_argument('--ckpt-path',type=str,default='models/evopose2d_S.h5')
    parser.add_argument('--alpha', type=float, default=0.8)
    args = parser.parse_args()

    # load the config .yaml file
    cfg.merge_from_file('configs/' + args.cfg)

    # load the trained model
    ckpt_path = args.ckpt_path
    model = tf.keras.models.load_model(ckpt_path)
    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]

    # load the dataset annotations
    coco = COCO(osp.join(args.coco_path, 'annotations', 'person_keypoints_val2017.json'))
    img_data = coco.loadImgs([args.img_id])[0]

    annotation = coco.loadAnns(coco.getAnnIds([args.img_id]))[0]
    bbox = annotation['bbox']
    kp = np.array(annotation['keypoints']).reshape(-1, 3)  # not used

    # get test image
    img_bytes = open(osp.join(args.coco_path, 'val2017', img_data['file_name']), 'rb').read()
    img = tf.image.decode_jpeg(img_bytes, channels=3)

    # preprocess
    _, norm_img, _, M, _ = preprocess(0, img, bbox, kp, 0., cfg.DATASET, split='val', predict_kp=True)
    M = np.expand_dims(np.array(M), axis=0)

    # generate heatmap predictions
    hms = model.predict(tf.expand_dims(norm_img, 0))

    # get keypoint predictions from heatmaps
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
    cv2.imwrite(img_data['file_name'], img)