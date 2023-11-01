import os
import os.path as osp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
import numpy as np
from dataset.dataloader import load_tfds
import math
import json
import cv2
from utils import detect_hardware, suppress_stdout
import pickle
from tqdm import tqdm


def get_preds(hms, Ms, input_shape, output_shape):
    preds = np.zeros((hms.shape[0], output_shape[-1], 3))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm = hms[i, :, :, j]
            idx = hm.argmax()
            y, x = np.unravel_index(idx, hm.shape)
            px = int(math.floor(x + 0.5))
            py = int(math.floor(y + 0.5))
            if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                diff = np.sign(diff)
                x += diff[0] * 0.25
                y += diff[1] * 0.25
            preds[i, j, :2] = [x * input_shape[1] / output_shape[1],
                              y * input_shape[0] / output_shape[0]]
            preds[i, j, -1] = hm.max() / 255

    # use inverse transform to map kp back to original image
    if Ms:
        for j in range(preds.shape[0]):
            M_inv = cv2.invertAffineTransform(Ms[j])
            preds[j, :, :2] = np.matmul(M_inv[:, :2], preds[j, :, :2].T).T + M_inv[:, 2].T
    return preds

def to_numpy(x, num_replicas):
    if num_replicas > 1:
        return np.concatenate(x.values)
    else:
        return x.numpy()
    
def validate(strategy, cfg, model=None, split='val'):
    cfg.DATASET.CACHE = False
    result_path = '{}/{}_{}.json'.format(cfg.MODEL.SAVE_DIR, cfg.MODEL.NAME, split)

    if split == 'val':
        with suppress_stdout():
            coco = COCO(cfg.DATASET.ANNOT)

    if model is None:
        with strategy.scope():
            model = tf.keras.models.load_model(
                osp.join(cfg.MODEL.SAVE_DIR, cfg.MODEL.NAME + '.h5'), compile=False)

    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]

    ds = load_tfds(cfg, split, det=cfg.VAL.DET,
                   predict_kp=True, drop_remainder=cfg.VAL.DROP_REMAINDER)
    ds = strategy.experimental_distribute_dataset(ds)

    @tf.function
    def predict(imgs, flip=False):
        if flip:
            imgs = imgs[:, :, ::-1, :]
        return model(imgs, training=False)

    results = []
    spv = cfg.DATASET.VAL_SAMPLES // cfg.VAL.BATCH_SIZE
    total = spv
    pbar = tqdm(enumerate(ds), total=total, leave=True, position=None,
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit="batch")
    for count, batch in pbar:
        pbar.set_description('Predicting ')
        ids, imgs, _, Ms, scores = batch

        ids = to_numpy(ids, strategy.num_replicas_in_sync)
        scores = to_numpy(scores, strategy.num_replicas_in_sync)
        Ms = to_numpy(Ms, strategy.num_replicas_in_sync)

        hms = strategy.run(predict, args=(imgs,))
        hms = to_numpy(hms, strategy.num_replicas_in_sync)

        if cfg.VAL.FLIP:
            flip_hms = strategy.run(predict, args=(imgs, True,))
            flip_hms = to_numpy(flip_hms, strategy.num_replicas_in_sync)
            flip_hms = flip_hms[:, :, ::-1, :]
            tmp = flip_hms.copy()
            for i in range(len(cfg.DATASET.KP_FLIP)):
                flip_hms[:, :, :, i] = tmp[:, :, :, cfg.DATASET.KP_FLIP[i]]
            # shift to align features
            flip_hms[:, :, 1:, :] = flip_hms[:, :, 0:-1, :].copy()
            hms = (hms + flip_hms) / 2.

        preds = get_preds(hms, Ms, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
        kp_scores = preds[:, :, -1].copy()

        # rescore
        rescored_score = np.zeros((len(kp_scores)))
        for i in range(len(kp_scores)):
            score_mask = kp_scores[i] > cfg.VAL.SCORE_THRESH
            if np.sum(score_mask) > 0:
                rescored_score[i] = np.mean(kp_scores[i][score_mask]) * scores[i]
        score_result = rescored_score

        for i in range(preds.shape[0]):
            results.append(dict(image_id=int(ids[i]),
                                category_id=1,
                                keypoints=preds[i].reshape(-1).tolist(),
                                score=float(score_result[i])))
        # if cfg.TRAIN.DISP:
        #     print('completed preds batch', count + 1)

    with open(result_path, 'w') as f:
        json.dump(results, f)

    if split == 'val':
        with suppress_stdout():
            result = coco.loadRes(result_path)
            cocoEval = COCOeval(coco, result, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        return cocoEval.stats[0]  # AP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', default=None)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('-c', '--cfg', required=True)  # yaml
    parser.add_argument('--det', type=int, default=-1)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--split', default='val')
    args = parser.parse_args()

    from dataset.coco import cn as cfg
    cfg.merge_from_file('configs/' + args.cfg)
    cfg.MODEL.NAME = args.cfg.split('.')[0]

    if args.ckpt:
        cfg.MODEL.NAME += '_{}'.format(args.ckpt)
    if args.det >= 0:
        cfg.VAL.DET = bool(args.det)
        
    if args.gpu is not None:
        strategy = tf.distribute.OneDeviceStrategy('/GPU:{}'.format(args.gpu))
    else:
        tpu, strategy = detect_hardware(args.tpu)

    if args.split == 'val':
        AP = validate(strategy, cfg, split='val')
        print('AP: {:.5f}'.format(AP))
    else:
        validate(strategy, cfg, split='test')