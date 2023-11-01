from yacs.config import CfgNode as CN

cn = CN(new_allowed=True)

cn.DATASET = CN(new_allowed=True)
cn.DATASET.NAME = 'COCO'
cn.DATASET.NUM_KEYPOINTS = 14
cn.DATASET.TFRECORDS = ''
cn.DATASET.ANNOT = 'coco2017/annotations/person_keypoints_val2017.json'
cn.DATASET.TRAIN_SAMPLES = 1125
cn.DATASET.VAL_SAMPLES = 225
cn.DATASET.KP_FLIP = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
cn.DATASET.KP_UPPER = [6, 7, 8, 9, 10, 11, 12, 13]
cn.DATASET.KP_LOWER = []
cn.DATASET.BGR = False
cn.DATASET.NORM = True
cn.DATASET.MEANS = [0.485, 0.456, 0.406]  # imagenet means RGB
cn.DATASET.STDS = [0.229, 0.224, 0.225]
cn.DATASET.INPUT_SHAPE = [256, 192, 3]
cn.DATASET.OUTPUT_SHAPE = [64, 48, 14]
cn.DATASET.ORIGINAL_SHAPE = [160, 120, 3]
cn.DATASET.SIGMA = 2 * cn.DATASET.OUTPUT_SHAPE[0] / 64
cn.DATASET.FLIP_PROB = 0.50# 0.5
cn.DATASET.HALF_BODY_PROB = 0.
cn.DATASET.HALF_BODY_MIN_KP = 8
cn.DATASET.SCALE_FACTOR = 0.3
cn.DATASET.ROT_PROB = 0.6
cn.DATASET.ROT_FACTOR = 40
cn.DATASET.COLOR_AUG = True
cn.DATASET.SAT  = [0.7, 1.3]
cn.DATASET.CONT = [0.8, 1.2]
cn.DATASET.BRI  = 0.1
cn.DATASET.HUE  = 0.01
cn.DATASET.CACHE = False
cn.DATASET.BFLOAT16 = False

cn.TRAIN = CN(new_allowed=True)
cn.TRAIN.BATCH_SIZE = 64
cn.TRAIN.BASE_LR = 0.00025
cn.TRAIN.SCALE_LR = True
cn.TRAIN.LR_SCHEDULE = 'warmup_piecewise'
cn.TRAIN.EPOCHS = 210
cn.TRAIN.DECAY_FACTOR = 0.1
cn.TRAIN.DECAY_EPOCHS = [170, 200]
cn.TRAIN.WARMUP_EPOCHS = 0
cn.TRAIN.WARMUP_FACTOR = 0.1
cn.TRAIN.DISP = True
cn.TRAIN.SEED = 0
cn.TRAIN.WD = 1e-5
cn.TRAIN.SAVE_EPOCHS = 0
cn.TRAIN.SAVE_META = False
cn.TRAIN.VAL = True

cn.VAL = CN(new_allowed=True)
cn.VAL.BATCH_SIZE = 64
cn.VAL.FLIP = True # True
cn.VAL.DROP_REMAINDER = False
cn.VAL.SCORE_THRESH = 0.2
cn.VAL.DET = True
cn.VAL.SAVE_BEST = True

cn.MODEL = CN(new_allowed=True)
cn.MODEL.TYPE = 'evopose'
cn.MODEL.LOAD_WEIGHTS = True
cn.MODEL.CKPT_PATH = 'models/coco_ckpt/evopose2d_L.h5'
cn.MODEL.PARENT = None
cn.MODEL.GENOTYPE = None
cn.MODEL.WIDTH_COEFFICIENT = 1.
cn.MODEL.DEPTH_COEFFICIENT = 1.
cn.MODEL.DEPTH_DIVISOR = 8
cn.MODEL.ACTIVATION = 'swish'
cn.MODEL.HEAD_BLOCKS = 3
cn.MODEL.HEAD_KERNEL = 3
cn.MODEL.HEAD_CHANNELS = 128
cn.MODEL.HEAD_ACTIVATION = 'swish'
cn.MODEL.FINAL_KERNEL = 3
cn.MODEL.SAVE_DIR = 'models'

cn.SEARCH = CN(new_allowed=True)