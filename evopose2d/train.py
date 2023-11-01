import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np, pandas as pd
from lr_schedules import WarmupCosineDecay, WarmupPiecewise
import os.path as osp
from utils import get_flops, detect_hardware
from dataset.dataloader import load_tfds
from dataset.coco import cn as cfg
from nets.simple_basline import SimpleBaseline
from nets.hrnet import HRNet
from nets.evopose2d import EvoPose
from time import time
import pickle
import argparse
from validate import validate
from tqdm import tqdm


@tf.function
def mse_loss(model, images, targets, valid, training=True):
    heatmaps = model(images, training=training)
    heatmaps = tf.cast(heatmaps, tf.float32)  # in case using bfloat16
    valid_mask = tf.reshape(valid, [tf.shape(images)[0], 1, 1, tf.shape(valid)[-1]])
    loss = tf.reduce_mean(tf.square(heatmaps - targets) * valid_mask)
    l2_loss = tf.cast(sum(model.losses), tf.float32)
    return loss, l2_loss


def train(strategy, cfg):
    os.makedirs(cfg.MODEL.SAVE_DIR, exist_ok=True)

    if cfg.DATASET.BFLOAT16:
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_policy(policy)

    tf.random.set_seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)

    meta_data = {'train_loss': [], 'val_loss': [], 'config': cfg}

    spe = int(np.ceil(cfg.DATASET.TRAIN_SAMPLES / cfg.TRAIN.BATCH_SIZE))
    spv = cfg.DATASET.VAL_SAMPLES // cfg.VAL.BATCH_SIZE

    if cfg.TRAIN.SCALE_LR:
        lr = cfg.TRAIN.BASE_LR * cfg.TRAIN.BATCH_SIZE / 32
        cfg.TRAIN.WARMUP_FACTOR = 32 / cfg.TRAIN.BATCH_SIZE
    else:
        lr = cfg.TRAIN.BASE_LR

    if cfg.TRAIN.LR_SCHEDULE == 'warmup_cosine_decay':
        lr_schedule = WarmupCosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(cfg.TRAIN.DECAY_EPOCHS) * spe,
            warmup_steps=cfg.TRAIN.WARMUP_EPOCHS * spe,
            warmup_factor=cfg.TRAIN.WARMUP_FACTOR)
    elif cfg.TRAIN.LR_SCHEDULE == 'warmup_piecewise':
        lr_schedule = WarmupPiecewise(
            boundaries=[x * spe for x in cfg.TRAIN.DECAY_EPOCHS],
            values=[lr, lr / 10, lr / 10 ** 2],
            warmup_steps=spe * cfg.TRAIN.WARMUP_EPOCHS,
            warmup_factor=cfg.TRAIN.WARMUP_FACTOR)
    else:
        lr_schedule = lr
        
    
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        if cfg.MODEL.TYPE == 'simple_baseline':
            model = SimpleBaseline(cfg)
        elif cfg.MODEL.TYPE == 'hrnet':
            model = HRNet(cfg)
        elif cfg.MODEL.TYPE == 'evopose':
            model = EvoPose(cfg)
        print("="*20+" Training "+"="*20)
        if len(cfg.MODEL.CKPT_PATH):
            print('Loading ckpt from {}'.format(cfg.MODEL.CKPT_PATH))
            model.load_weights(cfg.MODEL.CKPT_PATH, by_name=True, skip_mismatch=True)
        else:
            print('No ckpt')
        train_loss = tf.keras.metrics.Mean()
        val_loss   = tf.keras.metrics.Mean()

    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]
    cfg.DATASET.SIGMA = 2 * cfg.DATASET.OUTPUT_SHAPE[0] / 64

    meta_data['parameters'] = model.count_params()
    try:
        meta_data['flops'] = get_flops(model)
    except:
        meta_data['flops'] = -1

    train_ds = load_tfds(cfg, 'train')
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    train_iterator = iter(train_ds)

    if cfg.TRAIN.VAL:
        val_ds = load_tfds(cfg, 'val')
        val_ds = strategy.experimental_distribute_dataset(val_ds)

    @tf.function
    def train_step(train_iterator):
        def step_fn(inputs):
            imgs, targets, valid = inputs
            with tf.GradientTape() as tape:
                loss, l2_loss = mse_loss(model, imgs, targets, valid, training=True)
                scaled_loss = (loss + l2_loss) / strategy.num_replicas_in_sync
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            train_loss.update_state(loss)
        strategy.run(step_fn, args=(next(train_iterator),))

    @tf.function
    def val_step(dist_inputs):
        def step_fn(inputs):
            imgs, targets, valid = inputs
            loss, _ = mse_loss(model, imgs, targets, valid, training=False)
            val_loss.update_state(loss)
        strategy.run(step_fn, args=(dist_inputs,))

    print('Training {} ({:.2f}M / {:.2f}G) on {} for {} epochs'
          .format(cfg.MODEL.NAME, meta_data['parameters']/1e6,
                  meta_data['flops']/2/1e9, cfg.TRAIN.ACCELERATOR, cfg.TRAIN.EPOCHS))

    epoch = 1
    ts = time()
    print()
    if not cfg.TRAIN.VERBOSE:
        total = cfg.TRAIN.EPOCHS
        pbar  = tqdm(total=total, desc='Epoch ',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    while epoch <= cfg.TRAIN.EPOCHS:
        te = time()
        total = spe
        if cfg.TRAIN.VERBOSE:
            print('Epoch {}/{}'.format(epoch, cfg.TRAIN.EPOCHS))
            pbar = tqdm(range(spe), total=total, leave=True, position=None,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit="batch")
            it = pbar
        else:
            it = range(spe)
        for i in it:
            pbar.set_description('[Training] ') if cfg.TRAIN.VERBOSE else None
            train_step(train_iterator)
            # if cfg.TRAIN.DISP:
            #     print('epoch {} ({}/{}) | loss: {:.1f}'
            #           .format(epoch, i + 1, spe, train_loss.result().numpy()))
            pbar.set_postfix(loss=train_loss.result().numpy()) if cfg.TRAIN.VERBOSE else None
                
        meta_data['train_loss'].append(train_loss.result().numpy())

        if cfg.TRAIN.VAL:
            total = spv
            if cfg.TRAIN.VERBOSE:
                pbar = tqdm(enumerate(val_ds), total=total, leave=True, position=None,
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit="batch")
                it = pbar
            else:
                it = enumerate(val_ds)
            for i, batch in it:
                pbar.set_description('[Validating] ') if cfg.TRAIN.VERBOSE else None
                val_step(batch)
                # if cfg.TRAIN.DISP:
                #     print('val {} ({}/{}) | loss: {:.1f}'
                #       .format(epoch, i + 1, spv, val_loss.result().numpy()))
                pbar.set_postfix(loss=val_loss.result().numpy()) if cfg.TRAIN.VERBOSE else None
            meta_data['val_loss'].append(val_loss.result().numpy())

            if cfg.VAL.SAVE_BEST:
                if epoch == 1:
                    best_weights = model.get_weights()
                    best_loss = val_loss.result().numpy()
                    model.save(osp.join(cfg.MODEL.SAVE_DIR, '{}_{}_ckpt.h5'
                                        .format(cfg.MODEL.NAME, 'best')), save_format='h5')
                    print('Saved checkpoint to', osp.join(cfg.MODEL.SAVE_DIR, '{}_{}_ckpt.h5'
                                        .format(cfg.MODEL.NAME, 'best'))) if cfg.TRAIN.VERBOSE else None
                    if cfg.TRAIN.DISP:
                        print('Cached model weights') if cfg.TRAIN.VERBOSE else None
                elif val_loss.result().numpy() < best_loss:
                    best_weights = model.get_weights()
                    best_loss = val_loss.result().numpy()
                    model.save(osp.join(cfg.MODEL.SAVE_DIR, '{}_{}_ckpt.h5'
                                        .format(cfg.MODEL.NAME, 'best')), save_format='h5')
                    print('Saved checkpoint to', osp.join(cfg.MODEL.SAVE_DIR, '{}_{}_ckpt.h5'
                                        .format(cfg.MODEL.NAME, 'best'))) if cfg.TRAIN.VERBOSE else None
                    if cfg.TRAIN.DISP:
                        print('Cached model weights') if cfg.TRAIN.VERBOSE else None
                else:
                    print('No improvement') if cfg.TRAIN.VERBOSE else None

        train_loss.reset_states()
        val_loss.reset_states()

        if cfg.TRAIN.SAVE_EPOCHS and epoch % cfg.TRAIN.SAVE_EPOCHS == 0: # ===== save model
            model.save(osp.join(cfg.MODEL.SAVE_DIR, '{}_ckpt{:03d}.h5'
                                .format(cfg.MODEL.NAME, epoch)), save_format='h5')
            print('Saved checkpoint to', osp.join(cfg.MODEL.SAVE_DIR, '{}_ckpt{:03d}.h5'
                                            .format(cfg.MODEL.NAME, epoch))) if cfg.TRAIN.VERBOSE else None         
        if cfg.TRAIN.SAVE_META:
            pickle.dump(meta_data, open(osp.join(cfg.MODEL.SAVE_DIR,
                                                 '{}_meta.pkl'.format(cfg.MODEL.NAME)), 'wb'))

        if epoch > 1 and cfg.TRAIN.DISP:
            est_time = (cfg.TRAIN.EPOCHS - epoch) * (time() - te) / 3600
            print('Estimated time remaining: {:.2f} hrs'.format(est_time))

        epoch += 1
        print()
        if not cfg.TRAIN.VERBOSE:
            pbar.update(1)
        # end epoch-------------------------------------------------------------------------------------
    if not cfg.TRAIN.VERBOSE:
        pbar.close()

    meta_data['training_time'] = time() - ts

    if cfg.VAL.SAVE_BEST:
        model.set_weights(best_weights)
        
    del meta_data['config']
    pd.DataFrame(meta_data).to_csv('history.csv',index=False)

    return model, meta_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--tpu', default=None)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--verbose',action='store_true')
    args = parser.parse_args()

    tpu, strategy = detect_hardware(args.tpu)
    cfg.TRAIN.VERBOSE=args.verbose
    if tpu:
        cfg.TRAIN.ACCELERATOR = args.tpu
    else:
        cfg.TRAIN.ACCELERATOR = 'GPU/CPU'
    cfg.merge_from_file('configs/' + args.cfg)
    cfg.MODEL.NAME = args.cfg.split('.yaml')[0]
    model, meta_data = train(strategy, cfg)
    model.save(osp.join(cfg.MODEL.SAVE_DIR, '{}.h5'.format(cfg.MODEL.NAME)), save_format='h5')
    pickle.dump(meta_data, open(osp.join(cfg.MODEL.SAVE_DIR,
                                         '{}_meta.pkl'.format(cfg.MODEL.NAME)), 'wb'))

    if args.val:
        AP = validate(strategy, cfg, model)
        print('AP: {:.5f}'.format(AP))
        meta_data['AP'] = AP
        pickle.dump(meta_data, open(osp.join(cfg.MODEL.SAVE_DIR,
                                             '{}_meta.pkl'.format(cfg.MODEL.NAME)), 'wb'))
