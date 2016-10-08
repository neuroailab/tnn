"""
Main run module
"""

from __future__ import absolute_import, division, print_function

import os, json, argparse

import numpy as np
import tensorflow as tf

from tfutils import base, data
import model


def get_data(train=False,
             batch_size=256,
             data_path='',
             image_size_crop=224,
             num_train_images=2**20,
             num_valid_images=2**14,
             num_threads=4,
             random_crop=True,
             T_tot=8
             ):
    # Get images and labels for ImageNet.
    if train:
        subslice = range(num_train_images)
    else:
        subslice = range(num_train_images, num_train_images + num_valid_images)

    with tf.device("/gpu:0"):
        d = data.ImageNet(data_path,
                          subslice,
                          crop_size=image_size_crop,
                          batch_size=batch_size,
                          n_threads=num_threads)
    print('images and labels done')
    return [d.batch] * T_tot, d


def get_train_data(**params):
    return get_data(train=True, **params)


def get_valid_data(**params):
    return get_data(train=False, **params)


def get_valid_targets(inputs, outputs, **kwargs):
    top_1_ops = [tf.nn.in_top_k(output, inp['label'], 1)
                for inp, output in zip(inputs, outputs)]
    top_5_ops = [tf.nn.in_top_k(output, label, 5)
                for output, label in zip(inputs, outputs)]
    return {'top1': top_1_ops, 'top5': top_5_ops}


def get_learning_rate(num_batches_per_epoch=1,
                      learning_rate_base=.01,
                      learning_rate_decay_factor=.95,
                      num_epochs_per_decay=1
                      ):
    # Decay the learning rate exponentially based on the number of steps.
    if learning_rate_decay_factor is None:
        learning_rate = learning_rate_base  # just a constant.
    else:
        # Calculate the learning rate schedule.
        global_step = [v for v in tf.all_variables() if v.name == 'global_step:0'][0]
        learning_rate = tf.train.exponential_decay(
            learning_rate_base,  # Base learning rate.
            global_step,  # Current index into the dataset.
            num_batches_per_epoch * num_epochs_per_decay,  # Decay step (aka once every EPOCH)
            learning_rate_decay_factor,  # Decay rate.
            staircase=True)

    return learning_rate


def get_optimizer(loss, learning_rate=.01, momentum=.9, grad_clip=True):
    # Create optimizer for gradient descent.
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(loss)
    global_step = [v for v in tf.all_variables() if v.name == 'global_step:0'][0]
    if grad_clip:
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                      for grad, var in gvs if grad is not None]
        # gradient clipping. Some gradients returned are 'None' because
        # no relation between the variable and tot loss; so we skip those.
        optimizer = optimizer.apply_gradients(capped_gvs,
                                              global_step=global_step)
        print('Gradients clipped')
    else:
        optimizer = optimizer.apply_gradients(gvs, global_step=global_step)
        print('Gradients not clipped')

    return optimizer


def main(params):
    model_func_kwargs = {'model_base': model.alexnet,
                         'features_layer': None  # last layer
                         }
    model_func_kwargs.update(params['model'])

    data_func_kwargs = {'T_tot': params['model']['T_tot']}
    data_func_kwargs.update(params['data'])

    lr_func_kwargs = {'num_batches_per_epoch': params['num_train_batches']}
    lr_func_kwargs.update(params['learning_rate'])

    # to keep consistent count (of epochs passed, etc.)
    start_step = params['saver']['start_step'] if params['saver']['restore'] else 0
    end_step = params['num_epochs'] * params['num_train_batches']

    base.run_base(params,
             model_func=model.get_model,
             model_kwargs=model_func_kwargs,
             train_data_func=get_train_data,
             train_data_kwargs=params['data'],
             loss_func=model.get_loss,
             loss_kwargs=params['loss'],
             lr_func=get_learning_rate,
             lr_kwargs=lr_func_kwargs,
             opt_func=get_optimizer,
             opt_kwargs=params['optimizer'],
             saver_kwargs=params['saver'],
             train_targets_func=None,
             train_targets_kwargs={},
             valid_data_func=get_valid_data,
             valid_data_kwargs={},
             valid_targets_func=get_valid_data,
             valid_targets_kwargs={},
             thres_loss=params['thres_loss'],
             seed=params['seed'],
             start_step=start_step,
             end_step=end_step,
             log_device_placement=params['log_device_placement']
             )

if __name__ == '__main__':
    params = base.get_params()

    if params is None:
        from params import params
        outfile = 'params.json'

        with open(outfile, 'w') as f:
            json.dump(params, f)
        params = json.load(open(outfile))

    main(params)