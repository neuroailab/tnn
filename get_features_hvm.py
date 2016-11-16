""" Creates model from given checkpoint file, evaluates on hvm10 train or
test sets and saves features (at each time step) of a specified layer to
.npy files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import tensorflow as tf
import numpy as np
import pandas

import dldata
import dldata.stimulus_sets.hvm as hvm
from dldata.metrics.utils import compute_metric_base

import model


HVM10_TRAIN_PATH = '/home/mrui/bypass/consistency/images/'
META_TRAIN_PATH = '/home/mrui/bypass/consistency/metadata.pkl'
# META_TRAIN_PATH = '/mindhive/dicarlolab/u/qbilius/computed/mrui/metadata.pkl'
HVM10_TEST_PATH = '/home/mrui/bypass/consistency/images/'
META_TEST_PATH = '/home/mrui/bypass/consistency/hvm10test.pkl'
# META_TEST_PATH = '/mindhive/dicarlolab/u/qbilius/computed/mrui/hvm10test.pkl'

IMAGE_SIZE_ORIG = 256
PIXEL_DEPTH = 255


def get_features(params, outfile,
                 layer, train, test, checkpoint_dir=None, vars_path=None):

    assert (train or test), \
        'specify at least one set of images (train or test)'
    if train:  # get train images and do stuff
        with tf.Graph().as_default():
            # read in pickle file
            meta_pkl = pandas.read_pickle(META_TRAIN_PATH)
            # convert the rec_array to a DataFrame
            meta_data = pandas.DataFrame.from_records(meta_pkl)

            # get list of image paths for tf FileReader and make filename queue
            filenames_list = [HVM10_TRAIN_PATH + id + '.png' for id in
                              meta_data.id]
            filename_queue = tf.train.string_input_producer(filenames_list, shuffle=False)
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            print('Reading from training set')
            # Read image to tf object; if grayscale, convert to RGB (3 channel)
            img_raw = tf.image.decode_png(value, channels=3)
            np_features = _features(img_raw=img_raw,
                            num_imgs=len(filenames_list),
                            params=params, layer=layer,
                            checkpoint_dir=checkpoint_dir, vars_path=vars_path)
            for t in np_features.keys():
                np.save(outfile + 'train_' + str(t), np_features[t])

    if test:  # get test images and do stuff
        with tf.Graph().as_default():
            # read in pickle file
            meta_data = pandas.read_pickle(META_TEST_PATH)
            # get list of image paths for tf FileReader and make filename queue
            filenames_list = [HVM10_TEST_PATH + a.split('20110131/')[1]
                              for a in meta_data]
            filename_queue = tf.train.string_input_producer(filenames_list, shuffle=False)
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            print('Reading from test set')
            # Read image to tf object; if grayscale, convert to RGB (3 channel)
            img_raw = tf.image.decode_png(value, channels=3)
            np_features = _features(img_raw=img_raw,
                                    num_imgs=len(filenames_list),
                                    params=params, layer=layer,
                                    checkpoint_dir=checkpoint_dir,
                                    vars_path=vars_path)
            for t in np_features.keys():
                np.save(outfile + 'test_' + str(t), np_features[t])


def _features(img_raw, num_imgs, params, layer,
              checkpoint_dir=None, vars_path=None):
    """ Returns features in numpy array for the given image from the specified
    layer
    :params img_raw: from tf decoding the image pngs
    :param num_imgs: number of images to evaluate
    :param params: parameters dictionary
    :param layer: which layer to get features from
    :param checkpoint_dir: directory with checkpoint file
    :param vars_path: file containing tf saver's variable file
    """
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=params['log_device_placement'])) as sess:

        image = tf.expand_dims(img_raw, 0)  # add a batch dimension
        # need to explicitly specify the dimensions
        image = tf.reshape(image, shape=[1, IMAGE_SIZE_ORIG,
                                    IMAGE_SIZE_ORIG, 3])

        # center crop image and rescale pixels
        cropped_shape = [1, params['image_size_crop'],  # batch size 1
                         params['image_size_crop'], 3]
        r = (IMAGE_SIZE_ORIG - params['image_size_crop']) // 2
        image = tf.slice(image, begin=[0, r, r, 0],
                         size=cropped_shape)
        image = tf.to_float(image, name='ToFloat')  # convert to float
        image = (image - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

        print('image shape after reshape:', image.get_shape())
        print(image)

        # Input sequence
        if params['input_seq'] is None:
            input_seq = [image for t in range(0, params['T_tot'])]
        else:
            input_seq = params['input_seq']  # TODO: how to get other input
            # (todo) sequences? using functions?

        # Create model
        features = model._model(layers=params['layers'], layer_sizes=params[
            'layer_sizes'], bypasses=params['bypasses'], input_seq=input_seq,
                          T_tot=params['T_tot'],
                          initial_states=params['initial_states'],
                          features_layer=layer)
        # of the form: {t1: ___, t2: ___, ...}

        # Restore variables from checkpoint_dir or vars_path
        init = tf.initialize_all_variables()
        sess.run(init)
        print("We've initialized!")
        saver = tf.train.Saver()
        if checkpoint_dir is None:  # use vars_path
            restore_file = vars_path
        else:  # use checkpoint_dir to get most recent file output
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    restore_file = ckpt.model_checkpoint_path
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    restore_file = os.path.join(checkpoint_dir,
                                                ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return
        saver.restore(sess, restore_file)
        # extract global step from model checkpoint path name
        global_step = restore_file.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (restore_file, global_step))

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # each value in outarray is list of features for each image
        features_all = {t: [] for t in features.keys()}

        for i in range(num_imgs):
            features_val = sess.run(features)
            # outputted times = features.keys()
            for t in features.keys():
                features_all[t].append(features_val[t])
        coord.request_stop()
        coord.join(threads)

    # convert features into numpy arrays
    for t in features.keys():
        features_all[t] = np.asarray(features_all[t])
    return features_all


def run_behav():
    import sys
    sys.path.insert(0, '../')
    import train_clf
    import common

    train = common.HvM10Train()
    test = common.HvM10()

    train_meta = pandas.read_pickle(META_TRAIN_PATH)
    for tm1, tm2 in zip(train_meta, train.meta):
        assert tm1[-2]==tm2[-3]

    test_meta = pandas.read_pickle(META_TEST_PATH)
    for tm1, tm2 in zip(test_meta, test.meta['filename']):
        assert tm1==tm2

    train_feats = np.load('/mindhive/dicarlolab/u/qbilius/computed/mrui/anet_byp13_train_7.npy')
    train_feats = np.squeeze(train_feats)
    test_feats = np.load('/mindhive/dicarlolab/u/qbilius/computed/mrui/anet_byp13_test_7.npy')
    test_feats = np.squeeze(test_feats)
    import pdb; pdb.set_trace()

    clf = train_clf.Clf2AFC(nfeats=1000, norm=False)
    clf.fit(train_feats, train.meta['obj'], train.OBJS, decision_function_shape='ovr')
    confi = clf.predict_proba(test_feats, targets=test.meta['obj'], kind='2-way')
    conf_pred = pandas.Series(confi, index=test.meta['id'])
    agg = test.human_data.groupby('id').acc.mean()
    print(agg.corr(conf_pred))
    return conf_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser('logits.py')
    # path to parameters file
    parser.add_argument('-p', '--params', default='./params.json',
                        help='path to parameters file', dest='params',
                        required=True)
    # path to tf variable file or directory containing checkpoint file
    parser.add_argument('-c', '--checkpoint_dir', help='directory containing'
                        'checkpoint file to restore variables',
                        dest='checkpoint_dir')
    parser.add_argument('-v', '--vars_path', help='direct path to variable '
                      'file (instead of checkpoint dir). If both provided, '
                      'checkpoint is used',
                        dest='vars_path')

    # hvm10 training or testing images (or both)
    parser.add_argument('--train', dest='train', action='store_true',
                        default=False,
                        help='Use training image set')
    parser.add_argument('--test', dest='test', action='store_true',
                        default=False,
                        help='Use test image set')

    # output path for features and which features desired
    parser.add_argument('-o', '--out', help='path to output features file',
                        dest='out')
    parser.add_argument('-l', '--layer', help='layer to get feature from',
                        dest='layer')  # default None => take last logits
    # todo: make it take multiple layers

    args = parser.parse_args()
    if args.checkpoint_dir is None and args.vars_path is None:
        raise ValueError(
            'Please provide either checkpoint directory or path to'
            ' model parameters file ')
    with open(args.params, 'r') as f:
        params = json.load(f)
    # Convert keys from String to Int since JSON serialization turns
    # everything into strings.
    for entry in ['layers', 'layer_sizes']:
        params[entry] = {int(k): v for k, v in params[entry].items()}

    params['batch_size'] = 1
    for k,val in params['layers'].items():
        if 'output_size' in val[1]:
            params['layers'][k][1]['output_size'] = [1] + val[1]['output_size'][1:]
        if 'state_size' in val[1]:
            params['layers'][k][1]['state_size'] = [1] + val[1]['state_size'][1:]
    for k,val in params['layer_sizes'].items():
        if 'output' in val:
            params['layer_sizes'][k]['output'] = [1] + val['output'][1:]
        if 'state' in val:
            params['layer_sizes'][k]['state'] = [1] + val['state'][1:]
    # params['save_path'] = '/home/qbilius/mh17/computed/anet_byp1/outputs/anet_byp1'
    # print(params)
    get_features(params,
                 outfile=args.out,
                 layer=args.layer,
                 checkpoint_dir=args.checkpoint_dir,
                 vars_path=args.vars_path,
                 train=args.train,
                 test=args.test
                 )

