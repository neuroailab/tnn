""" Creates and evaluates model from given checkpoint file. May evaluate
once or multiple times

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model
import image_processing
import numpy as np
import time
import threading
import argparse
import json
import os
from datetime import datetime
import csv
import sys


def _eval_once(params, saver, top_1_ops, top_5_ops, checkpoint_dir,
               vars_path, data, enqueue_op):
    """
    Runs evaluation once
    :param params: dictionary of model and eval parameters
    :param saver: tf Saver object
    :param top_1_ops: tf Operation to count # labels in top 1
    :param top_5_ops: tf Operation to count # correct labels in top 5
    :param checkpoint_dir: Directory containing checkpoint file
    :param vars_path: path to variable file to restore (used if checkpoint_dir
    not specified)
    :param data: ImageNetSingle object to work with queues
    :param enqueue_op: tf Op for queuerunners
    """
    outfile_top1 = params['save_path'] + '_top1err.csv'
    outfile_top5 = params['save_path'] + '_top5err.csv'
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=params['log_device_placement'])) as sess:
        # restore variables from checkpoint_dir or vars_path
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
        # Assuming model_checkpoint_path looks something like:
        #   /my-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = restore_file.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (restore_file, global_step))

        # start queue runners (after initialization/restore step!)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(params['num_preprocess_threads']):
                thread = threading.Thread(target=data.load_and_enqueue,
                                          args=(sess, enqueue_op, coord))
                thread.daemon = True  # thread will close when parent quits
                thread.start()
                threads.append(thread)

            num_iter = params['num_validation_batches']
            # Counts the number of correct predictions; initialize counters
            t_keys = top_1_ops.keys()
            count_top_1 = {t: 0.0 for t in t_keys}
            count_top_5 = {t: 0.0 for t in t_keys}

            total_sample_count = num_iter * params['batch_size']
            step = 0
            print('%s: starting evaluation.' % (datetime.now()))
            sys.stdout.flush()  # flush the stdout buffer
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                top1_results = sess.run(top_1_ops)
                top5_results = sess.run(top_5_ops)

                for t in t_keys:
                    # sum # correct for num_iter times
                    count_top_1[t] += np.sum(top1_results[t])
                    count_top_5[t] += np.sum(top5_results[t])

                step += 1
                if step % 50 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 50.0
                    examples_per_sec = params['batch_size'] / \
                               sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()
                    sys.stdout.flush()  # flush the stdout buffer
            # Compute top 1 and top 5 error
            top1_error = {t: (1 - count_top_1[t] / total_sample_count) * 100.0
                          for t in t_keys}
            top5_error = {t: (1 - count_top_5[t] / total_sample_count) * 100.0
                          for t in t_keys}
            T_tot = max(t_keys)
            print('Step %.2f: top1_error = %.4f top5_error = %.4f [%d '
                  'examples]' % (int(global_step), top1_error[T_tot],
                                 top5_error[T_tot], total_sample_count))

            # write results to appropriate files.
            fieldnames = ['step'] + t_keys
            with open(outfile_top1, 'ab') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                write_dict = {'step': int(global_step)}
                write_dict.update(top1_error)
                writer.writerow(write_dict)
            with open(outfile_top5, 'ab') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                write_dict = {'step': int(global_step)}
                write_dict.update(top5_error)
                writer.writerow(write_dict)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def run_eval(params, eval_once, eval_interval_secs=None,
             checkpoint_dir=None,
             vars_path=None):
    """
    Evaluates model - top1 and top5 error -  either once or every
    eval_interval_secs. Saves errors for each time output to files.
    If params['eval_avg_crop'] == True, evaluates by taking average of
    logits from 5 crops (4 corners and 1 center similar to AlexNet)
    :param params: dictionary of model and eval parameters
    :param eval_once: If True, evaluates only once rather than every # secs
    :param eval_interval_secs: Number of seconds between evaluations (only
    matters if eval_once = False)
    :param checkpoint_dir: Directory containing checkpoint file
    :param vars_path: path to variable file to restore (used if checkpoint_dir
    not specified)
    """
    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]

        # Get images and labels for ImageNet.
        if params['eval_avg_crop']:
            print('Using avg of 5 crops')
            # take average of logits from 5 crops (4 corners and 1 center)
            # get original size (256) image then take random crop
            data, enqueue_op, images_batch, labels_batch = image_processing.inputs(
                train=False, data_path=params['data_path'],
                crop_size=None,  # take original image size
                batch_size=params['batch_size'],
                num_validation_batches=params['num_validation_batches'])

            # 5 Crops

            cropped_shape = [params['batch_size'], params['image_size_crop'],
                             params['image_size_crop'], params['num_channels']]
            spatial_orig = images_batch.get_shape().as_list()[1]  # 256
            # top left, bottom left, top right, bottom right, center slices
            slice_tl = tf.slice(images_batch, begin=[0, 0, 0, 0],
                            size=cropped_shape)
            slice_bl = tf.slice(images_batch, begin=[0, spatial_orig - params[
                                    'image_size_crop'], 0, 0],
                                size=cropped_shape)
            slice_tr = tf.slice(images_batch, begin=[0, 0, spatial_orig -
                                params['image_size_crop'], 0],
                                size=cropped_shape)
            slice_br = tf.slice(images_batch, begin=[0, spatial_orig - params[
                                'image_size_crop'], spatial_orig -
                                 params['image_size_crop'], 0],
                                size=cropped_shape)
            r = (spatial_orig - params['image_size_crop']) // 2
            slice_c = tf.slice(images_batch, begin=[0, r, r, 0],
                               size=cropped_shape)
            image_inputs = [slice_tl, slice_bl, slice_tr, slice_br, slice_c]

        else:  # get center cropped image
            print('Using center crop')
            data, enqueue_op, images_batch, labels_batch = image_processing.inputs(
                train=False, data_path=params['data_path'],
                crop_size=params['image_size_crop'],  # cropped size
                batch_size=params['batch_size'],
                num_validation_batches=params['num_validation_batches'])
            image_inputs = [images_batch]
        # image_inputs is a list of batches of images
        print('images and labels done')

        logits_list = []
        for i, img_batch in enumerate(image_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # Input sequence
            if params['input_seq'] is None:
                input_seq = [img_batch for t in range(0, params['T_tot'])]
            else:
                input_seq = params['input_seq']  # TODO: how to get other input
                # (todo) sequences? using functions?

            _logits = model._model(params['layers'], params['layer_sizes'],
                                    params['bypasses'], input_seq,
                                    params['T_tot'], params['initial_states'],
                                    num_labels=params['num_labels'])
            # _logits = dictionary {t: logits at time t}, collected in
            # logits_list (if using multiple crops)
            logits_list.append(_logits)

        # average logits
        logits = {}
        for t in _logits.keys():
            logits_t = [l[t] for l in logits_list]
            # take average of logits
            logits[t] = tf.div(tf.add_n(logits_t), float(len(logits_t)))

        # Collect # top1, top5 correct guesses, for each time point, in dict
        top_1_ops = {t: tf.nn.in_top_k(logit, labels_batch, 1)
                     for t, logit in logits.items()}
        top_5_ops = {t: tf.nn.in_top_k(logit, labels_batch, 5)
                     for t, logit in logits.items()}
        saver = tf.train.Saver()

        # Clear and write header to output files
        outfile_top1 = params['save_path'] + '_top1err.csv'
        outfile_top5 = params['save_path'] + '_top5err.csv'
        for outf in [outfile_top1, outfile_top5]:
            f = open(outf, "w+")
            fieldnames = ['step'] + [str(t) for t in top_1_ops.keys()]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.close()
        while True:
            _eval_once(params, saver, top_1_ops, top_5_ops, checkpoint_dir,
                       vars_path, data, enqueue_op)
            if eval_once:
                break
            time.sleep(eval_interval_secs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameters to JSON')
    parser.add_argument('-p', '--params', default='./params.json',
                        help='path to parameters file', dest='params',
                        required=True)
    parser.add_argument('-e', '--eval_once', dest='eval_once',
                        action='store_true', default=False, help='evaluate '
                        'model once or continuously [default].')
    parser.add_argument('-f', '--eval_interval_secs', help='Seconds between '
                        'evaluations', dest='eval_interval_secs', type=float)
    parser.add_argument('-c', '--checkpoint_dir', help='directory containing'
                        'checkpoint file to restore variables',
                        dest='checkpoint_dir')
    parser.add_argument('-v', '--vars_path', help='direct path to variable '
                      'file (instead of checkpoint dir). If both provided, '
                      'checkpoint is used',
                        dest='vars_path')
    args = parser.parse_args()
    if args.checkpoint_dir is None and args.vars_path is None:
        raise ValueError(
            'Please provide either checkpoint directory or path to'
            ' model parameters file ')
    print('eval once:', args.eval_once)
    with open(args.params, 'r') as f:
        params = json.load(f)
    # Convert keys from String to Int since JSON serialization turns
    # everything into strings.
    for entry in ['layers', 'layer_sizes']:
        params[entry] = {int(k): v for k, v in params[entry].items()}
    run_eval(params, eval_once=args.eval_once,
             eval_interval_secs=args.eval_interval_secs,
             checkpoint_dir=args.checkpoint_dir,
             vars_path=args.vars_path)
