""" Creates and trains model, possibly starting from given tf checkpoint"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model
import image_processing
import math
import numpy as np
import time
import sys
import threading
import argparse
import json


def run_train(params):
    """
    :param params: dictionary of params
    """
    # TODO: try having no gpu/cpu constraints and see effect; then try only gpu
    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]
        # create session
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=params['log_device_placement']))

        # Get images and labels for ImageNet.
        data, enqueue_op, images_batch, labels_batch = image_processing.inputs(
            train=True, params=params)
        print('images and labels done')

        # Input sequence
        if params['input_seq'] is None:
            input_seq = [images_batch for t in range(0, params['T_tot'])]
        else:
            input_seq = params['input_seq']  # TODO: how to get other input
            # sequences?

        logits = model._model(params['layers'], params['layer_sizes'],
                              params['bypasses'], input_seq,
                              params['T_tot'], params['initial_states'],
                              num_labels=params['num_labels'])

        # Calculate total loss (with time penalty)
        losses = {t: tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logit,
                                                           labels_batch),
            name='xentropy_loss_t' + str(t))
                  for t, logit in logits.items()}
        shortest_path = min(losses.keys())  # earliest time output
        losses_with_penalty = [tf.mul(loss, math.pow(params['time_penalty'],
                                                     t - shortest_path)) for
                               t, loss in losses.items()]
        total_loss = tf.add_n(losses_with_penalty, name='total_loss')

        # Create a variable to count the number of train() calls.
        # This equals the number of batches processed.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        
        # Calculate the learning rate schedule.
        num_batches_per_epoch = int((params['train_size'] /
                                     params['batch_size']))
        decay_steps = int(
            num_batches_per_epoch * params['num_epochs_per_decay'])

        # Decay the learning rate exponentially based on the number of steps.
        if params['learning_rate_decay_factor'] is None:
            learning_rate = params['learning_rate_base']  # just a constant.
        else:
            learning_rate = tf.train.exponential_decay(
                params['learning_rate_base'],  # Base learning rate.
                global_step,  # Current index into the dataset.
                decay_steps,  # Decay step (aka once every EPOCH)
                params['learning_rate_decay_factor'],  # Decay rate.
                staircase=True)

        # Create optimizer for gradient descent.
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=params['momentum'])
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(total_loss)
        if params['grad_clip']:
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

        if params['tensorboard']:  # save graph to tensorboard
            tf.train.SummaryWriter(params['checkpoint_dir'], sess.graph)

        # fetch dict for session to run
        fetch_dict = ({'opt': optimizer, 'lr': learning_rate})
        fetch_dict.update(losses)
        fetch_dict.update({'tot_loss': total_loss})

        saver = tf.train.Saver(max_to_keep=params['max_to_keep'])

        # initialize and/or restore variables for graph
        init = tf.initialize_all_variables()
        sess.run(init)
        print("Variables initialized")
        if params['restore_vars']:
            saver.restore(sess, params['restore_var_file'])
            print('Variables restored')

        # start queue runners (after initialization step!)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        threads = []
        try:
            for i in range(params['num_preprocess_threads']):
                thread = threading.Thread(target=data.load_and_enqueue,
                                          args=(sess, enqueue_op, coord))
                thread.daemon = True  # thread will close when parent quits
                thread.start()
                threads.append(thread)
            # Prepare output (of loss) to file
            write_out_freq = params['save_vars_freq'] / 10  # write loss->file
            tot_losses_len = int(math.floor(write_out_freq / params[
                'save_loss_freq']))
            tot_losses = np.zeros([tot_losses_len, 2])  # to write to file
            outfile_loss = params['save_path'] + '_loss.csv'  # to save losses
            f = open(outfile_loss, "w+")
            f.close()

            if params['restore_vars']:
                # to keep consistent count (of epochs passed, etc.)
                start_step = params['start_step'] + 1
            else:
                start_step = 1

            start_time_step = time.time()  # start timer
            for step in xrange(start_step, int(params['num_epochs'] *
                                               num_batches_per_epoch + 1)):
                # get run output as dictionary {'2': loss2, 'lr': lr, etc..}
                results = sess.run(fetch_dict)
                assert not np.isnan(results['tot_loss']), \
                    'Model diverged with loss = NaN'

                # save variables to checkpoint
                if params['save_vars'] and \
                                    step % params['save_vars_freq'] == 0:
                    saver.save(sess, save_path=params['save_path'],
                               global_step=global_step)
                    print('saved variable checkpoint')

                # write loss to file
                if params['save_loss'] and step % params[
                                                    'save_loss_freq'] == 0:
                    tot_losses[step // params['save_loss_freq'] %
                       tot_losses_len - 1, :] = [step, results['tot_loss']]

                print('Step %d total_loss: %.6f, lr: %.6f' % (
                    step, results['tot_loss'], results['lr']))
                elapsed_time_step = time.time() - start_time_step
                start_time_step = time.time()
                print('step %d, %.1f ms' % (step, 1000 * elapsed_time_step))

                if step % write_out_freq == 0:
                    # Write to file. only every EVAL_FREQUENCY to limit I/O
                    if params['save_loss']:
                        with open(outfile_loss, 'ab') as f_handle:
                            np.savetxt(
                                f_handle,  # file name
                                tot_losses,  # array to save
                                fmt='%.3f',  # formatting, 3 digits
                                delimiter=',',  # column delimiter
                                newline='\n')  # new line character
                sys.stdout.flush()  # flush the stdout buffer

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameters to JSON')
    parser.add_argument('-p', '--params', default='./params.json',
                        help='path to parameters file', dest='params',
                        required=True)

    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = json.load(f)
    # Convert keys from String to Int since JSON serialization turns
    # everything into strings.
    for entry in ['layers', 'layer_sizes']:
        params[entry] = {int(k): v for k, v in params[entry].items()}
    run_train(params)
