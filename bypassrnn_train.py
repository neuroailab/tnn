from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import bypassrnn_model

import bypassrnn_params as params
import image_processing
from imagenet_data import ImagenetData
import math
import numpy as np
import time
import sys
import threading
""" openmind (multi-gpu) version
"""


LOG_DEVICE_PLACEMENT = params.LOG_DEVICE_PLACEMENT #whether to log device placement

def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def run_train():
    """Note: Currently, train and eval use different (equiv.) methods of evaluating error. in future, we will transition
    train to use tf.nn.in_top_k as well.
    """""
    with tf.Graph().as_default(), tf.device('/cpu:0'):  # To have multiple graphs in same process
        # process data on cpu
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * params.NUM_GPUS.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = int((params.TRAIN_SIZE /
                                 params.BATCH_SIZE))
        decay_steps = int(num_batches_per_epoch * params.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.

        if params.LEARNING_RATE_DECAY_FACTOR is None:
            learning_rate = LEARNING_RATE_BASE  # just a constant.
        else:
            learning_rate = tf.train.exponential_decay(
                                        params.LEARNING_RATE_BASE,# Base learning rate.
                                        global_step,# Current index into the dataset.
                                        decay_steps,# Decay step (aka once every EPOCH)
                                        params.LEARNING_RATE_DECAY_FACTOR,# Decay rate.
                                        staircase=True)

        # Create optimizer for gradient descent.
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params.MOMENTUM)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Note: Switched from adam to momentum

        # Get images and labels for ImageNet and split the batch across GPUs.
        assert params.BATCH_SIZE % params.NUM_GPUS == 0, (
            'Batch size must be divisible by number of GPUs')
        split_batch_size = int(params.BATCH_SIZE / params.NUM_GPUS) # calculate batch_size/gpu
        data, enqueue_op, images_batch, labels_batch = image_processing.inputs(train=True)
        print('have obtained images and labels')

        # Split the batch of images and labels for towers.
        images_splits = tf.split(0, params.NUM_GPUS, images_batch)
        labels_splits = tf.split(0, params.NUM_GPUS, labels_batch)
        print('have split the images')

        # Calculate the gradients for each model tower.
        tower_grads = [] # keep track of gradients for all towers. (so we can avg later)
        for i in xrange(params.NUM_GPUS):
            with tf.device('/gpu:%d' % i): # Do training on GPU
                with tf.name_scope('%s_%d' % (params.TOWER_NAME, i)) as scope:
                    # Force all Variables to reside on the CPU.
                    with tf.device('/cpu:0'):
                        # Calculate the loss for one tower of the ImageNet model. This
                        # function constructs the entire ImageNet model but shares the
                        # variables across all towers.
                        fetch_dict = bypassrnn_model._model(params.LAYERS, params.LAYER_SIZES, params.BYPASSES, images_splits[i], labels_splits[i],
                                            train=True, keep_prob = params.KEEP_PROB, initial_states=None)
                        # Note: for now, keep_prob refers to after GAP layer. Can and will probably change in definition though.
                        # TODO- MAKE FETCH_DICT JUST BE OF PREDICTIONS.....
                    # Reuse variables for the next tower after creating first (shared btwn all)
                    tf.get_variable_scope().reuse_variables()
                    # Calculate the gradients for the batch of data on this ImageNet tower.
                    gvs = optimizer.compute_gradients(fetch_dict['tot_loss'])  # .minimize = compute gradients and apply them.
                    # Note: Unlike prev model, we apply gradients later, after averaging all tower gradients
                    tower_grads.append(gvs)# Keep track of the gradients across all towers.
        print('have gotten feed_dict as you already know')
        # calculate the mean of each gradient. Note: synchronization point across all towers.
        print('tower grads:', tower_grads)
        grads = _average_gradients(tower_grads)
        if params.GRAD_CLIP:
            capped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads if not grad is None]
            # gradient clipping. Note: some gradients returned are 'None' bcuz no relation btwn that var and tot loss; so we skip those.
            apply_grad_op = optimizer.apply_gradients(capped_grads, global_step = global_step) # returns an Operation to
            print('Gradients clipped')
        else:
            apply_grad_op = optimizer.apply_gradients(grads, global_step = global_step)
            print('Gradients not clipped')
        print('have applied gradients')
        fetch_dict.update({'opt': apply_grad_op, 'lr': learning_rate})


        saver = tf.train.Saver(max_to_keep=params.MAX_TO_KEEP)

        # Note: first, initialize all variables. THEN, if you want to restore existing vars, do so.
        # Create session and initialize variables
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init)
        print("We've initialized!")
        sys.stdout.flush()
        # restore model from checkpoint if RESTORE_VARS==True
        if params.RESTORE_VARS:
            if file is None:
                raise ValueError('Please provide a file from which to load variables!')
            # TODO: let this choose from checkpoint? At least in eval mode.
            restorer = tf.train.Saver() # currently, restore all variables NOTE: Can change to restore select vars if needed
            restorer.restore(sess, params.RESTORE_VAR_FILE)  # restore variables from restore_var_file to graph
            # Note: should also restore the global_step, at least in our current setup.
            print("Variables Restored!")

        # start queue runners (after initialization step!) to start filling up queues
        coord = tf.train.Coordinator()
        a = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            threads = []
            for i in range(params.NUM_PREPROCESS_THREADS):
                thread = threading.Thread(target=data.load_and_enqueue, args=(sess, enqueue_op, coord))
                thread.daemon = True # thread will close when parent quits
                thread.start()
                threads.append(thread)
            # Prepare output (of loss) to file
            write_out_freq = params.SAVE_VARS_FREQ / 10 # how often to write update losses to file.
            tot_losses_len = int(math.floor(write_out_freq/params.SAVE_LOSS_FREQ)) # to save a few re-computations. FLOOR since we start with step = 1
            tot_losses = np.zeros([tot_losses_len, 2]) # to write to file
            outfile_loss = params.SAVE_FILE + '_loss.csv' # file to save losses to.
            f = open(outfile_loss, "w+")
            f.close()

            # num training batches = total number of imgs (including epoch repeats)/batch size
            if params.RESTORE_VARS:
                start_step = params.START_STEP + 1  # to keep consistent count (of epochs passed, etc.)
            else:
                start_step = 1  # start from the very beginning, a very good place to start.

            start_time = start_time_step = time.time()  # start timer
            print('batch size per GPU: ', split_batch_size)
            for step in xrange(start_step, int(params.NUM_EPOCHS * num_batches_per_epoch + 1)):  # start with step=START_STEP or 1.
                print('STEP:', step, 'BEGINS')
                sys.stdout.flush()  # flush the stdout buffer
                # Note: we've already created fetch_dict from images/labels.
                results = sess.run(fetch_dict)  # results as a dictionary- {'loss_2': loss, 'lr': lr, etc..}
                assert not np.isnan(results['tot_loss']), 'Model diverged with loss = NaN'
                # Note: for training, we only look at final loss. Intermediate steps irrelevant. Predictions/error also irrelevant.
                if params.SAVE_LOSS and step%params.SAVE_LOSS_FREQ == 0: # only print output every so often as well.
                    tot_losses[step//params.SAVE_LOSS_FREQ%tot_losses_len - 1, :] = [step, results['tot_loss']] # save so we can write to file later
                print('Step %d total_loss: %.6f, lr: %.6f' % (
                    step, results['tot_loss'], results['lr']))  # last error rate
                sys.stdout.flush()  # flush the stdout buffer
                elapsed_time_step = time.time() - start_time_step
                start_time_step = time.time()
                print('step %d, %.1f ms' % (step, 1000 * elapsed_time_step))
                if step % params.SAVE_VARS_FREQ == 0:  # how often to SAVE OUR VARIABLES
                    if params.SAVE_VARS == True:
                        saver.save(sess, save_path=params.SAVE_FILE, global_step=global_step)
                        print('saved.')
                if step % write_out_freq == 0: # not really doing anything with Eval; just a good intermittent marker
                    elapsed_time = time.time() - start_time  # time between evaluations.
                    print('Step %d (epoch %.2f), avg step time:  %.1f ms' % (
                        step, float(step) / params.TRAIN_SIZE,  # # batches * (#epoch/batch)
                        1000 * elapsed_time / write_out_freq))  # average step time
                    # Write to file. Note: we only write every EVAL_FREQUENCY to limit I/O bottlenecking.
                    if params.SAVE_LOSS:
                        with open(outfile_loss, 'ab') as f_handle:
                            np.savetxt(
                                f_handle,  # file name
                                tot_losses,  # array to save
                                fmt='%.3f',  # formatting, 3 digits in this case
                                delimiter=',',  # column delimiter
                                newline='\n')  # new line character
                    # bypassrnn_eval._eval_once(eval_graph, eval_restorer, params.CHECKPOINT_DIR, top1_counts, top5_counts) # Evaluate
                    start_time = time.time()  # reset timer.
                sys.stdout.flush()  # flush the stdout buffer
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


if __name__ == "__main__":
    run_train()
