from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import math
import os.path
import time

import csv
import numpy as np
import tensorflow as tf
import bypassrnn_params as params
import bypassrnn_model
import image_processing

""" openmind (multi-gpu) version
- changing to run separately from train."""

T = params.T
params.EVAL_INTERVAL = 5 #60 * 5 # seconds between eval runs
params.EVAL_RUN_ONCE = False # run eval only once



def _eval_once(eval_restorer, checkpoint_dir, top1_counts, top5_counts):
    """
    checkpoint_dir = directory to read model checkpoint files to restore variables
    Will use tensorflow's tf.nn.in_top_k top help evaluate error (via accuracy)
    We will still use error as our metric (1 - accuracy) * 100%
    Following tf's Inception paradigm, when evaluating, we restore the model parameters from checkpoint,
    since we have created separate graphs for evaluation and training
    # Note: removed any loss evaluation.
    """


    outfile_top1 = params.SAVE_FILE + '_top1_val.csv'
    outfile_top5 = params.SAVE_FILE + '_top5_val.csv'

    with tf.Session() as sess: # Note: sessions own their variables, queues, readers, so we use WITH to release
        # resources when no longer required.
        # use eval_graph so we keep correct graph during eval vs training

        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                # Restores from checkpoint with absolute path.
                eval_restorer.restore(sess, ckpt.model_checkpoint_path)
            else:
                # Restores from checkpoint with relative path.
                eval_restorer.restore(sess, os.path.join(checkpoint_dir,
                                                 ckpt.model_checkpoint_path))

            # extract global_step from checkpoint path name (last #s following '-')
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        # start queue runners (after initialization step!) to start filling up queues
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        threads = []

        try:
            for i in range(params.NUM_PREPROCESS_THREADS):
                thread = threading.Thread(target=data.load_and_enqueue, args=(sess, enqueue_op, coord))
                thread.start()
                threads.append(thread)

            num_iter = int(math.ceil(params.NUM_EVAL_EXAMPLES / params.BATCH_SIZE))
            # Counts the number of correct predictions.
            count_top_1 = {}
            count_top_5 = {}
            counts_keys = top1_counts.keys() # so we don't have to keep re-accessing
            for t in counts_keys: # initialize counters
                count_top_1[t] = 0.0
                count_top_5[t] = 0.0
            total_sample_count = num_iter * params.BATCH_SIZE
            step = 0

            print('%s: starting evaluation.' % (datetime.now()))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                top1_results = sess.run(top1_counts)
                top5_results = sess.run(top5_counts)
                for t in counts_keys:
                    count_top_1[t] += np.sum(top1_results[t]) # sum correct
                    count_top_5[t] += np.sum(top5_results[t])
                step += 1
                if step % 50 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 50.0
                    examples_per_sec = params.BATCH_SIZE / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()
            top1_error = {}; top5_error = {} # clear every time we run a new eval
            # Compute precision @ 1.
            for t in counts_keys:
                top1_error[t] = (1 - count_top_1[t] / total_sample_count) * 100.0
                top5_error[t] = (1 - count_top_5[t] / total_sample_count) * 100.0
            print('%s: top1_error = %.4f top5_error = %.4f [%d examples]' %
                  (datetime.now(), top1_error[T], top5_error[T], total_sample_count))

            # append results to appropriate files.

            with open(outfile_top1, 'ab') as csvfile:
                fieldnames = counts_keys
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(top1_error)
            with open(outfile_top5, 'ab') as csvfile:
                fieldnames = counts_keys
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(top5_error)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    """ dataset = tf dataset with images/labels
    eval_fun = fetch_dict to be run for evaluation
    Creates a graph for evaluation, and returns the error_dict that includes intermediate and final predictions
    error calculations. (ops created under tf.Graph()) . Should be called only once (to create a graph)
    Then runs _eval_once every EVAL_INTERVAL seconds
    """""
    eval_graph = tf.Graph()
    with eval_graph.as_default(), tf.device('/cpu:0'):  # To have multiple graphs in same process
        # process data on cpu
        # Get images and labels from the hdf5
        data, enqueue_op, images_batch, labels_batch = image_processing.inputs(train=False) # Note: no preprocess threads needed since simple retrieval
        # TODO REMOVE WHEN DONE (see max label #)
        labels_max = tf.reduce_max(labels_batch)
        sess = tf.Session()
        import pdb; pdb.set_trace()
        print('ello??')
        print(sess.run(labels_max)) # TODO: Q-> why do you get stuck here?
        ########################################################
        top1_counts = {}; top5_counts = {}
        # Create a graph to compute predictions (Note: Since loss is irrelevant for evaluation, we will toggle it off
        # (note) when creating model)
        # Note: I guess we don't need to distribute across GPUs for simple evaluation.
        fetch_dict = bypassrnn_model._model(params.LAYERS, params.LAYER_SIZES, params.BYPASSES, images_batch, labels_batch, train=False,
                                keep_prob=None, initial_states=None)

        #for training eval (params.EVAL_INTERMED=false), fetch_dict['pred_(T)'] is only entry
        #for general purpose eval, there is fetch_dict['pred_5'], fetch_dict['pred_6']...

        eval_restorer = tf.train.Saver() # currently, restore all variables NOTE: Can change to restore select vars if needed

        # TODO: a file where each column is an entry of fetch_dict and the values are the prediction errors.
        if params.EVAL_INTERMED:
            for t in range(params.SHORTEST_PATH, T + 1):
                top1_counts[t] = tf.nn.in_top_k(fetch_dict['pred_' + str(t)], labels, 1)
                top5_counts[t] = tf.nn.in_top_k(fetch_dict['pred_' + str(t)], labels, 5)
        else: # only final prediction matters
            top1_counts[T] = tf.nn.in_top_k(fetch_dict['pred_' + str(T)], labels, 1)
            top5_counts[T] = tf.nn.in_top_k(fetch_dict['pred_' + str(T)], labels, 5)
        outfile_top1 = params.SAVE_FILE + '_top1_val.csv'
        outfile_top5 = params.SAVE_FILE + '_top5_val.csv'
        # clear output files and write header.
        counts_keys = top1_counts.keys()
        for outf in [outfile_top1, outfile_top5]:
            f = open(outf, "w+")
            fieldnames = [str(t) for t in counts_keys]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.close()
        while True:
            _eval_once(eval_restorer, checkpoint_dir=params.CHECKPOINT_DIR, top1_counts=top1_counts, top5_counts=top5_counts)
            if params.EVAL_RUN_ONCE:
                break
            time.sleep(params.EVAL_INTERVAL)

def run_eval():
    evaluate()


if __name__ == "__main__":
    run_eval()
