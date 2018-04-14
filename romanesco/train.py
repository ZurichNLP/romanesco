#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader, const
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph


def train(data: str, epochs: int = 10, save_to: str = 'model', log_to: str = 'logs', **kwargs):
    """Trains a language model.

    Arguments:
        data: the path to a plain text file containing all training data.
        num_epochs: number of times to iterate over all training data.
        save_to: the path to a folder where model parameters and vocabulary will
            be stored. Folder will be created if it doesn't exist yet.
        log_to: the path to a folder where all training logs will be stored.
            Folder will be created if it doesn't exist yet. Point tensorboard
            to this folder to monitor training.
    """
    vocab = Vocabulary()
    vocab.build(data, max_size=50000)
    vocab.save(os.path.join(save_to, 'vocab.json'))
    
    raw_data = reader.read(data, vocab)

    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    inputs, targets, train_fn, cost, prediction, summary = define_computation_graph(vocab.size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())

        for epoch in range(1, epochs + 1):
            total_cost = 0.0
            total_iter = 0
            for x, y in reader.iterate(raw_data, const.BATCH_SIZE, const.NUM_STEPS):
                c, p, _, s = session.run([cost, prediction, train_fn, summary],
                                         feed_dict={
                                             inputs: x,
                                             targets: y,
                                         })
                summary_writer.add_summary(s, total_iter)
                total_cost += c
                total_iter += 1
                if total_iter % 100 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)
            perplexity = np.exp(total_cost / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, 'model.epoch-{0}'.format(epoch)))
