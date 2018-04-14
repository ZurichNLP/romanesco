#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader
from romanesco.const import *
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph


def train(data: str, epochs: int = 10, batch_size: int = 50,
          save_to: str = 'model', log_to: str = 'logs', **kwargs):
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
    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    vocab = Vocabulary()
    vocab.build(data, max_size=50000)
    vocab.save(os.path.join(save_to, VOCAB_FILENAME))

    raw_data = reader.read(data, vocab)

    inputs, targets, loss, train_step, prediction, summary = define_computation_graph(vocab.size, batch_size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_iter = 0
            for x, y in reader.iterate(raw_data, batch_size, NUM_STEPS):
                l, p, _, s = session.run([loss, prediction, train_step, summary],
                                         feed_dict={
                                             inputs: x,
                                             targets: y,
                                         })
                summary_writer.add_summary(s, total_iter)
                total_loss += l
                total_iter += 1
                if total_iter % 100 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)
            perplexity = np.exp(total_loss / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, MODEL_FILENAME))
