#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader
from romanesco.const import *
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph


def train(data: str, epochs: int, batch_size: int, vocab_max_size: int,
          save_to: str, log_to: str, **kwargs):
    """Trains a language model. See argument description in `bin/romanesco`."""

    # create folders for model and logs if they don't exist yet
    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # create vocabulary to map words to ids
    vocab = Vocabulary()
    vocab.build(data, max_size=vocab_max_size)
    vocab.save(os.path.join(save_to, VOCAB_FILENAME))

    # convert training data to list of word ids
    raw_data = reader.read(data, vocab)

    # define computation graph
    inputs, targets, loss, train_step, _, summary = define_computation_graph(vocab.size, batch_size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())
        # iterate over training data `epoch` times
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_iter = 0
            for x, y in reader.iterate(raw_data, batch_size, NUM_STEPS):
                l, _, s = session.run([loss, train_step, summary],
                                      feed_dict={inputs: x, targets: y})
                summary_writer.add_summary(s, total_iter)
                total_loss += l
                total_iter += 1
                if total_iter % 100 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)
            perplexity = np.exp(total_loss / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, MODEL_FILENAME))
