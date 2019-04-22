#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph

from romanesco import const as C


def train(data: str,
          epochs: int = C.NUM_EPOCHS,
          batch_size: int = C.BATCH_SIZE,
          hidden_size: int = C.HIDDEN_SIZE,
          embedding_size: int = C.EMBEDDING_SIZE,
          vocab_max_size: int = C.VOCAB_SIZE,
          save_to: str = C.MODEL_PATH,
          log_to: str = C.LOGS_PATH,
          num_steps: int = C.NUM_STEPS,
          **kwargs):
    """Trains a language model. See argument description in `bin/romanesco`."""

    # create vocabulary to map words to ids
    vocab = Vocabulary()
    vocab.build(data, max_size=vocab_max_size)
    vocab.save(os.path.join(save_to, C.VOCAB_FILENAME))

    # convert training data to list of word ids
    raw_data = reader.read(data, vocab)

    # define computation graph
    inputs, targets, loss, train_step, _, summary = define_computation_graph(vocab_size=vocab.size,
                                                                             batch_size=batch_size,
                                                                             num_steps=num_steps,
                                                                             hidden_size=hidden_size,
                                                                             embedding_size=embedding_size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())
        # iterate over training data `epochs` times
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_iter = 0
            for x, y in reader.iterate(raw_data, batch_size, C.NUM_STEPS):
                l, _, s = session.run([loss, train_step, summary],
                                      feed_dict={inputs: x, targets: y})
                summary_writer.add_summary(s, total_iter)
                total_loss += l
                total_iter += 1
                if total_iter % 100 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)
            perplexity = np.exp(total_loss / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, C.MODEL_FILENAME))
