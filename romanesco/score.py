#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader
from romanesco import const as C
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph


def score(data: str,
          load_from: str = C.MODEL_PATH,
          batch_size: int = C.BATCH_SIZE,
          hidden_size: int = C.HIDDEN_SIZE,
          embedding_size: int = C.EMBEDDING_SIZE,
          num_steps: int = C.NUM_STEPS,
          **kwargs):
    """Scores a text using a trained language model. See argument description in `bin/romanesco`."""

    vocab = Vocabulary()
    vocab.load(os.path.join(load_from, C.VOCAB_FILENAME))

    raw_data = reader.read(data, vocab)
    data_length = len(raw_data)

    if data_length < num_steps:
        logging.warning("Length of input data is shorter than NUM_STEPS. Will try to reduce NUM_STEPS.")
        num_steps = data_length - 1

    if data_length < batch_size * num_steps:
        logging.warning("Length of input data is shorter than BATCH_SIZE * NUM_STEPS. Will try to set batch size to 1.")
        batch_size = 1

    inputs, targets, loss, _, _, _ = define_computation_graph(vocab_size=vocab.size,
                                                              batch_size=batch_size,
                                                              num_steps=num_steps,
                                                              hidden_size=hidden_size,
                                                              embedding_size=embedding_size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        total_loss = 0.0
        total_iter = 0
        for x, y in reader.iterate(raw_data, batch_size, num_steps):
            l = session.run([loss], feed_dict={inputs: x, targets: y})
            total_loss += l[0]
            total_iter += 1
        perplexity = np.exp(total_loss / total_iter)
        return perplexity
