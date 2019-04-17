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

    if len(raw_data) < num_steps:
        logging.warning("Length of input data is shorter than NUM_STEPS. Will pad input with <unk>.")
        # Source: https://stackoverflow.com/a/39923577/1987598
        unk_id = vocab.get_id(C.UNK)
        raw_data = (raw_data + num_steps * [unk_id])[:num_steps]

    if len(raw_data) < batch_size * num_steps:
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
