#!/usr/bin/env python3

import os
import logging

import numpy as np
import tensorflow as tf

from romanesco import reader
from romanesco.const import *
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph


def score(data: str, load_from: str, batch_size: int, **kwargs):
    """Scores a text using a trained language model. See argument description in `bin/romanesco`."""

    vocab = Vocabulary()
    vocab.load(os.path.join(load_from, 'vocab.json'))

    raw_data = reader.read(data, vocab)

    inputs, targets, loss, _, _, _ = define_computation_graph(vocab.size, batch_size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # load model
        saver.restore(session, os.path.join(load_from, MODEL_FILENAME))

        total_loss = 0.0
        total_iter = 0
        for x, y in reader.iterate(raw_data, batch_size, NUM_STEPS):
            l = session.run([loss], feed_dict={inputs: x, targets: y})
            total_loss += l[0]
            total_iter += 1
        perplexity = np.exp(total_loss / total_iter)
        return perplexity
