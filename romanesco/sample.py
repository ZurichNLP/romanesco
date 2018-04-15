#!/usr/bin/env python3

import os
import sys
import logging

import numpy as np
import tensorflow as tf

from romanesco.const import *
from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sample(length: int, load_from: str, first_symbol: str = None, **kwargs):
    """Generates a text by sampling from a trained language model.

    Arguments:
        length: the number of symbols to sample.
        load_from: the path to the folder with model parameters and vocabulary.
        first_symbol: the first symbol of the text to be generated.
    """
    vocab = Vocabulary()
    vocab.load(os.path.join(load_from, 'vocab.json'))

    inputs, targets, _, _, logits, _ = define_computation_graph(vocab.size, 1)

    saver = tf.train.Saver()

    sampled_sequence = []

    with tf.Session() as session:
        # load model
        saver.restore(session, os.path.join(load_from, MODEL_FILENAME))

        if first_symbol:
            try:
                sampled_symbol = vocab.get_id(first_symbol)
            except KeyError:
                logging.error('Unknown symbol `{0}`. Try with another start symbol.')
                sys.exit(0)
        else:
            sampled_symbol = vocab.get_random_id()

        x = np.array(np.zeros(NUM_STEPS, dtype=int)) # padding with zeros (UNK)
        y = np.array(np.zeros(NUM_STEPS, dtype=int)) # we don't care about gold targets here

        for _ in range(length):
            sampled_sequence.append(sampled_symbol)
            x = np.roll(x, -1)
            x[NUM_STEPS - 1] = sampled_symbol
            l = session.run([logits], feed_dict={inputs: [x], targets: [y]})
            next_symbol_logits = l[0][0][-1] # first returned session variable, first batch, last symbol
            next_symbol_probs = softmax(next_symbol_logits)
            sampled_symbol = np.random.choice(range(vocab.size), p=next_symbol_probs)

    words = vocab.get_words(sampled_sequence)
    return ' '.join(words)
