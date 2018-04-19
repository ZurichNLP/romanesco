#!/usr/bin/env python3

import os
import sys
import logging

import numpy as np
import tensorflow as tf

from typing import List

from romanesco.vocab import Vocabulary
from romanesco.compgraph import define_computation_graph

from romanesco import const as C


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sample(length: int, load_from: str, first_symbols: List[str] = [], **kwargs):
    """Generates a text by sampling from a trained language model. See argument
    description in `bin/romanesco`."""

    vocab = Vocabulary()
    vocab.load(os.path.join(load_from, 'vocab.json'))

    inputs, targets, _, _, logits, _ = define_computation_graph(vocab.size, 1)

    saver = tf.train.Saver()

    sampled_sequence = []

    with tf.Session() as session:
        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        if first_symbols != []:
            try:
                sampled_symbols = [vocab.get_id(symbol) for symbol in first_symbols]
            except KeyError:
                logging.error('Unknown symbol `{0}`. Try with another symbol.')
                sys.exit(0)
        else:
            # if no prime text, then just sample a single symbol
            sampled_symbols = [vocab.get_random_id()]

        x = np.array(np.zeros(C.NUM_STEPS, dtype=int)) # padding with zeros (UNK)
        y = np.array(np.zeros(C.NUM_STEPS, dtype=int)) # we don't care about gold targets here

        UNK_ID = vocab.get_id(C.UNK)

        sampled_symbol = sampled_symbols.pop(0)

        for _ in range(length):
            sampled_sequence.append(sampled_symbol)
            x = np.roll(x, -1)
            x[C.NUM_STEPS - 1] = sampled_symbol
            l = session.run([logits], feed_dict={inputs: [x], targets: [y]})
            next_symbol_logits = l[0][0][-1] # first returned session variable, first batch, last symbol
            next_symbol_probs = softmax(next_symbol_logits)

            try:
                sampled_symbol = sampled_symbols.pop(0)
            # list of priming symbols is exhausted
            except IndexError:
                # avoid generating unknown words
                sampled_symbol = UNK_ID
                while sampled_symbol == UNK_ID: # TODO: avoid infinite loop
                    sampled_symbol = np.random.choice(range(vocab.size), p=next_symbol_probs)

    words = vocab.get_words(sampled_sequence)
    return ' '.join(words).replace(' ' + EOS + ' ', '\n') # OPTIMIZE: remove <eos> at the very end
