#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from romanesco import const


def read_words(filename: str):
    """Reads a tokenised text.

    Args:
        filename: path to tokenised text file, one sentence per line.

    Returns:
        A single list for all tokens in all sentences, with sentence boundaries
        indicated by <eos> (end of sentence).
    """
    with tf.gfile.GFile(filename) as f:
        return f.read().replace("\n", const.EOS).split()


def read(filename: str, vocab):
    """Turns a tokenised text into a list of token ids.

    Args:
        filename: path to tokenised text file, one sentence per line.
        vocab: an instance of type romanesco.vocab.Vocabulary

    Returns:
        A single list of ids for all tokens in all sentences.
    """
    words = read_words(filename)
    return [vocab.get_id(word) for word in words]


def iterator(raw_data, batch_size: int, num_steps: int):
    """Yields batches from a dataset.

    Example:

     t=0  t=1    t=2  t=3     t=4
    [The, brown, fox, is,     quick]
    [The, red,   fox, jumped, high]

    batch[0] = [the, the]
    batch[1] = [brown, red]
    batch[2] = [fox, fox]
    batch[3] = [is, jumped]
    batch[4] = [quick, high]
    batch_size = 2, time_steps = 5
    """
    data_len = len(raw_data) - 1 # because y will be x, time shifted by 1
    num_batches = data_len // batch_size // num_steps

    data = np.array(raw_data)
    # [the brown fox is quick the red fox jumped high and went]
    x = data[0 : batch_size * num_batches * num_steps]
    # [the brown fox is quick the red fox jumped high]
    y = data[1 : batch_size * num_batches * num_steps + 1] # x, time shifted by one
    # [brown fox is quick the red fox jumped high and]
    x_seqs = x.reshape(num_batches * batch_size, num_steps)
    # [[the brown fox is     quick],
    #  [the red   fox jumped high]]
    y_seqs = y.reshape(num_batches * batch_size, num_steps)
    # [[brown fox is     quick the],
    #  [red   fox jumped high  and]]

    for i in range(num_batches):
        s = i * batch_size
        e = s + batch_size
        yield x_seqs[s : e].T, y_seqs[s : e].T
        # [[the,   the]
        #  [brown, red]
        #  [fox,   fox]
        #  [is,    jumped]
        #  [quick, high]] for x; equivalent shape for y
