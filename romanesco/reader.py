#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from romanesco import const as C


def read_words(filename: str):
    """Reads a tokenised text.

    Args:
        filename: path to tokenised text file, one sentence per line.

    Returns:
        A single list for all tokens in all sentences, with sentence boundaries
        indicated by <eos> (end of sentence).
    """
    with tf.gfile.GFile(filename) as f:
        return f.read().replace("\n", " " + C.EOS + " ").split()


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


def iterate(raw_data, batch_size: int, num_steps: int):
    """Yields sequences of length `num_step` for RNN training (or evaluation),
    in batches of size `batch_size`.

    Args:
        raw_data: the dataset (a list of numbers).
        batch_size: the batch size
        num_steps: number of time steps per example

    Yields:
        an (x, y) tuple, with x corresponding to inputs and y to expected
        outputs. y is x time shifted by one: y_0 = x_1, y_1 = x_2, etc. Both x
        and y are NumPy arrays of shape (num_steps, batch_size).

    Example:
        >>> raw_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        >>> i = iterate(raw_data, batch_size=3, num_steps=2)
        >>> batches = list(i)
        >>> len(batches)
        2
        >>> batches[0]
        ( [[ 0,  1],  [[ 1,  2],
           [ 5,  6],   [ 6,  7],
           [10, 11]],  [11, 12]] )
        >>> batches[1]
        ( [[ 2,  3],  [[ 3,  4],
           [ 7,  8],   [ 8,  9],
           [12, 13]]   [13, 14]] )
    """
    data_len = len(raw_data)
    num_batches = data_len // batch_size

    data = raw_data[0 : batch_size * num_batches] # cut off
    data = np.reshape(data, [batch_size, num_batches])

    # raw_data = [the cat sits on the mat and eats a tasty little tuna fish .]
    # data = [[the cat   sits   on  ]
    #         [the mat   and    eats]
    #         [a   tasty little tuna]]  with batch_size = 3

    num_batches_in_epoch = (num_batches - 1) // num_steps
    # -1 because y will be x, time shifted by 1

    for i in range(num_batches_in_epoch):
        s = i * num_steps # start
        e = s + num_steps # end
        yield data[:, s : e], data[:, s + 1 : e + 1]

        # ( [[the cat  ],  [[cat   sits],
        #    [the mat  ],   [mat   and ],
        #    [a   tasty]],  [tasty little]] )
