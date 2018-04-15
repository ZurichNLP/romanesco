#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco.const import *


def define_computation_graph(vocab_size: int, batch_size: int):

    # Placeholders for input and output
    inputs = tf.placeholder(tf.int32, shape=(batch_size, NUM_STEPS), name='x')  # (time, batch)
    targets = tf.placeholder(tf.int32, shape=(batch_size, NUM_STEPS), name='y') # (time, batch)

    with tf.name_scope('Embedding'):
        embedding = tf.get_variable('word_embedding', [vocab_size, HIDDEN_SIZE])
        input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

    with tf.name_scope('RNN'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=initial_state)

    with tf.name_scope('Final_Projection'):
        w = tf.get_variable('w', shape=(HIDDEN_SIZE, vocab_size))
        b = tf.get_variable('b', vocab_size)
        final_projection = lambda x: tf.matmul(x, w) + b
        logits = map_fn(final_projection, rnn_outputs)

    with tf.name_scope('Cost'):
        # weighted average cross-entropy (log-perplexity) per symbol
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=tf.ones([batch_size, NUM_STEPS]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return inputs, targets, loss, train_step, logits, summary
