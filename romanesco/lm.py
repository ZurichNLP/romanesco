#!/usr/bin/env python3

# pylint: disable=C0103

import os
import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco import reader
from romanesco.vocab import Vocabulary


NUM_STEPS = 35 # truncated backprop length
BATCH_SIZE = 20


def define_computation_graph(vocab_size):

    # Ugly hardcoded hyperparams
    LEARNING_RATE = 0.001
    VOCAB_SIZE = vocab_size # layer size
    INPUT_SIZE = VOCAB_SIZE # layer size
    HIDDEN_SIZE = 512 # layer size
    OUTPUT_SIZE = VOCAB_SIZE # layer size

    # Placeholders for input and output
    inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_STEPS), name='x')  # (time, batch)
    targets = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_STEPS), name='y') # (time, batch)

    with tf.name_scope('Embedding'):
        embedding = tf.get_variable('word_embedding', [VOCAB_SIZE, HIDDEN_SIZE])
        input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

    with tf.name_scope('RNN'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
        initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=initial_state)

    with tf.name_scope('Final_Projection'):
        w = tf.get_variable('w', shape=(HIDDEN_SIZE, VOCAB_SIZE))
        b = tf.get_variable('b', VOCAB_SIZE)
        final_projection = lambda x: tf.matmul(x, w) + b
        logits = map_fn(final_projection, rnn_outputs)

    with tf.name_scope('Prediction'):
        prediction = tf.argmax(logits, 2)

    with tf.name_scope('Cost'):
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=tf.ones([BATCH_SIZE, NUM_STEPS]),
                                                average_across_timesteps=False,
                                                average_across_batch=True)
        cost = tf.div(tf.reduce_sum(loss), BATCH_SIZE, name='cost')

    with tf.name_scope('Optimizer'):
        train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('cost', cost)
    summary = tf.summary.merge_all()

    return inputs, targets, train_fn, cost, prediction, summary


def train(data: str, epochs: int = 10, save_to: str = 'model', log_to: str = 'logs', **kwargs):
    """Trains a language model.

    Arguments:
        data: the path to a plain text file containing all training data.
        num_epochs: number of times to iterate over all training data.
        save_to: the path to a folder where model parameters and vocabulary will
            be stored. Folder will be created if it doesn't exist yet.
        log_to: the path to a folder where all training logs will be stored.
            Folder will be created if it doesn't exist yet. Point tensorboard
            to this folder to monitor training.
    """
    vocab = Vocabulary(data, max_size=50000)
    raw_data = reader.read(data, vocab)

    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    inputs, targets, train_fn, cost, prediction, summary = define_computation_graph(vocab.size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())

        for epoch in range(1, epochs + 1):
            total_cost = 0.0
            total_iter = 0
            for x, y in reader.iterate(raw_data, BATCH_SIZE, NUM_STEPS):
                c, p, _, s = session.run([cost, prediction, train_fn, summary],
                                         feed_dict={
                                             inputs: x,
                                             targets: y,
                                         })
                summary_writer.add_summary(s, total_iter)
                total_cost += c
                total_iter += 1
                if total_iter % 100 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)
            perplexity = np.exp(total_cost / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, 'model.epoch-{0}'.format(epoch)))
