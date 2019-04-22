#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python.ops.functional_ops import map_fn

from romanesco import const as C


def define_computation_graph(vocab_size: int = C.VOCAB_SIZE,
                             batch_size: int = C.BATCH_SIZE,
                             num_steps: int = C.NUM_STEPS,
                             hidden_size: int = C.HIDDEN_SIZE,
                             embedding_size: int = C.EMBEDDING_SIZE):

    # Placeholders for input and output
    inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='x')  # (batch, time)
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='y') # (batch, time)

    with tf.name_scope('Embedding'):
        embedding = tf.get_variable('word_embedding', [vocab_size, embedding_size])
        input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

    with tf.name_scope('RNN'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=initial_state)

    with tf.name_scope('Final_Projection'):
        w = tf.get_variable('w', shape=(hidden_size, vocab_size))
        b = tf.get_variable('b', vocab_size)
        final_projection = lambda x: tf.matmul(x, w) + b
        logits = map_fn(final_projection, rnn_outputs)

    with tf.name_scope('Cost'):
        # weighted average cross-entropy (log-perplexity) per symbol
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=tf.ones([batch_size, num_steps]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=C.LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return inputs, targets, loss, train_step, logits, summary
