#!/usr/bin/env python3

# pylint: disable=C0103

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco import reader
from romanesco.vocab import Vocabulary

# paths
DATA_TRAIN_PATH = '/home/user/laeubli/romanesco/data/train'
LOGS_PATH = '/home/user/laeubli/romanesco/logs'

vocab = Vocabulary(DATA_TRAIN_PATH, max_size=50000)
raw_data = reader.read(DATA_TRAIN_PATH, vocab)

# hyperparams
NUM_EPOCHS = 100
NUM_STEPS = 35 # truncated backprop length
BATCH_SIZE = 20
LEARNING_RATE = 0.001

# layer sizes
VOCAB_SIZE = vocab.size
INPUT_SIZE = VOCAB_SIZE
HIDDEN_SIZE = 512
OUTPUT_SIZE = VOCAB_SIZE

# Placeholders for input and output
inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_STEPS), name='x')  # (time, batch)
targets = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_STEPS), name='y') # (time, batch)

with tf.name_scope('Embedding'):
    embedding = tf.Variable(tf.random_uniform([VOCAB_SIZE, HIDDEN_SIZE], -1.0, 1.0),
                            dtype=tf.float32, name='embedding')
    input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

with tf.name_scope('RNN'):
    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=initial_state)

with tf.name_scope('Final_Projection'):
    w = tf.get_variable("w", shape=(HIDDEN_SIZE, VOCAB_SIZE))
    b = tf.get_variable("b", VOCAB_SIZE)
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
    cost = tf.div(tf.reduce_sum(loss), BATCH_SIZE, name="cost")

with tf.name_scope('Optimizer'):
    train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


# Logging of scalars
tf.summary.scalar("cost", cost)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as session:
    # init
    session.run(tf.global_variables_initializer())
    # write logs (for Tensorboard)
    summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())

    for epoch in range(20):
        total_cost = 0.0
        total_iter = 0
        for i, (x, y) in enumerate(reader.iterate(raw_data, BATCH_SIZE, NUM_STEPS)):
            c, p, _, summary = session.run([cost, prediction, train_fn, merged_summary_op],
            feed_dict={
                inputs: x,
                targets: y,
            })
            summary_writer.add_summary(summary, i)
            total_cost += c
            total_iter += 1
            if i % 100 == 0 and i > 0:
                print("Epoch={0}, iteration={1}, perplexity={2:.2f}".format(epoch, i, np.exp(total_cost / total_iter)))
                print("Y*: {0}\nY:  {1}".format(' '.join(vocab.get_words(y[0])),
                                                ' '.join(vocab.get_words(p[0]))))
        print("Epoch={0}, perplexity={1:.2f}".format(epoch, np.exp(total_cost / total_iter)))
