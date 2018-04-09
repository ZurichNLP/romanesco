#!/usr/bin/env python3

# pylint: disable=C0103

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops.functional_ops import map_fn

from romanesco import reader
from romanesco.vocab import Vocabulary

NUM_EPOCHS = 100
NUM_STEPS = 15 # truncated backprop length
BATCH_SIZE = 5
LEARNING_RATE = 0.01

# layer sizes
VOCAB_SIZE = 50000
INPUT_SIZE = VOCAB_SIZE
HIDDEN_SIZE = 512
OUTPUT_SIZE = VOCAB_SIZE

# Placeholders for input and output
inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_STEPS), name='x')  # (time, batch)
targets = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_STEPS), name='y') # (time, batch)

# Embedding layer
embedding = tf.Variable(tf.random_uniform([VOCAB_SIZE, HIDDEN_SIZE], -1.0, 1.0),
                        dtype=tf.float32, name='embedding')
input_embeddings = tf.nn.embedding_lookup(embedding, inputs)

# RNN
cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, input_embeddings, initial_state=initial_state)

# Output
w = tf.get_variable("w", shape=(HIDDEN_SIZE, VOCAB_SIZE))
b = tf.get_variable("b", VOCAB_SIZE)
final_projection = lambda x: tf.matmul(x, w) + b
logits = map_fn(final_projection, rnn_outputs)

loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                        targets=targets,
                                        weights=tf.ones([BATCH_SIZE, NUM_STEPS]))
cost = tf.div(tf.reduce_sum(loss), BATCH_SIZE, name="cost")

# Optimizer
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

training_data_filepath = '/Users/sam/Desktop/thedonald/data/train'
vocab = Vocabulary(training_data_filepath, max_size=VOCAB_SIZE)
raw_data = reader.read(training_data_filepath, vocab)

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for epoch in range(10):
    epoch_loss = 0
    for i, (x, y) in enumerate(reader.iterator(raw_data, BATCH_SIZE, NUM_STEPS)):
        epoch_loss += session.run([cost, train_fn], feed_dict={
            inputs: x,
            targets: y,
        })[0]
        if i % 100 == 0:
            print("Epoch={0}, iteration={1}, accumulated loss={2:.2f}".format(epoch, i, epoch_loss))
    print("Epoch={0}, total loss={1:.2f}".format(epoch, epoch_loss))
