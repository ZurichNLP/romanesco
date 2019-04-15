#!/usr/bin/env python3

EOS = '<eos>'
UNK = '<unk>'

# default values

MODEL_PATH = "model"
LOGS_PATH = "logs"

MODEL_FILENAME = 'model'
VOCAB_FILENAME = 'vocab.json'

CONFIG_FILENAME = 'config.json'

VOCAB_SIZE = 10000
BATCH_SIZE = 20

NUM_EPOCHS = 10

# num_steps and learning_rate are hardcoded here; at the moment,
# the only way to change them is to edit this file
NUM_STEPS = 35 # truncated backprop length
LEARNING_RATE = 0.0001

HIDDEN_SIZE = 1024  # RNN state size
EMBEDDING_SIZE = 256  # embedding vector size

SAMPLE_LENGTH = 100
