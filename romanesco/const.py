#!/usr/bin/env python3

EOS = '<eos>'
UNK = '<unk>'

# Ugly hardcoded hyperparameters
NUM_STEPS = 35 # truncated backprop length
BATCH_SIZE = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 512 # layer size
