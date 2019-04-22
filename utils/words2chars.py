#! /usr/bin/python3

import sys

for line in sys.stdin:
    line = line.strip()

    charred_words = []

    for word in line.split(" "):
        chars = list(word)
        charred_words.append(" ".join(chars))

    sys.stdout.write(" <space> ".join(charred_words) + "\n")
