#! /usr/bin/python3

import sys

for line in sys.stdin:

    line = line.strip()

    chars = line.split(" ")

    words = []
    word = []
    for char in chars:
        if char == "<space>":
            words.append("".join(word))
            word = []
            continue
        elif char == " ":
            continue
        word.append(char)

    sys.stdout.write(" ".join(words) + "\n")
