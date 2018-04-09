#!/usr/bin/env python3

from typing import List
from collections import Counter

from romanesco import const
from romanesco.reader import read_words


class Vocabulary:

    def __init__(self, filename):
        """Builds a vocabulary mapping words (tokens) to ids (integers) and vice
        versa. The more frequent a word, the lower its id. 0 is reserved for
        unknown words.

        Args:
            filename: path to tokenised text file, one sentence per line.
        """
        words = read_words(filename)
        word_counts = Counter(words)
        sorted_words = [word for word, _ in word_counts.most_common() if word != const.UNK]
        sorted_words = [const.UNK] + sorted_words
        self._id = {} # {word: id}
        self._word = {} # {id: word}
        for i, word in enumerate(sorted_words):
            self._id[word] = i
            self._word[i] = word

    @property
    def size(self):
        return len(self._id)

    def get_id(self, word: str):
        try:
            return self._id[word]
        except KeyError:
            return const.UNK

    def get_word(self, id: int):
        return self._word[id]

    def get_ids(self, words: List[str]):
        return [self.get_id(word) for word in words]

    def get_words(self, ids: List[int]):
        return [self.get_word(id) for id in ids]
