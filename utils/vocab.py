import torch
import pickle
import numpy as np

from utils.preprocessing import tokens


class CharsVocabulary:
    def __init__(self):
        self.idx2char = [tokens.zero_padding]

    def add_word(self, word):
        for char in word:
            if char not in self.idx2char:
                self.idx2char.append(char)

    def process(self):
        self.idx2char = list(sorted(self.idx2char, key=lambda c: ord(c)))
        self.char2idx = {char: i for i, char in enumerate(self.idx2char)}

    def set_idx2char(self, idx2char):
        # assert idx2char[0] == tokens.zero_padding
        self.idx2char = idx2char
        self.char2idx = {char: i for i, char in enumerate(self.idx2char)}

    def load(self, path):
        with open(path, "rb") as file:
            self.set_idx2char(pickle.load(file))

    def to_idx(self, word, word_len=None):
        if not word_len:
            word_len = len(word)
        ids = torch.zeros(word_len)
        for i, char in enumerate(word[:word_len]):
            try:
                ids[i] = self.char2idx[char]
            except KeyError:
                print(f"ERROR: Character '{char}' not recognized")
                return
        return ids.long()

    def to_chars(self, idx_tensor):
        chars = []
        for idx in idx_tensor:
            chars.append(self.idx2char[idx])
        return chars


class WordsVocabulary:
    def __init__(self):
        self.words_count = {}
        self.idx2word = [tokens.eos, tokens.unk_w]

    def add_word(self, word):
        if word in self.idx2word:
            return
        elif word in self.words_count.keys():
            self.words_count[word] += 1
        else:
            self.words_count[word] = 1

    def sort(self, max_words=None):
        if not max_words:
            max_words = len(self.words_count)

        self.words_count = {
            k: v
            for k, v in sorted(
                self.words_count.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        self.idx2word += list(self.words_count.keys())[:max_words]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def set_idx2word(self, idx2word):
        # assert idx2word[0] == tokens.eos
        # assert idx2word[1] == tokens.unk_w
        self.idx2word = idx2word
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def load(self, path):
        with open(path, "rb") as file:
            self.set_idx2word(pickle.load(file))

    def to_idx(self, word):
        try:
            idx = self.word2idx[word]
        except KeyError:
            idx = self.word2idx[tokens.unk_w]
        return idx

    def to_word(self, idx):
        return self.idx2word[idx]

    def output2word(self, output, n_best=1, no_unk=True):
        # output shape : (batch_size, seq_len, hidden_size)
        scores_idx = np.argsort(output[0, -1, :].numpy())[::-1]
        if no_unk:
            scores_idx = scores_idx[scores_idx != 1]

        best_words = [self.to_word(idx) for idx in scores_idx[:n_best]]
        return best_words
