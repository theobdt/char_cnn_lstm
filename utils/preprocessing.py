import pickle
import shutil
import torch
from collections import namedtuple
import os
import json


Tokens = namedtuple("Tokens", "zero_padding unk_w eos")
tokens = Tokens(zero_padding=" ", unk_w="__", eos="|")


def replace_token(word):
    if word == "<unk>":
        return tokens.unk_w
    if word == "</s>":
        return tokens.eos
    return word


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
        return ids


class WordsVocabulary:
    def __init__(self):
        self.words_count = {tokens.unk_w: 0}
        self.idx2word = [tokens.unk_w]

    def add_word(self, word):
        if word in self.words_count.keys():
            self.words_count[word] += 1
        else:
            self.words_count[word] = 1

    def sort(self, max_words):
        if max_words is None:
            max_words = len(self.words_count)

        self.words_count = {
            k: v
            for k, v in sorted(
                self.words_count.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        self.idx2word = list(self.words_count.keys())[:max_words]

        # make sure that tokens.unk_w is part of the vocabulary
        if tokens.unk_w not in self.idx2word:
            self.idx2word[-1] = tokens.unk_w
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def to_idx(self, word):
        try:
            idx = self.word2idx[word]
        except KeyError:
            idx = self.word2idx[tokens.unk_w]
        return idx


def create_objects(datasets, path, max_word_length, max_words):
    assert os.path.exists(path), "path not found"

    word_vocabulary = WordsVocabulary()
    char_vocabulary = CharsVocabulary()
    max_word_len_tmp = 0

    parameters = {}

    for mode, dataset in datasets.items():
        for word in dataset:
            word = replace_token(word)
            if len(word) > max_word_len_tmp:
                max_word_len_tmp = len(word)
            word_vocabulary.add_word(word)
            char_vocabulary.add_word(word)
        parameters[f"tokens_{mode}"] = len(dataset)
    char_vocabulary.process()
    word_vocabulary.sort(max_words)

    if max_word_length:
        word_length = min(max_word_length, max_word_len_tmp)
    else:
        word_length = max_word_len_tmp
    print(f"Word length : {word_length}")
    parameters["word_length"] = word_length
    parameters["max_word_length"] = word_length

    total_tokens = sum([len(ds) for mode, ds in datasets.items()])
    print(f"Total number of words : {total_tokens}")
    print(f"Size word vocabulary: {len(word_vocabulary.idx2word)}")
    print(f"Size character vocabulary: {len(char_vocabulary.idx2char)}\n")
    parameters[f"tokens_total"] = total_tokens

    for mode, dataset in datasets.items():
        output_chars = torch.zeros(
            (len(dataset), word_length), dtype=torch.uint8
        )
        output_words = torch.zeros(len(dataset), dtype=torch.long)

        for i, word in enumerate(dataset):
            word = replace_token(word)
            output_chars[i] = char_vocabulary.to_idx(word, word_length)
            output_words[i] = word_vocabulary.to_idx(word)
        path_chars = os.path.join(path, mode, "chars.pt")
        torch.save(output_chars, path_chars)
        path_words = os.path.join(path, mode, "words.pt")
        torch.save(output_words, path_words)
        print(f"Data {mode}: tensors saved to {path_chars} and {path_words}")
    path_char_voc = os.path.join(path, "idx2char.pkl")
    with open(path_char_voc, "wb") as file:
        pickle.dump(char_vocabulary.idx2char, file)
    print(f"\nCharacter vocabulary saved to {path_char_voc}")

    path_word_voc = os.path.join(path, "idx2word.pkl")
    with open(path_word_voc, "wb") as file:
        pickle.dump(word_vocabulary.idx2word, file)
    print(f"Word vocabulary saved to {path_word_voc}")

    parameters["size_char_vocab"] = len(char_vocabulary.idx2char)
    parameters["size_word_vocab"] = len(word_vocabulary.idx2word)

    path_parameters = os.path.join(path, "parameters.json")
    with open(path_parameters, "w") as file:
        json.dump(parameters, file, indent=4, sort_keys=True)
    print(f"Parameters saved to {path_parameters}")


def initialize_dataset(name, max_word_length=None, max_words=None):
    print(f"Initializing dataset {name} ..")
    if name == "penn-treebank":
        from torchnlp.datasets import penn_treebank_dataset

        train, val, test = penn_treebank_dataset(
            train=True, dev=True, test=True
        )
    else:
        raise ValueError(f"Dataset {name} not recognized")

    root_path = os.path.join("data", name, "objects")
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path + "/train", exist_ok=True)
    os.makedirs(root_path + "/val", exist_ok=True)
    os.makedirs(root_path + "/test", exist_ok=True)

    datasets = {"train": train, "val": val, "test": test}
    create_objects(datasets, root_path, max_word_length, max_words)


def load_params(dataset):
    path = os.path.join("data", dataset, "objects/parameters.json")
    with open(path, 'r') as file:
        dictionary = json.load(file)
    Parameters = namedtuple("Parameters", sorted(dictionary))
    params = Parameters(**dictionary)

    return params


if __name__ == "__main__":
    initialize_dataset("penn-treebank", 15)
    pass
