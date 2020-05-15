import pickle
import shutil
import torch
from collections import namedtuple
import nltk
import os
import yaml
import re
from tqdm import tqdm
from zipfile import ZipFile
import numpy as np


Tokens = namedtuple("Tokens", "zero_padding unk_w eos")
tokens = Tokens(zero_padding=" ", unk_w="__", eos="|")


def replace_token(word):
    if word == "<unk>":
        return tokens.unk_w
    elif word == "</s>":
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

    def set_idx2char(self, idx2char):
        assert idx2char[0] == tokens.zero_padding
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
        return ids

    def to_chars(self, idx_tensor):
        chars = []
        for idx in idx_tensor:
            chars.append(self.idx2char[idx])
        return chars

    # def encode_word(self, word, word_length):
    # encoded = torch.zeros(word_length)
    # for i in range(min(len(word), word_length)):
    # encoded[i] = self.to_idx(word[i])

    # return encoded.long()


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
        assert idx2word[0] == tokens.eos
        assert idx2word[1] == tokens.unk_w
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


def create_objects(
    corpus, datasets, path, max_word_length, max_words, compute_tensors=True
):
    assert os.path.exists(path), "path not found"

    word_vocabulary = WordsVocabulary()
    char_vocabulary = CharsVocabulary()
    max_word_len_tmp = 0

    parameters = {"corpus": corpus}
    print(f"Corpus : {corpus}")

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

    path_char_voc = os.path.join(path, "idx2char.pkl")
    with open(path_char_voc, "wb") as file:
        pickle.dump(char_vocabulary.idx2char, file)
    print(f"\nCharacter vocabulary saved to {path_char_voc}")

    path_word_voc = os.path.join(path, "idx2word.pkl")
    with open(path_word_voc, "wb") as file:
        pickle.dump(word_vocabulary.idx2word, file)
    print(f"Word vocabulary saved to {path_word_voc}")

    parameters["char_vocab_size"] = len(char_vocabulary.idx2char)
    parameters["word_vocab_size"] = len(word_vocabulary.idx2word)

    path_parameters = os.path.join(path, "data.yaml")
    save_params(parameters, path_parameters)
    print(f"Parameters saved to {path_parameters}")

    if compute_tensors:
        print("Computing tensors ..")
        for mode, dataset in datasets.items():
            output_chars = torch.zeros(
                (len(dataset), word_length), dtype=torch.uint8
            )
            output_words = torch.zeros(len(dataset), dtype=torch.long)

            for i, word in enumerate(tqdm(dataset, desc=mode)):
                word = replace_token(word)
                output_chars[i] = char_vocabulary.to_idx(word, word_length)
                output_words[i] = word_vocabulary.to_idx(word)
            path_chars = os.path.join(path, mode, "chars.pt")
            torch.save(output_chars, path_chars)
            path_words = os.path.join(path, mode, "words.pt")
            torch.save(output_words, path_words)
            print(f"Tensors {mode} saved to {path_chars} and {path_words}\n")
    return char_vocabulary, word_vocabulary


def initialize_dataset(
    corpus_name,
    max_word_length=None,
    max_words=None,
    data_splits=[0.7, 0.2, 0.1],
):
    print(f"Initializing dataset ..")
    assert abs(sum(data_splits) - 1) < 0.0001
    if corpus_name == "penn-treebank":
        nltk.download("treebank", "data/penn-treebank")
        from nltk.corpus import treebank as corpus

    elif corpus_name == "brown":
        nltk.download("brown", "data/brown")
        from nltk.corpus import brown as corpus

    else:
        raise ValueError(f"Corpus {corpus_name} not supported")

    # data preprocessing
    all_words = corpus.words()
    n_words = len(all_words)
    split_n = [int(s * n_words) for s in np.cumsum(data_splits)]
    train = clean_str(" ".join(all_words[: split_n[0]]))
    val = clean_str(" ".join(all_words[split_n[0] : split_n[1]]))
    test = clean_str(" ".join(all_words[split_n[1] :]))
    datasets = {
        "train": train.split(),
        "val": val.split(),
        "test": test.split(),
    }

    root_path = os.path.join("data", corpus_name, "objects")
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path + "/train", exist_ok=True)
    os.makedirs(root_path + "/val", exist_ok=True)
    os.makedirs(root_path + "/test", exist_ok=True)

    return create_objects(
        corpus_name, datasets, root_path, max_word_length, max_words
    )


def load_params(path):
    with open(path, "r") as file:
        dictionary = yaml.safe_load(file)

    return dictionary


def save_params(dictionary, path):
    with open(path, "w") as file:
        yaml.dump(dictionary, file)


def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/
    process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if tolower:
        string = string.lower()
    return string.strip()


def zipfolder(path_folder, name_zip):
    assert os.listdir(path_folder)
    last_folder = path_folder.split("/")[-1]
    files = os.listdir(path_folder)
    with ZipFile(name_zip, "w") as zipf:
        for file in files:
            path_file = os.path.join(path_folder, file)
            path_arc = os.path.join(last_folder, file)
            zipf.write(path_file, path_arc)


if __name__ == "__main__":
    initialize_dataset("penn-treebank", 15)
    pass
