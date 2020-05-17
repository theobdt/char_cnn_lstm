from collections import namedtuple
import re

Tokens = namedtuple("Tokens", "zero_padding unk_w eos")
tokens = Tokens(zero_padding=" ", unk_w="*", eos="|")


def replace_token(word):
    if word == "<unk>":
        return tokens.unk_w
    elif word == "</s>":
        return tokens.eos
    elif word == ".":
        return ""
    return word


def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/
    process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"``", " ", string)
    string = re.sub(r"''", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if tolower:
        string = string.lower()
    return string.strip()


def preprocess_sentence(sentence):
    replaced_sentence = [replace_token(word) for word in sentence]
    cleaned_sentence = clean_str(" ".join(replaced_sentence))
    return [tokens.eos] + cleaned_sentence.split()


if __name__ == "__main__":
    s = ["Hello", "I", "haven't!", "<unk>", ".", "</s>", "yes (no)?", "I'll"]
    print(preprocess_sentence(s))
